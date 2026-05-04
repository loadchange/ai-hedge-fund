from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING

from .base import DataSource, classify_ticker
from .yfinance_src import YFinanceSource
from .akshare_src import AkShareSource
from .tencent_src import TencentSource
from .baostock_src import BaostockSource

if TYPE_CHECKING:
    from src.data.models import Price, FinancialMetrics

logger = logging.getLogger(__name__)

# Price cross-validation threshold (percentage)
_PRICE_DIFF_THRESHOLD = 0.02  # 2%

# Cooldown for sources that hit rate limits (seconds)
_RATE_LIMIT_COOLDOWN = 300  # 5 minutes


class DataSourceManager:
    """Routes data requests to appropriate sources with fallback and cross-validation."""

    def __init__(self):
        self._sources: dict[str, DataSource] = {
            "yfinance": YFinanceSource(),
            "akshare": AkShareSource(),
            "tencent": TencentSource(),
            "baostock": BaostockSource(),
        }
        # Source priority per market. Free providers only.
        #
        # CN priority intentionally ends with yfinance: baostock +
        # akshare give the cleanest A-share data when reachable, but
        # they require routes to mainland Chinese servers that
        # GitHub-hosted runners can't always reach. yfinance pulls
        # .SS / .SZ tickers from Yahoo's international servers and
        # works from anywhere — it's the runner-side safety net so
        # CI never returns "no data" for a valid CN ticker.
        self._price_priority: dict[str, list[str]] = {
            "us": ["yfinance", "akshare"],
            "hk": ["tencent", "yfinance", "akshare"],
            "cn": ["baostock", "akshare", "yfinance"],
        }
        self._metrics_priority: dict[str, list[str]] = {
            "us": ["yfinance", "akshare"],
            "hk": ["akshare", "yfinance"],
            "cn": ["baostock", "akshare", "yfinance"],
        }
        # Track rate-limited sources: {source_name: cooldown_until_timestamp}
        self._rate_limited: dict[str, float] = {}

    def _is_rate_limited(self, source_name: str) -> bool:
        """Check if a source is in cooldown due to rate limiting."""
        until = self._rate_limited.get(source_name, 0)
        if time.time() < until:
            return True
        self._rate_limited.pop(source_name, None)
        return False

    def _mark_rate_limited(self, source_name: str):
        """Mark a source as rate-limited with cooldown."""
        self._rate_limited[source_name] = time.time() + _RATE_LIMIT_COOLDOWN
        logger.info("Source %s rate-limited, cooling down for %ds", source_name, _RATE_LIMIT_COOLDOWN)

    def _get_sources(
        self, market: str, source_type: str = "price"
    ) -> list[DataSource]:
        priority = (
            self._price_priority
            if source_type == "price"
            else self._metrics_priority
        )
        names = priority.get(market, priority["us"])
        return [
            self._sources[n]
            for n in names
            if n in self._sources and not self._is_rate_limited(n)
        ]

    # ── Prices ──────────────────────────────────────────────────────────

    def get_prices(
        self, ticker: str, start_date: str, end_date: str
    ) -> list["Price"]:
        """Fetch prices with fallback. After primary succeeds, tries one more source for cross-validation."""
        market = classify_ticker(ticker)
        sources = self._get_sources(market, "price")

        primary_result: list["Price"] = []
        all_results: dict[str, list["Price"]] = {}
        primary_done = False

        for source in sources:
            try:
                result = source.get_prices(ticker, start_date, end_date)
                if result:
                    all_results[source.name] = result
                    if not primary_result:
                        primary_result = result
                        logger.info(
                            "Prices for %s from %s: %d records",
                            ticker, source.name, len(result),
                        )
                    if primary_done:
                        # Cross-validation source succeeded, stop
                        break
                elif primary_done:
                    # Cross-validation source returned empty (e.g. rate limited), stop
                    break

                if primary_result and not primary_done:
                    # Primary done, mark it so next success/failure is the last
                    primary_done = True

            except Exception as e:
                err_msg = str(e).lower()
                if "rate" in err_msg or "429" in err_msg or "too many" in err_msg:
                    self._mark_rate_limited(source.name)
                logger.debug("Source %s failed for %s: %s", source.name, ticker, e)
                if primary_done:
                    # Cross-validation source threw error, stop
                    break

        # Cross-validate if we got data from multiple sources
        if len(all_results) > 1:
            self._cross_validate_prices(ticker, all_results)

        return primary_result

    def _cross_validate_prices(
        self, ticker: str, all_results: dict[str, list["Price"]]
    ) -> None:
        """Compare close prices across sources and log discrepancies."""
        source_names = list(all_results.keys())
        for i in range(len(source_names)):
            for j in range(i + 1, len(source_names)):
                name_a, name_b = source_names[i], source_names[j]
                prices_a = all_results[name_a]
                prices_b = all_results[name_b]

                # Build date-indexed lookup for source B
                b_by_date = {p.time: p.close for p in prices_b}

                mismatches = []
                for pa in prices_a:
                    cb = b_by_date.get(pa.time)
                    if cb and cb > 0 and pa.close > 0:
                        diff = abs(pa.close - cb) / pa.close
                        if diff > _PRICE_DIFF_THRESHOLD:
                            mismatches.append(
                                f"  {pa.time}: {name_a}={pa.close:.2f} vs {name_b}={cb:.2f} ({diff:.1%})"
                            )

                if mismatches:
                    logger.warning(
                        "Price cross-validation mismatch for %s (%s vs %s):\n%s",
                        ticker, name_a, name_b, "\n".join(mismatches[:5]),
                    )

    # ── Financial Metrics ───────────────────────────────────────────────

    def get_financial_metrics(
        self, ticker: str, end_date: str, period: str = "ttm", limit: int = 10
    ) -> list["FinancialMetrics"]:
        """Fetch financial metrics with fallback. Enriches HK metrics with Tencent real-time data."""
        market = classify_ticker(ticker)
        sources = self._get_sources(market, "metrics")

        result = []
        for source in sources:
            try:
                result = source.get_financial_metrics(ticker, end_date, period, limit)
                if result:
                    logger.info(
                        "Financial metrics for %s from %s: %d records",
                        ticker, source.name, len(result),
                    )
                    break
            except Exception as e:
                logger.warning("Source %s failed for %s metrics: %s", source.name, ticker, e)

        # Enrich HK/CN metrics with Tencent real-time valuation data
        if market in ("hk", "cn") and result:
            result = self._enrich_hk_metrics(ticker, result)

        return result

    def _enrich_hk_metrics(
        self, ticker: str, metrics: list["FinancialMetrics"]
    ) -> list["FinancialMetrics"]:
        """Enrich HK financial metrics with Tencent real-time quote data.

        Fills in: market_cap, price_to_earnings_ratio, price_to_book_ratio,
        and computes price_to_sales_ratio where possible.
        """
        tencent = self._sources.get("tencent")
        if not tencent:
            return metrics

        quote = tencent.get_realtime_quote(ticker)
        if not quote:
            return metrics

        market_cap = quote.get("market_cap")
        pe_ratio = quote.get("pe_ratio")
        pb_ratio = quote.get("pb_ratio")

        for m in metrics:
            if market_cap and not m.market_cap:
                m.market_cap = market_cap
            if pe_ratio and not m.price_to_earnings_ratio:
                m.price_to_earnings_ratio = pe_ratio
            if pb_ratio and not m.price_to_book_ratio:
                m.price_to_book_ratio = pb_ratio

            # Compute price_to_sales_ratio if we have market_cap and revenue
            # Revenue = OPERATE_INCOME (already mapped but not directly accessible here)
            # Skip for now - the agents can compute it from available data

        return metrics


# Singleton
_manager: DataSourceManager | None = None


def get_data_source_manager() -> DataSourceManager:
    global _manager
    if _manager is None:
        _manager = DataSourceManager()
    return _manager
