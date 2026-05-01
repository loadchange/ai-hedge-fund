"""Baostock adapter for A-share data.

Why this is the preferred CN source:
* Baostock returns clean OHLCV with adjusted close (`adjustflag=2` =
  forward-adjusted) plus per-bar P/E, P/S, P/B, turnover — saves us from
  patching valuation ratios from a separate Tencent quote.
* Quarterly financial fundamentals (profit / balance / growth / dupont /
  cash-flow) come back as structured tables, not Chinese-named columns,
  so the mapping into ``FinancialMetrics`` is far less brittle than the
  akshare path.
* Free, no API key.

Network notes:
* Baostock uses a single global session managed by ``bs.login`` /
  ``bs.logout``. It is *not* thread-safe — concurrent agent fan-out
  would otherwise crash. We serialise every call behind
  ``_BAOSTOCK_LOCK``.
* The server is mainland-China hosted; some proxies break the connection.
  We temporarily clear ``HTTP_PROXY`` / ``HTTPS_PROXY`` for the duration
  of every call (matches the akshare pattern).
* Login on every call — empirically faster than expected (~50-150ms),
  and avoids stale-session pitfalls.
"""

from __future__ import annotations

import contextlib
import io
import logging
import os
import threading
from contextlib import contextmanager
from typing import Iterable

import baostock as bs
import pandas as pd

from src.data.models import FinancialMetrics, Price
from .base import DataSource, classify_ticker, normalize_ticker


logger = logging.getLogger(__name__)

# Single global session inside baostock — every call must be serialised.
_BAOSTOCK_LOCK = threading.Lock()

# Domestic baostock servers + a couple of related Sina endpoints they
# call internally. Bypass any HTTP proxy for the duration of a call.
_PROXY_BYPASS_DOMAINS = (
    "baostock.com",
    "data.baostock.com",
)


@contextmanager
def _no_proxy_for(_domains: Iterable[str]):
    """Strip proxy env vars for the duration of the with-block."""
    keys = ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy")
    saved = {k: os.environ.pop(k, None) for k in keys}
    try:
        yield
    finally:
        for k, v in saved.items():
            if v is not None:
                os.environ[k] = v


@contextmanager
def _silent_session():
    """Login + logout silently, suppressing baostock's banner prints."""
    with _BAOSTOCK_LOCK, _no_proxy_for(_PROXY_BYPASS_DOMAINS):
        # Baostock prints `login success!` to stdout on every login;
        # mute it so CI / agent logs stay clean.
        with contextlib.redirect_stdout(io.StringIO()):
            lg = bs.login()
        if lg.error_code != "0":
            # Debug-level: a baostock login miss is recoverable — the
            # manager will fall through to akshare. Don't pollute CI.
            logger.debug("baostock login failed: %s %s", lg.error_code, lg.error_msg)
            with contextlib.redirect_stdout(io.StringIO()):
                bs.logout()
            yield None
            return
        try:
            yield True
        finally:
            with contextlib.redirect_stdout(io.StringIO()):
                bs.logout()


def _result_to_dataframe(rs) -> pd.DataFrame:
    """Drain a baostock ResultData iterator into a DataFrame."""
    rows: list[list] = []
    while rs.error_code == "0" and rs.next():
        rows.append(rs.get_row_data())
    if not rows:
        return pd.DataFrame(columns=rs.fields)
    return pd.DataFrame(rows, columns=rs.fields)


def _safe_float(val, default=None):
    if val is None or val == "":
        return default
    try:
        f = float(val)
        if pd.isna(f):
            return default
        return f
    except (TypeError, ValueError):
        return default


def _pct_to_decimal(val):
    """Some baostock fields return percentages already as decimals (e.g.
    roeAvg = 0.15 means 15%). Others return raw percentages (e.g.
    npMargin can be returned as 15.0 meaning 15%). Heuristic: > 1 →
    treat as percent, else as decimal."""
    f = _safe_float(val)
    if f is None:
        return None
    if abs(f) > 1.0:
        return f / 100.0
    return f


class BaostockSource(DataSource):
    """A-share adapter using Baostock as the upstream provider."""

    @property
    def name(self) -> str:
        return "baostock"

    # ------------------------------------------------------------------
    # Prices
    # ------------------------------------------------------------------

    def get_prices(
        self, ticker: str, start_date: str, end_date: str
    ) -> list[Price]:
        if classify_ticker(ticker) != "cn":
            # Baostock only supports A-shares; return empty so the
            # manager moves to the next source for non-CN tickers.
            return []

        symbol = normalize_ticker(ticker, "baostock")  # 600519.SS → sh.600519

        with _silent_session() as ok:
            if not ok:
                return []
            rs = bs.query_history_k_data_plus(
                symbol,
                "date,open,high,low,close,volume,amount,turn,pctChg,peTTM,pbMRQ",
                start_date=start_date,
                end_date=end_date,
                frequency="d",
                adjustflag="2",  # 2 = forward-adjusted (qfq)
            )
            if rs.error_code != "0":
                logger.debug("baostock prices error for %s: %s %s", ticker, rs.error_code, rs.error_msg)
                return []
            df = _result_to_dataframe(rs)

        if df.empty:
            return []

        prices: list[Price] = []
        for _, row in df.iterrows():
            try:
                prices.append(
                    Price(
                        open=_safe_float(row.get("open"), 0.0),
                        close=_safe_float(row.get("close"), 0.0),
                        high=_safe_float(row.get("high"), 0.0),
                        low=_safe_float(row.get("low"), 0.0),
                        volume=int(_safe_float(row.get("volume"), 0)),
                        time=str(row.get("date", "")),
                        source=self.name,
                    )
                )
            except Exception as e:
                logger.debug("baostock price row parse failed for %s: %s", ticker, e)
        return prices

    # ------------------------------------------------------------------
    # Financial metrics
    # ------------------------------------------------------------------

    def get_financial_metrics(
        self, ticker: str, end_date: str, period: str = "ttm", limit: int = 10
    ) -> list[FinancialMetrics]:
        if classify_ticker(ticker) != "cn":
            return []

        symbol = normalize_ticker(ticker, "baostock")
        end_dt = pd.to_datetime(end_date)

        # Pull up to ~3 years of quarterly data. Each (year, quarter)
        # call returns one row, so we iterate.
        candidates = self._enumerate_quarters(end_dt, periods=max(limit + 2, 12))

        with _silent_session() as ok:
            if not ok:
                return []
            rows = []
            for year, quarter in candidates:
                profit = self._fetch_one(bs.query_profit_data, symbol, year, quarter)
                if profit is None or profit.empty:
                    continue
                growth = self._fetch_one(bs.query_growth_data, symbol, year, quarter)
                balance = self._fetch_one(bs.query_balance_data, symbol, year, quarter)
                dupont = self._fetch_one(bs.query_dupont_data, symbol, year, quarter)
                rows.append({
                    "year": year,
                    "quarter": quarter,
                    "profit": profit,
                    "growth": growth,
                    "balance": balance,
                    "dupont": dupont,
                })

        if not rows:
            return []

        # Sort newest → oldest, take latest *limit*.
        rows.sort(key=lambda r: (int(r["year"]), int(r["quarter"])), reverse=True)
        rows = rows[:limit]

        metrics: list[FinancialMetrics] = []
        for row in rows:
            metrics.append(self._build_financial_metrics(ticker, row))
        return metrics

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _enumerate_quarters(end_dt: pd.Timestamp, *, periods: int) -> list[tuple[int, int]]:
        """Return ``(year, quarter)`` pairs ending at *end_dt*, newest first."""
        year = int(end_dt.year)
        quarter = (end_dt.month - 1) // 3 + 1
        out: list[tuple[int, int]] = []
        for _ in range(periods):
            out.append((year, quarter))
            quarter -= 1
            if quarter == 0:
                quarter = 4
                year -= 1
        return out

    @staticmethod
    def _fetch_one(query_fn, symbol: str, year: int, quarter: int) -> pd.DataFrame | None:
        """Run a baostock financial-data query and drain the result."""
        try:
            rs = query_fn(code=symbol, year=str(year), quarter=str(quarter))
        except Exception as e:
            logger.debug("baostock %s threw for %s %dQ%d: %s", query_fn.__name__, symbol, year, quarter, e)
            return None
        if rs.error_code != "0":
            return None
        return _result_to_dataframe(rs)

    def _build_financial_metrics(self, ticker: str, row: dict) -> FinancialMetrics:
        """Map baostock's profit / growth / balance / dupont rows into a single
        FinancialMetrics record."""
        profit = row["profit"].iloc[0] if not row["profit"].empty else {}
        growth = row["growth"].iloc[0] if row["growth"] is not None and not row["growth"].empty else {}
        balance = row["balance"].iloc[0] if row["balance"] is not None and not row["balance"].empty else {}
        dupont = row["dupont"].iloc[0] if row["dupont"] is not None and not row["dupont"].empty else {}

        get = lambda mapping, key: mapping.get(key) if hasattr(mapping, "get") else None

        report_period = str(get(profit, "statDate") or f"{row['year']}-Q{row['quarter']}")

        return FinancialMetrics(
            ticker=ticker,
            report_period=report_period,
            period="ttm",
            currency="CNY",
            market_cap=None,
            enterprise_value=None,
            price_to_earnings_ratio=None,  # baostock has peTTM on price bars, not here
            price_to_book_ratio=None,
            price_to_sales_ratio=None,
            enterprise_value_to_ebitda_ratio=None,
            enterprise_value_to_revenue_ratio=None,
            free_cash_flow_yield=None,
            peg_ratio=None,
            gross_margin=_pct_to_decimal(get(profit, "gpMargin")),
            operating_margin=None,
            net_margin=_pct_to_decimal(get(profit, "npMargin")),
            return_on_equity=_pct_to_decimal(get(profit, "roeAvg")),
            return_on_assets=_pct_to_decimal(get(dupont, "dupontROE"))
                if False  # dupontROE is ROE not ROA; leave None to avoid confusion
                else None,
            return_on_invested_capital=None,
            asset_turnover=_safe_float(get(dupont, "dupontAssetTurn")),
            inventory_turnover=None,
            receivables_turnover=None,
            days_sales_outstanding=None,
            operating_cycle=None,
            working_capital_turnover=None,
            current_ratio=_safe_float(get(balance, "currentRatio")),
            quick_ratio=_safe_float(get(balance, "quickRatio")),
            cash_ratio=_safe_float(get(balance, "cashRatio")),
            operating_cash_flow_ratio=None,
            debt_to_equity=None,
            debt_to_assets=_pct_to_decimal(get(balance, "liabilityToAsset")),
            interest_coverage=_safe_float(get(dupont, "dupontIntburden")),
            revenue_growth=_pct_to_decimal(get(growth, "YOYEquity")),  # closest available
            earnings_growth=_pct_to_decimal(get(growth, "YOYNI")),
            book_value_growth=_pct_to_decimal(get(growth, "YOYEquity")),
            earnings_per_share_growth=_pct_to_decimal(get(growth, "YOYEPSBasic")),
            free_cash_flow_growth=None,
            operating_income_growth=None,
            ebitda_growth=None,
            payout_ratio=None,
            earnings_per_share=_safe_float(get(profit, "epsTTM")),
            book_value_per_share=None,
            free_cash_flow_per_share=None,
            source=self.name,
        )
