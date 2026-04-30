from __future__ import annotations

import logging
import os
import threading
import time
from contextlib import contextmanager

import akshare as ak
import pandas as pd

# Global lock guarding all akshare calls. akshare ships with py-mini-racer
# (V8 isolate) for some Sina/Eastmoney endpoints that decrypt JS-based
# payloads. The V8 isolate pool is *not* thread-safe at initialization time —
# concurrent calls (e.g. langgraph fan-out across many agents on the same
# ticker) crash with `Check failed: !pool->IsInitialized()`. Serialising
# akshare access avoids the race; the perf cost is small because results are
# cached upstream after the first fetch.
_AKSHARE_LOCK = threading.Lock()

from src.data.models import FinancialMetrics, Price
from .base import DataSource, classify_ticker, get_proxy_dict, normalize_ticker


# ── Proxy bypass configuration ───────────────────────────────────────────────
# Domains that should NOT use the proxy (e.g. domestic APIs that break with proxy).
# Add entries here to skip proxy for specific hosts.
PROXY_BYPASS_DOMAINS: list[str] = [
    "money.finance.sina.com.cn",
    "stock.finance.sina.com.cn",
    "finance.sina.com.cn",
    "web.ifzq.gtimg.cn",
]


@contextmanager
def _no_proxy_for(domains: list[str]):
    """Temporarily remove proxy env vars so requests to *domains* bypass the proxy.

    Since akshare sets proxy globally via env vars and we can't control per-request
    routing, this context manager unsets the proxy vars, then restores them.
    """
    keys = ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy")
    saved = {k: os.environ.pop(k, None) for k in keys}
    try:
        yield
    finally:
        for k, v in saved.items():
            if v is not None:
                os.environ[k] = v


def _noop_tqdm(iterable, *args, **kwargs):
    """Passthrough wrapper that suppresses tqdm progress bars."""
    return iterable


# Modules that `from akshare.utils.tqdm import get_tqdm` — need per-module patching
_TQDM_MODULES = [
    "akshare.stock_fundamental.stock_finance_sina",
    "akshare.stock_fundamental.stock_ipo_review",
    "akshare.stock_fundamental.stock_ipo_tutor",
    "akshare.stock_fundamental.stock_profit_forecast_em",
    "akshare.stock_fundamental.stock_register_em",
]


@contextmanager
def _suppress_tqdm():
    """Temporarily patch akshare's tqdm with a no-op to hide progress bars."""
    import importlib
    originals = {}
    patched = []
    for mod_name in _TQDM_MODULES:
        try:
            mod = importlib.import_module(mod_name)
            if hasattr(mod, "get_tqdm"):
                originals[mod_name] = mod.get_tqdm
                mod.get_tqdm = lambda enable=True: _noop_tqdm
                patched.append(mod_name)
        except ImportError:
            pass
    try:
        yield
    finally:
        for mod_name in patched:
            mod = importlib.import_module(mod_name)
            mod.get_tqdm = originals[mod_name]

logger = logging.getLogger(__name__)


def _safe_float(val, default=None):
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return default
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


def _pct_to_decimal(val):
    """Convert percentage value (e.g. 46.9) to decimal (0.469)."""
    f = _safe_float(val)
    if f is None:
        return None
    if abs(f) > 1:
        return f / 100.0
    return f


class AkShareSource(DataSource):
    """Adapter for AkShare (supports US + HK stocks)."""

    def __init__(self):
        # AkShare uses requests internally; set proxy via env vars
        proxies = get_proxy_dict()
        if proxies:
            os.environ.setdefault("HTTP_PROXY", proxies.get("http", ""))
            os.environ.setdefault("HTTPS_PROXY", proxies.get("https", ""))

    @property
    def name(self) -> str:
        return "akshare"

    # ── Prices ──────────────────────────────────────────────────────────

    def get_prices(
        self, ticker: str, start_date: str, end_date: str
    ) -> list[Price]:
        market = classify_ticker(ticker)
        if market == "hk":
            return self._get_hk_prices(ticker, start_date, end_date)
        if market == "cn":
            return self._get_cn_prices(ticker, start_date, end_date)
        return self._get_us_prices(ticker, start_date, end_date)

    def _get_us_prices(
        self, ticker: str, start_date: str, end_date: str
    ) -> list[Price]:
        try:
            with _AKSHARE_LOCK:
                df = ak.stock_us_daily(symbol=ticker, adjust="qfq")
            if df.empty or "date" not in df.columns:
                return []

            df["date"] = pd.to_datetime(df["date"])
            start_dt = pd.to_datetime(start_date)
            end_dt = pd.to_datetime(end_date)
            df = df[(df["date"] >= start_dt) & (df["date"] <= end_dt)]

            prices = []
            for _, row in df.iterrows():
                prices.append(
                    Price(
                        open=_safe_float(row.get("open"), 0.0),
                        close=_safe_float(row.get("close"), 0.0),
                        high=_safe_float(row.get("high"), 0.0),
                        low=_safe_float(row.get("low"), 0.0),
                        volume=int(row.get("volume", 0)),
                        time=row["date"].strftime("%Y-%m-%d"),
                        source=self.name,
                    )
                )
            return prices
        except Exception as e:
            logger.debug("AkShare US price error for %s: %s", ticker, e)
            return []

    def _get_hk_prices(
        self, ticker: str, start_date: str, end_date: str
    ) -> list[Price]:
        try:
            symbol = normalize_ticker(ticker, "akshare")  # 9988.HK -> 9988
            symbol = symbol.zfill(5)  # 9988 -> 09988, 100 -> 00100
            start_fmt = start_date.replace("-", "")
            end_fmt = end_date.replace("-", "")

            # Try stock_hk_daily first, fallback to stock_hk_hist
            try:
                with _AKSHARE_LOCK:
                    df = ak.stock_hk_daily(symbol=symbol, adjust="qfq")
                if not df.empty and "date" in df.columns:
                    df["date"] = pd.to_datetime(df["date"])
                    start_dt = pd.to_datetime(start_date)
                    end_dt = pd.to_datetime(end_date)
                    df = df[(df["date"] >= start_dt) & (df["date"] <= end_dt)]
            except Exception:
                with _AKSHARE_LOCK:
                    df = ak.stock_hk_hist(
                        symbol=symbol, period="daily",
                        start_date=start_fmt, end_date=end_fmt, adjust="qfq"
                    )
                if not df.empty and "date" in df.columns:
                    df["date"] = pd.to_datetime(df["date"])
                    start_dt = pd.to_datetime(start_date)
                    end_dt = pd.to_datetime(end_date)
                    df = df[(df["date"] >= start_dt) & (df["date"] <= end_dt)]

            if df.empty or "date" not in df.columns:
                return []

            prices = []
            for _, row in df.iterrows():
                prices.append(
                    Price(
                        open=_safe_float(row.get("open"), 0.0),
                        close=_safe_float(row.get("close"), 0.0),
                        high=_safe_float(row.get("high"), 0.0),
                        low=_safe_float(row.get("low"), 0.0),
                        volume=int(row.get("volume", 0)),
                        time=row["date"].strftime("%Y-%m-%d"),
                        source=self.name,
                    )
                )
            return prices
        except Exception as e:
            logger.debug("AkShare HK price error for %s: %s", ticker, e)
            return []

    def _get_cn_prices(
        self, ticker: str, start_date: str, end_date: str
    ) -> list[Price]:
        try:
            symbol = normalize_ticker(ticker, "akshare")  # 600519 -> sh600519
            # Lock first: see _fetch_cn_indicator_with_retry for rationale.
            with _AKSHARE_LOCK, _no_proxy_for(PROXY_BYPASS_DOMAINS):
                df = ak.stock_zh_a_daily(symbol=symbol, adjust="qfq")
            if df.empty or "date" not in df.columns:
                return []

            df["date"] = pd.to_datetime(df["date"])
            start_dt = pd.to_datetime(start_date)
            end_dt = pd.to_datetime(end_date)
            df = df[(df["date"] >= start_dt) & (df["date"] <= end_dt)]

            prices = []
            for _, row in df.iterrows():
                prices.append(
                    Price(
                        open=_safe_float(row.get("open"), 0.0),
                        close=_safe_float(row.get("close"), 0.0),
                        high=_safe_float(row.get("high"), 0.0),
                        low=_safe_float(row.get("low"), 0.0),
                        volume=int(row.get("volume", 0)),
                        time=row["date"].strftime("%Y-%m-%d"),
                        source=self.name,
                    )
                )
            return prices
        except Exception as e:
            logger.debug("AkShare CN price error for %s: %s", ticker, e)
            return []

    # ── Financial Metrics ───────────────────────────────────────────────

    def get_financial_metrics(
        self, ticker: str, end_date: str, period: str = "ttm", limit: int = 10
    ) -> list[FinancialMetrics]:
        market = classify_ticker(ticker)
        if market == "hk":
            return self._get_hk_financial_metrics(ticker, end_date, limit)
        if market == "cn":
            return self._get_cn_financial_metrics(ticker, end_date, limit)
        return self._get_us_financial_metrics(ticker, end_date, limit)

    def _get_us_financial_metrics(
        self, ticker: str, end_date: str, limit: int
    ) -> list[FinancialMetrics]:
        try:
            with _AKSHARE_LOCK:
                df = ak.stock_financial_us_analysis_indicator_em(symbol=ticker)
            if df.empty:
                return []

            end_dt = pd.to_datetime(end_date)
            df["REPORT_DATE"] = pd.to_datetime(df["REPORT_DATE"], errors="coerce")
            df = df[df["REPORT_DATE"] <= end_dt].head(limit)

            metrics = []
            for _, row in df.iterrows():
                report_period = row["REPORT_DATE"].strftime("%Y-%m-%d") if pd.notna(row.get("REPORT_DATE")) else end_date
                revenue = _safe_float(row.get("OPERATE_INCOME"))
                net_profit = _safe_float(row.get("PARENT_HOLDER_NETPROFIT"))
                gross_margin = _pct_to_decimal(row.get("GROSS_PROFIT_RATIO"))
                net_margin = _pct_to_decimal(row.get("NET_PROFIT_RATIO"))

                metrics.append(
                    FinancialMetrics(
                        ticker=ticker,
                        report_period=report_period,
                        period="ttm",
                        currency=str(row.get("CURRENCY_ABBR", "USD")),
                        market_cap=None,
                        enterprise_value=None,
                        price_to_earnings_ratio=None,
                        price_to_book_ratio=None,
                        price_to_sales_ratio=None,
                        enterprise_value_to_ebitda_ratio=None,
                        enterprise_value_to_revenue_ratio=None,
                        free_cash_flow_yield=None,
                        peg_ratio=None,
                        gross_margin=gross_margin,
                        operating_margin=None,
                        net_margin=net_margin,
                        return_on_equity=_pct_to_decimal(row.get("ROE_AVG")),
                        return_on_assets=_pct_to_decimal(row.get("ROA")),
                        return_on_invested_capital=None,
                        asset_turnover=_safe_float(row.get("TOTAL_ASSETS_TR")),
                        inventory_turnover=_safe_float(row.get("INVENTORY_TR")),
                        receivables_turnover=_safe_float(row.get("ACCOUNTS_RECE_TR")),
                        days_sales_outstanding=_safe_float(row.get("ACCOUNTS_RECE_TDAYS")),
                        current_ratio=_safe_float(row.get("CURRENT_RATIO")),
                        quick_ratio=_safe_float(row.get("SPEED_RATIO")),
                        cash_ratio=None,
                        operating_cash_flow_ratio=_safe_float(row.get("OCF_LIQDEBT")),
                        debt_to_equity=None,
                        debt_to_assets=_pct_to_decimal(row.get("DEBT_ASSET_RATIO")),
                        interest_coverage=None,
                        revenue_growth=_pct_to_decimal(row.get("OPERATE_INCOME_YOY")),
                        earnings_growth=_pct_to_decimal(row.get("PARENT_HOLDER_NETPROFIT_YOY")),
                        earnings_per_share=_safe_float(row.get("BASIC_EPS")),
                        book_value_per_share=None,
                        source=self.name,
                    )
                )
            return metrics
        except Exception as e:
            logger.debug("AkShare US metrics error for %s: %s", ticker, e)
            return []

    def _get_hk_financial_metrics(
        self, ticker: str, end_date: str, limit: int
    ) -> list[FinancialMetrics]:
        """HK financial metrics from AkShare (limited data available)."""
        try:
            symbol = normalize_ticker(ticker, "akshare")
            symbol = symbol.zfill(5)  # 100 -> 00100, 9988 -> 09988
            with _AKSHARE_LOCK:
                df = ak.stock_financial_hk_analysis_indicator_em(symbol=symbol)
            if df.empty:
                return []

            metrics = []
            for _, row in df.head(limit).iterrows():
                report_period = str(row.get("REPORT_DATE", end_date))
                # Normalize report_period to YYYY-MM-DD
                if " " in report_period:
                    report_period = report_period.split(" ")[0]
                revenue = _safe_float(row.get("OPERATE_INCOME"))
                revenue_growth = _pct_to_decimal(row.get("OPERATE_INCOME_YOY"))
                earnings_growth = _pct_to_decimal(row.get("HOLDER_PROFIT_YOY"))

                metrics.append(
                    FinancialMetrics(
                        ticker=ticker,
                        report_period=report_period,
                        period="ttm",
                        currency=str(row.get("CURRENCY", "HKD")),
                        market_cap=None,
                        enterprise_value=None,
                        price_to_earnings_ratio=None,
                        price_to_book_ratio=None,
                        price_to_sales_ratio=None,
                        enterprise_value_to_ebitda_ratio=None,
                        enterprise_value_to_revenue_ratio=None,
                        free_cash_flow_yield=None,
                        peg_ratio=None,
                        gross_margin=_pct_to_decimal(row.get("GROSS_PROFIT_RATIO")),
                        operating_margin=None,
                        net_margin=_pct_to_decimal(row.get("NET_PROFIT_RATIO")),
                        return_on_equity=_pct_to_decimal(row.get("ROE_AVG")),
                        return_on_assets=_pct_to_decimal(row.get("ROA")),
                        return_on_invested_capital=_pct_to_decimal(row.get("ROIC_YEARLY")),
                        asset_turnover=None,
                        inventory_turnover=None,
                        receivables_turnover=None,
                        days_sales_outstanding=None,
                        operating_cycle=None,
                        working_capital_turnover=None,
                        current_ratio=_safe_float(row.get("CURRENT_RATIO")),
                        quick_ratio=None,
                        cash_ratio=None,
                        operating_cash_flow_ratio=_safe_float(row.get("OCF_SALES")),
                        debt_to_equity=None,
                        debt_to_assets=_pct_to_decimal(row.get("DEBT_ASSET_RATIO")),
                        interest_coverage=None,
                        revenue_growth=revenue_growth,
                        earnings_growth=earnings_growth,
                        book_value_growth=None,
                        earnings_per_share_growth=None,
                        free_cash_flow_growth=None,
                        operating_income_growth=None,
                        ebitda_growth=None,
                        payout_ratio=None,
                        earnings_per_share=_safe_float(row.get("BASIC_EPS")),
                        book_value_per_share=_safe_float(row.get("BPS")),
                        free_cash_flow_per_share=_safe_float(row.get("PER_NETCASH_OPERATE")),
                        source=self.name,
                    )
                )
            return metrics
        except Exception as e:
            logger.debug("AkShare HK metrics error for %s: %s", ticker, e)
            return []

    def _fetch_cn_indicator_with_retry(
        self, code: str, max_retries: int = 3, base_delay: float = 2.0
    ) -> pd.DataFrame:
        """Fetch CN financial indicator data with retry for transient SSL/network errors."""
        for attempt in range(max_retries):
            try:
                # Acquire the akshare lock FIRST so the tqdm-suppression patch
                # (which mutates module-level attributes) is applied serially.
                # Otherwise concurrent threads racing through _suppress_tqdm
                # save each other's noop as "original" and the restore step
                # leaves the module in patched state — or worse, leaks the bar.
                with _AKSHARE_LOCK, _no_proxy_for(PROXY_BYPASS_DOMAINS), _suppress_tqdm():
                    return ak.stock_financial_analysis_indicator(symbol=code, start_year="2020")
            except Exception as e:
                err_msg = str(e).lower()
                is_transient = any(k in err_msg for k in (
                    "ssl", "eof", "connection", "timeout", "reset", "retry",
                    "429", "rate", "too many", "proxy",
                ))
                if attempt < max_retries - 1 and is_transient:
                    delay = base_delay * (2 ** attempt)
                    logger.info("Retry %d/%d for CN indicator %s after %.1fs: %s",
                                attempt + 1, max_retries, code, delay, e)
                    time.sleep(delay)
                else:
                    raise
        return pd.DataFrame()

    def _get_cn_financial_metrics(
        self, ticker: str, end_date: str, limit: int
    ) -> list[FinancialMetrics]:
        """A-share financial metrics from AkShare (stock_financial_analysis_indicator)."""
        try:
            code = ticker.replace(".SS", "").replace(".SZ", "").replace(".ss", "").replace(".sz", "")
            if not code.isdigit():
                code = normalize_ticker(ticker, "akshare").lstrip("shsz")
            df = self._fetch_cn_indicator_with_retry(code)
            if df.empty:
                return []

            end_dt = pd.to_datetime(end_date)
            df["日期"] = pd.to_datetime(df["日期"], errors="coerce")
            df = df[df["日期"] <= end_dt].sort_values("日期", ascending=False).head(limit)

            metrics = []
            for _, row in df.iterrows():
                report_period = row["日期"].strftime("%Y-%m-%d") if pd.notna(row.get("日期")) else end_date

                metrics.append(
                    FinancialMetrics(
                        ticker=ticker,
                        report_period=report_period,
                        period="ttm",
                        currency="CNY",
                        market_cap=None,
                        enterprise_value=None,
                        price_to_earnings_ratio=None,
                        price_to_book_ratio=None,
                        price_to_sales_ratio=None,
                        enterprise_value_to_ebitda_ratio=None,
                        enterprise_value_to_revenue_ratio=None,
                        free_cash_flow_yield=None,
                        peg_ratio=None,
                        gross_margin=_pct_to_decimal(row.get("销售毛利率(%)")),
                        operating_margin=_pct_to_decimal(row.get("营业利润率(%)")),
                        net_margin=_pct_to_decimal(row.get("销售净利率(%)")),
                        return_on_equity=_pct_to_decimal(row.get("净资产收益率(%)")),
                        return_on_assets=_pct_to_decimal(row.get("总资产利润率(%)")),
                        return_on_invested_capital=None,
                        asset_turnover=_safe_float(row.get("总资产周转率(次)")),
                        inventory_turnover=_safe_float(row.get("存货周转率(次)")),
                        receivables_turnover=_safe_float(row.get("应收账款周转率(次)")),
                        days_sales_outstanding=_safe_float(row.get("应收账款周转天数(天)")),
                        operating_cycle=None,
                        working_capital_turnover=None,
                        current_ratio=_safe_float(row.get("流动比率")),
                        quick_ratio=_safe_float(row.get("速动比率")),
                        cash_ratio=_pct_to_decimal(row.get("现金比率(%)")),
                        operating_cash_flow_ratio=None,
                        debt_to_equity=None,
                        debt_to_assets=_pct_to_decimal(row.get("资产负债率(%)")),
                        interest_coverage=None,
                        revenue_growth=_pct_to_decimal(row.get("主营业务收入增长率(%)")),
                        earnings_growth=_pct_to_decimal(row.get("净利润增长率(%)")),
                        book_value_growth=_pct_to_decimal(row.get("净资产增长率(%)")),
                        earnings_per_share_growth=None,
                        free_cash_flow_growth=None,
                        operating_income_growth=None,
                        ebitda_growth=None,
                        payout_ratio=None,
                        earnings_per_share=_safe_float(row.get("每股收益_调整后(元)")),
                        book_value_per_share=_safe_float(row.get("每股净资产_调整前(元)")),
                        free_cash_flow_per_share=_safe_float(row.get("每股经营性现金流(元)")),
                        source=self.name,
                    )
                )
            return metrics
        except Exception as e:
            logger.debug("AkShare CN metrics error for %s: %s", ticker, e)
            return []
