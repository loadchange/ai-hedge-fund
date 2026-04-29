from __future__ import annotations

import logging
from datetime import datetime

import requests

from src.data.models import Price
from .base import DataSource, classify_ticker, get_proxy_dict, normalize_ticker

logger = logging.getLogger(__name__)

_BASE_URL = "https://web.ifzq.gtimg.cn/appstock/app"


def _safe_float(val, default=None):
    if val is None:
        return default
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


class TencentSource(DataSource):
    """Adapter for Tencent Finance (ifzq.gtimg.cn) - supports US + HK stocks."""

    @property
    def name(self) -> str:
        return "tencent"

    def get_prices(
        self, ticker: str, start_date: str, end_date: str
    ) -> list[Price]:
        market = classify_ticker(ticker)
        if market == "hk":
            return self._get_hk_prices(ticker, start_date, end_date)
        return self._get_us_prices(ticker, start_date, end_date)

    def get_financial_metrics(
        self, ticker: str, end_date: str, period: str = "ttm", limit: int = 10
    ) -> list:
        # Tencent Finance does not provide historical financial metrics
        return []

    def get_realtime_quote(self, ticker: str) -> dict | None:
        """Fetch real-time quote from Tencent Finance (HK + CN stocks).

        Returns dict with: current_price, market_cap, pe_ratio, pb_ratio, dividend_yield
        Tencent qt array field mapping:
            [3]=current_price, [39]=PE, [43]=PB, [44]=market_cap(亿), [47]=dividend_yield%
        """
        market = classify_ticker(ticker)
        try:
            if market == "hk":
                symbol = normalize_ticker(ticker, "tencent").zfill(5)
                key = f"hk{symbol}"
                url = f"{_BASE_URL}/hkfqkline/get?param={key},day,,,1,qfq"
            elif market == "cn":
                symbol = normalize_ticker(ticker, "tencent")  # sh600519 / sz002594
                key = symbol
                url = f"{_BASE_URL}/fqkline/get?param={key},day,,,1,qfq"
            else:
                return None

            resp = requests.get(url, timeout=10)
            if resp.status_code != 200:
                return None

            data = resp.json()
            if data.get("code") != 0:
                return None

            qt = data.get("data", {}).get(key, {}).get("qt", {}).get(key, [])
            if len(qt) < 45:
                return None

            current_price = _safe_float(qt[3])
            market_cap_yi = _safe_float(qt[44])  # 亿元
            pe_ratio = _safe_float(qt[39])
            pb_ratio = _safe_float(qt[43])

            result = {}
            if current_price:
                result["current_price"] = current_price
            if market_cap_yi:
                currency = "HKD" if market == "hk" else "CNY"
                result["market_cap"] = market_cap_yi * 1e8  # 亿 -> 元
                result["currency"] = currency
            if pe_ratio:
                result["pe_ratio"] = pe_ratio
            if pb_ratio:
                result["pb_ratio"] = pb_ratio
            if market == "hk" and len(qt) > 47:
                dividend_yield_pct = _safe_float(qt[47])
                if dividend_yield_pct:
                    result["dividend_yield"] = dividend_yield_pct / 100.0

            return result if result else None
        except Exception as e:
            logger.warning("Tencent realtime quote error for %s: %s", ticker, e)
            return None

    # ── HK Prices ─────────────────────────────────────────────────────

    def _get_hk_prices(
        self, ticker: str, start_date: str, end_date: str
    ) -> list[Price]:
        """Fetch HK stock prices from Tencent Finance.

        API: https://web.ifzq.gtimg.cn/appstock/app/hkfqkline/get
        Response data format: [date, open, close, high, low, volume, {}, change, turnover]
        """
        try:
            symbol = normalize_ticker(ticker, "tencent")  # 9988.HK -> 9988
            # Tencent requires 5-digit HK codes: 0100 -> 00100, 9988 -> 09988
            symbol = symbol.zfill(5)
            url = (
                f"{_BASE_URL}/hkfqkline/get"
                f"?param=hk{symbol},day,{start_date},,500,qfq"
            )
            resp = requests.get(url, timeout=15)
            if resp.status_code != 200:
                logger.warning("Tencent HK price HTTP %d for %s", resp.status_code, ticker)
                return []

            data = resp.json()
            if data.get("code") != 0:
                logger.warning("Tencent HK price error for %s: %s", ticker, data.get("msg"))
                return []

            key = f"hk{symbol}"  # e.g. hk09988
            days = data.get("data", {}).get(key, {}).get("qfqday", [])
            if not days:
                return []

            start_dt = datetime.strptime(start_date, "%Y-%m-%d")
            end_dt = datetime.strptime(end_date, "%Y-%m-%d")

            prices = []
            for row in days:
                # [date, open, close, high, low, volume, {}, change, turnover]
                if len(row) < 6:
                    continue
                date_str = str(row[0])
                row_dt = datetime.strptime(date_str, "%Y-%m-%d")
                if row_dt < start_dt or row_dt > end_dt:
                    continue

                prices.append(
                    Price(
                        open=_safe_float(row[1], 0.0),
                        close=_safe_float(row[2], 0.0),
                        high=_safe_float(row[3], 0.0),
                        low=_safe_float(row[4], 0.0),
                        volume=int(_safe_float(row[5], 0)),
                        time=date_str,
                        source=self.name,
                    )
                )
            return prices
        except Exception as e:
            logger.warning("Tencent HK price error for %s: %s", ticker, e)
            return []

    # ── US Prices ─────────────────────────────────────────────────────

    def _get_us_prices(
        self, ticker: str, start_date: str, end_date: str
    ) -> list[Price]:
        """Fetch US stock prices from Tencent Finance.

        API: https://web.ifzq.gtimg.cn/appstock/app/fqkline/get
        Response data format: [date, open, close, high, low, volume]
        """
        try:
            url = (
                f"{_BASE_URL}/fqkline/get"
                f"?param=us{ticker},day,{start_date},,500,qfq"
            )
            resp = requests.get(url, timeout=15)
            if resp.status_code != 200:
                logger.warning("Tencent US price HTTP %d for %s", resp.status_code, ticker)
                return []

            data = resp.json()
            if data.get("code") != 0:
                logger.warning("Tencent US price error for %s: %s", ticker, data.get("msg"))
                return []

            key = f"us{ticker}"
            days = data.get("data", {}).get(key, {}).get("day", [])
            if not days:
                # Try qfqday as fallback
                days = data.get("data", {}).get(key, {}).get("qfqday", [])
            if not days:
                return []

            start_dt = datetime.strptime(start_date, "%Y-%m-%d")
            end_dt = datetime.strptime(end_date, "%Y-%m-%d")

            prices = []
            for row in days:
                # [date, open, close, high, low, volume]
                if len(row) < 6:
                    continue
                date_str = str(row[0])
                row_dt = datetime.strptime(date_str, "%Y-%m-%d")
                if row_dt < start_dt or row_dt > end_dt:
                    continue

                prices.append(
                    Price(
                        open=_safe_float(row[1], 0.0),
                        close=_safe_float(row[2], 0.0),
                        high=_safe_float(row[3], 0.0),
                        low=_safe_float(row[4], 0.0),
                        volume=int(_safe_float(row[5], 0)),
                        time=date_str,
                        source=self.name,
                    )
                )
            return prices
        except Exception as e:
            logger.warning("Tencent US price error for %s: %s", ticker, e)
            return []
