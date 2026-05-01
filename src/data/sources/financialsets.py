"""Financial Datasets API adapter.

Implements two interfaces:
  * ``DataSource`` — required by ``DataSourceManager`` for multi-source
    orchestration (US prices + financial metrics, with cross-source
    fallback driven by ticker classification).
  * ``DataClient`` (Protocol) — a richer single-provider surface
    (news / insider trades / company facts / earnings / market cap)
    consumed by the quantitative modules (signals, features, event_study).

Network behaviour:
  * Persistent ``requests.Session`` for connection reuse.
  * 429 retry with backoff ``(5, 15, 30)`` seconds (3 retries total).
  * Connection / DNS / timeout errors return ``None``, never raise — the
    multi-source manager treats that as "skip this provider".
  * Proxy is honoured when ``HTTP_PROXY`` / ``HTTPS_PROXY`` is set.
"""

from __future__ import annotations

import logging
import os
import time

import requests

from src.data.models import (
    CompanyFacts,
    CompanyFactsResponse,
    CompanyNews,
    CompanyNewsResponse,
    Earnings,
    FinancialMetrics,
    FinancialMetricsResponse,
    InsiderTrade,
    InsiderTradeResponse,
    Price,
    PriceResponse,
)
from .base import DataSource, get_proxy_dict

logger = logging.getLogger(__name__)


class FinancialDatasetsSource(DataSource):
    """Adapter for ``https://api.financialdatasets.ai`` (US stocks)."""

    BASE_URL = "https://api.financialdatasets.ai"
    _RETRY_DELAYS = (5, 15, 30)

    def __init__(self, api_key: str | None = None, timeout: float = 30.0) -> None:
        self._api_key = api_key or os.environ.get("FINANCIAL_DATASETS_API_KEY", "")
        self._timeout = timeout
        self._session = requests.Session()
        if self._api_key:
            self._session.headers["X-API-KEY"] = self._api_key

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def __enter__(self) -> "FinancialDatasetsSource":
        return self

    def __exit__(self, *_args) -> None:
        self.close()

    def close(self) -> None:
        self._session.close()

    @property
    def name(self) -> str:
        return "financialdatasets"

    # ------------------------------------------------------------------
    # DataSource (multi-source manager contract)
    # ------------------------------------------------------------------

    def get_prices(
        self, ticker: str, start_date: str, end_date: str
    ) -> list[Price]:
        resp = self._request(
            "GET",
            "/prices/",
            params={
                "ticker": ticker,
                "interval": "day",
                "interval_multiplier": 1,
                "start_date": start_date,
                "end_date": end_date,
            },
        )
        if resp is None:
            return []
        try:
            prices = PriceResponse(**resp.json()).prices
        except Exception as e:
            logger.warning("financialdatasets price parse error for %s: %s", ticker, e)
            return []
        for p in prices:
            p.source = self.name
        return prices

    def get_financial_metrics(
        self, ticker: str, end_date: str, period: str = "ttm", limit: int = 10
    ) -> list[FinancialMetrics]:
        resp = self._request(
            "GET",
            "/financial-metrics/",
            params={
                "ticker": ticker,
                "report_period_lte": end_date,
                "period": period,
                "limit": limit,
            },
        )
        if resp is None:
            return []
        try:
            metrics = FinancialMetricsResponse(**resp.json()).financial_metrics
        except Exception as e:
            logger.warning("financialdatasets metrics parse error for %s: %s", ticker, e)
            return []
        for m in metrics:
            m.source = self.name
        return metrics

    # ------------------------------------------------------------------
    # DataClient (richer single-provider surface)
    # ------------------------------------------------------------------

    def get_news(
        self,
        ticker: str,
        end_date: str,
        start_date: str | None = None,
        limit: int = 1000,
    ) -> list[CompanyNews]:
        params: dict = {"ticker": ticker, "end_date": end_date, "limit": limit}
        if start_date is not None:
            params["start_date"] = start_date
        resp = self._request("GET", "/news/", params=params)
        if resp is None:
            return []
        try:
            return CompanyNewsResponse(**resp.json()).news
        except Exception as e:
            logger.warning("financialdatasets news parse error for %s: %s", ticker, e)
            return []

    def get_insider_trades(
        self,
        ticker: str,
        end_date: str,
        start_date: str | None = None,
        limit: int = 1000,
    ) -> list[InsiderTrade]:
        params: dict = {"ticker": ticker, "filing_date_lte": end_date, "limit": limit}
        if start_date is not None:
            params["filing_date_gte"] = start_date
        resp = self._request("GET", "/insider-trades/", params=params)
        if resp is None:
            return []
        try:
            return InsiderTradeResponse(**resp.json()).insider_trades
        except Exception as e:
            logger.warning("financialdatasets insider parse error for %s: %s", ticker, e)
            return []

    def get_company_facts(self, ticker: str) -> CompanyFacts | None:
        resp = self._request("GET", "/company/facts/", params={"ticker": ticker})
        if resp is None:
            return None
        try:
            return CompanyFactsResponse(**resp.json()).company_facts
        except Exception as e:
            logger.warning("financialdatasets facts parse error for %s: %s", ticker, e)
            return None

    def get_earnings(self, ticker: str) -> Earnings | None:
        resp = self._request("GET", "/earnings", params={"ticker": ticker})
        if resp is None:
            return None
        try:
            payload = resp.json().get("earnings")
            return Earnings(**payload) if payload else None
        except Exception as e:
            logger.warning("financialdatasets earnings parse error for %s: %s", ticker, e)
            return None

    def get_market_cap(self, ticker: str, end_date: str) -> float | None:
        """Return market cap from company facts, falling back to financial metrics."""
        facts = self.get_company_facts(ticker)
        if facts is not None and facts.market_cap is not None:
            return facts.market_cap
        metrics = self.get_financial_metrics(ticker, end_date, limit=1)
        if metrics and metrics[0].market_cap is not None:
            return metrics[0].market_cap
        return None

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _request(
        self,
        method: str,
        path: str,
        *,
        params: dict | None = None,
    ) -> requests.Response | None:
        """HTTP request with retry on 429. Never raises."""
        url = self.BASE_URL + path
        proxies = get_proxy_dict()

        for attempt, delay in enumerate((*self._RETRY_DELAYS, None)):
            try:
                resp = self._session.request(
                    method,
                    url,
                    params=params,
                    proxies=proxies,
                    timeout=self._timeout,
                )
            except requests.exceptions.SSLError as e:
                logger.debug("SSL error via proxy on %s, retrying without proxy: %s", url, e)
                try:
                    resp = self._session.request(
                        method, url, params=params, proxies=None, timeout=self._timeout,
                    )
                except requests.exceptions.RequestException as e2:
                    logger.debug("HTTP %s %s failed after SSL fallback: %s", method, path, e2)
                    return None
            except requests.exceptions.RequestException as e:
                # DNS / connection / timeout — let the manager fall back.
                logger.debug("HTTP %s %s failed: %s", method, path, e)
                return None

            if resp.status_code == 429 and delay is not None:
                logger.info(
                    "Rate limited (429) on %s, retrying in %ds (attempt %d/%d)",
                    path, delay, attempt + 1, len(self._RETRY_DELAYS),
                )
                time.sleep(delay)
                continue

            if resp.status_code >= 400:
                logger.debug("%s %s returned HTTP %d", method, path, resp.status_code)
                return None

            return resp

        logger.warning("Rate limit exhausted after %d retries on %s", len(self._RETRY_DELAYS), path)
        return None
