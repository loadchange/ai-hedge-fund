from __future__ import annotations

import logging
import os
import time

import requests

from src.data.models import (
    FinancialMetrics,
    FinancialMetricsResponse,
    Price,
    PriceResponse,
)
from .base import DataSource, get_proxy_dict

logger = logging.getLogger(__name__)


class FinancialDatasetsSource(DataSource):
    """Adapter for financialdatasets.ai API (US stocks only)."""

    @property
    def name(self) -> str:
        return "financialdatasets"

    def _headers(self) -> dict:
        api_key = os.environ.get("FINANCIAL_DATASETS_API_KEY")
        return {"X-API-KEY": api_key} if api_key else {}

    def _request(self, url: str, max_retries: int = 3) -> requests.Response | None:
        for attempt in range(max_retries + 1):
            response = requests.get(url, headers=self._headers(), proxies=get_proxy_dict())
            if response.status_code == 429 and attempt < max_retries:
                time.sleep(60 + 30 * attempt)
                continue
            if response.status_code == 200:
                return response
            return None
        return None

    def get_prices(
        self, ticker: str, start_date: str, end_date: str
    ) -> list[Price]:
        url = (
            f"https://api.financialdatasets.ai/prices/"
            f"?ticker={ticker}&interval=day&interval_multiplier=1"
            f"&start_date={start_date}&end_date={end_date}"
        )
        resp = self._request(url)
        if not resp:
            return []
        try:
            prices = PriceResponse(**resp.json()).prices
            for p in prices:
                p.source = self.name
            return prices
        except Exception as e:
            logger.warning("financialdatasets price parse error for %s: %s", ticker, e)
            return []

    def get_financial_metrics(
        self, ticker: str, end_date: str, period: str = "ttm", limit: int = 10
    ) -> list[FinancialMetrics]:
        url = (
            f"https://api.financialdatasets.ai/financial-metrics/"
            f"?ticker={ticker}&report_period_lte={end_date}"
            f"&limit={limit}&period={period}"
        )
        resp = self._request(url)
        if not resp:
            return []
        try:
            metrics = FinancialMetricsResponse(**resp.json()).financial_metrics
            for m in metrics:
                m.source = self.name
            return metrics
        except Exception as e:
            logger.warning("financialdatasets metrics parse error for %s: %s", ticker, e)
            return []
