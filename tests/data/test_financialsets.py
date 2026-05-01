"""Live integration tests against api.financialdatasets.ai.

Skipped automatically when ``FINANCIAL_DATASETS_API_KEY`` is not set so
unit-test runs (CI without the secret, local dev) stay green.
"""

from __future__ import annotations

import os

import pytest

from src.data.sources.financialsets import FinancialDatasetsSource

requires_fd_key = pytest.mark.skipif(
    not os.environ.get("FINANCIAL_DATASETS_API_KEY"),
    reason="FINANCIAL_DATASETS_API_KEY not set",
)

TICKERS = ["AAPL", "MSFT", "NVDA", "JPM", "XOM"]
PRICE_START = "2024-01-01"
PRICE_END = "2026-04-15"


@pytest.fixture(scope="module")
def fd():
    with FinancialDatasetsSource() as client:
        yield client


@requires_fd_key
@pytest.mark.parametrize("ticker", TICKERS)
def test_prices(fd: FinancialDatasetsSource, ticker: str) -> None:
    prices = fd.get_prices(ticker, PRICE_START, PRICE_END)
    assert len(prices) > 0, f"No prices for {ticker}"
    dates = [p.time for p in prices]
    print(f"  {ticker} prices: {len(prices)} bars  [{dates[0]} → {dates[-1]}]")


@requires_fd_key
@pytest.mark.parametrize("ticker", TICKERS)
def test_financial_metrics(fd: FinancialDatasetsSource, ticker: str) -> None:
    metrics = fd.get_financial_metrics(ticker, PRICE_END, period="ttm", limit=4)
    assert len(metrics) > 0, f"No metrics for {ticker}"
    m = metrics[0]
    populated = [
        f for f in (
            "market_cap", "price_to_earnings_ratio", "return_on_equity",
            "gross_margin", "debt_to_equity", "revenue_growth",
        )
        if getattr(m, f) is not None
    ]
    periods = [m.report_period for m in metrics]
    print(f"  {ticker} metrics: {len(metrics)} periods  [{periods[-1]} → {periods[0]}]")
    print(f"    Key fields: {', '.join(populated)}")


@requires_fd_key
@pytest.mark.parametrize("ticker", TICKERS)
def test_earnings(fd: FinancialDatasetsSource, ticker: str) -> None:
    earnings = fd.get_earnings(ticker)
    assert earnings is not None, f"No earnings for {ticker}"
    print(f"  {ticker} earnings: period={earnings.fiscal_period}  report={earnings.report_period}")
    if earnings.quarterly:
        q = earnings.quarterly
        print(f"    Q: rev={q.revenue}  EPS={q.earnings_per_share}  surprise={q.eps_surprise}")


@requires_fd_key
@pytest.mark.parametrize("ticker", TICKERS)
def test_news(fd: FinancialDatasetsSource, ticker: str) -> None:
    news = fd.get_news(ticker, PRICE_END, limit=5)
    assert len(news) > 0, f"No news for {ticker}"
    sources = set(n.source for n in news if n.source)
    print(f"  {ticker} news: {len(news)} articles  sources={sources}")


@requires_fd_key
@pytest.mark.parametrize("ticker", TICKERS)
def test_insider_trades(fd: FinancialDatasetsSource, ticker: str) -> None:
    trades = fd.get_insider_trades(ticker, PRICE_END, limit=5)
    assert len(trades) > 0, f"No insider trades for {ticker}"
    names = set(t.name for t in trades if t.name)
    print(f"  {ticker} insider trades: {len(trades)} records  insiders={len(names)} unique")


@requires_fd_key
@pytest.mark.parametrize("ticker", TICKERS)
def test_company_facts(fd: FinancialDatasetsSource, ticker: str) -> None:
    facts = fd.get_company_facts(ticker)
    assert facts is not None, f"No facts for {ticker}"
    assert facts.sector is not None, f"No sector for {ticker}"
    print(f"  {ticker}: {facts.name}  sector={facts.sector}  exchange={facts.exchange}")
