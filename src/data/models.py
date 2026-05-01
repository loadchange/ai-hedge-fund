from pydantic import BaseModel


class Price(BaseModel):
    open: float
    close: float
    high: float
    low: float
    volume: int
    time: str
    source: str | None = None


class PriceResponse(BaseModel):
    ticker: str
    prices: list[Price]


class FinancialMetrics(BaseModel):
    ticker: str
    report_period: str
    period: str
    currency: str
    market_cap: float | None
    enterprise_value: float | None
    price_to_earnings_ratio: float | None
    price_to_book_ratio: float | None
    price_to_sales_ratio: float | None
    enterprise_value_to_ebitda_ratio: float | None
    enterprise_value_to_revenue_ratio: float | None
    free_cash_flow_yield: float | None
    peg_ratio: float | None
    gross_margin: float | None
    operating_margin: float | None
    net_margin: float | None
    return_on_equity: float | None
    return_on_assets: float | None
    return_on_invested_capital: float | None
    asset_turnover: float | None
    inventory_turnover: float | None
    receivables_turnover: float | None
    days_sales_outstanding: float | None
    operating_cycle: float | None
    working_capital_turnover: float | None
    current_ratio: float | None
    quick_ratio: float | None
    cash_ratio: float | None
    operating_cash_flow_ratio: float | None
    debt_to_equity: float | None
    debt_to_assets: float | None
    interest_coverage: float | None
    revenue_growth: float | None
    earnings_growth: float | None
    book_value_growth: float | None
    earnings_per_share_growth: float | None
    free_cash_flow_growth: float | None
    operating_income_growth: float | None
    ebitda_growth: float | None
    payout_ratio: float | None
    earnings_per_share: float | None
    book_value_per_share: float | None
    free_cash_flow_per_share: float | None
    source: str | None = None


class FinancialMetricsResponse(BaseModel):
    financial_metrics: list[FinancialMetrics]


class LineItem(BaseModel):
    ticker: str
    report_period: str
    period: str
    currency: str

    # Allow additional fields dynamically
    model_config = {"extra": "allow"}


class LineItemResponse(BaseModel):
    search_results: list[LineItem]


class InsiderTrade(BaseModel):
    ticker: str
    issuer: str | None
    name: str | None
    title: str | None
    is_board_director: bool | None
    transaction_date: str | None
    transaction_shares: float | None
    transaction_price_per_share: float | None
    transaction_value: float | None
    shares_owned_before_transaction: float | None
    shares_owned_after_transaction: float | None
    security_title: str | None
    filing_date: str


class InsiderTradeResponse(BaseModel):
    insider_trades: list[InsiderTrade]


class CompanyNews(BaseModel):
    ticker: str
    title: str
    author: str | None = None
    source: str
    date: str
    url: str
    sentiment: str | None = None


class CompanyNewsResponse(BaseModel):
    news: list[CompanyNews]


class CompanyFacts(BaseModel):
    ticker: str
    name: str
    cik: str | None = None
    industry: str | None = None
    sector: str | None = None
    category: str | None = None
    exchange: str | None = None
    is_active: bool | None = None
    listing_date: str | None = None
    location: str | None = None
    market_cap: float | None = None
    number_of_employees: int | None = None
    sec_filings_url: str | None = None
    sic_code: str | None = None
    sic_industry: str | None = None
    sic_sector: str | None = None
    website_url: str | None = None
    weighted_average_shares: int | None = None


class CompanyFactsResponse(BaseModel):
    company_facts: CompanyFacts


class EarningsData(BaseModel):
    """Financial data for a single earnings period (quarterly or annual).

    Returned as the ``quarterly`` / ``annual`` payload nested inside ``Earnings``.
    """

    model_config = {"extra": "ignore"}

    # Actuals vs. estimates
    revenue: float | None = None
    estimated_revenue: float | None = None
    revenue_surprise: str | None = None  # "BEAT" | "MISS" | "MEET"
    earnings_per_share: float | None = None
    estimated_earnings_per_share: float | None = None
    eps_surprise: str | None = None

    # Income statement
    net_income: float | None = None
    gross_profit: float | None = None
    operating_income: float | None = None
    weighted_average_shares: float | None = None
    weighted_average_shares_diluted: float | None = None
    free_cash_flow: float | None = None

    # Balance sheet
    cash_and_equivalents: float | None = None
    total_debt: float | None = None
    total_assets: float | None = None
    total_liabilities: float | None = None
    shareholders_equity: float | None = None

    # Cash flow
    net_cash_flow_from_operations: float | None = None
    capital_expenditure: float | None = None
    net_cash_flow_from_investing: float | None = None
    net_cash_flow_from_financing: float | None = None
    change_in_cash_and_equivalents: float | None = None

    # Period-over-period changes (omitted by API when null)
    revenue_chg: float | None = None
    net_income_chg: float | None = None
    operating_income_chg: float | None = None
    gross_profit_chg: float | None = None
    free_cash_flow_chg: float | None = None


class Earnings(BaseModel):
    """Earnings response wrapping a quarterly + annual snapshot per ticker."""

    model_config = {"extra": "ignore"}

    ticker: str
    report_period: str
    fiscal_period: str | None = None
    currency: str | None = None
    quarterly: EarningsData | None = None
    annual: EarningsData | None = None


class Filing(BaseModel):
    """SEC filing metadata returned by /filings."""

    model_config = {"extra": "ignore"}

    ticker: str | None = None
    cik: str | None = None
    accession_number: str | None = None
    filing_type: str | None = None
    filing_date: str | None = None
    report_period: str | None = None
    document_count: int | None = None
    is_xbrl: bool | None = None
    url: str | None = None


class AnalystEstimate(BaseModel):
    """Single analyst estimate from /analyst-estimates."""

    model_config = {"extra": "ignore"}

    fiscal_period: str | None = None
    period: str | None = None
    revenue: float | None = None
    earnings_per_share: float | None = None


class Position(BaseModel):
    cash: float = 0.0
    shares: int = 0
    ticker: str


class Portfolio(BaseModel):
    positions: dict[str, Position]  # ticker -> Position mapping
    total_cash: float = 0.0


class AnalystSignal(BaseModel):
    signal: str | None = None
    confidence: float | None = None
    reasoning: dict | str | None = None
    max_position_size: float | None = None  # For risk management signals


class TickerAnalysis(BaseModel):
    ticker: str
    analyst_signals: dict[str, AnalystSignal]  # agent_name -> signal mapping


class AgentStateData(BaseModel):
    tickers: list[str]
    portfolio: Portfolio
    start_date: str
    end_date: str
    ticker_analyses: dict[str, TickerAnalysis]  # ticker -> analysis mapping


class AgentStateMetadata(BaseModel):
    show_reasoning: bool = False
    model_config = {"extra": "allow"}
