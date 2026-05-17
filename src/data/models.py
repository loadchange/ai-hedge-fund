from pydantic import BaseModel


class Price(BaseModel):
    open: float
    close: float
    high: float
    low: float
    volume: int
    time: str
    source: str | None = None


class FinancialMetrics(BaseModel):
    """Free providers fill what they can — every numeric field defaults to None.

    Agents must guard for missing data (``if metric.gross_margin is not
    None: …``) since each free provider exposes a different subset.
    """

    ticker: str
    report_period: str
    period: str
    currency: str
    market_cap: float | None = None
    enterprise_value: float | None = None
    price_to_earnings_ratio: float | None = None
    price_to_book_ratio: float | None = None
    price_to_sales_ratio: float | None = None
    enterprise_value_to_ebitda_ratio: float | None = None
    enterprise_value_to_revenue_ratio: float | None = None
    free_cash_flow_yield: float | None = None
    peg_ratio: float | None = None
    gross_margin: float | None = None
    operating_margin: float | None = None
    net_margin: float | None = None
    return_on_equity: float | None = None
    return_on_assets: float | None = None
    return_on_invested_capital: float | None = None
    asset_turnover: float | None = None
    inventory_turnover: float | None = None
    receivables_turnover: float | None = None
    days_sales_outstanding: float | None = None
    operating_cycle: float | None = None
    working_capital_turnover: float | None = None
    current_ratio: float | None = None
    quick_ratio: float | None = None
    cash_ratio: float | None = None
    operating_cash_flow_ratio: float | None = None
    debt_to_equity: float | None = None
    debt_to_assets: float | None = None
    interest_coverage: float | None = None
    revenue_growth: float | None = None
    earnings_growth: float | None = None
    book_value_growth: float | None = None
    earnings_per_share_growth: float | None = None
    free_cash_flow_growth: float | None = None
    operating_income_growth: float | None = None
    ebitda_growth: float | None = None
    payout_ratio: float | None = None
    earnings_per_share: float | None = None
    book_value_per_share: float | None = None
    free_cash_flow_per_share: float | None = None
    source: str | None = None


class LineItem(BaseModel):
    """One fiscal-period snapshot of a company's reported financials.

    Every numeric column is declared up front and defaults to ``None`` so
    that any data source (yfinance / akshare / tencent / future sources)
    can fill the subset it knows about and agents can do
    ``if item.revenue: …`` on every field without crashing on missing
    keys.  Treat this as the authoritative LineItem schema — adding a
    new field here is the right way to expose it to agents.
    """

    ticker: str
    report_period: str
    period: str
    currency: str

    # Income statement
    revenue: float | None = None
    cost_of_revenue: float | None = None
    gross_profit: float | None = None
    operating_income: float | None = None
    operating_expense: float | None = None
    research_and_development: float | None = None
    depreciation_and_amortization: float | None = None
    interest_expense: float | None = None
    net_income: float | None = None
    ebit: float | None = None
    ebitda: float | None = None
    earnings_per_share: float | None = None

    # Balance sheet
    total_assets: float | None = None
    current_assets: float | None = None
    cash_and_equivalents: float | None = None
    inventory: float | None = None
    accounts_receivable: float | None = None
    total_liabilities: float | None = None
    current_liabilities: float | None = None
    accounts_payable: float | None = None
    total_debt: float | None = None
    long_term_debt: float | None = None
    shareholders_equity: float | None = None
    outstanding_shares: float | None = None
    working_capital: float | None = None
    goodwill_and_intangible_assets: float | None = None
    intangible_assets: float | None = None
    book_value_per_share: float | None = None

    # Cash flow
    operating_cash_flow: float | None = None
    operating_cash_flow_per_share: float | None = None
    capital_expenditure: float | None = None
    free_cash_flow: float | None = None
    dividends_and_other_cash_distributions: float | None = None
    issuance_or_purchase_of_equity_shares: float | None = None

    # Ratios / derived
    gross_margin: float | None = None
    operating_margin: float | None = None
    net_margin: float | None = None
    debt_to_equity: float | None = None
    return_on_equity: float | None = None
    return_on_assets: float | None = None
    return_on_invested_capital: float | None = None

    source: str | None = None


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


class CompanyNews(BaseModel):
    ticker: str
    title: str
    author: str | None = None
    source: str
    date: str
    url: str
    sentiment: str | None = None


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


class ChipDistribution(BaseModel):
    """A-share chip distribution snapshot (筹码分布)."""
    ticker: str
    date: str
    profit_ratio: float | None = None
    avg_cost: float | None = None
    concentration_90: float | None = None
    concentration_70: float | None = None
    cost_range_90_low: float | None = None
    cost_range_90_high: float | None = None
    cost_range_70_low: float | None = None
    cost_range_70_high: float | None = None
    source: str | None = None


class CapitalFlowRecord(BaseModel):
    """A-share daily capital flow (资金流向)."""
    ticker: str
    date: str
    close_price: float | None = None
    change_pct: float | None = None
    main_net_inflow: float | None = None
    main_net_pct: float | None = None
    super_large_net: float | None = None
    super_large_pct: float | None = None
    large_net: float | None = None
    large_pct: float | None = None
    medium_net: float | None = None
    medium_pct: float | None = None
    small_net: float | None = None
    small_pct: float | None = None
    main_net_5d: float | None = None
    main_net_10d: float | None = None
    source: str | None = None


class SectorRanking(BaseModel):
    """Industry sector ranking (板块排名)."""
    sector_name: str
    change_pct: float | None = None
    latest_price: float | None = None
    up_count: int | None = None
    down_count: int | None = None
    rank: int | None = None
    source: str | None = None


class DragonTigerRecord(BaseModel):
    """Dragon Tiger Board (龙虎榜) appearance record."""
    ticker: str
    date: str
    is_on_lhb: bool = False
    recent_count_30d: int = 0
    latest_date: str | None = None
    buy_amount: float | None = None
    sell_amount: float | None = None
    net_amount: float | None = None
    reason: str | None = None
    source: str | None = None


class AgentStateMetadata(BaseModel):
    show_reasoning: bool = False
    model_config = {"extra": "allow"}
