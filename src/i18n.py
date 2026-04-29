"""Simple i18n module for CLI output language switching."""

import json

_current_lang = "en"

# Progress status message translations
STATUS_MESSAGES = {
    "zhCN": {
        # Fetching
        "Fetching company news": "获取公司新闻",
        "Fetching financial data": "获取财务数据",
        "Fetching financial line items": "获取财务明细",
        "Fetching financial metrics": "获取财务指标",
        "Fetching insider trades": "获取内部交易",
        "Fetching line items": "获取明细项目",
        "Fetching market cap": "获取市值",
        "Fetching price data": "获取价格数据",
        "Fetching price data and calculating volatility": "获取价格数据并计算波动率",
        "Fetching recent price data for momentum": "获取近期价格数据以计算动量",
        "Gathering comprehensive line items": "收集综合明细",
        "Gathering financial line items": "收集财务明细",
        "Getting market cap": "获取市值",
        # Analyzing
        "Aggregating signals": "汇总信号",
        "Analyzing Graham valuation": "格雷厄姆估值分析",
        "Analyzing activism potential": "激进投资潜力分析",
        "Analyzing antifragility": "反脆弱性分析",
        "Analyzing balance sheet": "资产负债表分析",
        "Analyzing balance sheet and capital structure": "资产负债表与资本结构分析",
        "Analyzing book value growth": "账面价值增长分析",
        "Analyzing business predictability": "业务可预测性分析",
        "Analyzing business quality": "业务质量分析",
        "Analyzing cash flow": "现金流分析",
        "Analyzing cash yield and valuation": "现金收益率与估值分析",
        "Analyzing competitive moat": "竞争护城河分析",
        "Analyzing consistency": "一致性分析",
        "Analyzing contrarian sentiment": "逆向情绪分析",
        "Analyzing convexity": "凸性分析",
        "Analyzing disruptive potential": "颠覆性潜力分析",
        "Analyzing downside protection": "下行保护分析",
        "Analyzing earnings stability": "盈利稳定性分析",
        "Analyzing financial health": "财务健康分析",
        "Analyzing financial strength": "财务实力分析",
        "Analyzing fragility": "脆弱性分析",
        "Analyzing fundamentals": "基本面分析",
        "Analyzing growth": "成长性分析",
        "Analyzing growth & momentum": "成长与动量分析",
        "Analyzing growth & quality": "成长与质量分析",
        "Analyzing growth and reinvestment": "成长与再投资分析",
        "Analyzing innovation-driven growth": "创新驱动增长分析",
        "Analyzing insider activity": "内部人活动分析",
        "Analyzing management actions": "管理层行为分析",
        "Analyzing management efficiency & leverage": "管理效率与杠杆分析",
        "Analyzing management quality": "管理层质量分析",
        "Analyzing margins & stability": "利润率与稳定性分析",
        "Analyzing moat strength": "护城河强度分析",
        "Analyzing price data": "价格数据分析",
        "Analyzing pricing power": "定价能力分析",
        "Analyzing profitability": "盈利能力分析",
        "Analyzing risk profile": "风险画像分析",
        "Analyzing risk-reward": "风险收益分析",
        "Analyzing sentiment": "情绪分析",
        "Analyzing skin in the game": "利益绑定分析",
        "Analyzing tail risk": "尾部风险分析",
        "Analyzing trading patterns": "交易模式分析",
        "Analyzing valuation (Fisher style)": "估值分析（费雪风格）",
        "Analyzing valuation (focus on PEG)": "估值分析（PEG）",
        "Analyzing valuation ratios": "估值比率分析",
        "Analyzing value": "价值分析",
        "Analyzing volatility": "波动率分析",
        "Analyzing volatility regime": "波动率机制分析",
        "Assessing potential to double": "翻倍潜力评估",
        "Assessing relative valuation": "相对估值评估",
        # Calculating / Generating
        "Calculating final signal": "计算最终信号",
        "Calculating intrinsic value": "计算内在价值",
        "Calculating intrinsic value & margin of safety": "计算内在价值与安全边际",
        "Calculating intrinsic value (DCF)": "计算内在价值（DCF）",
        "Calculating mean reversion": "计算均值回归",
        "Calculating momentum": "计算动量",
        "Calculating Munger-style valuation": "芒格风格估值",
        "Calculating trend signals": "计算趋势信号",
        "Calculating valuation & high-growth scenario": "估值与高增长情景",
        "Calculating volatility- and correlation-adjusted limits": "计算波动率与相关性调整限额",
        "Combining signals": "合并信号",
        "Generating trading decisions": "生成交易决策",
        "Processing analyst signals": "处理分析师信号",
        "Performing Druckenmiller-style valuation": "德鲁肯米勒风格估值",
        "Scanning for black swan signals": "扫描黑天鹅信号",
        "Statistical analysis": "统计分析",
        "Generating LLM output": "生成 LLM 输出",
        # Agent-specific generating
        "Generating Ben Graham analysis": "生成格雷厄姆分析",
        "Generating Bill Ackman analysis": "生成阿克曼分析",
        "Generating Cathie Wood analysis": "生成伍德分析",
        "Generating Charlie Munger analysis": "生成芒格分析",
        "Generating Damodaran analysis": "生成达莫达兰分析",
        "Generating Jhunjhunwala analysis": "生成琼琼瓦拉分析",
        "Generating Nassim Taleb analysis": "生成塔勒布分析",
        "Generating Pabrai analysis": "生成帕布雷分析",
        "Generating Peter Lynch analysis": "生成林奇分析",
        "Generating Phil Fisher-style analysis": "生成费雪分析",
        "Generating Stanley Druckenmiller analysis": "生成德鲁肯米勒分析",
        "Generating Warren Buffett analysis": "生成巴菲特分析",
        # Failed
        "Failed: All valuation methods zero": "失败：所有估值方法为零",
        "Failed: Insufficient financial line items": "失败：财务明细不足",
        "Failed: Market cap unavailable": "失败：市值不可用",
        "Failed: No financial metrics found": "失败：未找到财务指标",
        "Failed: No price data found": "失败：未找到价格数据",
        "Failed: No valid price data": "失败：无有效价格数据",
        "Failed: Not enough financial metrics": "失败：财务指标不足",
        # Warning
        "Warning: Insufficient price data": "警告：价格数据不足",
        "Warning: No price data found": "警告：未找到价格数据",
    },
}

# Agent display name translations
AGENT_NAMES = {
    "zhCN": {
        "Aswath Damodaran": "阿斯瓦斯·达莫达兰",
        "Ben Graham": "本杰明·格雷厄姆",
        "Bill Ackman": "比尔·阿克曼",
        "Cathie Wood": "凯茜·伍德",
        "Charlie Munger": "查理·芒格",
        "Michael Burry": "迈克尔·伯里",
        "Mohnish Pabrai": "莫尼什·帕布雷",
        "Nassim Taleb": "纳西姆·塔勒布",
        "Peter Lynch": "彼得·林奇",
        "Phil Fisher": "菲利普·费雪",
        "Rakesh Jhunjhunwala": "拉凯什·琼琼瓦拉",
        "Stanley Druckenmiller": "斯坦利·德鲁肯米勒",
        "Warren Buffett": "沃伦·巴菲特",
        "Technical Analyst": "技术分析师",
        "Fundamentals Analyst": "基本面分析师",
        "Growth Analyst": "成长分析师",
        "Growth": "成长分析师",
        "News Sentiment Analyst": "新闻情绪分析师",
        "News Sentiment": "新闻情绪分析师",
        "Sentiment Analyst": "情绪分析师",
        "Sentiment": "情绪分析师",
        "Valuation Analyst": "估值分析师",
        "Valuation": "估值分析师",
        "Risk Management": "风险管理",
        "Portfolio Manager": "投资组合经理",
    },
}

TRANSLATIONS = {
    "en": {
        # Display - main output
        "analysis_for": "Analysis for",
        "agent_analysis": "AGENT ANALYSIS",
        "trading_decision": "TRADING DECISION",
        "portfolio_summary": "PORTFOLIO SUMMARY",
        "portfolio_strategy": "Portfolio Strategy",
        "no_decisions": "No trading decisions available",
        # Table headers - agent analysis
        "agent": "Agent",
        "signal": "Signal",
        "confidence": "Confidence",
        "reasoning": "Reasoning",
        # Table headers - trading decision
        "action": "Action",
        "quantity": "Quantity",
        # Table headers - portfolio summary
        "ticker": "Ticker",
        "bullish": "Bullish",
        "bearish": "Bearish",
        "neutral": "Neutral",
        # Backtester
        "backtest_completed": "Backtest completed successfully!",
        "backtest_interrupted": "Backtest interrupted by user.",
        "partial_results": "Partial results available.",
        "could_not_generate": "Could not generate partial results: {}",
        "initial_value": "Initial Portfolio Value",
        "final_value": "Final Portfolio Value",
        "total_return": "Total Return",
        "cash_balance": "Cash Balance",
        "total_position_value": "Total Position Value",
        "total_value": "Total Value",
        "portfolio_return": "Portfolio Return",
        "benchmark_return": "Benchmark Return",
        "sharpe_ratio": "Sharpe Ratio",
        "sortino_ratio": "Sortino Ratio",
        "max_drawdown": "Max Drawdown",
        # CLI prompts
        "select_analysts": "Select your AI analysts.",
        "select_model": "Select your LLM model:",
        "select_ollama_model": "Select your Ollama model:",
        "using_model": "Using specified model:",
        "using_ollama": "Using Ollama for local LLM inference.",
        "selected_model": "Selected model:",
        "selected_ollama_model": "Selected Ollama model:",
        "model_not_found": "Model '{}' not found. Please select a model.",
        "interrupt_exiting": "Interrupt received. Exiting...",
        "custom_model_prompt": "Enter the custom model name:",
        "selected_analysts": "Selected analysts:",
        # CLI help text
        "help_tickers": "Comma-separated list of stock ticker symbols (e.g., AAPL,MSFT,GOOGL)",
        "help_analysts": "Comma-separated list of analysts to use (e.g., michael_burry,other_analyst)",
        "help_analysts_all": "Use all available analysts (overrides --analysts)",
        "help_ollama": "Use Ollama for local LLM inference",
        "help_model": "Model name to use (e.g., gpt-4o)",
        "help_lang": "Language for output (en, zhCN)",
        "help_show_reasoning": "Show reasoning from each agent",
        "help_show_agent_graph": "Show the agent graph",
        "help_initial_cash": "Initial cash position (alias: --initial-capital). Defaults to 100000.0",
        "help_margin_requirement": "Initial margin requirement ratio for shorts (e.g., 0.5 for 50%%). Defaults to 0.0",
        "help_start_date": "Start date (YYYY-MM-DD)",
        "help_end_date": "End date (YYYY-MM-DD)",
        # Progress
        "done": "Done",
        "error": "Error",
    },
    "zhCN": {
        # Display - main output
        "analysis_for": "分析",
        "agent_analysis": "智能体分析",
        "trading_decision": "交易决策",
        "portfolio_summary": "投资组合摘要",
        "portfolio_strategy": "投资策略",
        "no_decisions": "无可用交易决策",
        # Table headers - agent analysis
        "agent": "智能体",
        "signal": "信号",
        "confidence": "信心度",
        "reasoning": "推理",
        # Table headers - trading decision
        "action": "操作",
        "quantity": "数量",
        # Table headers - portfolio summary
        "ticker": "股票代码",
        "bullish": "看涨",
        "bearish": "看跌",
        "neutral": "中性",
        # Backtester
        "backtest_completed": "回测完成！",
        "backtest_interrupted": "回测被用户中断。",
        "partial_results": "部分结果可用。",
        "could_not_generate": "无法生成部分结果：{}",
        "initial_value": "初始投资组合价值",
        "final_value": "最终投资组合价值",
        "total_return": "总收益",
        "cash_balance": "现金余额",
        "total_position_value": "总持仓价值",
        "total_value": "总资产",
        "portfolio_return": "组合收益",
        "benchmark_return": "基准收益",
        "sharpe_ratio": "夏普比率",
        "sortino_ratio": "索提诺比率",
        "max_drawdown": "最大回撤",
        # CLI prompts
        "select_analysts": "选择你的 AI 分析师。",
        "select_model": "选择你的 LLM 模型：",
        "select_ollama_model": "选择你的 Ollama 模型：",
        "using_model": "使用指定模型：",
        "using_ollama": "使用 Ollama 进行本地 LLM 推理。",
        "selected_model": "已选择模型：",
        "selected_ollama_model": "已选择 Ollama 模型：",
        "model_not_found": "未找到模型 '{}'，请选择一个模型。",
        "interrupt_exiting": "收到中断信号，正在退出...",
        "custom_model_prompt": "请输入自定义模型名称：",
        "selected_analysts": "已选择分析师：",
        # CLI help text
        "help_tickers": "逗号分隔的股票代码列表 (如: AAPL,MSFT,GOOGL)",
        "help_analysts": "逗号分隔的分析师列表 (如: michael_burry,other_analyst)",
        "help_analysts_all": "使用所有可用分析师 (覆盖 --analysts)",
        "help_ollama": "使用 Ollama 进行本地 LLM 推理",
        "help_model": "使用的模型名称 (如: gpt-4o)",
        "help_lang": "输出语言 (en, zhCN)",
        "help_show_reasoning": "显示每个智能体的推理过程",
        "help_show_agent_graph": "显示智能体关系图",
        "help_initial_cash": "初始现金头寸 (别名: --initial-capital)，默认 100000.0",
        "help_margin_requirement": "做空初始保证金比例 (如: 0.5 代表 50%%)，默认 0.0",
        "help_start_date": "开始日期 (YYYY-MM-DD)",
        "help_end_date": "结束日期 (YYYY-MM-DD)",
        # Progress
        "done": "完成",
        "error": "错误",
    },
}


def set_lang(lang: str):
    """Set the current language for all output."""
    global _current_lang
    _current_lang = lang


def get_text(key: str, *args) -> str:
    """Get translated text for the given key, with optional format args."""
    text = TRANSLATIONS.get(_current_lang, TRANSLATIONS["en"]).get(key, key)
    if args:
        text = text.format(*args)
    return text


def get_lang() -> str:
    """Get the current language code."""
    return _current_lang


def get_lang_instruction() -> str:
    """Return a prompt instruction for the LLM to use the current language."""
    if _current_lang == "zhCN":
        return (
            "\n\n"
            "【语言要求】你必须使用简体中文撰写所有文本内容，包括 reasoning、summary、conclusion 等字段。"
            "禁止使用英文撰写分析内容。所有输出必须是中文。"
        )
    return ""


def translate_agent_name(name: str) -> str:
    """Translate an agent display name to the current language."""
    mapping = AGENT_NAMES.get(_current_lang, {})
    return mapping.get(name, name)


def translate_status(status: str) -> str:
    """Translate a progress status message to the current language."""
    mapping = STATUS_MESSAGES.get(_current_lang, {})
    return mapping.get(status, status)


def translate_signal(signal: str) -> str:
    """Translate signal enum (BULLISH/BEARISH/NEUTRAL) to current language."""
    key = signal.lower()
    translated = get_text(key)
    return translated if translated != key else signal


def translate_action(action: str) -> str:
    """Translate action enum (BUY/SELL/HOLD/SHORT/COVER) to current language."""
    mapping = {
        "zhCN": {
            "BUY": "买入", "SELL": "卖出", "HOLD": "持有",
            "SHORT": "做空", "COVER": "平仓",
        },
    }
    lang_map = mapping.get(_current_lang, {})
    return lang_map.get(action.upper(), action)


def _try_parse_json(s: str):
    """Try to parse JSON from a string. Returns parsed object or None."""
    if not s or not s.strip():
        return None
    stripped = s.strip()
    # Direct parse
    try:
        return json.loads(stripped)
    except (json.JSONDecodeError, TypeError):
        pass
    # Brace-matching extraction
    start = stripped.find("{")
    if start == -1:
        start = stripped.find("[")
    if start != -1:
        open_ch = stripped[start]
        close_ch = "}" if open_ch == "{" else "]"
        depth = 0
        for i, ch in enumerate(stripped[start:], start):
            if ch == open_ch:
                depth += 1
            elif ch == close_ch:
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(stripped[start:i + 1])
                    except (json.JSONDecodeError, TypeError):
                        return None
    return None


def _strip_json_syntax(s: str) -> str:
    """Remove leftover JSON key-value syntax from a string, keeping the values."""
    import re
    # Remove patterns like "key": or 'key': at the start of lines
    s = re.sub(r'^\s*"[^"]*"\s*:\s*', '', s, flags=re.MULTILINE)
    # Remove trailing commas
    s = re.sub(r',\s*$', '', s, flags=re.MULTILINE)
    # Remove standalone braces/brackets at start/end
    s = re.sub(r'^\s*[\{\[\}\]]\s*$', '', s, flags=re.MULTILINE)
    return s.strip()


def _fmt_num(val) -> str:
    """Format a numeric value: round floats, strip trailing zeros."""
    if isinstance(val, int):
        return str(val)
    if isinstance(val, float):
        if val == 0:
            return "0"
        if abs(val) >= 100:
            s = f"{val:.1f}"
        elif abs(val) >= 1:
            s = f"{val:.2f}"
        elif abs(val) >= 0.01:
            s = f"{val:.4f}"
        else:
            s = f"{val:.6f}"
        return s.rstrip("0").rstrip(".")
    return str(val)


def summarize_json_reasoning(reasoning, depth: int = 0) -> str:
    """Convert any reasoning value to a natural-language string.

    Handles: str (plain or JSON-encoded), dict, list, and nested combinations.
    Recursion is capped at depth=3 to avoid runaway expansion.
    Preserves ALL content — nothing is truncated or omitted.
    """
    if depth > 3:
        return str(reasoning)

    # ── str: try JSON parse, otherwise return as-is ──
    if isinstance(reasoning, str):
        stripped = reasoning.strip()
        if not stripped:
            return ""
        # Try to parse as JSON
        parsed = _try_parse_json(stripped)
        if parsed is not None:
            return summarize_json_reasoning(parsed, depth + 1)
        # Plain text — return as-is (no truncation)
        return reasoning

    # ── list: summarize each element ──
    if isinstance(reasoning, list):
        items = []
        for item in reasoning:
            s = summarize_json_reasoning(item, depth + 1).strip()
            if s:
                items.append(s)
        return "\n".join(items) if len(items) > 3 else "; ".join(items)

    # ── non-dict: str() it ──
    if not isinstance(reasoning, dict):
        return str(reasoning)

    # ── dict: ordered key extraction ──
    PRIORITY_KEYS = (
        "reasoning", "summary", "conclusion", "analysis", "assessment",
        "verdict", "recommendation", "rationale", "explanation",
        "overview", "description", "thesis",
        "final_analysis", "final_verdict",
    )
    DETAIL_KEYS = (
        "key_points", "key_factors", "strengths", "weaknesses",
        "risks", "catalysts", "details", "notes", "observations",
        "insights", "comments",
        "historical_growth", "growth_valuation", "margin_expansion",
        "insider_conviction", "financial_health",
    )
    SECONDARY_KEYS = (
        "signal", "confidence", "action", "note", "comment",
        "observation", "insight", "weighted_score", "score",
    )

    parts = []

    # First pass: priority keys — these carry the main message
    for key in PRIORITY_KEYS:
        if key not in reasoning:
            continue
        val = reasoning[key]
        if isinstance(val, str):
            parts.append(val)
        elif isinstance(val, (list, dict)):
            s = summarize_json_reasoning(val, depth + 1)
            if s:
                parts.append(s)

    # Second pass: detail keys — lists of points, factors, etc.
    for key in DETAIL_KEYS:
        if key not in reasoning:
            continue
        val = reasoning[key]
        if isinstance(val, list):
            items = []
            for item in val:
                s = summarize_json_reasoning(item, depth + 1).strip()
                if s:
                    items.append(s)
            if items:
                parts.append("; ".join(items))
        elif isinstance(val, str) and len(val) > 1:
            parts.append(val)
        elif isinstance(val, dict):
            s = summarize_json_reasoning(val, depth + 1)
            if s:
                parts.append(s)
        elif isinstance(val, (int, float)):
            parts.append(f"{key}: {_fmt_num(val)}")

    # Third pass: secondary keys — only if we need more content, and skip bare signal labels
    if len(" ".join(parts)) < 60:
        for key in SECONDARY_KEYS:
            if key not in reasoning:
                continue
            val = reasoning[key]
            if isinstance(val, str) and len(val) > 10:  # only substantial text, not "bullish"/"neutral"
                parts.append(val)

    # Fourth pass: catch remaining dict values (skip pure metadata keys)
    META_KEYS = ("signal", "confidence", "score", "weighted_score", "weight")
    for key, val in reasoning.items():
        if key in PRIORITY_KEYS or key in DETAIL_KEYS or key in SECONDARY_KEYS:
            continue
        if isinstance(val, dict):
            s = summarize_json_reasoning(val, depth + 1)
            if s and len(s) > 1:
                parts.append(s)
        elif isinstance(val, str) and len(val) > 3 and key not in META_KEYS:
            parts.append(val)

    result = " ".join(parts)
    # If still nothing, try to build a summary from scores/metrics
    if not result and isinstance(reasoning, dict):
        metric_parts = []
        for key, val in reasoning.items():
            if isinstance(val, (int, float)) and key not in ("signal", "confidence"):
                metric_parts.append(f"{key}: {_fmt_num(val)}")
            elif isinstance(val, dict):
                for k2, v2 in val.items():
                    if isinstance(v2, (int, float)):
                        metric_parts.append(f"{k2}: {_fmt_num(v2)}")
                    elif isinstance(v2, str) and len(v2) > 1:
                        metric_parts.append(v2)
        result = "; ".join(metric_parts) if metric_parts else json.dumps(reasoning, ensure_ascii=False)
    # Clean up any leftover JSON syntax
    result = _strip_json_syntax(result)
    return result
