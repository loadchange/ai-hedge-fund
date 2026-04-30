"""Duan Yongping (段永平) agent.

Methodology distilled from Duan's public talks (Snowball / Sina) and his
"stop doing list":

  1. 本分 (integrity / focus) — management does the right thing, sticks to
     the core business, doesn't chase fads.
  2. 不懂不投 (only invest in what you understand) — simple, predictable
     business models with long-term demand.
  3. 生意模式 (business quality) — pricing power, durable margins, low
     reinvestment needs.
  4. owner earnings — judge real free cash flow, not accounting profit.
  5. 长期持有 (multi-year horizon) — focus on 5–10 year cash flow, ignore
     short-term price.
  6. stop doing list — exclude leveraged, capital-intensive, dilutive,
     declining businesses.

Five 0–5 point dimensions feed a numeric score; the LLM then writes the
in-character reasoning in Duan's plain-spoken Chinese style.
"""

from __future__ import annotations

import json
from typing_extensions import Literal

from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from src.graph.state import AgentState, show_agent_reasoning
from src.tools.api import get_financial_metrics, get_market_cap, search_line_items
from src.utils.api_key import get_api_key_from_state
from src.utils.llm import call_llm
from src.utils.progress import progress


class DuanYongpingSignal(BaseModel):
    signal: Literal["bullish", "bearish", "neutral"]
    confidence: int = Field(description="Confidence 0-100")
    reasoning: str = Field(description="Reasoning for the decision")


# ── Agent entry point ─────────────────────────────────────────────────────────


def duan_yongping_agent(state: AgentState, agent_id: str = "duan_yongping_agent"):
    """Score companies through Duan Yongping's lens and let the LLM voice it."""
    data = state["data"]
    end_date = data["end_date"]
    tickers = data["tickers"]
    api_key = get_api_key_from_state(state, "FINANCIAL_DATASETS_API_KEY")

    duan_analysis: dict[str, dict] = {}

    for ticker in tickers:
        progress.update_status(agent_id, ticker, "Fetching financial metrics")
        metrics = get_financial_metrics(ticker, end_date, period="ttm", limit=10, api_key=api_key)

        progress.update_status(agent_id, ticker, "Gathering financial line items")
        line_items = search_line_items(
            ticker,
            [
                "revenue",
                "net_income",
                "operating_income",
                "free_cash_flow",
                "capital_expenditure",
                "total_debt",
                "shareholders_equity",
                "outstanding_shares",
                "issuance_or_purchase_of_equity_shares",
                "dividends_and_other_cash_distributions",
                "cash_and_equivalents",
            ],
            end_date,
            period="ttm",
            limit=10,
            api_key=api_key,
        )

        progress.update_status(agent_id, ticker, "Getting market cap")
        market_cap = get_market_cap(ticker, end_date, api_key=api_key)

        progress.update_status(agent_id, ticker, "Analyzing business quality")
        quality = analyze_business_quality(metrics, line_items)

        progress.update_status(agent_id, ticker, "Analyzing management actions")
        mgmt = analyze_management_actions(line_items)

        progress.update_status(agent_id, ticker, "Analyzing business predictability")
        predictability = analyze_long_term_consistency(line_items, metrics)

        progress.update_status(agent_id, ticker, "Analyzing cash yield and valuation")
        cash_yield = analyze_cash_yield(line_items, market_cap)

        progress.update_status(agent_id, ticker, "Analyzing stop doing list")
        stop_doing = analyze_stop_doing_list(metrics, line_items)

        total_score = (
            quality["score"]
            + mgmt["score"]
            + predictability["score"]
            + cash_yield["score"]
            + stop_doing["score"]
        )
        max_score = (
            quality["max_score"]
            + mgmt["max_score"]
            + predictability["max_score"]
            + cash_yield["max_score"]
            + stop_doing["max_score"]
        )

        analysis_for_llm = {
            "ticker": ticker,
            "score": total_score,
            "max_score": max_score,
            "business_quality": quality,
            "management_actions": mgmt,
            "long_term_consistency": predictability,
            "cash_yield": cash_yield,
            "stop_doing": stop_doing,
            "market_cap": market_cap,
        }

        progress.update_status(agent_id, ticker, "Generating Duan Yongping analysis")
        output = generate_duan_yongping_output(
            ticker=ticker,
            analysis_data=analysis_for_llm,
            state=state,
            agent_id=agent_id,
        )

        duan_analysis[ticker] = {
            "signal": output.signal,
            "confidence": output.confidence,
            "reasoning": output.reasoning,
        }

        progress.update_status(agent_id, ticker, "Done", analysis=output.reasoning)

    message = HumanMessage(content=json.dumps(duan_analysis), name=agent_id)

    if state["metadata"]["show_reasoning"]:
        show_agent_reasoning(duan_analysis, agent_id)

    state["data"]["analyst_signals"][agent_id] = duan_analysis
    progress.update_status(agent_id, None, "Done")

    return {"messages": [message], "data": state["data"]}


# ── Analysis dimensions ───────────────────────────────────────────────────────


def analyze_business_quality(metrics: list, line_items: list) -> dict:
    """生意模式: pricing power + capital efficiency + durable margins (0-5)."""
    if not metrics:
        return {"score": 0, "max_score": 5, "details": "No metrics available"}

    score = 0
    reasons: list[str] = []

    # Stable, high gross margin → pricing power / brand moat
    gross_margins = [m.gross_margin for m in metrics if m.gross_margin is not None]
    if gross_margins:
        avg_gm = sum(gross_margins) / len(gross_margins)
        if avg_gm > 0.4:
            score += 2
            reasons.append(f"高毛利率均值 {avg_gm:.1%}，具备品牌/定价权特征")
        elif avg_gm > 0.25:
            score += 1
            reasons.append(f"毛利率 {avg_gm:.1%}，定价权一般")
        else:
            reasons.append(f"毛利率仅 {avg_gm:.1%}，缺乏定价权")

    # Long-term ROE > 15%
    roes = [m.return_on_equity for m in metrics if m.return_on_equity is not None]
    if len(roes) >= 5:
        high = sum(1 for r in roes if r > 0.15)
        if high / len(roes) >= 0.8:
            score += 2
            reasons.append(f"ROE 长期>15% ({high}/{len(roes)} 期)，资本回报优秀")
        elif high / len(roes) >= 0.5:
            score += 1
            reasons.append(f"ROE>15% 占比 {high}/{len(roes)}，回报尚可")
        else:
            reasons.append(f"ROE 长期不达标，资本回报弱")
    elif roes:
        if roes[0] and roes[0] > 0.15:
            score += 1
            reasons.append(f"最新 ROE {roes[0]:.1%}，但历史数据不足")

    # Capital intensity: low capex/revenue → 轻资产
    capex_ratios = []
    for it in line_items[:5]:
        cap = getattr(it, "capital_expenditure", None)
        rev = getattr(it, "revenue", None)
        if cap is not None and rev and rev > 0:
            capex_ratios.append(abs(cap) / rev)
    if capex_ratios:
        avg_cap = sum(capex_ratios) / len(capex_ratios)
        if avg_cap < 0.05:
            score += 1
            reasons.append(f"capex/收入仅 {avg_cap:.1%}，轻资产生意")
        elif avg_cap > 0.15:
            reasons.append(f"capex/收入 {avg_cap:.1%}，重资产、再投入大")

    return {
        "score": min(score, 5),
        "max_score": 5,
        "details": "; ".join(reasons) if reasons else "数据不足",
    }


def analyze_management_actions(line_items: list) -> dict:
    """管理层行为: 回购、分红、不滥发股票 (0-5)."""
    if not line_items:
        return {"score": 0, "max_score": 5, "details": "No line item data"}

    score = 0
    reasons: list[str] = []

    # Look across recent periods for share repurchases vs issuance
    repurchases = 0
    issuances = 0
    for it in line_items[:5]:
        ev = getattr(it, "issuance_or_purchase_of_equity_shares", None)
        if ev is None:
            continue
        if ev < 0:
            repurchases += 1
        elif ev > 0:
            issuances += 1

    if repurchases >= 3:
        score += 2
        reasons.append(f"近期持续回购 ({repurchases} 期)，管理层认为股价被低估")
    elif repurchases >= 1 and issuances == 0:
        score += 1
        reasons.append("有回购、无增发，对股东友好")
    elif issuances >= 3:
        reasons.append(f"频繁增发稀释股东 ({issuances} 期)")

    # Dividends signal real cash flow + shareholder return
    dividends = sum(
        1 for it in line_items[:5]
        if getattr(it, "dividends_and_other_cash_distributions", None) and
        it.dividends_and_other_cash_distributions < 0
    )
    if dividends >= 3:
        score += 2
        reasons.append(f"持续分红 ({dividends} 期)，现金流真实")
    elif dividends >= 1:
        score += 1
        reasons.append(f"有分红 ({dividends} 期)")

    # Outstanding shares trend: shrinking is good
    shares = [it.outstanding_shares for it in line_items if getattr(it, "outstanding_shares", None)]
    if len(shares) >= 3 and shares[0] < shares[-1]:
        score += 1
        reasons.append("流通股数下降，长期回购")
    elif len(shares) >= 3 and shares[0] > shares[-1] * 1.05:
        reasons.append("流通股数明显上升，存在稀释")

    return {
        "score": min(score, 5),
        "max_score": 5,
        "details": "; ".join(reasons) if reasons else "数据不足",
    }


def analyze_long_term_consistency(line_items: list, metrics: list) -> dict:
    """长期一致性: 收入/利润可预测，波动小 (0-5)."""
    if len(line_items) < 4:
        return {"score": 0, "max_score": 5, "details": "历史数据不足"}

    score = 0
    reasons: list[str] = []

    # Revenue stability: each period should be >= the next (chronological)
    revenues = [it.revenue for it in line_items if getattr(it, "revenue", None)]
    if len(revenues) >= 4:
        growing = sum(1 for i in range(len(revenues) - 1) if revenues[i] >= revenues[i + 1])
        rate = growing / (len(revenues) - 1)
        if rate >= 0.8:
            score += 2
            reasons.append(f"收入持续增长 ({growing}/{len(revenues) - 1} 期)")
        elif rate >= 0.5:
            score += 1
            reasons.append(f"收入大体增长 ({growing}/{len(revenues) - 1} 期)")
        else:
            reasons.append("收入增长不稳定")

    # Net income positivity + growth
    nis = [it.net_income for it in line_items if getattr(it, "net_income", None) is not None]
    if len(nis) >= 4:
        positive = sum(1 for n in nis if n > 0)
        if positive == len(nis):
            score += 2
            reasons.append(f"近{len(nis)}期持续盈利")
        elif positive >= len(nis) * 0.75:
            score += 1
            reasons.append(f"多数期盈利 ({positive}/{len(nis)})")
        else:
            reasons.append(f"盈利不稳定 ({positive}/{len(nis)} 期为正)")

    # Operating margin stability — coefficient of variation
    op_margins = [m.operating_margin for m in metrics if m.operating_margin is not None]
    if len(op_margins) >= 4:
        avg = sum(op_margins) / len(op_margins)
        if avg > 0:
            var = sum((x - avg) ** 2 for x in op_margins) / len(op_margins)
            cv = (var ** 0.5) / avg
            if cv < 0.3:
                score += 1
                reasons.append(f"经营利润率波动小 (CV={cv:.2f})，业务可预测")

    return {
        "score": min(score, 5),
        "max_score": 5,
        "details": "; ".join(reasons) if reasons else "数据不足",
    }


def analyze_cash_yield(line_items: list, market_cap: float | None) -> dict:
    """Owner-earnings 视角: FCF/市值收益率 + 净现金 (0-5)."""
    if not line_items or not market_cap or market_cap <= 0:
        return {"score": 0, "max_score": 5, "details": "无市值或现金流数据"}

    score = 0
    reasons: list[str] = []

    # FCF yield = latest FCF / market cap
    latest = line_items[0]
    fcf = getattr(latest, "free_cash_flow", None)
    if fcf and fcf > 0:
        fcf_yield = fcf / market_cap
        if fcf_yield > 0.08:
            score += 3
            reasons.append(f"FCF 收益率 {fcf_yield:.1%}，明显便宜")
        elif fcf_yield > 0.05:
            score += 2
            reasons.append(f"FCF 收益率 {fcf_yield:.1%}，估值合理")
        elif fcf_yield > 0.03:
            score += 1
            reasons.append(f"FCF 收益率 {fcf_yield:.1%}，估值偏贵但可接受")
        else:
            reasons.append(f"FCF 收益率仅 {fcf_yield:.1%}，估值偏高")
    elif fcf is not None and fcf <= 0:
        reasons.append("最近一期 FCF 为负，需要消耗现金运营")

    # Multi-period FCF positivity — true earning power
    fcfs = [getattr(it, "free_cash_flow", None) for it in line_items[:5]]
    pos = sum(1 for f in fcfs if f and f > 0)
    if pos == len(fcfs) and len(fcfs) >= 4:
        score += 1
        reasons.append("近 5 年 FCF 持续为正")

    # Net cash position: cash > debt → 安全边际之一
    cash = getattr(latest, "cash_and_equivalents", None)
    debt = getattr(latest, "total_debt", None)
    if cash is not None and debt is not None:
        net_cash = cash - debt
        if net_cash > 0:
            score += 1
            reasons.append(f"净现金 {net_cash:,.0f}，资产负债表稳健")
        elif debt > cash * 3:
            reasons.append(f"债务/现金 = {debt / max(cash, 1):.1f}x，杠杆偏高")

    return {
        "score": min(score, 5),
        "max_score": 5,
        "details": "; ".join(reasons) if reasons else "数据不足",
    }


def analyze_stop_doing_list(metrics: list, line_items: list) -> dict:
    """Stop doing list: 命中负面清单则扣分；干净则给满分 (0-5).

    判据 (每项命中 -1)：
      - 高负债：debt/equity > 1
      - 重资产周期股：毛利<20% 且 capex/收入>10%
      - 利润下滑：净利连续下降
      - 频繁稀释股东：连续 3 期增发
      - 经营现金流为负
    """
    if not metrics or not line_items:
        return {"score": 0, "max_score": 5, "details": "数据不足"}

    base = 5
    flags: list[str] = []

    latest_m = metrics[0]
    latest_l = line_items[0]

    # 1. High leverage
    if latest_m.debt_to_equity and latest_m.debt_to_equity > 1.0:
        base -= 1
        flags.append(f"杠杆偏高 D/E={latest_m.debt_to_equity:.1f}")

    # 2. Capital-intensive cyclical
    cap = getattr(latest_l, "capital_expenditure", None)
    rev = getattr(latest_l, "revenue", None)
    gm = latest_m.gross_margin
    if cap and rev and rev > 0 and gm is not None:
        capex_ratio = abs(cap) / rev
        if gm < 0.2 and capex_ratio > 0.1:
            base -= 1
            flags.append(f"低毛利({gm:.1%})+高 capex({capex_ratio:.1%})，重资产周期")

    # 3. Earnings decline trend
    nis = [it.net_income for it in line_items[:4] if getattr(it, "net_income", None) is not None]
    if len(nis) >= 3 and all(nis[i] < nis[i + 1] for i in range(len(nis) - 1)):
        base -= 1
        flags.append("净利润连续下滑")

    # 4. Persistent dilution
    issuances = sum(
        1 for it in line_items[:3]
        if getattr(it, "issuance_or_purchase_of_equity_shares", None) and
        it.issuance_or_purchase_of_equity_shares > 0
    )
    if issuances >= 3:
        base -= 1
        flags.append("近 3 期持续增发")

    # 5. Persistently negative FCF
    neg_fcf = sum(
        1 for it in line_items[:3]
        if getattr(it, "free_cash_flow", None) is not None and it.free_cash_flow < 0
    )
    if neg_fcf >= 3:
        base -= 1
        flags.append("近 3 期 FCF 为负")

    base = max(base, 0)
    if not flags:
        details = "未触发任何负面清单"
    else:
        details = "; ".join(flags)

    return {
        "score": base,
        "max_score": 5,
        "details": details,
        "flags": flags,
    }


# ── LLM voice ────────────────────────────────────────────────────────────────


def generate_duan_yongping_output(
    ticker: str,
    analysis_data: dict,
    state: AgentState,
    agent_id: str = "duan_yongping_agent",
) -> DuanYongpingSignal:
    """Wrap the numeric analysis with Duan's plain-spoken Chinese voice."""

    template = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are Duan Yongping (段永平), the Chinese value investor known for buying
Apple, Moutai and Kweichow Moutai-style "do the right thing" businesses.
Decide bullish, bearish, or neutral using only the provided facts and your
core principles.

Core principles:
1. 本分 (do the right thing) — only invest when management focuses on the
   core business, treats shareholders fairly, and avoids the "stop doing
   list" (leveraged growth, fad-chasing, ego acquisitions).
2. 不懂不投 (only invest in what you understand) — the business model must
   be explainable in one or two sentences. If you cannot, default to neutral
   and explicitly say it is outside your circle of competence.
3. 生意模式 (business quality) — durable pricing power, stable gross margins,
   low reinvestment needs. The business should not depend on a star CEO.
4. owner earnings — judge real free cash flow, not accounting profit. FCF
   yield > 5% is the bar for a reasonable price.
5. 长期持有 (long horizon) — think 5–10 years; ignore short-term swings.
   A great business at a fair price beats a mediocre business at a cheap
   price.

Signal rules:
- bullish: high business-quality score AND shareholder-friendly management
  AND reasonable/cheap FCF yield AND no stop-doing flags.
- bearish: multiple stop-doing flags hit (high debt / earnings decline /
  persistent dilution / capital-intensive cyclical), or the business model
  itself is poor.
- neutral: decent business at an unattractive price, OR the business is
  outside your circle of competence (state this explicitly).

Confidence scale:
- 90–100: exceptional business + outstanding management + cheap — "would
  hold for 10 years with conviction".
- 70–89: good business, fair price, would buy with moderate sizing.
- 50–69: mixed signals or near the edge of your circle of competence.
- 30–49: mediocre business or expensive valuation.
- 10–29: multiple negative-list flags, or entirely outside your circle.

Style requirements:
- Plain-spoken, direct, no flowery language; cite one or two key facts in
  ordinary words, do not list numbers.
- Keep reasoning under 80 characters.
- Voice should sound like Duan: humble, calm, almost conversational.
- Return JSON only, no other text."""
            ),
            (
                "human",
                """Ticker: {ticker}

Facts (already scored against Duan Yongping's criteria):
{analysis}

Return exactly:
{{
  "signal": "bullish" | "bearish" | "neutral",
  "confidence": int,
  "reasoning": "short justification in Duan Yongping's voice"
}}"""
            ),
        ]
    )

    prompt = template.invoke({
        "ticker": ticker,
        "analysis": json.dumps(analysis_data, ensure_ascii=False, indent=2, default=str),
    })

    def default():
        return DuanYongpingSignal(
            signal="neutral",
            confidence=50,
            reasoning="Insufficient data; staying on the sidelines.",
        )

    return call_llm(
        prompt=prompt,
        pydantic_model=DuanYongpingSignal,
        agent_name=agent_id,
        state=state,
        default_factory=default,
    )
