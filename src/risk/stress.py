"""Scenario stress testing.

Apply a historical shock template (2008 GFC, 2020 COVID, 2022 rate hikes,
2025 trade-war flash crash) to a current portfolio of ticker exposures
and report the implied P&L impact. Useful before sizing a position to
gauge how much cash drawdown a known-shape event would inflict.

These templates are conservative averages from observed peak-to-trough
moves on broad indices; users can subclass or pass custom :class:`StressScenario`
instances for sector-specific stress.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping


@dataclass(frozen=True)
class StressScenario:
    """A named historical shock and its sector / asset-class shocks.

    ``shocks`` maps an asset class label (or specific ticker) to its
    fractional return during the scenario. Lookup falls back to ``"default"``.
    """

    name: str
    description: str
    shocks: dict[str, float] = field(default_factory=dict)

    def shock_for(self, key: str) -> float:
        return self.shocks.get(key, self.shocks.get("default", 0.0))


def default_scenarios() -> list[StressScenario]:
    """Library of canonical scenarios.

    Numbers come from peak-to-trough single-day or week moves on broad
    indices and benchmark sector ETFs. They're **rules of thumb**, not
    a substitute for a proper Monte Carlo or historical simulation.
    """
    return [
        StressScenario(
            name="2008 GFC (Sep-Oct 2008)",
            description="Lehman/AIG collapse — broad equity drawdown, financials hardest hit.",
            shocks={
                "default": -0.40,
                "financial": -0.55,
                "energy": -0.45,
                "consumer_staples": -0.20,
                "utilities": -0.25,
                "technology": -0.35,
            },
        ),
        StressScenario(
            name="2020 COVID (Feb-Mar 2020)",
            description="Pandemic shock — global equities -34% peak-to-trough in 5 weeks.",
            shocks={
                "default": -0.34,
                "energy": -0.50,
                "financial": -0.40,
                "real_estate": -0.42,
                "technology": -0.27,
                "healthcare": -0.22,
                "consumer_staples": -0.18,
            },
        ),
        StressScenario(
            name="2022 Rate Hikes (Jan-Oct 2022)",
            description="Fed tightening cycle — bonds & long-duration equities sell off together.",
            shocks={
                "default": -0.20,
                "technology": -0.32,
                "consumer_discretionary": -0.30,
                "financial": -0.18,
                "energy": 0.30,  # the one sector that ripped
                "utilities": -0.07,
            },
        ),
        StressScenario(
            name="2025 Trade-War Flash (Apr 2025)",
            description="Tariff escalation — single-week -12% on broad indices.",
            shocks={
                "default": -0.12,
                "consumer_discretionary": -0.18,
                "technology": -0.16,
                "industrials": -0.15,
                "consumer_staples": -0.05,
                "utilities": -0.04,
            },
        ),
    ]


def apply_scenario(
    exposures: Mapping[str, float],
    scenario: StressScenario,
    sector_map: Mapping[str, str] | None = None,
) -> dict:
    """Compute the dollar P&L impact of *scenario* on the given *exposures*.

    Args:
        exposures: ``{ticker: signed_dollar_exposure}`` (positive long,
            negative short).
        scenario: the scenario to apply.
        sector_map: optional ``{ticker: sector_label}``. When a ticker
            isn't in the map, the scenario's ``"default"`` shock is used.

    Returns:
        Dict with ``total_pnl``, ``per_ticker`` (dict ticker → pnl),
        ``shock_used`` (dict ticker → fractional shock applied).
    """
    sector_map = sector_map or {}
    per_ticker: dict[str, float] = {}
    shock_used: dict[str, float] = {}

    for ticker, exposure in exposures.items():
        sector = sector_map.get(ticker, "default")
        shock = scenario.shock_for(sector)
        # P&L = exposure × return. Short positions (negative exposure)
        # gain when shock < 0, which falls out of the multiplication.
        pnl = float(exposure) * shock
        per_ticker[ticker] = pnl
        shock_used[ticker] = shock

    return {
        "scenario": scenario.name,
        "description": scenario.description,
        "total_pnl": sum(per_ticker.values()),
        "per_ticker": per_ticker,
        "shock_used": shock_used,
    }
