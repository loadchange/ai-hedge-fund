"""Transaction-cost models for the backtester.

Two implementations ship out of the box:

* :class:`FixedBpsCostModel` — flat rate (default 10 bps of notional).
  Reasonable starting point for US equities; matches the v2 README's
  "10 bps fixed" target.
* :class:`SpreadPlusImpactModel` — half-spread + square-root market
  impact. Plug a venue-specific spread / participation estimate in.

Custom models implement :class:`TransactionCostModel.estimate` and the
backtester calls them on every executed trade.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


class TransactionCostModel(ABC):
    """Strategy interface for per-trade cost estimation."""

    @abstractmethod
    def estimate(
        self,
        *,
        ticker: str,
        action: str,
        quantity: int,
        price: float,
    ) -> float:
        """Return the dollar cost of executing this trade. Always non-negative."""
        ...


@dataclass(frozen=True)
class FixedBpsCostModel(TransactionCostModel):
    """Flat ``bps`` of notional traded.

    ``10 bps`` ≈ 0.10% of trade value. For a $10,000 buy that's $10.

    Holds, zero-quantity trades, and zero-price entries cost nothing.
    """

    bps: float = 10.0

    def estimate(
        self,
        *,
        ticker: str,  # noqa: ARG002
        action: str,
        quantity: int,
        price: float,
    ) -> float:
        if quantity <= 0 or price <= 0 or action == "hold":
            return 0.0
        notional = abs(int(quantity)) * float(price)
        return notional * (float(self.bps) / 10_000.0)


@dataclass(frozen=True)
class SpreadPlusImpactModel(TransactionCostModel):
    """Half-spread + sqrt-impact cost.

    ``cost = (half_spread_bps + impact_coef × √(qty / adv)) × notional``

    Where:
    * ``half_spread_bps`` is the typical bid-ask half-spread (e.g. 1-3 bps
      for US large-caps, 10-20 bps for small-caps).
    * ``impact_coef`` (in bps) governs how the cost scales with order size
      relative to *adv* (average daily volume). The default ``10 bps``
      matches the rough Almgren-Chriss reduced-form ballpark for liquid
      US names.

    ADV defaults to 1M shares when not provided per call.
    """

    half_spread_bps: float = 2.0
    impact_coef: float = 10.0
    adv: float = 1_000_000.0

    def estimate(
        self,
        *,
        ticker: str,  # noqa: ARG002
        action: str,
        quantity: int,
        price: float,
    ) -> float:
        if quantity <= 0 or price <= 0 or action == "hold":
            return 0.0
        qty = abs(int(quantity))
        notional = qty * float(price)
        adv = max(float(self.adv), 1.0)
        impact_bps = float(self.impact_coef) * (qty / adv) ** 0.5
        total_bps = float(self.half_spread_bps) + impact_bps
        return notional * (total_bps / 10_000.0)


def build_cost_model(name: str, *, bps: float = 10.0) -> TransactionCostModel:
    """CLI-friendly factory."""
    name = (name or "fixed").lower()
    if name in ("fixed", "fixed_bps"):
        return FixedBpsCostModel(bps=bps)
    if name in ("spread", "spread_impact", "spread_plus_impact"):
        return SpreadPlusImpactModel(half_spread_bps=bps / 2, impact_coef=bps)
    raise ValueError(f"Unknown cost model {name!r}; use 'fixed' or 'spread'.")
