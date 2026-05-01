"""Risk analytics demo: drawdown stats + scenario stress on a synthetic curve."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.risk import (
    apply_scenario,
    default_scenarios,
    drawdown_stats,
    drawdown_series,
    kelly_fraction,
    vol_targeted_size,
)


def synthetic_equity_curve(n_days: int = 504, seed: int = 7) -> pd.Series:
    """Two years of daily NLV with a deliberate ~30% drawdown around day 200."""
    np.random.seed(seed)
    dates = pd.date_range("2024-01-01", periods=n_days, freq="B")
    ret = np.random.normal(0.0005, 0.012, n_days)
    # Inject a 30-day drawdown around day 200.
    ret[200:230] -= 0.012
    nlv = 100_000.0 * (1 + ret).cumprod()
    return pd.Series(nlv, index=dates, name="NLV")


def main() -> None:
    print("=== Drawdown analytics ===")
    nlv = synthetic_equity_curve()
    stats = drawdown_stats(nlv)
    print(f"  Equity curve: {len(nlv)} days, ${nlv.iloc[0]:,.0f} → ${nlv.iloc[-1]:,.0f}")
    print(f"  Max drawdown:        {stats.max_drawdown:.2%}  on {stats.max_drawdown_date.date()}")
    print(f"  Peak before max DD:  {stats.peak_date_before_max.date()}")
    print(f"  Current drawdown:    {stats.current_drawdown:.2%}")
    print(f"  Longest underwater:  {stats.longest_underwater_days} days")
    print(f"  Underwater right now: {stats.underwater_days} days\n")

    print("=== Scenario stress on $100k tech-heavy book ===")
    exposures = {
        "AAPL": 25_000,
        "MSFT": 25_000,
        "NVDA": 30_000,
        "JPM": 10_000,
        "XOM": 10_000,
    }
    sectors = {
        "AAPL": "technology", "MSFT": "technology", "NVDA": "technology",
        "JPM": "financial",   "XOM": "energy",
    }
    print(f"  Book: ${sum(exposures.values()):,} long; tech tilt 80%.\n")

    for scenario in default_scenarios():
        result = apply_scenario(exposures, scenario, sector_map=sectors)
        pnl_pct = result["total_pnl"] / sum(exposures.values()) * 100
        print(f"  {scenario.name:35} → ${result['total_pnl']:>+12,.0f}  ({pnl_pct:+.1f}%)")

    print("\n=== Position sizing ===")
    print(f"  Kelly (win 60%, b=2):                      {kelly_fraction(0.6, 2.0):.4f}")
    print(f"  Kelly (win 40%, b=1.5, no edge):           {kelly_fraction(0.4, 1.5):.4f}")
    print(f"  Vol-target ($100k book, 30% asset vol):    ${vol_targeted_size(100_000, asset_annual_vol=0.30):,.0f}")
    print(f"  Vol-target ($100k book, 15% asset vol):    ${vol_targeted_size(100_000, asset_annual_vol=0.15):,.0f}")


if __name__ == "__main__":
    main()
