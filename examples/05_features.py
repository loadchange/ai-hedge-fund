"""Feature engineering demo: SUE + KPI momentum + Granger causality."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.features import (
    compute_kpi_momentum,
    cross_correlation_at_lag,
    granger_causality,
    standardised_unexpected_earnings,
)


# ---------------------------------------------------------------------------
# 1. SUE — Standardised Unexpected Earnings
# ---------------------------------------------------------------------------

print("=== SUE on a synthetic earnings history ===")
# Eight quarters of EPS — actuals trending up, with a final-quarter blowout.
actuals = [1.00, 1.04, 1.09, 1.14, 1.20, 1.27, 1.35, 1.55]
estimates = [1.00, 1.03, 1.08, 1.13, 1.19, 1.26, 1.34, 1.45]
sue = standardised_unexpected_earnings(actuals, estimates)
print(f"  Actuals:   {actuals}")
print(f"  Estimates: {estimates}")
print(f"  Latest surprise = {actuals[-1] - estimates[-1]:+.2f}")
print(f"  SUE (std-units): {sue:.2f}  (>+1 = significant beat)\n")


# ---------------------------------------------------------------------------
# 2. KPI momentum — z-scored YoY changes across multiple metrics
# ---------------------------------------------------------------------------

print("=== KPI momentum ===")


class FakeMetric:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


# 9 quarters of metrics — gentle trend, then a final-quarter acceleration.
metrics = [
    FakeMetric(revenue=100, earnings_per_share=1.00, gross_margin=0.30, operating_margin=0.10),
    FakeMetric(revenue=105, earnings_per_share=1.05, gross_margin=0.31, operating_margin=0.11),
    FakeMetric(revenue=110, earnings_per_share=1.10, gross_margin=0.30, operating_margin=0.10),
    FakeMetric(revenue=115, earnings_per_share=1.15, gross_margin=0.32, operating_margin=0.12),
    FakeMetric(revenue=120, earnings_per_share=1.22, gross_margin=0.32, operating_margin=0.12),
    FakeMetric(revenue=130, earnings_per_share=1.35, gross_margin=0.33, operating_margin=0.13),
    FakeMetric(revenue=140, earnings_per_share=1.45, gross_margin=0.34, operating_margin=0.14),
    FakeMetric(revenue=150, earnings_per_share=1.55, gross_margin=0.35, operating_margin=0.15),
    FakeMetric(revenue=170, earnings_per_share=1.85, gross_margin=0.36, operating_margin=0.16),
]

momentum = compute_kpi_momentum(metrics)
print("  Latest YoY z-score per metric:")
for field, z in momentum.items():
    print(f"    {field:25} {'n/a' if z is None else f'{z:+.2f} σ'}")


# ---------------------------------------------------------------------------
# 3. Lead-lag + Granger causality between two synthetic series
# ---------------------------------------------------------------------------

print("\n=== Lead-lag detection ===")
np.random.seed(0)
T = 250
x = pd.Series(np.random.randn(T))           # leading indicator
y = x.shift(2) + np.random.normal(0, 0.4, T)  # follower lags x by 2 days

print("  x leads y by 2 days. Cross-correlations at various lags:")
for lag in (-2, 0, 2, 5):
    corr = cross_correlation_at_lag(x, y, lag=lag)
    print(f"    lag = {lag:+}: corr(x[t], y[t+lag]) = {corr:+.3f}")

print("\n=== Granger causality ===")
gc_xy = granger_causality(x, y, max_lag=3)
gc_yx = granger_causality(y, x, max_lag=3)
print(f"  x → y: F = {gc_xy['f_stat']:>7.2f}, p = {gc_xy['p_value']:.4f}  (expect significant)")
print(f"  y → x: F = {gc_yx['f_stat']:>7.2f}, p = {gc_yx['p_value']:.4f}  (expect insignificant)")
