"""Event study demo: market model fit + AR / CAR / CAAR + significance tests."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.event_study import (
    compute_abnormal_returns,
    compute_caar,
    compute_car,
    fit_market_model,
    t_test_car,
    wilcoxon_signed_rank_car,
)


def synthetic_returns(true_alpha=0.0001, true_beta=1.2, n_days=500, seed=42):
    """Synthetic asset returns generated as α + β·r_market + ε,
    with a +2% abnormal return injected on day n-5 as the 'event'."""
    np.random.seed(seed)
    dates = pd.date_range("2024-01-01", periods=n_days, freq="B")
    bench = pd.Series(np.random.normal(0.0003, 0.01, n_days), index=dates, name="SPY")
    asset = pd.Series(
        true_alpha + true_beta * bench + np.random.normal(0, 0.005, n_days),
        index=dates, name="AAPL",
    )
    event_idx = n_days - 5
    asset.iloc[event_idx] += 0.02  # the surprise
    return asset, bench, dates[event_idx]


def main() -> None:
    asset, bench, event_date = synthetic_returns()
    print(f"Synthetic asset: {len(asset)} days, true α=0.0001, β=1.2.\n")

    print("=== Market model fit (252-day estimation, 30-day gap) ===")
    mm = fit_market_model(asset, bench, estimation_window=252, gap=30)
    print(f"  Fitted α = {mm.alpha:.5f}  (true 0.0001)")
    print(f"  Fitted β = {mm.beta:.4f}   (true 1.2)")
    print(f"  Residual σ = {mm.residual_std:.5f}, n = {mm.n_obs}\n")

    print("=== Abnormal returns around the event ===")
    ar = compute_abnormal_returns(asset, bench, mm)
    print(f"  AR series: {len(ar)} obs, mean = {ar.mean():.5f}, std = {ar.std():.5f}")
    print(f"  AR on event day ({event_date.date()}): {ar.loc[event_date]:+.4f}  (target +0.0200)\n")

    print("=== CAR around event windows ===")
    for window in [(-1, +1), (-3, +3), (-5, +5)]:
        car = compute_car(ar, event_window=window, event_index=event_date)
        print(f"  CAR{window}: {car:+.4f}")

    print("\n=== Cross-sectional CAAR (multiple synthetic events) ===")
    # Sample 8 events at different dates, compute CAR for each, then CAAR.
    cars = [
        compute_car(ar, event_window=(-3, +3), event_index=ar.index[i])
        for i in range(len(ar) - 100, len(ar) - 10, 10)
    ]
    print(f"  Per-event CARs: {[f'{c:+.4f}' for c in cars]}")
    print(f"  CAAR: {compute_caar(cars):+.4f}")

    print("\n=== Significance tests on the cross-section ===")
    t_result = t_test_car(cars)
    w_result = wilcoxon_signed_rank_car(cars)
    print(f"  t-test:    mean = {t_result['mean']:+.4f}, t = {t_result['t_stat']:.2f}, p = {t_result['p_value']:.4f}")
    print(f"  Wilcoxon:  median = {w_result['median']:+.4f}, p = {w_result['p_value']:.4f}")


if __name__ == "__main__":
    main()
