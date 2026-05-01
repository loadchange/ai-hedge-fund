"""Market model — α + β fit on a clean pre-event window."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class MarketModel:
    """Result of a market-model regression.

    Attributes:
        alpha: intercept (mean abnormal daily return when benchmark = 0).
        beta: sensitivity to benchmark.
        residual_std: residual standard deviation (used for t-stats).
        n_obs: observations used in the fit.
    """

    alpha: float
    beta: float
    residual_std: float
    n_obs: int

    def predict(self, benchmark_returns: np.ndarray) -> np.ndarray:
        """Expected return given benchmark returns: ``α + β · r_market``."""
        return self.alpha + self.beta * np.asarray(benchmark_returns, dtype=float)


def fit_market_model(
    asset_returns: pd.Series,
    benchmark_returns: pd.Series,
    *,
    estimation_window: int = 252,
    gap: int = 30,
    event_idx: int | None = None,
) -> MarketModel:
    """Fit ``r_asset = α + β · r_benchmark + ε`` over a clean window.

    Args:
        asset_returns: full asset return series, indexed by date.
        benchmark_returns: same length & index as *asset_returns*.
        estimation_window: number of trailing observations to use (default 252).
        gap: gap (in trading days) between the end of the estimation
            window and the event; protects against contamination by
            pre-event drift. Default 30.
        event_idx: integer position of the event in the series. When
            ``None`` (default) the most recent ``estimation_window``
            samples ending at ``len - gap`` are used.

    Returns:
        :class:`MarketModel` with α, β, residual std, and observation count.
    """
    asset = pd.Series(asset_returns).astype(float)
    bench = pd.Series(benchmark_returns).astype(float)

    df = pd.concat([asset, bench], axis=1).dropna()
    df.columns = ["asset", "bench"]
    n = len(df)
    if n < estimation_window // 2:
        raise ValueError(f"Need at least {estimation_window // 2} aligned obs, got {n}")

    if event_idx is None:
        end = max(0, n - gap)
    else:
        end = max(0, event_idx - gap)
    start = max(0, end - estimation_window)
    window = df.iloc[start:end]
    if len(window) < 30:
        raise ValueError(
            f"Estimation window too small ({len(window)} obs); need at least 30"
        )

    x = window["bench"].to_numpy()
    y = window["asset"].to_numpy()
    x_mean = x.mean()
    y_mean = y.mean()
    var_x = ((x - x_mean) ** 2).sum()
    if var_x <= 0:
        # Benchmark is constant — fall back to mean-only model.
        beta = 0.0
        alpha = y_mean
    else:
        beta = float(((x - x_mean) * (y - y_mean)).sum() / var_x)
        alpha = float(y_mean - beta * x_mean)

    residuals = y - (alpha + beta * x)
    residual_std = float(np.std(residuals, ddof=2)) if len(residuals) > 2 else float("nan")

    return MarketModel(
        alpha=alpha,
        beta=beta,
        residual_std=residual_std,
        n_obs=len(window),
    )
