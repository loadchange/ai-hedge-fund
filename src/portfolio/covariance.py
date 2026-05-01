"""Covariance estimators — sample, Ledoit-Wolf shrinkage, Marchenko-Pastur cleaning.

Raw sample covariance is unbiased but **very** noisy when ``n_assets``
approaches ``n_observations`` — eigenvalues at the extremes are heavily
mis-estimated, which then propagates into terrible mean-variance
weights. Use shrinkage or eigenvalue cleaning.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def sample_covariance(returns: pd.DataFrame) -> np.ndarray:
    """Naive sample covariance (annualised, 252 trading days).

    *returns* is a wide ``DataFrame[date, ticker]`` of daily returns.
    """
    if returns.empty or returns.shape[1] == 0:
        return np.zeros((0, 0))
    cov_daily = returns.cov().to_numpy()
    return cov_daily * 252.0


def ledoit_wolf_shrinkage(returns: pd.DataFrame) -> np.ndarray:
    """Ledoit-Wolf 2003 shrinkage to constant-correlation target.

    Annualised. Shrinks the sample covariance toward a structured
    target (constant pairwise correlation, sample variances retained)
    by an analytically-optimal mixing parameter that minimises Frobenius
    distance.

    Falls back to sample covariance when the input has < 3 observations.
    """
    if returns.empty or returns.shape[0] < 3 or returns.shape[1] < 2:
        return sample_covariance(returns)

    x = returns.to_numpy()
    n, p = x.shape
    x_centered = x - x.mean(axis=0)

    # Sample covariance.
    sample = (x_centered.T @ x_centered) / n

    # Constant-correlation target.
    var = np.diag(sample)
    sqrt_var = np.sqrt(np.maximum(var, 1e-12))
    corr = sample / np.outer(sqrt_var, sqrt_var)
    np.fill_diagonal(corr, 1.0)
    avg_corr = (corr.sum() - p) / (p * (p - 1))
    target = avg_corr * np.outer(sqrt_var, sqrt_var)
    np.fill_diagonal(target, var)

    # Optimal shrinkage intensity (Ledoit-Wolf 2003 formulas).
    y = x_centered ** 2
    phi_mat = (y.T @ y) / n - sample ** 2
    phi = phi_mat.sum()

    rho_diag = np.diag(phi_mat).sum()
    rho_off = 0.0
    for i in range(p):
        for j in range(p):
            if i == j:
                continue
            term = (
                (sqrt_var[j] / sqrt_var[i])
                * ((x_centered[:, i] ** 3 * x_centered[:, j]).mean() - sample[i, i] * sample[i, j])
                + (sqrt_var[i] / sqrt_var[j])
                * ((x_centered[:, j] ** 3 * x_centered[:, i]).mean() - sample[j, j] * sample[i, j])
            )
            rho_off += term
    rho = rho_diag + (avg_corr / 2.0) * rho_off

    gamma = float(((sample - target) ** 2).sum())
    if gamma <= 0:
        kappa = 0.0
    else:
        kappa = (phi - rho) / gamma
    delta = max(0.0, min(1.0, kappa / n))

    shrunk = delta * target + (1 - delta) * sample
    return shrunk * 252.0


def marchenko_pastur_clean(
    returns: pd.DataFrame, *, q: float | None = None
) -> np.ndarray:
    """Eigenvalue clipping based on the Marchenko-Pastur distribution.

    Replaces the bulk of "noise" eigenvalues (those below the upper edge
    of the MP distribution) with their mean, leaving the "signal"
    eigenvalues unchanged. *q* is ``n_assets / n_observations``; defaults
    to the ratio implied by the input shape.
    """
    if returns.empty or returns.shape[0] < 2 or returns.shape[1] < 2:
        return sample_covariance(returns)

    x = returns.to_numpy()
    n, p = x.shape
    if q is None:
        q = p / n
    sigma2 = np.var(x, axis=0).mean()
    upper_edge = sigma2 * (1.0 + np.sqrt(q)) ** 2

    cov = (x - x.mean(axis=0)).T @ (x - x.mean(axis=0)) / n
    # Symmetrise (covariance can pick up tiny floating-point asymmetries).
    cov = 0.5 * (cov + cov.T)
    eigvals, eigvecs = np.linalg.eigh(cov)

    noise_mask = eigvals < upper_edge
    if noise_mask.any():
        noise_mean = eigvals[noise_mask].mean()
        eigvals[noise_mask] = noise_mean

    cleaned = eigvecs @ np.diag(eigvals) @ eigvecs.T
    return 0.5 * (cleaned + cleaned.T) * 252.0
