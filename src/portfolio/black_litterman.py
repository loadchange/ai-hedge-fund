"""Black-Litterman expected-return blender.

Combines a market-implied prior (from current weights + risk aversion)
with subjective views (from analyst signals) into a posterior expected-
return vector, which is then fed to mean-variance.

Reference: He & Litterman (1999).

Posterior::

    μ_post = [ (τΣ)⁻¹ + PᵀΩ⁻¹P ]⁻¹ [ (τΣ)⁻¹ Π + PᵀΩ⁻¹ Q ]

where:
    Π  = market-implied excess returns = δ Σ w_market
    P  = views matrix (n_views × n_assets)
    Q  = views vector (n_views,)
    Ω  = view uncertainty (diag of view variances)
    τ  = small constant (typical 0.025-0.05)
    δ  = risk-aversion coefficient (typical 2.5)
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from src.portfolio.optimizer import BaseOptimizer, MeanVarianceOptimizer
from src.signals.types import PortfolioTarget


class BlackLittermanOptimizer(BaseOptimizer):
    """BL-blended mean-variance optimizer.

    Pass ``views_matrix`` (P) and ``views_vector`` (Q) at solve time; if
    omitted the posterior collapses to the market-implied prior and the
    output is a tilted equal-weight portfolio.
    """

    def __init__(
        self,
        tau: float = 0.05,
        risk_aversion: float = 2.5,
        position_cap: float = 0.25,
        allow_short: bool = True,
    ) -> None:
        self.tau = tau
        self.risk_aversion = risk_aversion
        self.position_cap = position_cap
        self.allow_short = allow_short

    def solve(
        self,
        tickers: Sequence[str],
        expected_returns: np.ndarray,  # noqa: ARG002 — BL derives μ from market weights
        cov_matrix: np.ndarray,
        *,
        market_weights: np.ndarray | None = None,
        views_matrix: np.ndarray | None = None,
        views_vector: np.ndarray | None = None,
        view_confidences: np.ndarray | None = None,
        **_kwargs,
    ) -> PortfolioTarget:
        n = len(tickers)
        if n == 0:
            return PortfolioTarget(weights={})

        Sigma = 0.5 * (cov_matrix + cov_matrix.T) + 1e-8 * np.eye(n)
        if market_weights is None:
            market_weights = np.full(n, 1.0 / n)

        # Market-implied prior: Π = δ Σ w_mkt
        prior = self.risk_aversion * Sigma @ market_weights

        if views_matrix is None or views_vector is None or len(views_vector) == 0:
            posterior = prior
        else:
            P = np.asarray(views_matrix)
            Q = np.asarray(views_vector)
            tau_sigma = self.tau * Sigma

            if view_confidences is None:
                # Default uncertainty: diag(P τΣ Pᵀ).
                Omega = np.diag(np.diag(P @ tau_sigma @ P.T))
            else:
                # Higher confidence → smaller variance.
                conf = np.asarray(view_confidences, dtype=float)
                conf = np.clip(conf, 0.05, 1.0)
                base = np.diag(P @ tau_sigma @ P.T)
                Omega = np.diag(base / conf)

            tau_sigma_inv = np.linalg.pinv(tau_sigma)
            omega_inv = np.linalg.pinv(Omega) if Omega.size else np.zeros_like(Sigma)

            posterior_precision = tau_sigma_inv + P.T @ omega_inv @ P
            posterior_mean = (
                np.linalg.pinv(posterior_precision)
                @ (tau_sigma_inv @ prior + P.T @ omega_inv @ Q)
            )
            posterior = posterior_mean

        # Hand off to mean-variance with the posterior expected returns.
        mvo = MeanVarianceOptimizer(
            position_cap=self.position_cap,
            allow_short=self.allow_short,
            risk_aversion=self.risk_aversion,
        )
        return mvo.solve(tickers, posterior, Sigma)
