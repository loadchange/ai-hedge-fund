"""Risk-parity portfolio: equal risk contribution from each asset.

Each asset's contribution to total portfolio variance is forced equal,
so the portfolio doesn't lean on any single asset (or factor) for its
risk. Useful when you don't trust expected-return estimates — RP only
needs a covariance matrix.

Solved with cvxpy via the convex log-formulation::

    minimise  ½ wᵀ Σ w − (1/n) Σ ln(w_i)
    subject to w ≥ 0

The first-order conditions of this objective are exactly the risk-parity
condition (equal marginal risk × weight for every asset). Long-only by
construction.
"""

from __future__ import annotations

from typing import Sequence

import cvxpy as cp
import numpy as np

from src.portfolio.optimizer import BaseOptimizer
from src.signals.types import PortfolioTarget


class RiskParityOptimizer(BaseOptimizer):
    """Equal-risk-contribution long-only portfolio.

    *expected_returns* is accepted to satisfy the BaseOptimizer interface
    but ignored — RP doesn't use it.
    """

    def __init__(self, position_cap: float = 0.5) -> None:
        self.position_cap = position_cap

    def solve(
        self,
        tickers: Sequence[str],
        expected_returns: np.ndarray,  # noqa: ARG002 — unused, RP ignores returns
        cov_matrix: np.ndarray,
        **_kwargs,
    ) -> PortfolioTarget:
        n = len(tickers)
        if n == 0:
            return PortfolioTarget(weights={})

        # Normalise to a one-unit weight vector via the log-formulation,
        # then rescale to sum=1.
        w = cp.Variable(n, pos=True)
        cov_psd = 0.5 * (cov_matrix + cov_matrix.T) + 1e-8 * np.eye(n)
        risk = 0.5 * cp.quad_form(w, cp.psd_wrap(cov_psd))
        log_term = cp.sum(cp.log(w)) / n

        constraints = [w >= 1e-6, w <= self.position_cap * n]  # cap before rescaling
        problem = cp.Problem(cp.Minimize(risk - log_term), constraints)
        try:
            problem.solve(solver=cp.CLARABEL)
        except cp.SolverError:
            problem.solve(solver=cp.SCS)

        if w.value is None:
            # Fallback: equal weights.
            equal_w = 1.0 / n
            return PortfolioTarget(weights={t: equal_w for t in tickers})

        raw = np.maximum(w.value, 0.0)
        normed = raw / raw.sum() if raw.sum() > 0 else np.full(n, 1.0 / n)
        # Apply the post-scale position cap and renormalise.
        normed = np.minimum(normed, self.position_cap)
        if normed.sum() > 0:
            normed = normed / normed.sum()

        weights = {ticker: float(normed[i]) for i, ticker in enumerate(tickers)}
        port_var = float(weights_array(normed) @ cov_psd @ weights_array(normed))
        return PortfolioTarget(
            weights=weights,
            expected_return=None,
            expected_risk=float(np.sqrt(port_var)),
        )


def weights_array(arr: np.ndarray) -> np.ndarray:
    """Tiny helper so the call site reads naturally."""
    return arr
