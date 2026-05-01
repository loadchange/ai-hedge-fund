"""Mean-variance portfolio optimization.

Built on cvxpy so adding new constraints (CVaR, tracking error, sector
caps, transaction costs) is mechanical.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Sequence

import cvxpy as cp
import numpy as np

from src.signals.types import PortfolioTarget


class BaseOptimizer(ABC):
    """Common interface for all portfolio optimizers."""

    @abstractmethod
    def solve(
        self,
        tickers: Sequence[str],
        expected_returns: np.ndarray,
        cov_matrix: np.ndarray,
        **kwargs,
    ) -> PortfolioTarget:
        ...


class MeanVarianceOptimizer(BaseOptimizer):
    """Markowitz minimum-variance with target-return constraint.

    Default formulation::

        minimise   wᵀ Σ w
        subject to wᵀ μ ≥ target_return
                   sum(w) = 1
                   |w_i|   ≤ position_cap        (per-asset)

    Set ``allow_short=False`` to restrict to long-only (``w_i ≥ 0``).
    Pass ``risk_aversion`` instead of ``target_return`` for an
    unconstrained-return formulation: ``minimise risk - λ · return``.
    """

    def __init__(
        self,
        position_cap: float = 0.25,
        allow_short: bool = True,
        target_return: float | None = None,
        risk_aversion: float | None = None,
    ) -> None:
        if target_return is None and risk_aversion is None:
            risk_aversion = 1.0  # default: balanced trade-off
        self.position_cap = position_cap
        self.allow_short = allow_short
        self.target_return = target_return
        self.risk_aversion = risk_aversion

    def solve(
        self,
        tickers: Sequence[str],
        expected_returns: np.ndarray,
        cov_matrix: np.ndarray,
        **_kwargs,
    ) -> PortfolioTarget:
        n = len(tickers)
        if n == 0:
            return PortfolioTarget(weights={})
        if expected_returns.shape != (n,):
            raise ValueError(f"expected_returns shape {expected_returns.shape} != ({n},)")
        if cov_matrix.shape != (n, n):
            raise ValueError(f"cov_matrix shape {cov_matrix.shape} != ({n}, {n})")

        w = cp.Variable(n)
        # Symmetrise + lift slightly off the boundary so cvxpy's PSD
        # check doesn't reject a numerically-noisy sample matrix.
        cov_psd = 0.5 * (cov_matrix + cov_matrix.T) + 1e-8 * np.eye(n)
        risk = cp.quad_form(w, cp.psd_wrap(cov_psd))
        ret = expected_returns @ w

        constraints = [cp.sum(w) == 1]
        if self.allow_short:
            constraints += [w >= -self.position_cap, w <= self.position_cap]
        else:
            constraints += [w >= 0, w <= self.position_cap]
        if self.target_return is not None:
            constraints.append(ret >= self.target_return)

        if self.target_return is not None:
            objective = cp.Minimize(risk)
        else:
            objective = cp.Minimize(risk - self.risk_aversion * ret)

        problem = cp.Problem(objective, constraints)
        try:
            problem.solve(solver=cp.CLARABEL)
        except cp.SolverError:
            problem.solve(solver=cp.SCS)

        if w.value is None:
            return PortfolioTarget(weights={})

        weights = {ticker: float(w.value[i]) for i, ticker in enumerate(tickers)}
        return PortfolioTarget(
            weights=weights,
            expected_return=float(ret.value) if ret.value is not None else None,
            expected_risk=float(np.sqrt(risk.value)) if risk.value is not None else None,
        )
