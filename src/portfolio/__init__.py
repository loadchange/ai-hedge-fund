"""Portfolio construction and optimization.

Three optimizers are exposed; pick whichever fits your use case:

* :class:`MeanVarianceOptimizer` — classic Markowitz; minimise variance
  subject to a target expected return + weight bounds.
* :class:`RiskParityOptimizer` — equal risk contribution; useful when you
  don't trust your expected-return estimates.
* :class:`BlackLittermanOptimizer` — combines a market-implied prior
  with subjective views (e.g. from the persona LLM agents) into a
  posterior expected-return vector, then feeds it to mean-variance.

All three return a :class:`PortfolioTarget` with weights in ``[-cap, +cap]``
and (where applicable) the realised expected return / risk.

Covariance estimators live in :mod:`src.portfolio.covariance` —
shrinkage (Ledoit-Wolf) and eigenvalue cleaning (Marchenko-Pastur) are
both implemented because raw sample covariance is notoriously noisy
when the number of assets approaches the number of observations.
"""

from src.portfolio.black_litterman import BlackLittermanOptimizer
from src.portfolio.covariance import (
    ledoit_wolf_shrinkage,
    marchenko_pastur_clean,
    sample_covariance,
)
from src.portfolio.optimizer import BaseOptimizer, MeanVarianceOptimizer
from src.portfolio.risk_parity import RiskParityOptimizer
from src.portfolio.views import build_views_from_signals
from src.signals.types import PortfolioTarget

__all__ = [
    "BaseOptimizer",
    "BlackLittermanOptimizer",
    "MeanVarianceOptimizer",
    "PortfolioTarget",
    "RiskParityOptimizer",
    "build_views_from_signals",
    "ledoit_wolf_shrinkage",
    "marchenko_pastur_clean",
    "sample_covariance",
]
