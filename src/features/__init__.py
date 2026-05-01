"""Feature engineering primitives.

Three families:

* :mod:`src.features.earnings_surprise` — SUE (standardised unexpected
  earnings) and PEAD-style drift over a configurable post-announcement
  window.
* :mod:`src.features.kpi_momentum` — z-scored YoY changes in headline
  KPIs (revenue, EPS, margins) for cross-sectional ranking.
* :mod:`src.features.lead_lag` — pairwise cross-correlation at multiple
  lags + Granger-causality test, useful for "does sector A's price move
  predict sector B?" questions.

These are the inputs that the BaseSignal subclasses (Phase 2) and the
event-study workflows (Phase 7) compose into trading signals.
"""

from src.features.earnings_surprise import (
    pead_drift,
    standardised_unexpected_earnings,
)
from src.features.kpi_momentum import compute_kpi_momentum
from src.features.lead_lag import (
    cross_correlation_at_lag,
    granger_causality,
    lead_lag_matrix,
)

__all__ = [
    "compute_kpi_momentum",
    "cross_correlation_at_lag",
    "granger_causality",
    "lead_lag_matrix",
    "pead_drift",
    "standardised_unexpected_earnings",
]
