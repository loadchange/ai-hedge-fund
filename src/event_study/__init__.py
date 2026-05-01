"""Event study framework — measure abnormal returns around discrete events.

The classic Brown & Warner (1985) / MacKinlay (1997) toolkit:

1. Estimate a *market model* (asset_return = α + β · benchmark_return + ε)
   from a clean pre-event window (default: 252 days ending 30 days before
   the event).
2. Compute *abnormal returns* (AR) on the event window as the residual
   between actual and model-implied return.
3. Aggregate to *cumulative abnormal returns* (CAR) per event, then
   *cross-sectionally average* (CAAR) across events of the same type.
4. Significance-test with a t-statistic (parametric) or signed-rank
   (non-parametric).

Use case: "do BEAT-EPS surprises produce a measurable +α drift in the
following 5 trading days?", "do cluster insider buys precede outperformance?"
"""

from src.event_study.abnormal_returns import (
    compute_abnormal_returns,
    compute_caar,
    compute_car,
)
from src.event_study.market_model import MarketModel, fit_market_model
from src.event_study.significance import t_test_car, wilcoxon_signed_rank_car

__all__ = [
    "MarketModel",
    "compute_abnormal_returns",
    "compute_car",
    "compute_caar",
    "fit_market_model",
    "t_test_car",
    "wilcoxon_signed_rank_car",
]
