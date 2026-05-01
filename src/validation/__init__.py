"""Walk-forward validation tooling — CPCV + PBO.

Two pieces, both from López de Prado's *Advances in Financial Machine
Learning* (chapters 7 & 11):

* :class:`CombinatorialPurgedKFold` — generates train/test splits where
  test groups are purged & embargoed around training data, eliminating
  leakage from temporally-correlated samples.
* :func:`compute_pbo` — Probability of Backtest Overfitting from a
  matrix of in-sample / out-of-sample Sharpe ratios across CPCV splits.
  ``pbo > 0.5`` means the strategy is more likely to underperform OOS
  than to repeat its IS rank.

Plus a thin :func:`evaluate_signal` runner that ties the two together:
"how well does my signal generalise?"
"""

from src.validation.cpcv import CombinatorialPurgedKFold, generate_splits
from src.validation.pbo import compute_pbo, deflated_sharpe_ratio
from src.validation.runner import evaluate_signal

__all__ = [
    "CombinatorialPurgedKFold",
    "compute_pbo",
    "deflated_sharpe_ratio",
    "evaluate_signal",
    "generate_splits",
]
