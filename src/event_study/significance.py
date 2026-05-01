"""Significance tests for cross-sectional CARs.

Two flavours:

* :func:`t_test_car` — parametric, assumes CARs are i.i.d. normal.
  Powerful when the assumption holds; biased when there are extreme
  outliers (one big winner / loser).
* :func:`wilcoxon_signed_rank_car` — non-parametric, tests whether the
  *median* CAR differs from zero. More robust to outliers.
"""

from __future__ import annotations

import math

import numpy as np
import scipy.stats as stats


def t_test_car(cars: list[float]) -> dict:
    """Two-sided one-sample t-test against H₀: mean CAR = 0.

    Returns ``{"mean": float, "std": float, "t_stat": float, "p_value": float, "n": int}``.
    """
    arr = np.asarray(cars, dtype=float)
    arr = arr[~np.isnan(arr)]
    n = len(arr)
    if n < 2:
        return {
            "mean": float(arr.mean()) if n else float("nan"),
            "std": float("nan"),
            "t_stat": float("nan"),
            "p_value": float("nan"),
            "n": n,
        }
    mean = float(arr.mean())
    std = float(arr.std(ddof=1))
    if std <= 0:
        return {"mean": mean, "std": 0.0, "t_stat": float("nan"), "p_value": float("nan"), "n": n}

    t_stat = mean / (std / math.sqrt(n))
    # Two-sided p-value from Student's t distribution.
    p_value = 2.0 * (1.0 - stats.t.cdf(abs(t_stat), df=n - 1))
    return {"mean": mean, "std": std, "t_stat": float(t_stat), "p_value": float(p_value), "n": n}


def wilcoxon_signed_rank_car(cars: list[float]) -> dict:
    """Wilcoxon signed-rank test against H₀: median CAR = 0.

    Returns ``{"median": float, "stat": float, "p_value": float, "n": int}``.
    Robust to outliers (uses ranks of |CAR|, not values).
    """
    arr = np.asarray(cars, dtype=float)
    arr = arr[~np.isnan(arr)]
    n = len(arr)
    if n < 2:
        return {
            "median": float(np.median(arr)) if n else float("nan"),
            "stat": float("nan"),
            "p_value": float("nan"),
            "n": n,
        }
    if (arr == 0).all():
        return {"median": 0.0, "stat": float("nan"), "p_value": float("nan"), "n": n}
    try:
        result = stats.wilcoxon(arr, zero_method="wilcox")
        return {
            "median": float(np.median(arr)),
            "stat": float(result.statistic),
            "p_value": float(result.pvalue),
            "n": n,
        }
    except ValueError:
        return {
            "median": float(np.median(arr)),
            "stat": float("nan"),
            "p_value": float("nan"),
            "n": n,
        }
