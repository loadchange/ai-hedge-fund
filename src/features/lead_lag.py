"""Cross-sector lead/lag detection.

Two complementary tools:

* :func:`cross_correlation_at_lag` — correlation of ``X[t]`` with
  ``Y[t + lag]``. ``lag > 0`` means *X leads Y*; ``lag < 0`` means *Y
  leads X*.
* :func:`granger_causality` — formally tests whether lagged values of
  ``X`` improve a regression of ``Y`` on its own lags. Returns the
  F-statistic and p-value at the requested max-lag.

Use case: do semiconductor equipment sales lead the broader tech ETF by
1-2 quarters? Does crude lead energy-sector equity by a few weeks?
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd
import scipy.stats as stats


def cross_correlation_at_lag(
    x: pd.Series,
    y: pd.Series,
    *,
    lag: int,
) -> float:
    """Pearson correlation of ``x[t]`` with ``y[t + lag]``.

    ``lag > 0`` ⇒ does *x* lead *y* by *lag* periods?
    ``lag = 0`` ⇒ contemporaneous correlation.
    Returns ``nan`` when there's < 5 overlapping observations.
    """
    a = pd.Series(x).astype(float)
    b = pd.Series(y).astype(float)
    if lag > 0:
        b = b.shift(-lag)
    elif lag < 0:
        a = a.shift(lag)  # shifts forward
    paired = pd.concat([a, b], axis=1).dropna()
    if len(paired) < 5:
        return float("nan")
    arr_a = paired.iloc[:, 0].to_numpy()
    arr_b = paired.iloc[:, 1].to_numpy()
    if arr_a.std() == 0 or arr_b.std() == 0:
        return float("nan")
    return float(np.corrcoef(arr_a, arr_b)[0, 1])


def lead_lag_matrix(
    series_by_label: dict[str, pd.Series],
    *,
    lags: Iterable[int] = (-5, -1, 0, +1, +5),
) -> pd.DataFrame:
    """Pairwise correlations across labels at multiple lags.

    Returns a long-form DataFrame with columns ``leader``, ``follower``,
    ``lag``, ``correlation``. ``lag > 0`` means *leader* leads *follower*.
    """
    labels = list(series_by_label.keys())
    rows: list[dict] = []
    for i, a_label in enumerate(labels):
        for b_label in labels[i + 1:]:
            for lag in lags:
                corr = cross_correlation_at_lag(
                    series_by_label[a_label],
                    series_by_label[b_label],
                    lag=lag,
                )
                rows.append(
                    {
                        "leader": a_label,
                        "follower": b_label,
                        "lag": lag,
                        "correlation": corr,
                    }
                )
    return pd.DataFrame(rows)


def granger_causality(
    cause: pd.Series,
    effect: pd.Series,
    *,
    max_lag: int = 5,
) -> dict:
    """Granger F-test: do lagged values of ``cause`` help predict ``effect``?

    Implements the standard restricted/unrestricted OLS comparison
    without dragging in statsmodels for the F distribution.

    Returns ``{"f_stat": float, "p_value": float, "lag": int, "n": int}``.
    ``p_value < 0.05`` is conventionally taken as evidence that ``cause``
    Granger-causes ``effect``.
    """
    a = pd.Series(cause).astype(float)
    b = pd.Series(effect).astype(float)
    paired = pd.concat([a, b], axis=1).dropna()
    if len(paired) < max_lag * 4:
        return {"f_stat": float("nan"), "p_value": float("nan"), "lag": max_lag, "n": len(paired)}

    a_arr = paired.iloc[:, 0].to_numpy()
    b_arr = paired.iloc[:, 1].to_numpy()
    n = len(b_arr) - max_lag

    # Build design matrices.
    Y = b_arr[max_lag:]
    # Restricted: regress effect on its own lags only (+ constant).
    X_r = np.column_stack([np.ones(n)] + [b_arr[max_lag - k - 1: -k - 1] for k in range(max_lag)])
    # Unrestricted: add lagged cause.
    X_u = np.column_stack(
        [np.ones(n)]
        + [b_arr[max_lag - k - 1: -k - 1] for k in range(max_lag)]
        + [a_arr[max_lag - k - 1: -k - 1] for k in range(max_lag)]
    )

    rss_r = _ols_rss(X_r, Y)
    rss_u = _ols_rss(X_u, Y)
    if rss_u <= 0 or rss_r <= rss_u:
        return {"f_stat": float("nan"), "p_value": float("nan"), "lag": max_lag, "n": n}

    df1 = max_lag
    df2 = n - X_u.shape[1]
    if df2 <= 0:
        return {"f_stat": float("nan"), "p_value": float("nan"), "lag": max_lag, "n": n}

    f_stat = ((rss_r - rss_u) / df1) / (rss_u / df2)
    p_value = float(1.0 - stats.f.cdf(f_stat, df1, df2))
    return {"f_stat": float(f_stat), "p_value": p_value, "lag": max_lag, "n": n}


def _ols_rss(X: np.ndarray, y: np.ndarray) -> float:
    """OLS residual sum of squares via least-squares.

    Suppresses the ``divide``/``overflow``/``invalid`` warnings that
    occasionally fire on near-singular design matrices — the result is
    still numerically valid (or NaN, which the caller handles).
    """
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        coef, *_ = np.linalg.lstsq(X, y, rcond=None)
        residuals = y - X @ coef
        return float((residuals ** 2).sum())
