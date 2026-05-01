"""Convert analyst signals into Black-Litterman views.

Each LLM persona / quant signal that emits a ``{signal, confidence}``
payload becomes a P-row + Q-entry pair: P picks the ticker, Q is the
expected excess return implied by the signal direction & magnitude.

The mapping is deliberately conservative — a fully bullish signal at
100% confidence translates to a +5% excess return view. Shrink with
``return_scale`` if you want a tighter posterior.
"""

from __future__ import annotations

from typing import Iterable, Mapping

import numpy as np


def build_views_from_signals(
    tickers: list[str],
    analyst_signals: Mapping[str, Mapping[str, dict]],
    *,
    skip_agents: Iterable[str] = ("risk_management_agent",),
    return_scale: float = 0.05,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Aggregate per-agent signals into BL ``(P, Q, confidences)``.

    For each (agent, ticker) pair with a non-neutral signal, emits one
    view: P-row picks the ticker (1.0), Q is ``±confidence × return_scale``,
    and the per-view confidence is the mean confidence of all contributing
    signals (clamped to [0.05, 1.0]).

    Returns ``None`` when there are no usable views (so callers can skip
    BL and fall back to plain mean-variance).

    *analyst_signals* shape::

        {agent_id: {ticker: {"signal": "bullish"/"bearish"/"neutral",
                              "confidence": float (in % or [0,1])}}}

    Confidences in either units are normalised to ``[0, 1]``.
    """
    skip = set(skip_agents)
    rows: list[tuple[int, float, float]] = []  # (ticker_idx, view_q, confidence)
    ticker_idx = {t: i for i, t in enumerate(tickers)}

    for agent_id, by_ticker in analyst_signals.items():
        if agent_id in skip:
            continue
        for ticker, payload in (by_ticker or {}).items():
            if ticker not in ticker_idx:
                continue
            sig = (payload or {}).get("signal", "")
            sig_lc = str(sig).lower()
            if sig_lc not in ("bullish", "bearish"):
                continue
            conf = _normalise_confidence(payload.get("confidence"))
            if conf <= 0:
                continue
            sign = 1.0 if sig_lc == "bullish" else -1.0
            rows.append((ticker_idx[ticker], sign * conf * return_scale, conf))

    if not rows:
        return None

    n_assets = len(tickers)
    n_views = len(rows)
    P = np.zeros((n_views, n_assets))
    Q = np.zeros(n_views)
    confidences = np.zeros(n_views)
    for i, (idx, q, conf) in enumerate(rows):
        P[i, idx] = 1.0
        Q[i] = q
        confidences[i] = conf
    return P, Q, confidences


def _normalise_confidence(value) -> float:
    """Confidence may arrive as 0–1 (quant signals) or 0–100 (LLM agents)."""
    if value is None:
        return 0.0
    try:
        f = float(value)
    except (TypeError, ValueError):
        return 0.0
    if f != f:  # NaN
        return 0.0
    if f > 1.0:
        f = f / 100.0
    return max(0.0, min(1.0, f))
