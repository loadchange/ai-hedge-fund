"""Combine multiple signal outputs into a single composite signal.

Re-exports the legacy ``weighted_signal_combination`` helper used by
``src/agents/technicals.py`` so existing backtest behaviour is preserved
exactly.
"""

from __future__ import annotations

from typing import Mapping

from src.signals.types import SignalResult
from src.signals.utils import safe_float


def weighted_signal_combination(
    signals: Mapping[str, dict],
    weights: Mapping[str, float],
) -> dict:
    """Combine ``{strategy: {signal, confidence, ...}}`` signals into one.

    Inputs match the legacy dict format used by ``technical_analyst_agent``;
    this preserves bit-for-bit compatibility with existing backtest output.

    Returns ``{"signal": "bullish"|"bearish"|"neutral", "confidence": float}``.
    """
    signal_values = {"bullish": 1, "neutral": 0, "bearish": -1}

    weighted_sum = 0.0
    total_confidence = 0.0

    for strategy, signal in signals.items():
        numeric_signal = signal_values.get(signal.get("signal", "neutral"), 0)
        weight = weights.get(strategy, 0.0)
        confidence = safe_float(signal.get("confidence"), 0.0)

        weighted_sum += numeric_signal * weight * confidence
        total_confidence += weight * confidence

    final_score = weighted_sum / total_confidence if total_confidence > 0 else 0.0

    if final_score > 0.2:
        signal = "bullish"
    elif final_score < -0.2:
        signal = "bearish"
    else:
        signal = "neutral"

    return {
        "signal": signal,
        "confidence": safe_float(abs(final_score), 0.0),
    }


def signal_result_to_legacy(result: SignalResult) -> dict:
    """Convert a :class:`SignalResult` into the legacy dict format expected by
    ``weighted_signal_combination``.

    Uses ``direction`` + ``confidence`` directly (not ``value``) so the
    output is bit-identical to ``src/agents/technicals.py``'s historical
    behaviour, including neutral cases where ``confidence == 0.5``.
    """
    return {
        "signal": result.direction,
        "confidence": safe_float(result.confidence, 0.0),
        "metrics": dict(result.components),
    }
