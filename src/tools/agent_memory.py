"""Agent memory system for confidence calibration.

Stores past analysis results per ticker in local JSON files under
``data/agent_memory/{ticker}.json``. When enough historical data
accumulates (>= 5 records with actual 20d returns), the system
calibrates raw confidence scores based on historical accuracy.

Enabled via ``ENABLE_AGENT_MEMORY=1`` env var. Off by default.
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_MEMORY_DIR = Path("data/agent_memory")
_MIN_SAMPLES = 5


def _get_memory_path(ticker: str) -> Path:
    safe = ticker.replace(".", "_").replace("/", "_").replace("\\", "_")
    return _MEMORY_DIR / f"{safe}.json"


def load_memory(ticker: str) -> dict:
    """Load memory JSON for a ticker, returning empty structure if none exists."""
    path = _get_memory_path(ticker)
    if path.exists():
        try:
            with open(path) as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            logger.debug("AgentMemory load error for %s: %s", ticker, e)
    return {"ticker": ticker, "records": []}


def save_memory(ticker: str, memory: dict) -> None:
    """Persist memory to disk atomically."""
    _MEMORY_DIR.mkdir(parents=True, exist_ok=True)
    path = _get_memory_path(ticker)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(memory, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def record_analysis(
    ticker: str,
    date: str,
    signal: str,
    confidence: float,
    value: float,
    price: float,
) -> None:
    """Append a new analysis record. Skips if a record for the same date exists."""
    mem = load_memory(ticker)
    records: list[dict] = mem.get("records", [])

    # Don't duplicate same-day records
    for r in records:
        if r.get("date") == date:
            return

    records.append({
        "date": date,
        "signal": signal,
        "confidence": confidence,
        "value": value,
        "price_at_analysis": price,
        "price_5d_after": None,
        "price_10d_after": None,
        "price_20d_after": None,
        "actual_return_5d": None,
        "actual_return_10d": None,
        "actual_return_20d": None,
        "calibrated": False,
    })
    mem["records"] = records
    save_memory(ticker, mem)


def backfill_actuals(ticker: str, prices_df: pd.DataFrame) -> int:
    """Fill actual returns for past records where price data is available.

    Returns count of records backfilled.
    """
    mem = load_memory(ticker)
    records: list[dict] = mem.get("records", [])
    if not records:
        return 0

    df = prices_df.copy()
    if df.index.name != "Date":
        if "Date" in df.columns:
            df = df.set_index("Date")
        else:
            df.index = pd.to_datetime(df.index)
    df.index = pd.to_datetime(df.index)

    updated = 0
    for r in records:
        if r.get("calibrated"):
            continue
        analysis_date = pd.Timestamp(r["date"])

        def _price_after(days: int) -> float | None:
            future = df[df.index > analysis_date].head(days)
            if len(future) < max(days // 2, 2):
                return None
            return float(future["close"].iloc[-1])

        p5 = _price_after(5)
        p10 = _price_after(10)
        p20 = _price_after(20)

        if p20 is not None:
            r["price_5d_after"] = p5
            r["price_10d_after"] = p10
            r["price_20d_after"] = p20
            entry = r["price_at_analysis"]
            if entry and entry > 0:
                r["actual_return_5d"] = round((p5 - entry) / entry * 100, 2) if p5 else None
                r["actual_return_10d"] = round((p10 - entry) / entry * 100, 2) if p10 else None
                r["actual_return_20d"] = round((p20 - entry) / entry * 100, 2) if p20 else None
            r["calibrated"] = True
            updated += 1

    if updated:
        mem["records"] = records
        save_memory(ticker, mem)

    return updated


def compute_calibration_score(ticker: str) -> float:
    """Compute historical direction accuracy [0, 1].

    Uses 20d returns as the truth metric. Returns 0.5 if < _MIN_SAMPLES.
    """
    mem = load_memory(ticker)
    records: list[dict] = mem.get("records", [])
    calibrated = [r for r in records if r.get("calibrated") and r.get("actual_return_20d") is not None]
    if len(calibrated) < _MIN_SAMPLES:
        return 0.5

    correct = 0
    for r in calibrated:
        ret_20d = r["actual_return_20d"]
        signal = r["signal"]
        if (signal == "bullish" and ret_20d > 0) or (signal == "bearish" and ret_20d < 0):
            correct += 1
        elif signal == "neutral":
            correct += 0.5  # neutral gets partial credit if return is flat

    return correct / len(calibrated)


def adjust_confidence(base_confidence: float, calibration_score: float) -> float:
    """Calibrate confidence based on historical accuracy.

    - accuracy > 0.6: boost confidence up to +20%
    - accuracy < 0.4: reduce confidence up to -20%
    - otherwise: no adjustment
    """
    if calibration_score > 0.6:
        # Map [0.6, 1.0] -> [1.0, 1.2]
        factor = 1.0 + (calibration_score - 0.6) * 0.5
        return min(1.0, base_confidence * factor)
    if calibration_score < 0.4:
        # Map [0.0, 0.4] -> [0.8, 1.0]
        factor = 1.0 - (0.4 - calibration_score) * 0.5
        return max(0.1, base_confidence * factor)
    return base_confidence
