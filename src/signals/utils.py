"""Shared helpers for technical signals (RSI, EMA, ADX, ATR, Bollinger, Hurst).

Extracted from ``src/agents/technicals.py`` so individual signal modules
can compose them without an import cycle.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Generic numeric helpers
# ---------------------------------------------------------------------------


def safe_float(value, default: float = 0.0) -> float:
    """Convert to float, returning *default* for NaN / None / invalid input."""
    try:
        if value is None or pd.isna(value) or np.isnan(value):
            return default
        return float(value)
    except (ValueError, TypeError, OverflowError):
        return default


def normalize_pandas(obj):
    """Convert pandas Series / DataFrame to native Python primitives."""
    if isinstance(obj, pd.Series):
        return obj.tolist()
    if isinstance(obj, pd.DataFrame):
        return obj.to_dict("records")
    if isinstance(obj, dict):
        return {k: normalize_pandas(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [normalize_pandas(item) for item in obj]
    return obj


# ---------------------------------------------------------------------------
# Technical indicators (operate on OHLCV DataFrame with at least 'close').
# ---------------------------------------------------------------------------


def calculate_rsi(prices_df: pd.DataFrame, period: int = 14) -> pd.Series:
    delta = prices_df["close"].diff()
    gain = delta.where(delta > 0, 0).fillna(0)
    loss = (-delta.where(delta < 0, 0)).fillna(0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def calculate_bollinger_bands(
    prices_df: pd.DataFrame, window: int = 20
) -> tuple[pd.Series, pd.Series]:
    sma = prices_df["close"].rolling(window).mean()
    std_dev = prices_df["close"].rolling(window).std()
    upper = sma + (std_dev * 2)
    lower = sma - (std_dev * 2)
    return upper, lower


def calculate_ema(df: pd.DataFrame, window: int) -> pd.Series:
    return df["close"].ewm(span=window, adjust=False).mean()


def calculate_adx(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """Average Directional Index. Returns DataFrame with columns adx, +di, -di."""
    df = df.copy()
    df["high_low"] = df["high"] - df["low"]
    df["high_close"] = (df["high"] - df["close"].shift()).abs()
    df["low_close"] = (df["low"] - df["close"].shift()).abs()
    df["tr"] = df[["high_low", "high_close", "low_close"]].max(axis=1)

    df["up_move"] = df["high"] - df["high"].shift()
    df["down_move"] = df["low"].shift() - df["low"]

    df["plus_dm"] = np.where(
        (df["up_move"] > df["down_move"]) & (df["up_move"] > 0), df["up_move"], 0
    )
    df["minus_dm"] = np.where(
        (df["down_move"] > df["up_move"]) & (df["down_move"] > 0), df["down_move"], 0
    )

    df["+di"] = 100 * (
        df["plus_dm"].ewm(span=period).mean() / df["tr"].ewm(span=period).mean()
    )
    df["-di"] = 100 * (
        df["minus_dm"].ewm(span=period).mean() / df["tr"].ewm(span=period).mean()
    )
    df["dx"] = 100 * (df["+di"] - df["-di"]).abs() / (df["+di"] + df["-di"])
    df["adx"] = df["dx"].ewm(span=period).mean()

    return df[["adx", "+di", "-di"]]


def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Average True Range."""
    high_low = df["high"] - df["low"]
    high_close = (df["high"] - df["close"].shift()).abs()
    low_close = (df["low"] - df["close"].shift()).abs()
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    return ranges.max(axis=1).rolling(period).mean()


def calculate_hurst_exponent(price_series: pd.Series, max_lag: int = 20) -> float:
    """Hurst exponent. ``< 0.5`` mean-reverting, ``= 0.5`` random walk, ``> 0.5`` trending."""
    lags = range(2, max_lag)
    tau = [
        max(
            1e-8,
            np.sqrt(np.std(np.subtract(price_series[lag:], price_series[:-lag]))),
        )
        for lag in lags
    ]
    try:
        slope, _intercept = np.polyfit(np.log(lags), np.log(tau), 1)
        return float(slope)
    except (ValueError, RuntimeWarning):
        return 0.5
