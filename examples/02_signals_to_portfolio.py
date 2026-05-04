"""Signals → views → Black-Litterman → mean-variance, end-to-end (no LLM).

Pulls a year of US prices, runs five technical signals on each ticker,
turns the resulting bullish/bearish/neutral verdicts into Black-
Litterman views, and solves for optimal weights.

Demonstrates the integration point: persona LLM agents and quant
``BaseSignal`` subclasses produce the *same* signal shape, and the
optimizer doesn't care which one generated it.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from src.portfolio import (
    BlackLittermanOptimizer,
    MeanVarianceOptimizer,
    build_views_from_signals,
    ledoit_wolf_shrinkage,
)
from src.signals import (
    MomentumSignal,
    MeanReversionSignal,
    StatArbSignal,
    TrendFollowingSignal,
    VolatilitySignal,
)
from src.signals.composite import signal_result_to_legacy
from src.tools.api import get_prices, prices_to_df


# Use a fixed historical window so yfinance / Tencent both have data to
# return; manager picks whichever source replies first with bars.
TICKERS = ["AAPL", "MSFT", "NVDA", "JPM", "XOM"]
END = "2025-04-01"
START = "2025-01-01"
SIGNAL_CLASSES = [
    TrendFollowingSignal, MeanReversionSignal, MomentumSignal,
    VolatilitySignal, StatArbSignal,
]


def fetch_returns_panel(tickers: list[str]) -> pd.DataFrame:
    """Pull aligned daily returns for the given tickers."""
    series: dict[str, pd.Series] = {}
    for t in tickers:
        prices = get_prices(t, START, END)
        if not prices:
            print(f"  ⚠ no price data for {t}, skipping")
            continue
        df = prices_to_df(prices)
        series[t] = df["close"].pct_change().dropna()
    return pd.DataFrame(series).dropna(how="any")


def run_signals(tickers: list[str]) -> dict:
    """Run each signal on each ticker; return a fake-`analyst_signals` dict."""
    out: dict[str, dict] = {}
    for sig_cls in SIGNAL_CLASSES:
        sig = sig_cls()
        agent_id = f"{sig.name}_signal"
        per_ticker: dict[str, dict] = {}
        for t in tickers:
            prices = get_prices(t, START, END)
            if not prices:
                continue
            df = prices_to_df(prices)
            try:
                result = sig.compute_from_prices(df)
            except Exception as e:
                print(f"  ⚠ {sig.name} failed on {t}: {e}")
                continue
            legacy = signal_result_to_legacy(result)
            # confidence is 0-1 here; matches what build_views_from_signals expects
            per_ticker[t] = {
                "signal": legacy["signal"],
                "confidence": legacy["confidence"],
            }
        out[agent_id] = per_ticker
    return out


def main() -> int:
    print(f"Fetching returns panel for {TICKERS} ({START} → {END})…")
    returns = fetch_returns_panel(TICKERS)
    if returns.empty or returns.shape[1] < 2:
        print("Not enough data to demo the pipeline. Try again later "
              "(yfinance / akshare may be rate-limiting).")
        return 1
    available = list(returns.columns)
    print(f"  Got {len(returns)} aligned bars across {len(available)} tickers.\n")

    print("Running technical signals…")
    signals = run_signals(available)
    for agent_id, per_ticker in signals.items():
        verdicts = ", ".join(
            f"{t}={p['signal'][:4]}{p['confidence']*100:.0f}%"
            for t, p in per_ticker.items()
        )
        print(f"  {agent_id:24} {verdicts}")
    print()

    # Annualised mean returns + covariance.
    mu = returns.mean().to_numpy() * 252
    cov = ledoit_wolf_shrinkage(returns)

    # 1) Plain mean-variance
    print("Mean-variance (no views) → cap 30%, λ = 2:")
    mvo = MeanVarianceOptimizer(position_cap=0.30, risk_aversion=2.0)
    target = mvo.solve(available, mu, cov)
    for t, w in target.weights.items():
        print(f"  {t}: {w:+.3f}")
    print(f"  → expected return {target.expected_return:.4f}, risk {target.expected_risk:.4f}\n")

    # 2) Black-Litterman with the signals as views
    views = build_views_from_signals(available, signals)
    if views is None:
        print("BL skipped — all signals were neutral.")
        return 0
    P, Q, conf = views
    print(f"Built {len(Q)} BL views from technical signals.")

    bl = BlackLittermanOptimizer(position_cap=0.30, risk_aversion=2.5)
    target_bl = bl.solve(
        available, mu, cov,
        views_matrix=P, views_vector=Q, view_confidences=conf,
    )
    print("Black-Litterman (signals as views) → cap 30%:")
    for t, w in target_bl.weights.items():
        print(f"  {t}: {w:+.3f}")
    if target_bl.expected_return is not None:
        print(f"  → expected return {target_bl.expected_return:.4f}, risk {target_bl.expected_risk:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
