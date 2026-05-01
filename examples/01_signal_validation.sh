#!/usr/bin/env bash
#
# Validation CLI demos — CPCV + PBO + Deflated Sharpe.
# No LLM cost; needs network access for price data.

set -euo pipefail

cd "$(git rev-parse --show-toplevel)"

echo "=================================================================="
echo "1. Single signal × single ticker — momentum on AAPL, 3y window"
echo "=================================================================="
uv run python -m src.validation.cli evaluate \
  --signal momentum \
  --ticker AAPL \
  --start 2023-06-01 --end 2025-04-01 \
  --n-splits 6 --n-test-splits 2 \
  --rolling-window 180 \
  --out /tmp/val_momentum_aapl.json

echo
echo "=================================================================="
echo "2. Multi-signal sweep — does any technical signal survive CPCV?"
echo "=================================================================="
uv run python -m src.validation.cli evaluate \
  --signal trend,mean_reversion,momentum,volatility,stat_arb \
  --ticker AAPL \
  --start 2023-06-01 --end 2025-04-01 \
  --n-splits 6 --n-test-splits 2 \
  --rolling-window 180 \
  --out /tmp/val_sweep_aapl.json

echo
echo "=================================================================="
echo "3. Cross-ticker — momentum on three names"
echo "=================================================================="
uv run python -m src.validation.cli evaluate \
  --signal momentum \
  --ticker AAPL,MSFT,NVDA \
  --start 2023-06-01 --end 2025-04-01 \
  --n-splits 6 --n-test-splits 2 \
  --rolling-window 180 \
  --out /tmp/val_momentum_3names.json

echo
echo "Done. JSON results written to /tmp/val_*.json"
