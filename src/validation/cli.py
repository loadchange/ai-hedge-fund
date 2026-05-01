"""CLI for signal validation: ``python -m src.validation.cli evaluate ...``.

Examples::

    uv run python -m src.validation.cli evaluate --signal momentum \
        --ticker AAPL --start 2020-01-01 --end 2025-01-01 \
        --n-splits 8 --n-test-splits 2

    # Compare multiple signals on the same ticker
    uv run python -m src.validation.cli evaluate --signal momentum,trend,mean_reversion \
        --ticker AAPL --start 2020-01-01 --end 2025-01-01

Writes ``validation_results.json`` next to the output for downstream
tooling (issue bot artifact, dashboards).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from src.signals import SIGNAL_REGISTRY
from src.validation.runner import evaluate_signal


def _print_result(result: dict) -> None:
    if "error" in result:
        print(f"  ⚠ {result['ticker']} / {result['signal']}: {result['error']}")
        return
    is_mean = result["is_sharpe_mean"]
    oos_mean = result["oos_sharpe_mean"]
    pbo = result["pbo"]["pbo"]
    dsr = result["deflated_sharpe"]
    print(
        f"  {result['ticker']:8} {result['signal']:18}  "
        f"IS Sharpe={is_mean:+.2f}  OOS Sharpe={oos_mean:+.2f}  "
        f"PBO={pbo:.2f}  DSR={dsr:.2f}"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="src.validation.cli", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    eval_p = sub.add_parser("evaluate", help="Evaluate a signal via CPCV + PBO")
    eval_p.add_argument(
        "--signal",
        required=True,
        help=f"Signal key(s), comma-separated. Available: {','.join(sorted(SIGNAL_REGISTRY))}",
    )
    eval_p.add_argument("--ticker", required=True, help="Ticker (US). Comma-separated for batch.")
    eval_p.add_argument("--start", required=True, dest="start_date")
    eval_p.add_argument("--end", required=True, dest="end_date")
    eval_p.add_argument("--n-splits", type=int, default=8)
    eval_p.add_argument("--n-test-splits", type=int, default=2)
    eval_p.add_argument("--embargo-pct", type=float, default=0.01)
    eval_p.add_argument("--rolling-window", type=int, default=60)
    eval_p.add_argument("--out", default="validation_results.json")

    args = parser.parse_args(argv)

    signals = [s.strip() for s in args.signal.split(",") if s.strip()]
    tickers = [t.strip() for t in args.ticker.split(",") if t.strip()]

    print(
        f"Evaluating {len(signals)} signal(s) × {len(tickers)} ticker(s)\n"
        f"  Window: {args.start_date} → {args.end_date}\n"
        f"  CPCV: {args.n_splits} splits, {args.n_test_splits} test splits, embargo {args.embargo_pct:.0%}\n"
    )

    results: list[dict] = []
    for ticker in tickers:
        for sig in signals:
            try:
                result = evaluate_signal(
                    sig,
                    ticker,
                    start_date=args.start_date,
                    end_date=args.end_date,
                    n_splits=args.n_splits,
                    n_test_splits=args.n_test_splits,
                    embargo_pct=args.embargo_pct,
                    rolling_window=args.rolling_window,
                )
            except Exception as e:  # noqa: BLE001
                result = {
                    "signal": sig,
                    "ticker": ticker,
                    "error": f"{type(e).__name__}: {e}",
                }
            _print_result(result)
            results.append(result)

    Path(args.out).write_text(
        json.dumps(results, ensure_ascii=False, default=str, indent=2),
        encoding="utf-8",
    )
    print(f"\nFull results written to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
