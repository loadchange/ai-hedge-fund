# Examples

End-to-end demos that exercise the **non-LLM** quantitative modules.
None of these scripts call an LLM, so you can run them without any
LLM API key.

The only thing they need is **network access** for price data —
yfinance / akshare / baostock / tencent are all free and require no
key.

| File | What it demonstrates |
|---|---|
| `01_signal_validation.sh` | `python -m src.validation.cli evaluate` — CPCV + PBO + Deflated Sharpe across signals & tickers |
| `02_signals_to_portfolio.py` | Pull prices, run a few `BaseSignal` subclasses, build a Black-Litterman view from those signals, and solve mean-variance |
| `03_risk_analytics.py` | Drawdown stats + scenario stress test on a synthetic equity curve |
| `04_event_study.py` | Fit a market model, compute CAR around a fake event, run t-test + Wilcoxon |
| `05_features.py` | SUE, KPI momentum (z-scored YoY), Granger causality between two series |

Run any one with:

```bash
uv run bash examples/01_signal_validation.sh
uv run python examples/02_signals_to_portfolio.py
# …
```
