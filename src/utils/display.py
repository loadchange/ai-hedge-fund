import os
import json
from colorama import Fore, Style
from rich.table import Table
from rich.text import Text
from rich.console import Console
from rich import box

from .analysts import ANALYST_ORDER
from src.i18n import get_text, translate_agent_name, translate_signal, translate_action, summarize_json_reasoning

_console = Console()


# ── Helpers ───────────────────────────────────────────────────────────────────

def _strip_color(s: str) -> str:
    """Remove colorama ANSI codes from a string."""
    for code in (Fore.BLACK, Fore.RED, Fore.GREEN, Fore.YELLOW, Fore.BLUE,
                 Fore.MAGENTA, Fore.CYAN, Fore.WHITE, Fore.RESET,
                 Style.BRIGHT, Style.DIM, Style.NORMAL, Style.RESET_ALL):
        s = s.replace(code, "")
    return s


def _to_rich(text: str) -> Text:
    """Convert a colorama-formatted string to a rich Text object."""
    return Text.from_ansi(str(text))


def sort_agent_signals(signals):
    """Sort agent signals in a consistent order."""
    analyst_order = {display: idx for idx, (display, _) in enumerate(ANALYST_ORDER)}
    analyst_order["Risk Management"] = len(ANALYST_ORDER)
    return sorted(signals, key=lambda x: analyst_order.get(x[0], 999))


def _get_color_for_signal(signal_type: str):
    return {"BULLISH": Fore.GREEN, "BEARISH": Fore.RED, "NEUTRAL": Fore.YELLOW}.get(signal_type, Fore.WHITE)


def _get_color_for_action(action: str):
    return {"BUY": Fore.GREEN, "SELL": Fore.RED, "HOLD": Fore.YELLOW,
            "COVER": Fore.GREEN, "SHORT": Fore.RED}.get(action, Fore.WHITE)


# ── Table builder ─────────────────────────────────────────────────────────────

def _make_table(headers: list[str], rows: list[list[str]], colalign: list[str] | None = None,
                expand_col: int | None = None, plain_cols: set[int] | None = None,
                show_header: bool = True) -> Table:
    """Build a rich Table that auto-sizes to terminal width.

    Args:
        expand_col: Index of the column that should expand to fill remaining space.
        plain_cols: Set of column indices that contain plain text (no ANSI codes).
        show_header: Whether to show the header row.
    """
    if plain_cols is None:
        plain_cols = set()
    table = Table(box=box.ROUNDED, show_header=show_header, header_style="bold", pad_edge=False, show_lines=True,
                  expand=expand_col is not None)
    ncols = len(headers)
    if colalign is None:
        colalign = ["left"] * ncols
    while len(colalign) < ncols:
        colalign.append("left")

    justify_map = {"left": "left", "right": "right", "center": "center"}

    for i, header in enumerate(headers):
        table.add_column(
            _to_rich(header),
            justify=justify_map.get(colalign[i], "left"),
            no_wrap=False,
            ratio=1 if i == expand_col else None,
            overflow="fold" if i == expand_col else "ellipsis",
        )
    for row in rows:
        cells = []
        for i, cell in enumerate(row):
            if i in plain_cols:
                cells.append(Text(str(cell)))
            else:
                cells.append(_to_rich(cell))
        table.add_row(*cells)
    return table


def _print_table(headers: list[str], rows: list[list[str]], colalign: list[str] | None = None,
                 expand_col: int | None = None, plain_cols: set[int] | None = None,
                 show_header: bool = True):
    """Render and print a table using rich (auto-width, CJK-safe)."""
    table = _make_table(headers, rows, colalign, expand_col=expand_col, plain_cols=plain_cols, show_header=show_header)
    _console.print(table)


# ── Main output ──────────────────────────────────────────────────────────────

def print_trading_output(result: dict) -> None:
    """Print formatted trading results with colored tables for multiple tickers."""
    decisions = result.get("decisions")
    if not decisions:
        print(f"{Fore.RED}{get_text('no_decisions')}{Style.RESET_ALL}")
        return

    for ticker, decision in decisions.items():
        print(f"\n{Fore.WHITE}{Style.BRIGHT}{get_text('analysis_for')} {Fore.CYAN}{ticker}{Style.RESET_ALL}")
        print(f"{Fore.WHITE}{Style.BRIGHT}{'=' * 50}{Style.RESET_ALL}")

        # ── Agent Analysis Table ─────────────────────────────────────────
        table_data = []
        for agent, signals in result.get("analyst_signals", {}).items():
            if ticker not in signals:
                continue
            if agent == "risk_management_agent":
                continue

            signal = signals[ticker]
            agent_name = agent.replace("_agent", "").replace("_", " ").title()
            agent_name = translate_agent_name(agent_name)
            signal_type = signal.get("signal", "").upper()
            signal_display = translate_signal(signal_type)
            confidence = signal.get("confidence", 0)
            signal_color = _get_color_for_signal(signal_type)

            reasoning_raw = signal.get("reasoning", "")
            reasoning_str = summarize_json_reasoning(reasoning_raw)

            table_data.append([
                f"{Fore.CYAN}{agent_name}{Style.RESET_ALL}",
                f"{signal_color}{signal_display}{Style.RESET_ALL}",
                f"{Fore.WHITE}{confidence}%{Style.RESET_ALL}",
                reasoning_str,  # plain text — no ANSI, Rich handles width correctly
            ])

        table_data = sort_agent_signals(table_data)

        print(f"\n{Fore.WHITE}{Style.BRIGHT}{get_text('agent_analysis')}:{Style.RESET_ALL} [{Fore.CYAN}{ticker}{Style.RESET_ALL}]")
        _print_table(
            headers=[get_text("agent"), get_text("signal"), get_text("confidence"), get_text("reasoning")],
            rows=table_data,
            colalign=["left", "center", "right", "left"],
            expand_col=3,  # reasoning column expands
            plain_cols={3},  # reasoning is plain text, no ANSI
        )

        # ── Trading Decision Table ───────────────────────────────────────
        action = decision.get("action", "").upper()
        action_display = translate_action(action)
        action_color = _get_color_for_action(action)
        reasoning = summarize_json_reasoning(decision.get("reasoning", ""))

        decision_data = [
            [get_text("action"),     f"{action_color}{action_display}{Style.RESET_ALL}"],
            [get_text("quantity"),   f"{action_color}{decision.get('quantity', 0)}{Style.RESET_ALL}"],
            [get_text("confidence"), f"{Fore.WHITE}{decision.get('confidence', 0):.1f}%{Style.RESET_ALL}"],
            [get_text("reasoning"),  reasoning],
        ]

        print(f"\n{Fore.WHITE}{Style.BRIGHT}{get_text('trading_decision')}:{Style.RESET_ALL} [{Fore.CYAN}{ticker}{Style.RESET_ALL}]")
        _print_table(
            headers=["", ""],
            show_header=False,
            rows=decision_data,
            colalign=["left", "left"],
            expand_col=1,  # value column expands
        )

    # ── Portfolio Summary ────────────────────────────────────────────────
    print(f"\n{Fore.WHITE}{Style.BRIGHT}{get_text('portfolio_summary')}:{Style.RESET_ALL}")

    portfolio_manager_reasoning = None
    for ticker, decision in decisions.items():
        if decision.get("reasoning"):
            portfolio_manager_reasoning = decision.get("reasoning")
            break

    analyst_signals = result.get("analyst_signals", {})
    portfolio_data = []
    for ticker, decision in decisions.items():
        action = decision.get("action", "").upper()
        action_display = translate_action(action)
        action_color = _get_color_for_action(action)

        bullish = bearish = neutral = 0
        if analyst_signals:
            for agent, signals in analyst_signals.items():
                if ticker in signals:
                    sig = signals[ticker].get("signal", "").upper()
                    if sig == "BULLISH":
                        bullish += 1
                    elif sig == "BEARISH":
                        bearish += 1
                    elif sig == "NEUTRAL":
                        neutral += 1

        portfolio_data.append([
            f"{Fore.CYAN}{ticker}{Style.RESET_ALL}",
            f"{action_color}{action_display}{Style.RESET_ALL}",
            f"{action_color}{decision.get('quantity')}{Style.RESET_ALL}",
            f"{Fore.WHITE}{decision.get('confidence'):.1f}%{Style.RESET_ALL}",
            f"{Fore.GREEN}{bullish}{Style.RESET_ALL}",
            f"{Fore.RED}{bearish}{Style.RESET_ALL}",
            f"{Fore.YELLOW}{neutral}{Style.RESET_ALL}",
        ])

    _print_table(
        headers=[
            get_text("ticker"), get_text("action"), get_text("quantity"),
            get_text("confidence"), get_text("bullish"), get_text("bearish"),
            get_text("neutral"),
        ],
        rows=portfolio_data,
        colalign=["left", "center", "right", "right", "center", "center", "center"],
    )

    # ── Portfolio Strategy ───────────────────────────────────────────────
    if portfolio_manager_reasoning:
        reasoning_str = summarize_json_reasoning(portfolio_manager_reasoning)
        print(f"\n{Fore.WHITE}{Style.BRIGHT}{get_text('portfolio_strategy')}:{Style.RESET_ALL}")
        print(f"{Fore.CYAN}{reasoning_str}{Style.RESET_ALL}")


# ── Backtest output ──────────────────────────────────────────────────────────

def print_backtest_results(table_rows: list) -> None:
    """Print the backtest results in a nicely formatted table."""
    os.system("cls" if os.name == "nt" else "clear")

    ticker_rows = []
    summary_rows = []
    for row in table_rows:
        if isinstance(row[1], str) and "PORTFOLIO SUMMARY" in row[1]:
            summary_rows.append(row)
        else:
            ticker_rows.append(row)

    if summary_rows:
        latest_summary = max(summary_rows, key=lambda r: r[0])
        print(f"\n{Fore.WHITE}{Style.BRIGHT}{get_text('portfolio_summary')}:{Style.RESET_ALL}")

        position_str = latest_summary[7].split("$")[1].split(Style.RESET_ALL)[0].replace(",", "")
        cash_str     = latest_summary[8].split("$")[1].split(Style.RESET_ALL)[0].replace(",", "")
        total_str    = latest_summary[9].split("$")[1].split(Style.RESET_ALL)[0].replace(",", "")

        print(f"{get_text('cash_balance')}: {Fore.CYAN}${float(cash_str):,.2f}{Style.RESET_ALL}")
        print(f"{get_text('total_position_value')}: {Fore.YELLOW}${float(position_str):,.2f}{Style.RESET_ALL}")
        print(f"{get_text('total_value')}: {Fore.WHITE}${float(total_str):,.2f}{Style.RESET_ALL}")
        print(f"{get_text('portfolio_return')}: {latest_summary[10]}")
        if len(latest_summary) > 14 and latest_summary[14]:
            print(f"{get_text('benchmark_return')}: {latest_summary[14]}")
        if latest_summary[11]:
            print(f"{get_text('sharpe_ratio')}: {latest_summary[11]}")
        if latest_summary[12]:
            print(f"{get_text('sortino_ratio')}: {latest_summary[12]}")
        if latest_summary[13]:
            print(f"{get_text('max_drawdown')}: {latest_summary[13]}")

    print("\n" * 2)

    _print_table(
        headers=[
            "Date", get_text("ticker"), get_text("action"), get_text("quantity"),
            "Price", "Long Shares", "Short Shares", "Position Value",
        ],
        rows=ticker_rows,
        colalign=["left", "left", "center", "right", "right", "right", "right", "right"],
    )

    print("\n" * 4)


def format_backtest_row(
    date: str, ticker: str, action: str, quantity: float, price: float,
    long_shares: float = 0, short_shares: float = 0, position_value: float = 0,
    is_summary: bool = False, total_value: float = None, return_pct: float = None,
    cash_balance: float = None, total_position_value: float = None,
    sharpe_ratio: float = None, sortino_ratio: float = None, max_drawdown: float = None,
    benchmark_return_pct: float | None = None,
) -> list[any]:
    """Format a row for the backtest results table."""
    action_color = {
        "BUY": Fore.GREEN, "COVER": Fore.GREEN,
        "SELL": Fore.RED, "SHORT": Fore.RED, "HOLD": Fore.WHITE,
    }.get(action.upper(), Fore.WHITE)

    if is_summary:
        return_color = Fore.GREEN if return_pct >= 0 else Fore.RED
        benchmark_str = ""
        if benchmark_return_pct is not None:
            bench_color = Fore.GREEN if benchmark_return_pct >= 0 else Fore.RED
            benchmark_str = f"{bench_color}{benchmark_return_pct:+.2f}%{Style.RESET_ALL}"
        return [
            date,
            f"{Fore.WHITE}{Style.BRIGHT}PORTFOLIO SUMMARY{Style.RESET_ALL}",
            "", "", "", "", "",
            f"{Fore.YELLOW}${total_position_value:,.2f}{Style.RESET_ALL}",
            f"{Fore.CYAN}${cash_balance:,.2f}{Style.RESET_ALL}",
            f"{Fore.WHITE}${total_value:,.2f}{Style.RESET_ALL}",
            f"{return_color}{return_pct:+.2f}%{Style.RESET_ALL}",
            f"{Fore.YELLOW}{sharpe_ratio:.2f}{Style.RESET_ALL}" if sharpe_ratio is not None else "",
            f"{Fore.YELLOW}{sortino_ratio:.2f}{Style.RESET_ALL}" if sortino_ratio is not None else "",
            f"{Fore.RED}{max_drawdown:.2f}%{Style.RESET_ALL}" if max_drawdown is not None else "",
            benchmark_str,
        ]
    else:
        action_display = translate_action(action.upper())
        return [
            date,
            f"{Fore.CYAN}{ticker}{Style.RESET_ALL}",
            f"{action_color}{action_display}{Style.RESET_ALL}",
            f"{action_color}{quantity:,.0f}{Style.RESET_ALL}",
            f"{Fore.WHITE}{price:,.2f}{Style.RESET_ALL}",
            f"{Fore.GREEN}{long_shares:,.0f}{Style.RESET_ALL}",
            f"{Fore.RED}{short_shares:,.0f}{Style.RESET_ALL}",
            f"{Fore.YELLOW}${position_value:,.2f}{Style.RESET_ALL}",
        ]
