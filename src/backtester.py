import sys

from colorama import Fore, Style

from src.main import run_hedge_fund
from src.backtesting.costs import build_cost_model
from src.backtesting.engine import BacktestEngine
from src.backtesting.types import PerformanceMetrics
from src.cli.input import (
    parse_cli_inputs,
)
from src.i18n import get_text


def run_backtest(backtester: BacktestEngine) -> PerformanceMetrics | None:
    """Run the backtest with graceful KeyboardInterrupt handling."""
    try:
        performance_metrics = backtester.run_backtest()
        print(f"\n{Fore.GREEN}{get_text('backtest_completed')}{Style.RESET_ALL}")

        # Surface transaction costs alongside the rest of the summary so
        # cost-aware tuning is visible at the CLI level (issue-bot
        # markdown reply has shown them since Phase 5).
        try:
            total_costs = backtester.get_total_transaction_costs()
            if total_costs > 0:
                breakdown = backtester.get_costs_by_action()
                detail = ", ".join(
                    f"{action}=${amount:,.2f}"
                    for action, amount in breakdown.items() if amount > 0
                )
                print(
                    f"{Fore.CYAN}Total Transaction Costs: ${total_costs:,.2f}"
                    f"{(' (' + detail + ')') if detail else ''}{Style.RESET_ALL}"
                )
        except Exception:  # pragma: no cover — purely cosmetic readout
            pass

        return performance_metrics
    except KeyboardInterrupt:
        print(f"\n\n{Fore.YELLOW}{get_text('backtest_interrupted')}{Style.RESET_ALL}")

        # Try to show any partial results that were computed
        try:
            portfolio_values = backtester.get_portfolio_values()
            if len(portfolio_values) > 1:
                print(f"{Fore.GREEN}{get_text('partial_results')}{Style.RESET_ALL}")

                # Show basic summary from the available portfolio values
                first_value = portfolio_values[0]["Portfolio Value"]
                last_value = portfolio_values[-1]["Portfolio Value"]
                total_return = ((last_value - first_value) / first_value) * 100

                print(f"{Fore.CYAN}{get_text('initial_value')}: ${first_value:,.2f}{Style.RESET_ALL}")
                print(f"{Fore.CYAN}{get_text('final_value')}: ${last_value:,.2f}{Style.RESET_ALL}")
                print(f"{Fore.CYAN}{get_text('total_return')}: {total_return:+.2f}%{Style.RESET_ALL}")
        except Exception as e:
            print(f"{Fore.RED}{get_text('could_not_generate', str(e))}{Style.RESET_ALL}")

        sys.exit(0)


### Run the Backtest #####
if __name__ == "__main__":
    inputs = parse_cli_inputs(
        description="Run backtesting simulation",
        require_tickers=False,
        default_months_back=1,
        include_graph_flag=False,
        include_reasoning_flag=False,
    )

    cost_model = build_cost_model(inputs.cost_model, bps=inputs.cost_bps)

    # Create and run the backtester
    backtester = BacktestEngine(
        agent=run_hedge_fund,
        tickers=inputs.tickers,
        start_date=inputs.start_date,
        end_date=inputs.end_date,
        initial_capital=inputs.initial_cash,
        model_name=inputs.model_name,
        model_provider=inputs.model_provider,
        selected_analysts=inputs.selected_analysts,
        initial_margin_requirement=inputs.margin_requirement,
        cost_model=cost_model,
    )

    # Run the backtest with graceful exit handling
    performance_metrics = run_backtest(backtester)
