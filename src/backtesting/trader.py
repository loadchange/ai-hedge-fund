from __future__ import annotations

from .costs import TransactionCostModel
from .portfolio import Portfolio
from .types import Action, ActionLiteral


class TradeExecutor:
    """Executes trades against a Portfolio with Backtester-identical semantics.

    When *cost_model* is provided, every executed trade charges its
    estimated transaction cost back to the portfolio cash and the total
    cost is tracked on this executor (``total_costs_paid`` / ``costs_by_action``).
    """

    def __init__(self, cost_model: TransactionCostModel | None = None) -> None:
        self.cost_model = cost_model
        self.total_costs_paid: float = 0.0
        self.costs_by_action: dict[str, float] = {
            "buy": 0.0, "sell": 0.0, "short": 0.0, "cover": 0.0,
        }

    def execute_trade(
        self,
        ticker: str,
        action: ActionLiteral,
        quantity: float,
        current_price: float,
        portfolio: Portfolio,
    ) -> int:
        if quantity is None or quantity <= 0:
            return 0

        try:
            action_enum = Action(action) if not isinstance(action, Action) else action
        except Exception:
            action_enum = Action.HOLD

        if action_enum == Action.BUY:
            executed = portfolio.apply_long_buy(ticker, int(quantity), float(current_price))
        elif action_enum == Action.SELL:
            executed = portfolio.apply_long_sell(ticker, int(quantity), float(current_price))
        elif action_enum == Action.SHORT:
            executed = portfolio.apply_short_open(ticker, int(quantity), float(current_price))
        elif action_enum == Action.COVER:
            executed = portfolio.apply_short_cover(ticker, int(quantity), float(current_price))
        else:
            return 0

        # Charge the trade cost (if any) after a successful fill.
        if executed > 0 and self.cost_model is not None:
            cost = self.cost_model.estimate(
                ticker=ticker,
                action=action_enum.value,
                quantity=executed,
                price=float(current_price),
            )
            if cost > 0:
                portfolio.deduct_cash(cost)
                self.total_costs_paid += cost
                key = action_enum.value
                if key in self.costs_by_action:
                    self.costs_by_action[key] += cost

        return executed
