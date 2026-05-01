"""Position sizing helpers — Kelly criterion + volatility targeting."""

from __future__ import annotations


def kelly_fraction(win_rate: float, win_loss_ratio: float, *, fractional: float = 0.5) -> float:
    """Kelly bet fraction (clamped to ``[0, 1]``).

    The classic formula is ``f* = p - q/b`` where ``p`` is win rate,
    ``q = 1 - p``, and ``b`` is win/loss ratio. Most practitioners use a
    *fractional* Kelly (default 0.5) to reduce variance — the literature
    shows full Kelly is too aggressive for any drawdown sensitivity.

    Returns 0 when the edge is non-positive (don't bet).
    """
    if win_rate <= 0 or win_rate >= 1 or win_loss_ratio <= 0:
        return 0.0
    full_kelly = win_rate - (1.0 - win_rate) / win_loss_ratio
    if full_kelly <= 0:
        return 0.0
    return max(0.0, min(1.0, full_kelly * fractional))


def vol_targeted_size(
    portfolio_value: float,
    *,
    target_annual_vol: float = 0.15,
    asset_annual_vol: float,
    cap_fraction: float = 0.25,
) -> float:
    """Dollar position size for a given asset to hit *target_annual_vol*.

    Equation: ``position = portfolio_value × (target_vol / asset_vol)``,
    clipped to *cap_fraction* of the portfolio. ``asset_annual_vol = 0``
    returns 0 (no signal, no risk model).

    Defaults imply: target 15% portfolio vol, single-position cap 25%.
    """
    if portfolio_value <= 0 or asset_annual_vol <= 0:
        return 0.0
    raw = portfolio_value * (target_annual_vol / asset_annual_vol)
    return max(0.0, min(raw, portfolio_value * cap_fraction))
