"""Multi-market trading calendar for CN/US/HK markets.

Provides trading day detection, market open/close checks, and date arithmetic
using the ``holidays`` library for holiday data. Falls back to simple weekday
checks when ``holidays`` is not installed.
"""

from __future__ import annotations

import logging
from datetime import date, datetime, time, timedelta
from typing import Literal
from zoneinfo import ZoneInfo

logger = logging.getLogger(__name__)

Market = Literal["cn", "us", "hk"]

_MARKET_TIMEZONES: dict[str, ZoneInfo] = {
    "cn": ZoneInfo("Asia/Shanghai"),
    "us": ZoneInfo("America/New_York"),
    "hk": ZoneInfo("Asia/Hong_Kong"),
}

_MARKET_HOURS: dict[str, tuple[time, time]] = {
    "cn": (time(9, 30), time(15, 0)),
    "us": (time(9, 30), time(16, 0)),
    "hk": (time(9, 30), time(16, 0)),
}

# ── Holiday helpers ────────────────────────────────────────────────────────────

def _build_holiday_set(market: Market, years: range) -> set[date]:
    """Build a set of holiday dates for the given market and year range."""
    try:
        import holidays as _holidays
    except ImportError:
        logger.warning("holidays library not installed; falling back to weekday-only check")
        return set()

    if market == "cn":
        cal = _holidays.China(years=years)
    elif market == "us":
        cal = _holidays.UnitedStates(years=years, observed=True)
    elif market == "hk":
        cal = _holidays.HongKong(years=years)
    else:
        return set()
    return set(cal.keys())


# ── Calendar class ─────────────────────────────────────────────────────────────

class TradingCalendar:
    """Multi-market trading day detection."""

    def __init__(self, year_start: int = 2020, year_end: int = 2030):
        years = range(year_start, year_end + 1)
        self._holidays: dict[Market, set[date]] = {
            "cn": _build_holiday_set("cn", years),
            "us": _build_holiday_set("us", years),
            "hk": _build_holiday_set("hk", years),
        }

    def _is_holiday(self, d: date, market: Market) -> bool:
        return d in self._holidays.get(market, set())

    def is_trading_day(self, d: date, market: Market = "us") -> bool:
        """Return True if *d* is a trading day (weekday and not a holiday)."""
        if d.weekday() >= 5:
            return False
        return not self._is_holiday(d, market)

    def next_trading_day(self, d: date, market: Market = "us") -> date:
        """Return the next trading day strictly after *d*."""
        candidate = d + timedelta(days=1)
        while not self.is_trading_day(candidate, market):
            candidate += timedelta(days=1)
        return candidate

    def previous_trading_day(self, d: date, market: Market = "us") -> date:
        """Return the previous trading day strictly before *d*."""
        candidate = d - timedelta(days=1)
        while not self.is_trading_day(candidate, market):
            candidate -= timedelta(days=1)
        return candidate

    def trading_days_between(self, start: date, end: date, market: Market = "us") -> list[date]:
        """Return all trading days in [start, end] inclusive."""
        if start > end:
            return []
        days = []
        candidate = start
        while candidate <= end:
            if self.is_trading_day(candidate, market):
                days.append(candidate)
            candidate += timedelta(days=1)
        return days

    def is_market_open(self, market: Market, now: datetime | None = None) -> bool:
        """Check if the market is currently open."""
        tz = _MARKET_TIMEZONES[market]
        if now is None:
            now = datetime.now(tz)
        else:
            now = now.astimezone(tz)

        local_date = now.date()
        if not self.is_trading_day(local_date, market):
            return False

        open_time, close_time = _MARKET_HOURS[market]
        local_time = now.time()
        return open_time <= local_time <= close_time


# ── Singleton ──────────────────────────────────────────────────────────────────

_calendar: TradingCalendar | None = None


def get_trading_calendar() -> TradingCalendar:
    global _calendar
    if _calendar is None:
        _calendar = TradingCalendar()
    return _calendar
