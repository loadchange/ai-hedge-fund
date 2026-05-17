"""Daily analysis scheduler with trading-calendar awareness.

Runs the hedge fund analysis pipeline on a configurable schedule,
skipping non-trading days. Supports cron-like scheduling via the
``schedule`` library and a continuous watch mode during market hours.
"""

from __future__ import annotations

import logging
import signal
import time
from datetime import date, datetime
from typing import Callable, Literal

from src.data.trading_calendar import Market, get_trading_calendar

logger = logging.getLogger(__name__)


class DailyScheduler:
    """Runs analysis on trading days at configured times."""

    def __init__(
        self,
        tickers: list[str],
        markets: list[Market] | None = None,
        run_fn: Callable | None = None,
        schedule_time: str = "09:30",
        timezone_name: str = "Asia/Shanghai",
    ):
        """
        Args:
            tickers: Tickers to analyze.
            markets: Which markets to check for trading day.
            run_fn: Callable that runs the analysis.
            schedule_time: HH:MM in local time.
            timezone_name: Timezone for schedule.
        """
        self.tickers = tickers
        self.markets = markets or ["us", "cn", "hk"]
        self.run_fn = run_fn
        self.schedule_time = schedule_time
        self.timezone_name = timezone_name
        self._running = False

    def should_run_today(self) -> bool:
        """Check if today is a trading day for any of the configured markets."""
        today = date.today()
        cal = get_trading_calendar()
        return any(cal.is_trading_day(today, m) for m in self.markets)

    def run_once(self) -> dict | None:
        """Run analysis once (for manual trigger or testing)."""
        if not self.should_run_today():
            logger.info("Today is not a trading day for any configured market. Skipping.")
            return None
        if not self.run_fn:
            logger.warning("No run_fn configured. Nothing to do.")
            return None
        try:
            return self.run_fn()
        except Exception as exc:
            logger.exception("Scheduled analysis failed: %s", exc)
            return None

    def start(self) -> None:
        """Start the scheduler loop (blocking)."""
        import schedule as _schedule

        _schedule.every().day.at(self.schedule_time).do(self._scheduled_run)
        self._running = True
        logger.info(
            "Scheduler started: will run at %s (%s) on trading days for markets %s",
            self.schedule_time, self.timezone_name, self.markets,
        )

        def _shutdown(signum, frame):
            logger.info("Received signal %s, shutting down scheduler.", signum)
            self._running = False

        signal.signal(signal.SIGINT, _shutdown)
        signal.signal(signal.SIGTERM, _shutdown)

        while self._running:
            _schedule.run_pending()
            time.sleep(30)

    def stop(self) -> None:
        """Stop the scheduler."""
        self._running = False

    def _scheduled_run(self) -> None:
        """Execute a scheduled run."""
        logger.info("Scheduled run triggered at %s", datetime.now().isoformat())
        result = self.run_once()
        if result is not None:
            logger.info("Scheduled analysis completed successfully")
        else:
            logger.info("Scheduled analysis skipped or failed")


class WatchMode:
    """Continuous re-run every N minutes during market hours."""

    def __init__(
        self,
        interval_minutes: int,
        markets: list[Market] | None = None,
        run_fn: Callable | None = None,
    ):
        self.interval_minutes = max(1, interval_minutes)
        self.markets = markets or ["us", "cn", "hk"]
        self.run_fn = run_fn
        self._running = False

    def start(self) -> None:
        """Start watch mode loop (blocking)."""
        cal = get_trading_calendar()
        self._running = True
        logger.info("Watch mode started: every %d minutes during market hours", self.interval_minutes)

        def _shutdown(signum, frame):
            self._running = False

        signal.signal(signal.SIGINT, _shutdown)
        signal.signal(signal.SIGTERM, _shutdown)

        while self._running:
            any_open = any(cal.is_market_open(m) for m in self.markets)
            if any_open:
                logger.info("Market is open — running analysis")
                try:
                    if self.run_fn:
                        self.run_fn()
                except Exception as exc:
                    logger.exception("Watch mode analysis failed: %s", exc)
            else:
                logger.debug("No market currently open — waiting")

            for _ in range(self.interval_minutes * 60 // 30):
                if not self._running:
                    return
                time.sleep(30)
