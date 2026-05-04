"""Base class for all quantitative signals.

Two extension points:

* :meth:`BaseSignal.compute_from_prices` — synchronous, takes a price
  ``DataFrame`` already in memory. Used by the technical-analysis pipeline
  where the LangGraph agent has already pulled the price history.
* :meth:`BaseSignal.compute` — pulls whatever data the signal needs via the
  shared cache + ``DataSourceManager``. Used by validation /
  event-study / batch backtest runners that don't have a DataFrame yet.

Subclasses override at least one of the two; the unimplemented one raises
:class:`NotImplementedError`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Literal

import numpy as np
import pandas as pd

from src.signals.types import SignalResult

SignalKind = Literal["technical", "fundamental"]


class BaseSignal(ABC):
    """Abstract base for quantitative signals (no LLM, pure math).

    The ``kind`` property splits signals into two families that are
    evaluated very differently:

    * ``technical`` — derived purely from price/volume time series.
      Implements :meth:`compute_from_prices`, so the validation runner
      can roll a window forward day-by-day and CPCV produces a daily
      strategy return.
    * ``fundamental`` — derived from reported financials, analyst
      estimates, or earnings dates. Updates on report dates rather than
      every trading day, so day-by-day rolling is meaningless. The
      validation runner refuses these in CPCV mode.

    Concrete signals must override ``name`` and ``kind``; price-driven
    ones additionally override ``compute_from_prices``.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Signal identifier (e.g. ``"trend"``, ``"value"``)."""
        ...

    @property
    @abstractmethod
    def kind(self) -> SignalKind:
        """``"technical"`` for price-driven, ``"fundamental"`` otherwise."""
        ...

    # ------------------------------------------------------------------
    # Public compute methods — at least one should be implemented.
    # ------------------------------------------------------------------

    def compute(self, ticker: str, end_date: str, **kwargs) -> SignalResult:
        """Compute the signal for *ticker* as-of *end_date*.

        Default implementation pulls a 1-year window of prices and
        delegates to :meth:`compute_from_prices`. Fundamental signals
        override this to pull the appropriate dataset (financial metrics,
        earnings, etc.).
        """
        from datetime import datetime
        from dateutil.relativedelta import relativedelta
        from src.tools.api import get_prices, prices_to_df

        start = (datetime.strptime(end_date, "%Y-%m-%d") - relativedelta(years=1)).strftime("%Y-%m-%d")
        prices = get_prices(ticker, start, end_date)
        if not prices:
            return self._empty_result()
        return self.compute_from_prices(prices_to_df(prices))

    def compute_from_prices(self, prices_df: pd.DataFrame) -> SignalResult:
        """Compute the signal from an in-memory OHLCV DataFrame.

        Override in price-driven signals (trend / momentum / volatility / …).
        Default raises ``NotImplementedError`` so signals that only support
        the ticker-based API fail loudly when misused.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support compute_from_prices()."
        )

    # ------------------------------------------------------------------
    # Helpers shared across concrete signals.
    # ------------------------------------------------------------------

    def _empty_result(self) -> SignalResult:
        return SignalResult(signal_name=self.name, value=0.0, components={})

    @staticmethod
    def _safe_float(value, default: float = 0.0) -> float:
        """Convert to float, returning *default* for NaN / None / errors."""
        if value is None:
            return default
        try:
            f = float(value)
            return default if (np.isnan(f) or np.isinf(f)) else f
        except (ValueError, TypeError):
            return default

    @staticmethod
    def _percentile_rank(value: float, values: list[float]) -> float:
        """Return the percentile rank (0-100) of *value* within *values*."""
        if not values:
            return 50.0
        below = sum(1 for v in values if v < value)
        return (below / len(values)) * 100.0

    @staticmethod
    def _normalize_to_signal(raw: float, low: float = -1.0, high: float = 1.0) -> float:
        """Clamp *raw* into ``[low, high]``."""
        if np.isnan(raw) or np.isinf(raw):
            return 0.0
        return max(low, min(high, raw))

    @staticmethod
    def _sigmoid(x: float, scale: float = 5.0) -> float:
        """Map an unbounded value into ``(-1, +1)`` via scaled tanh."""
        if np.isnan(x) or np.isinf(x):
            return 0.0
        return float(np.tanh(x * scale))

    @staticmethod
    def _signal_to_value(signal: str, confidence: float) -> float:
        """Convert a categorical (``bullish``/``bearish``/``neutral``) +
        ``[0, 1]`` confidence into a signed ``[-1, +1]`` value."""
        confidence = max(0.0, min(1.0, confidence or 0.0))
        if signal == "bullish":
            return confidence
        if signal == "bearish":
            return -confidence
        return 0.0

    @staticmethod
    def _value_to_signal(value: float) -> tuple[str, float]:
        """Inverse of :meth:`_signal_to_value`. Returns ``(signal, confidence)``.

        Threshold ``0.2`` matches the legacy ``weighted_signal_combination``
        cut-off so existing backtest output is preserved.
        """
        if np.isnan(value) or np.isinf(value):
            return "neutral", 0.0
        if value > 0.2:
            return "bullish", abs(value)
        if value < -0.2:
            return "bearish", abs(value)
        return "neutral", abs(value)
