"""Trading calendar helpers.

This module provides NYSE (XNYS) session-aware date arithmetic.

Why:
- Many parts of the pipeline historically used pandas business-day offsets
  (e.g., `BDay`) which do NOT match actual exchange sessions.
- For rolling-origin / walk-forward simulation, we need true session counts
  (e.g., "21 trading days", "63 trading days").

Implementation:
- Preferred backend: `exchange_calendars` (industry standard; accurate holidays).
- Fallback backend (only if dependency is missing): Mon–Fri business days.

All returned timestamps are tz-naive session labels (midnight dates).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd


@dataclass(frozen=True)
class TradingCalendar:
    """A minimal interface for session arithmetic."""

    name: str

    def session_on_or_after(self, ts: pd.Timestamp) -> pd.Timestamp:  # pragma: no cover
        raise NotImplementedError

    def session_on_or_before(self, ts: pd.Timestamp) -> pd.Timestamp:  # pragma: no cover
        raise NotImplementedError

    def add_sessions(self, ts: pd.Timestamp, n: int) -> pd.Timestamp:  # pragma: no cover
        raise NotImplementedError


class _ExchangeCalendarsXNYS(TradingCalendar):
    def __init__(self) -> None:
        super().__init__(name="XNYS")
        import exchange_calendars as xcals  # type: ignore

        self._cal = xcals.get_calendar("XNYS")
        self._sessions = self._cal.sessions
        self._first_session = pd.Timestamp(self._sessions[0])
        self._last_session = pd.Timestamp(self._sessions[-1])
        self._fallback = _WeekdayCalendar()

    @staticmethod
    def _norm(ts: pd.Timestamp) -> pd.Timestamp:
        ts = pd.Timestamp(ts)
        if getattr(ts, "tzinfo", None) is not None:
            ts = ts.tz_localize(None)
        return ts.normalize()

    def session_on_or_after(self, ts: pd.Timestamp) -> pd.Timestamp:
        norm = self._norm(ts)
        if norm < self._first_session or norm > self._last_session:
            return self._fallback.session_on_or_after(norm)
        return pd.Timestamp(self._cal.date_to_session(norm, direction="next"))

    def session_on_or_before(self, ts: pd.Timestamp) -> pd.Timestamp:
        norm = self._norm(ts)
        if norm < self._first_session or norm > self._last_session:
            return self._fallback.session_on_or_before(norm)
        return pd.Timestamp(self._cal.date_to_session(norm, direction="previous"))

    def add_sessions(self, ts: pd.Timestamp, n: int) -> pd.Timestamp:
        base = self.session_on_or_after(ts)
        # If the base session is outside XNYS coverage, use the fallback.
        if base < self._first_session or base > self._last_session:
            return self._fallback.add_sessions(base, n)

        pos = int(self._sessions.get_loc(base))
        new_pos = pos + int(n)
        if new_pos < 0 or new_pos >= len(self._sessions):
            raise IndexError(f"Session index out of range: {base} + {n}")
        return pd.Timestamp(self._sessions[new_pos])


class _WeekdayCalendar(TradingCalendar):
    def __init__(self) -> None:
        super().__init__(name="WEEKDAY")

    @staticmethod
    def _norm(ts: pd.Timestamp) -> pd.Timestamp:
        ts = pd.Timestamp(ts)
        if getattr(ts, "tzinfo", None) is not None:
            ts = ts.tz_localize(None)
        return ts.normalize()

    def session_on_or_after(self, ts: pd.Timestamp) -> pd.Timestamp:
        d = self._norm(ts)
        while d.weekday() >= 5:
            d = d + pd.Timedelta(days=1)
        return d

    def session_on_or_before(self, ts: pd.Timestamp) -> pd.Timestamp:
        d = self._norm(ts)
        while d.weekday() >= 5:
            d = d - pd.Timedelta(days=1)
        return d

    def add_sessions(self, ts: pd.Timestamp, n: int) -> pd.Timestamp:
        # Mon–Fri only; does not skip holidays.
        d = self.session_on_or_after(ts)
        step = 1 if n >= 0 else -1
        remaining = abs(int(n))
        while remaining > 0:
            d = d + pd.Timedelta(days=step)
            if d.weekday() < 5:
                remaining -= 1
        return d


_CACHED: Optional[TradingCalendar] = None


def get_trading_calendar() -> TradingCalendar:
    """Return the best available calendar.

    Prefers XNYS via exchange_calendars; falls back to weekday-only.
    """

    global _CACHED
    if _CACHED is not None:
        return _CACHED

    try:
        _CACHED = _ExchangeCalendarsXNYS()
    except Exception:
        _CACHED = _WeekdayCalendar()

    return _CACHED


def session_on_or_after(ts: pd.Timestamp) -> pd.Timestamp:
    return get_trading_calendar().session_on_or_after(pd.Timestamp(ts))


def session_on_or_before(ts: pd.Timestamp) -> pd.Timestamp:
    return get_trading_calendar().session_on_or_before(pd.Timestamp(ts))


def add_sessions(ts: pd.Timestamp, n: int) -> pd.Timestamp:
    return get_trading_calendar().add_sessions(pd.Timestamp(ts), int(n))
