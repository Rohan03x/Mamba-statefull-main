from datetime import time as dt_time
from datetime import datetime
from typing import Iterable, Union, Optional

import pandas as pd
import pytz

try:  # exchange_calendars is an installed dependency in this repo
    import exchange_calendars as xcals  # type: ignore
except Exception:  # pragma: no cover
    xcals = None  # type: ignore

NY = pytz.timezone("America/New_York")
UTC = pytz.UTC

_XNYS_CAL = None


def nyse_sessions_in_range(
    start: Union[str, pd.Timestamp],
    end: Union[str, pd.Timestamp],
    *,
    tz_aware_utc: bool = True,
) -> pd.DatetimeIndex:
    """Return NYSE trading sessions between start and end (inclusive).

    - Uses `exchange_calendars` when available (preferred; handles holidays).
    - Falls back to pandas business days if calendar is unavailable.

    When tz_aware_utc=True, returns a tz-aware index localized to UTC midnight.
    This matches the output of `to_nyse_close_index*` helpers.
    """

    start_ts = pd.to_datetime(start)
    end_ts = pd.to_datetime(end)

    # exchange_calendars expects tz-naive session labels.
    # Accept tz-aware inputs by converting to UTC and dropping tz info.
    if getattr(start_ts, "tz", None) is not None:
        start_ts = start_ts.tz_convert(UTC).tz_localize(None)
    if getattr(end_ts, "tz", None) is not None:
        end_ts = end_ts.tz_convert(UTC).tz_localize(None)

    start_ts = start_ts.normalize()
    end_ts = end_ts.normalize()
    if start_ts > end_ts:
        start_ts, end_ts = end_ts, start_ts

    idx: pd.DatetimeIndex
    if xcals is not None:
        global _XNYS_CAL
        if _XNYS_CAL is None:
            _XNYS_CAL = xcals.get_calendar("XNYS")
        try:
            idx = _XNYS_CAL.sessions_in_range(start_ts, end_ts)  # type: ignore[attr-defined]
        except Exception:
            idx = pd.date_range(start=start_ts, end=end_ts, freq="B")
    else:
        idx = pd.date_range(start=start_ts, end=end_ts, freq="B")

    if len(idx) == 0:
        idx = pd.DatetimeIndex([end_ts])

    if tz_aware_utc:
        # exchange_calendars returns tz-naive session labels.
        if getattr(idx, "tz", None) is None:
            return idx.tz_localize(UTC)
        return idx.tz_convert(UTC)

    # tz-naive
    if getattr(idx, "tz", None) is not None:
        return idx.tz_convert(UTC).tz_localize(None)
    return idx


def to_nyse_close_index_from_series(series: pd.Series, *, align_sessions: bool = True) -> pd.Series:
    """Return a copy of the series indexed by the NYSE close date (stored as UTC midnight).
    Assumes the original index is tz-naive or UTC timestamps.
    """
    idx = pd.to_datetime(series.index)
    if idx.tz is None:
        # Heuristic: date-like indices (00:00:00) are typically already session labels.
        # Treat them as NY-local rather than UTC to avoid shifting to the prior day.
        if len(idx) and (idx.hour == 0).all() and (idx.minute == 0).all() and (idx.second == 0).all():
            idx = idx.tz_localize(NY)
        else:
            idx = idx.tz_localize(UTC).tz_convert(NY)
    else:
        # IMPORTANT:
        # Some families produce tz-aware UTC midnight indices as *session labels*
        # (e.g., a pd.date_range(..., tz='UTC')). Converting those to NY will
        # shift them to the prior NY date (00:00 UTC == 20:00 prior day NY),
        # truncating the last day and misaligning cache coverage.
        if (
            len(idx)
            and (idx.hour == 0).all()
            and (idx.minute == 0).all()
            and (idx.second == 0).all()
            and str(getattr(idx, "tz", "")) == "UTC"
        ):
            idx = idx.tz_localize(None).tz_localize(NY)
        else:
            idx = idx.tz_convert(NY)
    ny_dates = idx.normalize()
    utc_midnight = ny_dates.tz_convert(UTC).normalize()
    s = series.copy()
    s.index = utc_midnight
    s = s.sort_index()
    if align_sessions and not s.empty:
        sessions = nyse_sessions_in_range(s.index.min(), s.index.max(), tz_aware_utc=True)
        s = s[~s.index.duplicated(keep="last")].reindex(sessions)
    return s


def to_nyse_close_index(df: pd.DataFrame, *, align_sessions: bool = True) -> pd.DataFrame:
    """Set a DataFrame's index to NYSE close date (as UTC midnight).

    When align_sessions=True, the output is reindexed to actual NYSE trading
    sessions (drops weekends/holidays, adds missing sessions as NaN).
    """
    if df.empty:
        return df
    idx = pd.to_datetime(df.index)
    if idx.tz is None:
        # Heuristic: date-like indices (00:00:00) are typically already session labels.
        # Treat them as NY-local rather than UTC to avoid shifting to the prior day.
        if len(idx) and (idx.hour == 0).all() and (idx.minute == 0).all() and (idx.second == 0).all():
            idx = idx.tz_localize(NY)
        else:
            idx = idx.tz_localize(UTC).tz_convert(NY)
    else:
        # See note in to_nyse_close_index_from_series: tz-aware UTC midnight indices
        # are often already session labels and should not be converted to NY.
        if (
            len(idx)
            and (idx.hour == 0).all()
            and (idx.minute == 0).all()
            and (idx.second == 0).all()
            and str(getattr(idx, "tz", "")) == "UTC"
        ):
            idx = idx.tz_localize(None).tz_localize(NY)
        else:
            idx = idx.tz_convert(NY)
    ny_dates = idx.normalize()
    utc_midnight = ny_dates.tz_convert(UTC).normalize()
    out = df.copy()
    out.index = utc_midnight
    out = out.sort_index()
    if align_sessions and not out.empty:
        sessions = nyse_sessions_in_range(out.index.min(), out.index.max(), tz_aware_utc=True)
        out = out[~out.index.duplicated(keep="last")].reindex(sessions)
    return out


def lag_if_post_close(series: pd.Series, days: int = 1) -> pd.Series:
    """Apply a +days shift to represent next-day availability for post-close publications (e.g., FRED)."""
    return series.shift(days)


def to_market_session(
    timestamps: Union[pd.Series, pd.Index, Iterable],
    *,
    cutoff: dt_time = dt_time(16, 0, 0),
    tz: str = "America/New_York",
) -> pd.Series:
    """Map timestamps to market session dates with a strict cutoff.

    Any publication that lands *on or after* ``cutoff`` is attributed to the
    following market session to avoid look-ahead bias.

    Args:
        timestamps: Iterable of timestamps (tz-aware or naive). Naive values
            are assumed to be UTC.
        cutoff: Market close time used as the rollover boundary.
        tz: Trading timezone (default NYSE).

    Returns:
        Pandas Series of session dates localized to UTC midnight and returned
        as tz-naive timestamps for downstream compatibility.
    """

    if isinstance(timestamps, (datetime, pd.Timestamp)):
        timestamps = [timestamps]
        single_value = True
    else:
        single_value = False

    series = pd.to_datetime(timestamps, utc=False)
    if isinstance(series, pd.Timestamp):
        series = pd.Series([series])
        single_value = True
    elif isinstance(series, pd.DatetimeIndex):
        series = pd.Series(series)
    elif not isinstance(series, pd.Series):
        series = pd.Series(series)

    if series.empty:
        result = pd.Series([], dtype="datetime64[ns]")
        return result.iloc[0] if single_value else result

    if series.dt.tz is None:
        series = series.dt.tz_localize(UTC)
    else:
        series = series.dt.tz_convert(UTC)

    trading_tz = pytz.timezone(tz)
    local = series.dt.tz_convert(trading_tz)

    cutoff_seconds = cutoff.hour * 3600 + cutoff.minute * 60 + cutoff.second
    local_seconds = (
        local.dt.hour * 3600 + local.dt.minute * 60 + local.dt.second
    )
    after_cutoff = local_seconds >= cutoff_seconds

    session_local = local.dt.floor('D') + pd.to_timedelta(after_cutoff.astype(int), unit="D")
    session_utc = session_local.dt.tz_convert(UTC).dt.floor('D')
    session_naive = session_utc.dt.tz_localize(None)

    result = pd.Series(session_naive, index=series.index)
    if single_value:
        return result.iloc[0]
    return result
