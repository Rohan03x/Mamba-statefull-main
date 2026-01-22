"""Exchange calendar structure (holidays, early closes).

Family: exchange_calendar

Intent:
- Provide calendar-aware structural features to reduce subtle bugs.
- Deterministic (no vendor latency / missingness).

Implementation note:
- Prefer EODHD Exchanges API when available (holidays + early closes), and union
    it with `exchange_calendars` as a deterministic fallback/sanity anchor.

Features:
- exchange_calendar_is_holiday_adjacent (±1 calendar day from a holiday)
- exchange_calendar_days_to_holiday (calendar days to next holiday; 9999 if none)
- exchange_calendar_days_since_holiday (calendar days since last holiday; 9999 if none)
- exchange_calendar_is_half_day (early close session)

Critical fixes (Jan 2026):
- Replace 9999 sentinels with bounded proximity/recency signals:
  - holiday_prox = exp(-min(days_to_holiday, 60) / 3)  → peaks near holiday
  - holiday_recency = exp(-min(days_since_holiday, 60) / 3)  → decays after holiday
- Add calendar_liquidity_stress composite for position sizing / turnover control:
  - calendar_liquidity_stress = clip(max(is_half_day, is_holiday_adjacent) + 0.5*holiday_prox + 0.3*holiday_recency, 0, 1)

Routing: Portfolio parquet ONLY (execution/microstructure context, not Mamba alpha)

DST transition flag intentionally omitted (daily panel only).
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from src.dcf_lab.utils.timealign import nyse_sessions_in_range
from src.data_sources.eodhd_provider import EODHDProvider


# ---------------------------------------------------------------------------
# Bounded proximity / recency transforms (replace 9999 sentinels)
# ---------------------------------------------------------------------------

def _holiday_prox(days_to: pd.Series, *, cap: float = 60.0, tau: float = 3.0) -> pd.Series:
    """Convert days_to_holiday (with 9999 sentinel) to bounded proximity in [0,1].

    Formula: exp(-min(days_to, cap) / tau)
    - peaks at 1.0 when holiday is today (days_to=0)
    - decays quickly (tau=3) because holiday effects are very local
    - 9999 values → effectively 0.0
    """
    clamped = days_to.clip(lower=0.0, upper=cap).replace(9999.0, cap).fillna(cap)
    return np.exp(-clamped / tau)


def _holiday_recency(days_since: pd.Series, *, cap: float = 60.0, tau: float = 3.0) -> pd.Series:
    """Convert days_since_holiday (with 9999 sentinel) to bounded recency in [0,1].

    Formula: exp(-min(days_since, cap) / tau)
    - peaks at 1.0 when holiday just passed (days_since=0)
    - decays quickly (tau=3) because post-holiday effects are brief
    - 9999 values → effectively 0.0
    """
    clamped = days_since.clip(lower=0.0, upper=cap).replace(9999.0, cap).fillna(cap)
    return np.exp(-clamped / tau)


def _infer_exchange(symbol: str) -> str:
    # Best-effort mapping; pipeline is currently US/NYSE-session aligned.
    # Keep as a hook if multi-exchange support is added later.
    if not symbol:
        return "XNYS"
    suf = (symbol.split(".")[-1] if "." in symbol else "").upper()
    if suf in {"US", "NYSE", "NASDAQ", "BATS", "ARCA"}:
        return "XNYS"
    return "XNYS"


def _infer_eodhd_exchange_code(symbol: str) -> str:
    # EODHD exchange codes: for US equities, the unified exchange is "US".
    # Keep a hook for multi-exchange support.
    if not symbol:
        return "US"
    suf = (symbol.split(".")[-1] if "." in symbol else "").upper()
    # If caller already passed e.g. AAPL.US or MSFT.NYSE, respect it.
    if suf in {"US", "NYSE", "NASDAQ", "BATS", "ARCA", "OTCQB", "OTCQX", "PINK", "LSE", "XETRA"}:
        return suf
    return "US"


def fetch(symbol: str, start: str, end: str, *, exchange: Optional[str] = None) -> Optional[pd.DataFrame]:
    # We emit NYSE-session indexed features; this is consistent with the rest of the pipeline.
    sessions = nyse_sessions_in_range(start, end, tz_aware_utc=False)
    if len(sessions) == 0:
        return None

    ex = exchange or _infer_exchange(symbol)
    eodhd_exchange = _infer_eodhd_exchange_code(symbol)

    # Holidays/early closes are defined on (normalized) session dates.
    start_ts = pd.Timestamp(sessions[0]).normalize()
    end_ts = pd.Timestamp(sessions[-1]).normalize()

    cal = None
    cal_err = None
    try:
        import exchange_calendars as xc

        cal = xc.get_calendar(ex)
    except Exception as exc:
        cal_err = exc

    # EODHD exchange-details (holidays + early closes).
    eodhd_holidays: pd.DatetimeIndex = pd.DatetimeIndex([])
    eodhd_early_close: set[pd.Timestamp] = set()
    eodhd_ok = False
    try:
        prov = EODHDProvider()
        if prov.api_key:
            hdf = prov.get_exchange_holidays(eodhd_exchange, start_date=start_ts.strftime("%Y-%m-%d"), end_date=end_ts.strftime("%Y-%m-%d"))
            if hdf is not None and not hdf.empty and "date" in hdf.columns:
                eodhd_holidays = pd.DatetimeIndex(pd.to_datetime(hdf["date"], errors="coerce").dropna().dt.normalize().unique())
            edf = prov.get_exchange_early_close_days(
                eodhd_exchange,
                start_date=start_ts.strftime("%Y-%m-%d"),
                end_date=end_ts.strftime("%Y-%m-%d"),
            )
            if edf is not None and not edf.empty and "date" in edf.columns:
                eodhd_early_close = set(pd.to_datetime(edf["date"], errors="coerce").dropna().dt.normalize())
            eodhd_ok = True
    except Exception:
        # Best-effort only.
        eodhd_ok = False

    # Holidays (exclude weekends implicitly because exchange holiday calendars exclude them).

    hol = pd.DatetimeIndex([])
    if cal is not None:
        try:
            hol = hol.union(pd.DatetimeIndex(cal.regular_holidays.holidays(start_ts, end_ts)))
        except Exception:
            pass
        try:
            adh = pd.DatetimeIndex([pd.Timestamp(x).normalize() for x in getattr(cal, "adhoc_holidays", [])])
            hol = hol.union(adh[(adh >= start_ts) & (adh <= end_ts)])
        except Exception:
            pass

    # Union vendor + deterministic.
    if len(eodhd_holidays) > 0:
        hol = hol.union(eodhd_holidays)

    hol = pd.DatetimeIndex(sorted(pd.Timestamp(x).normalize() for x in hol.unique()))
    hol_set = set(hol)

    # Holiday-adjacent: ±1 calendar day around holiday dates.
    prev_day = sessions - pd.Timedelta(days=1)
    next_day = sessions + pd.Timedelta(days=1)
    is_adj = pd.Series([(d in hol_set) or (n in hol_set) for d, n in zip(prev_day, next_day)], index=sessions).astype(float)

    # Days to/since holiday.
    never = 9999.0
    if len(hol) == 0:
        days_to = pd.Series(float(never), index=sessions)
        days_since = pd.Series(float(never), index=sessions)
    else:
        # Next holiday
        idx_next = hol.searchsorted(sessions.values, side="left")
        next_vals = np.full(len(sessions), np.nan, dtype="datetime64[ns]")
        mask_next = idx_next < len(hol)
        next_vals[mask_next] = hol.values[idx_next[mask_next]]
        days_to = pd.Series((pd.to_datetime(next_vals) - sessions).days, index=sessions).astype(float)
        days_to = days_to.replace([np.inf, -np.inf], np.nan).fillna(float(never))

        # Previous holiday
        idx_prev = hol.searchsorted(sessions.values, side="right") - 1
        prev_vals = np.full(len(sessions), np.nan, dtype="datetime64[ns]")
        mask_prev = idx_prev >= 0
        prev_vals[mask_prev] = hol.values[idx_prev[mask_prev]]
        days_since = pd.Series((sessions - pd.to_datetime(prev_vals)).days, index=sessions).astype(float)
        days_since = days_since.replace([np.inf, -np.inf], np.nan).fillna(float(never))

    # Half-days: prefer EODHD early close days when present; otherwise (or additionally)
    # detect early close sessions via exchange_calendars schedule.
    is_half = pd.Series(0.0, index=sessions)
    if eodhd_early_close:
        is_half = pd.Series([1.0 if pd.Timestamp(d).normalize() in eodhd_early_close else 0.0 for d in sessions], index=sessions)
    try:
        if cal is not None:
            sched = cal.schedule.loc[start_ts.strftime("%Y-%m-%d"):end_ts.strftime("%Y-%m-%d")]
            if "close" in sched.columns and len(sched) > 0:
                close = sched["close"].copy()
                close_ny = close.dt.tz_convert("America/New_York")
                close_hm = close_ny.dt.strftime("%H:%M")
                early = (close_hm != "16:00").astype(float)
                early.index = pd.to_datetime(sched.index, errors="coerce").tz_localize(None).normalize()
                # Union with EODHD if both exist.
                is_half = np.maximum(is_half.values.astype(float), early.reindex(sessions).fillna(0.0).values.astype(float))
                is_half = pd.Series(is_half, index=sessions)
    except Exception:
        pass

    out = pd.DataFrame(index=sessions)
    has_any = bool((cal is not None) or eodhd_ok)
    out["exchange_calendar_has_data"] = 1.0 if has_any else 0.0
    out["exchange_calendar_is_holiday_adjacent"] = is_adj
    out["exchange_calendar_days_to_holiday"] = days_to
    out["exchange_calendar_days_since_holiday"] = days_since
    out["exchange_calendar_is_half_day"] = is_half

    # -------------------------------------------------------------------------
    # Bounded proximity/recency (replace 9999 sentinels for safe aggregation)
    # Holiday effects are very local → tau=3 days
    # -------------------------------------------------------------------------
    out["exchange_calendar_holiday_prox"] = _holiday_prox(days_to, cap=60.0, tau=3.0)
    out["exchange_calendar_holiday_recency"] = _holiday_recency(days_since, cap=60.0, tau=3.0)

    # -------------------------------------------------------------------------
    # Composite liquidity stress (for position sizing / turnover control)
    # calendar_liquidity_stress = max(is_half_day, is_holiday_adjacent) + 0.5*holiday_prox + 0.3*holiday_recency
    # Clipped to [0, 1]
    # -------------------------------------------------------------------------
    base_stress = np.maximum(is_half.values.astype(float), is_adj.values.astype(float))
    prox_contrib = 0.5 * out["exchange_calendar_holiday_prox"].values.astype(float)
    recency_contrib = 0.3 * out["exchange_calendar_holiday_recency"].values.astype(float)
    out["exchange_calendar_liquidity_stress"] = pd.Series(
        np.clip(base_stress + prox_contrib + recency_contrib, 0.0, 1.0),
        index=sessions
    )

    sources = []
    if eodhd_ok:
        sources.append(f"eodhd.exchange-details.{eodhd_exchange}")
    if cal is not None:
        sources.append(f"exchange_calendars.{ex}")

    out.attrs["telemetry"] = {
        "status": "ok" if has_any else "dormant:no_sources",
        "sources": sources,
        "eodhd": {"holidays": int(len(eodhd_holidays)), "early_closes": int(len(eodhd_early_close))},
        "exchange_calendars": {"enabled": bool(cal is not None), "error": (str(cal_err) if cal_err else "")},
    }
    out.attrs["provenance"] = {"source": "+".join(sources) if sources else "none", "leakage": "none (calendar metadata)"}
    out.attrs["feature_counts"] = {"generated": int(len(out.columns))}
    return out
