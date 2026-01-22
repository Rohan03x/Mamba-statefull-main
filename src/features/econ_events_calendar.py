"""Economic events calendar + surprise features (EODHD).

Family: econ_events_calendar

Data source:
- EODHD Economic Events Data API
  https://eodhd.com/financial-apis/economic-events-data-api/

Design goals:
- Daily (NYSE session-aligned) macro release schedule features
- "No lookahead" policy:
  - Schedule-derived features (days_to_next, days_since_last, pre/post windows) are safe unshifted.
  - Realized surprises are applied on the *next* NYSE session after the release session.

Critical fixes (Jan 2026):
- Replace 9999 sentinels with bounded proximity/recency signals:
  - prox_next_* = exp(-min(days_to_next, 252) / 5)  → peaks when event is imminent
  - recency_last_* = exp(-min(days_since_last, 252) / 10)  → decays after event
- Add composite macro features for policy state:
  - macro_upcoming_major = max(prox_next_{fomc, cpi, nfp})
  - macro_shock_major = max(pulse_strength_{fomc, cpi, nfp})
  - macro_surprise_signed = importance-weighted sum of pulse surprises

Coverage note:
- EODHD economic-events coverage starts around 2020.
- This family is symbol-agnostic (macro), but emitted per-symbol for pipeline consistency.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from src.dcf_lab.utils.timealign import nyse_sessions_in_range
from src.dcf_lab.utils.trading_calendar import add_sessions, session_on_or_after


def _rolling_windows(start: pd.Timestamp, end: pd.Timestamp, *, window_days: int) -> Iterable[Tuple[pd.Timestamp, pd.Timestamp]]:
    win = int(max(1, window_days))
    cur = pd.Timestamp(start).normalize()
    end = pd.Timestamp(end).normalize()
    while cur <= end:
        cur_end = min(end, cur + pd.Timedelta(days=win - 1))
        yield cur, cur_end
        cur = cur_end + pd.Timedelta(days=1)


def _dedupe_eodhd_events(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    # Prefer stable identifiers when available, otherwise fall back to a composite key.
    seen: set[tuple[str, str, str, str]] = set()
    out: List[Dict[str, Any]] = []
    for r in rows:
        if not isinstance(r, dict):
            continue
        event_id = _to_str(r.get("event_id") or r.get("eventId") or r.get("id") or r.get("ID"))
        date_key = _to_str(r.get("date") or r.get("datetime") or r.get("Date"))
        country_key = _to_str(r.get("country") or r.get("Country")).strip().upper()
        indicator = _to_str(r.get("event") or r.get("type") or r.get("name") or r.get("title"))
        k = (event_id, date_key, country_key, indicator)
        if k in seen:
            continue
        seen.add(k)
        out.append(r)
    return out


def _fetch_eodhd_economic_events_time_sliced(
    provider: Any,
    *,
    start_date: str,
    end_date: str,
    country: str,
    window_days: int = 30,
    limit: int = 1000,
    hard_cap_threshold: int = 950,
) -> Optional[List[Dict[str, Any]]]:
    """Fetch economic-events using *time windows only* (offset must stay at 0).

    Non-negotiable rule:
    - Never rely on offset pagination for /economic-events.
    - Always use offset=0.
    - Window by time such that each call returns well under the endpoint's caps.
    """

    try:
        start_ts = pd.Timestamp(start_date).normalize()
        end_ts = pd.Timestamp(end_date).normalize()
    except Exception:
        return None

    if pd.isna(start_ts) or pd.isna(end_ts) or start_ts > end_ts:
        return None

    # EODHD supports limit up to 1000; keep it explicit.
    page_limit = int(max(1, min(1000, limit)))
    hard_cap = int(max(1, min(page_limit, hard_cap_threshold)))
    win_days = int(max(1, window_days))

    # Provider uses `_make_request('economic-events', params)`.
    make_request = getattr(provider, "_make_request", None)
    if not callable(make_request):
        return None

    raw: List[Dict[str, Any]] = []
    for w_start, w_end in _rolling_windows(start_ts, end_ts, window_days=win_days):
        params: Dict[str, Any] = {
            "from": w_start.strftime("%Y-%m-%d"),
            "to": w_end.strftime("%Y-%m-%d"),
            "country": str(country).strip().upper(),
            "limit": page_limit,
            # Always offset=0. Never paginate.
            "offset": 0,
        }

        data = make_request("economic-events", params)
        if data is None:
            return None

        if not isinstance(data, list):
            # Defensive: some APIs wrap lists, but EODHD typically returns list.
            if isinstance(data, dict):
                for key in ("data", "results", "items"):
                    payload = data.get(key)
                    if isinstance(payload, list):
                        data = payload
                        break

        if not isinstance(data, list):
            return None

        batch = [x for x in data if isinstance(x, dict)]
        raw.extend(batch)

        # Ingestion-time assertion: if a single window approaches the hard cap,
        # assume truncation risk and fail loudly.
        if len(batch) >= hard_cap:
            raise RuntimeError(
                f"EODHD economic-events window {params['from']}..{params['to']} returned {len(batch)} rows "
                f"(>= hard_cap_threshold={hard_cap}). Reduce window_days (current={win_days}) to avoid truncation."
            )

    return _dedupe_eodhd_events(raw)


# ---------------------------------------------------------------------------
# Bounded proximity / recency transforms (replace 9999 sentinels)
# ---------------------------------------------------------------------------
# Major event types for composite features
_MAJOR_EVENT_TYPES = {"fomc", "cpi", "nfp"}

# Importance weights for weighted surprise composites
_EVENT_IMPORTANCE_WEIGHTS = {
    "fomc": 3.0,
    "cpi": 3.0,
    "nfp": 3.0,
    "gdp": 2.5,
    "pce": 2.0,
    "unemployment": 2.0,
    "retail_sales": 1.5,
    "ism": 1.5,
    "core_cpi": 2.0,
}


def _bounded_proximity(days_to: pd.Series, *, cap: float = 252.0, tau: float = 5.0) -> pd.Series:
    """Convert days_to_next (with 9999 sentinel) to bounded proximity in [0,1].

    Formula: exp(-min(days_to, cap) / tau)
    - peaks at 1.0 when days_to=0 (event imminent)
    - decays to ~0.05 when days_to >= tau*3
    - 9999 values → effectively 0.0
    """
    clamped = days_to.clip(lower=0.0, upper=cap).fillna(cap)
    return np.exp(-clamped / tau)


def _bounded_recency(days_since: pd.Series, *, cap: float = 252.0, tau: float = 10.0) -> pd.Series:
    """Convert days_since_last (with 9999 sentinel) to bounded recency in [0,1].

    Formula: exp(-min(days_since, cap) / tau)
    - peaks at 1.0 when days_since=0 (event just happened)
    - decays to ~0.05 when days_since >= tau*3
    - 9999 values → effectively 0.0
    """
    clamped = days_since.clip(lower=0.0, upper=cap).fillna(cap)
    return np.exp(-clamped / tau)


@dataclass(frozen=True)
class _EventType:
    key: str
    patterns: Tuple[str, ...]


_EVENT_TYPES: Tuple[_EventType, ...] = (
    _EventType("cpi", ("consumer price index", "cpi")),
    _EventType("core_cpi", ("core consumer price index", "core cpi")),
    _EventType("pce", ("pce", "personal consumption expenditures")),
    _EventType("nfp", ("nonfarm payroll", "non-farm payroll", "nfp")),
    _EventType("unemployment", ("unemployment rate", "jobless rate")),
    _EventType("gdp", ("gross domestic product", "gdp")),
    _EventType("fomc", ("fomc", "fed", "federal open market committee", "interest rate decision")),
    _EventType("retail_sales", ("retail sales",)),
    _EventType("ism", ("ism", "pmi")),
)


def _to_float(v: Any) -> float:
    if v is None:
        return 0.0
    try:
        x = float(v)
        if np.isfinite(x):
            return x
        return 0.0
    except Exception:
        try:
            x = float(pd.to_numeric(v, errors="coerce"))
            if np.isfinite(x):
                return x
        except Exception:
            return 0.0
    return 0.0


def _to_str(v: Any) -> str:
    if v is None:
        return ""
    try:
        return str(v)
    except Exception:
        return ""


def _parse_date(v: Any) -> Optional[pd.Timestamp]:
    if v is None:
        return None
    ts = pd.to_datetime(v, errors="coerce")
    if pd.isna(ts):
        return None
    return pd.Timestamp(ts).normalize()


def _infer_importance_weight(row: Dict[str, Any]) -> float:
    for k in ("importance", "impact", "volatility", "severity"):
        if k in row and row.get(k) is not None:
            val = _to_str(row.get(k)).strip().lower()
            if not val:
                continue
            if val in {"high", "3", "3.0"}:
                return 3.0
            if val in {"medium", "2", "2.0"}:
                return 2.0
            if val in {"low", "1", "1.0"}:
                return 1.0
            try:
                w = float(val)
                if np.isfinite(w) and w > 0:
                    return float(min(3.0, max(1.0, w)))
            except Exception:
                continue
    return 1.0


def _classify_event(name: str) -> Optional[str]:
    n = (name or "").strip().lower()
    if not n:
        return None

    # Prefer more specific matches first.
    if "core" in n and "cpi" in n:
        return "core_cpi"
    if "core consumer price index" in n:
        return "core_cpi"

    for spec in _EVENT_TYPES:
        for p in spec.patterns:
            if p in n:
                # Disambiguate CPI vs core CPI.
                if spec.key == "cpi" and "core" in n:
                    continue
                return spec.key
    return None


def _days_to_next_event(sessions: pd.DatetimeIndex, event_sessions: pd.DatetimeIndex, never: float = 9999.0) -> pd.Series:
    if len(sessions) == 0:
        return pd.Series(dtype=float)
    if len(event_sessions) == 0:
        return pd.Series(float(never), index=sessions)

    event_sessions = pd.DatetimeIndex(sorted(pd.Timestamp(x).normalize() for x in event_sessions.unique()))
    # Searchsorted indices of next event for each session.
    idx = event_sessions.searchsorted(sessions.values, side="left")
    out = np.full(len(sessions), float(never), dtype=float)

    # Map sessions->position for distance.
    sess_pos = pd.Series(np.arange(len(sessions), dtype=int), index=sessions)
    for i, j in enumerate(idx):
        if j >= len(event_sessions):
            continue
        next_sess = event_sessions[j]
        try:
            out[i] = float(sess_pos[next_sess] - i)
        except Exception:
            # next_sess might be outside sessions range
            continue

    return pd.Series(out, index=sessions)


def _days_since_last_event(sessions: pd.DatetimeIndex, event_sessions: pd.DatetimeIndex, never: float = 9999.0) -> pd.Series:
    if len(sessions) == 0:
        return pd.Series(dtype=float)
    if len(event_sessions) == 0:
        return pd.Series(float(never), index=sessions)

    event_sessions = pd.DatetimeIndex(sorted(pd.Timestamp(x).normalize() for x in event_sessions.unique()))
    idx = event_sessions.searchsorted(sessions.values, side="right") - 1
    out = np.full(len(sessions), float(never), dtype=float)

    sess_pos = pd.Series(np.arange(len(sessions), dtype=int), index=sessions)
    for i, j in enumerate(idx):
        if j < 0:
            continue
        prev_sess = event_sessions[j]
        try:
            out[i] = float(i - sess_pos[prev_sess])
        except Exception:
            continue

    return pd.Series(out, index=sessions)


def _time_window_zscore(events: pd.Series, window_days: int = 1825, *, min_periods: int = 3) -> pd.Series:
    # events: indexed by event session only.
    if events.empty:
        return events
    win = f"{int(window_days)}D"
    mean = events.rolling(win, min_periods=min_periods).mean()
    std = events.rolling(win, min_periods=min_periods).std(ddof=0).replace(0.0, np.nan)
    z = (events - mean) / std
    return z.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def fetch(
    symbol: str,
    start: str,
    end: str,
    *,
    country: str = "US",
    include_types: Optional[Sequence[str]] = None,
    pre_window_sessions: int = 3,
    post_window_sessions: int = 3,
    zscore_window_days: int = 1825,
) -> Optional[pd.DataFrame]:
    """Fetch economic events calendar + surprise features.

    Args:
        symbol: unused (macro), kept for pipeline signature compatibility.
        start/end: YYYY-MM-DD bounds
        country: ISO-3166 alpha-2 (default "US")
        include_types: optional subset of event keys to include
        pre_window_sessions: mark 1 if within N sessions *before* a release
        post_window_sessions: mark 1 if within N sessions *after* a release
        zscore_window_days: rolling history window for surprise z-score (approx 5y)

    Returns:
        DataFrame indexed by tz-naive NYSE sessions.
    """

    def _stub(
        sess: pd.DatetimeIndex,
        *,
        status: str,
        note: str,
        country: str,
        include_types: Optional[Sequence[str]],
        pre_window_sessions: int,
        post_window_sessions: int,
    ) -> pd.DataFrame:
        sess = pd.DatetimeIndex([pd.Timestamp(x).normalize() for x in sess])
        out = pd.DataFrame(index=sess)
        out["econ_events_calendar_has_data"] = 0.0

        pre_n = int(max(0, pre_window_sessions))
        post_n = int(max(0, post_window_sessions))
        never = 9999.0

        if include_types is None:
            types = [x.key for x in _EVENT_TYPES]
        else:
            allowed = {str(x).strip().lower() for x in include_types if x}
            types = [x.key for x in _EVENT_TYPES if x.key in allowed]

        for et in sorted(set(types)):
            out[f"econ_events_calendar_days_to_next_{et}"] = float(never)
            out[f"econ_events_calendar_days_since_last_{et}"] = float(never)
            out[f"econ_events_calendar_pre_window_{pre_n}d_{et}"] = 0.0
            out[f"econ_events_calendar_post_window_{post_n}d_{et}"] = 0.0
            out[f"econ_events_calendar_surprise_{et}"] = 0.0
            out[f"econ_events_calendar_surprise_z_{et}"] = 0.0
            out[f"econ_events_calendar_surprise_w_{et}"] = 0.0
            out[f"econ_events_calendar_surprise_w_z_{et}"] = 0.0
            out[f"econ_events_calendar_surprise_has_forecast_{et}"] = 0.0
            out[f"econ_events_calendar_event_occurrence_{et}"] = 0.0
            out[f"econ_events_calendar_pulse_occurrence_{et}"] = 0.0
            out[f"econ_events_calendar_pulse_surprise_{et}"] = 0.0
            out[f"econ_events_calendar_pulse_strength_{et}"] = 0.0
            # Bounded proximity/recency (replace 9999 sentinels)
            out[f"econ_events_calendar_prox_next_{et}"] = 0.0
            out[f"econ_events_calendar_recency_last_{et}"] = 0.0

        # Composite macro features (major events only: fomc, cpi, nfp)
        out["econ_events_calendar_macro_upcoming_major"] = 0.0
        out["econ_events_calendar_macro_shock_major"] = 0.0
        out["econ_events_calendar_macro_surprise_signed"] = 0.0

        out.attrs["telemetry"] = {
            "status": status,
            "source": "EODHD.economic-events",
            "country": str(country),
            "note": note,
        }
        out.attrs["feature_counts"] = {"generated": int(len(out.columns))}
        out.attrs["provenance"] = {"source": "EODHD.economic-events", "leakage": "surprise shifted to next session"}
        return out

    try:
        from src.data_sources.eodhd_provider import get_eodhd_provider  # type: ignore
    except Exception:
        sessions = nyse_sessions_in_range(start, end, tz_aware_utc=False)
        if len(sessions) == 0:
            return None
        return _stub(
            sessions,
            status="dormant:provider_unavailable",
            note="EODHD provider import failed",
            country=country,
            include_types=include_types,
            pre_window_sessions=pre_window_sessions,
            post_window_sessions=post_window_sessions,
        )

    sessions = nyse_sessions_in_range(start, end, tz_aware_utc=False)
    if len(sessions) == 0:
        return None

    # Extend history to compute z-scores robustly.
    lookback_days = int(max(800, zscore_window_days + 120))
    fetch_start = (pd.Timestamp(start) - pd.Timedelta(days=lookback_days)).strftime("%Y-%m-%d")

    sessions_ext = nyse_sessions_in_range(fetch_start, end, tz_aware_utc=False)
    if len(sessions_ext) == 0:
        return None

    provider = get_eodhd_provider()
    if not getattr(provider, "api_key", None):
        return _stub(
            sessions,
            status="dormant:missing_api_key",
            note="EODHD api_key missing",
            country=country,
            include_types=include_types,
            pre_window_sessions=pre_window_sessions,
            post_window_sessions=post_window_sessions,
        )

    # EODHD economic-events coverage starts around 2020, and the endpoint has a
    # hard pagination cap. Avoid silent truncation by chunking requests.
    eod_min = pd.Timestamp("2020-01-01")
    fetch_start_eff = max(pd.Timestamp(fetch_start), eod_min).strftime("%Y-%m-%d")

    # Safe default: 30-day windows. (US macro can be heavy; tighten further if needed.)
    # This function intentionally never uses offset pagination.
    raw = _fetch_eodhd_economic_events_time_sliced(
        provider,
        start_date=fetch_start_eff,
        end_date=end,
        country=country,
        window_days=30,
        limit=1000,
        hard_cap_threshold=950,
    )
    if raw is None:
        return _stub(
            sessions,
            status="error:fetch_failed",
            note="EODHD economic-events fetch failed (None)",
            country=country,
            include_types=include_types,
            pre_window_sessions=pre_window_sessions,
            post_window_sessions=post_window_sessions,
        )

    out = pd.DataFrame(index=sessions_ext)
    out["econ_events_calendar_has_data"] = 1.0

    # If provider returns empty, emit shape-stable zero features.
    if len(raw) == 0:
        out.attrs["telemetry"] = {
            "status": "ok",
            "source": "EODHD.economic-events",
            "note": "empty payload",
        }
        out = out.loc[(out.index >= sessions[0]) & (out.index <= sessions[-1])].copy()
        out.attrs["feature_counts"] = {"generated": int(len(out.columns))}
        out.attrs["provenance"] = {"source": "EODHD.economic-events", "leakage": "surprise shifted to next session"}
        return out

    rows = [x for x in raw if isinstance(x, dict)]
    if not rows:
        return None

    # Parse payload into normalized events.
    parsed: List[Dict[str, Any]] = []
    for r in rows:
        name = _to_str(r.get("event") or r.get("type") or r.get("name") or r.get("title"))
        dt = _parse_date(r.get("date") or r.get("datetime") or r.get("Date"))
        if dt is None:
            continue

        etype = _classify_event(name)
        if etype is None:
            continue

        parsed.append(
            {
                "event_type": etype,
                "event_name": name,
                "event_date": dt,
                "actual": _to_float(r.get("actual") or r.get("Actual")),
                # EODHD uses 'estimate' for consensus/forecast.
                "forecast": _to_float(
                    r.get("forecast")
                    or r.get("Forecast")
                    or r.get("estimate")
                    or r.get("Estimate")
                ),
                "previous": _to_float(r.get("previous") or r.get("Previous")),
                "has_forecast": 1.0
                if (
                    r.get("forecast") is not None
                    or r.get("Forecast") is not None
                    or r.get("estimate") is not None
                    or r.get("Estimate") is not None
                )
                else 0.0,
                "importance_weight": _infer_importance_weight(r),
            }
        )

    if not parsed:
        # Nothing matched our tracked types; keep has_data=1 but no signals.
        out.attrs["telemetry"] = {
            "status": "ok",
            "source": "EODHD.economic-events",
            "note": "no tracked event types matched",
        }
        out = out.loc[(out.index >= sessions[0]) & (out.index <= sessions[-1])].copy()
        out.attrs["feature_counts"] = {"generated": int(len(out.columns))}
        out.attrs["provenance"] = {"source": "EODHD.economic-events", "leakage": "surprise shifted to next session"}
        return out

    ev = pd.DataFrame(parsed)
    ev["release_session"] = ev["event_date"].apply(session_on_or_after)
    ev = ev.dropna(subset=["release_session"]).copy()
    if ev.empty:
        return None

    if include_types is not None:
        allowed = {str(x).strip().lower() for x in include_types if x}
        ev = ev.loc[ev["event_type"].isin(allowed)].copy()

    # Surprise definition: actual - forecast (0 if no forecast)
    ev["surprise"] = ev["actual"] - ev["forecast"]
    ev.loc[ev["has_forecast"] <= 0.0, "surprise"] = 0.0
    ev["surprise_weighted"] = ev["surprise"] * ev["importance_weight"]

    # Aggregate per release session and event type.
    grouped = (
        ev.groupby(["event_type", "release_session"], as_index=False)
        .agg(
            surprise=("surprise", "sum"),
            surprise_weighted=("surprise_weighted", "sum"),
            has_forecast=("has_forecast", "max"),
            event_occurrence=("event_type", "size"),
        )
        .sort_values(["event_type", "release_session"])
    )

    # Build per-type schedule features and surprise pulses.
    never = 9999.0
    pre_n = int(max(0, pre_window_sessions))
    post_n = int(max(0, post_window_sessions))

    for et in sorted(grouped["event_type"].unique().tolist()):
        rel = grouped.loc[grouped["event_type"] == et].copy()
        release_sessions = pd.DatetimeIndex(pd.to_datetime(rel["release_session"], errors="coerce").dropna().unique())
        release_sessions = pd.DatetimeIndex([pd.Timestamp(x).normalize() for x in release_sessions])

        # Schedule distances (safe, unshifted)
        d_next = _days_to_next_event(sessions_ext, release_sessions, never=never)
        d_prev = _days_since_last_event(sessions_ext, release_sessions, never=never)

        out[f"econ_events_calendar_days_to_next_{et}"] = d_next
        out[f"econ_events_calendar_days_since_last_{et}"] = d_prev

        # Windows (exclude day0 to avoid intraday leakage):
        # pre: 1..pre_n sessions before
        if pre_n > 0:
            out[f"econ_events_calendar_pre_window_{pre_n}d_{et}"] = ((d_next >= 1.0) & (d_next <= float(pre_n))).astype(float)
        else:
            out[f"econ_events_calendar_pre_window_0d_{et}"] = 0.0

        # post: 1..post_n sessions after
        if post_n > 0:
            out[f"econ_events_calendar_post_window_{post_n}d_{et}"] = ((d_prev >= 1.0) & (d_prev <= float(post_n))).astype(float)
        else:
            out[f"econ_events_calendar_post_window_0d_{et}"] = 0.0

        # Realized surprise pulses (apply on next session after release session)
        # Build sparse series on release sessions only for stable zscore.
        rel_series = rel.set_index("release_session")["surprise"].astype(float)
        rel_series.index = pd.DatetimeIndex([pd.Timestamp(x).normalize() for x in rel_series.index])
        rel_series = rel_series.sort_index()

        rel_w_series = rel.set_index("release_session")["surprise_weighted"].astype(float)
        rel_w_series.index = pd.DatetimeIndex([pd.Timestamp(x).normalize() for x in rel_w_series.index])
        rel_w_series = rel_w_series.sort_index()

        rel_has_fc = rel.set_index("release_session")["has_forecast"].astype(float)
        rel_has_fc.index = pd.DatetimeIndex([pd.Timestamp(x).normalize() for x in rel_has_fc.index])
        rel_has_fc = rel_has_fc.sort_index()

        rel_occ = rel.set_index("release_session")["event_occurrence"].astype(float)
        rel_occ.index = pd.DatetimeIndex([pd.Timestamp(x).normalize() for x in rel_occ.index])
        rel_occ = (rel_occ > 0.0).astype(float).sort_index()

        # NOTE: EODHD economic-events history can be truncated (pagination caps).
        # Using a lower min_periods prevents all-zero z-scores when only a few
        # releases are available per event type.
        z = _time_window_zscore(rel_series, window_days=zscore_window_days, min_periods=3)
        z_w = _time_window_zscore(rel_w_series, window_days=zscore_window_days, min_periods=3)

        # Shift to next session.
        eff_idx = [add_sessions(session_on_or_after(ts), 1) for ts in rel_series.index]
        eff_idx = pd.DatetimeIndex([pd.Timestamp(x).normalize() for x in eff_idx])

        surprise_eff = pd.Series(rel_series.values, index=eff_idx)
        surprise_w_eff = pd.Series(rel_w_series.values, index=eff_idx)
        z_eff = pd.Series(z.values, index=eff_idx)
        z_w_eff = pd.Series(z_w.values, index=eff_idx)
        has_fc_eff = pd.Series(rel_has_fc.values, index=eff_idx)
        occ_eff = pd.Series(rel_occ.values, index=eff_idx)

        out[f"econ_events_calendar_surprise_{et}"] = surprise_eff.reindex(sessions_ext).fillna(0.0)
        out[f"econ_events_calendar_surprise_z_{et}"] = z_eff.reindex(sessions_ext).fillna(0.0)
        out[f"econ_events_calendar_surprise_w_{et}"] = surprise_w_eff.reindex(sessions_ext).fillna(0.0)
        out[f"econ_events_calendar_surprise_w_z_{et}"] = z_w_eff.reindex(sessions_ext).fillna(0.0)
        out[f"econ_events_calendar_surprise_has_forecast_{et}"] = has_fc_eff.reindex(sessions_ext).fillna(0.0)

        # Split concepts:
        # A) Occurrence: did an event happen (shifted to next session)
        out[f"econ_events_calendar_event_occurrence_{et}"] = occ_eff.reindex(sessions_ext).fillna(0.0)

        # B) Surprise: magnitude is already captured in econ_events_calendar_surprise_* (0 when missing)
        # Pulse logic (shifted)
        out[f"econ_events_calendar_pulse_occurrence_{et}"] = (
            out[f"econ_events_calendar_event_occurrence_{et}"] > 0.0
        ).astype(float)
        out[f"econ_events_calendar_pulse_surprise_{et}"] = out[f"econ_events_calendar_surprise_{et}"].astype(float)

        # Pulse strength: abs(surprise) when available, else baseline on occurrence.
        baseline = 1.0
        has_fc_now = out[f"econ_events_calendar_surprise_has_forecast_{et}"] > 0.0
        occ_now = out[f"econ_events_calendar_event_occurrence_{et}"] > 0.0
        strength = np.zeros(len(out), dtype=float)
        try:
            s_abs = np.abs(pd.to_numeric(out[f"econ_events_calendar_surprise_{et}"], errors="coerce").fillna(0.0).to_numpy())
        except Exception:
            s_abs = np.zeros(len(out), dtype=float)
        strength = np.where(occ_now & (~has_fc_now), float(baseline), strength)
        strength = np.where(occ_now & has_fc_now, s_abs, strength)
        out[f"econ_events_calendar_pulse_strength_{et}"] = strength

        # ---------------------------------------------------------------------
        # Bounded proximity/recency (replace 9999 sentinels for safe aggregation)
        # prox_next: peaks when event is imminent (tau=5 days)
        # recency_last: peaks after event, decays over ~2 weeks (tau=10 days)
        # ---------------------------------------------------------------------
        out[f"econ_events_calendar_prox_next_{et}"] = _bounded_proximity(d_next, cap=252.0, tau=5.0)
        out[f"econ_events_calendar_recency_last_{et}"] = _bounded_recency(d_prev, cap=252.0, tau=10.0)

    # -------------------------------------------------------------------------
    # Composite macro features (aggregate major event signals for policy state)
    # -------------------------------------------------------------------------
    major_event_types = [et for et in _MAJOR_EVENT_TYPES if f"econ_events_calendar_prox_next_{et}" in out.columns]

    # macro_upcoming_major: max proximity across major events (fomc, cpi, nfp)
    if major_event_types:
        prox_cols = [out[f"econ_events_calendar_prox_next_{et}"] for et in major_event_types]
        out["econ_events_calendar_macro_upcoming_major"] = pd.concat(prox_cols, axis=1).max(axis=1).fillna(0.0)
    else:
        out["econ_events_calendar_macro_upcoming_major"] = 0.0

    # macro_shock_major: max pulse strength across major events
    if major_event_types:
        shock_cols = [out[f"econ_events_calendar_pulse_strength_{et}"] for et in major_event_types if f"econ_events_calendar_pulse_strength_{et}" in out.columns]
        if shock_cols:
            out["econ_events_calendar_macro_shock_major"] = pd.concat(shock_cols, axis=1).max(axis=1).fillna(0.0)
        else:
            out["econ_events_calendar_macro_shock_major"] = 0.0
    else:
        out["econ_events_calendar_macro_shock_major"] = 0.0

    # macro_surprise_signed: importance-weighted sum of pulse surprises (for directional macro impulse)
    all_event_types = [et for et in [spec.key for spec in _EVENT_TYPES] if f"econ_events_calendar_pulse_surprise_{et}" in out.columns]
    weighted_surprises = []
    for et in all_event_types:
        weight = _EVENT_IMPORTANCE_WEIGHTS.get(et, 1.0)
        pulse_col = f"econ_events_calendar_pulse_surprise_{et}"
        if pulse_col in out.columns:
            weighted_surprises.append(out[pulse_col].fillna(0.0) * weight)
    if weighted_surprises:
        out["econ_events_calendar_macro_surprise_signed"] = sum(weighted_surprises).fillna(0.0)
    else:
        out["econ_events_calendar_macro_surprise_signed"] = 0.0

    out.attrs["telemetry"] = {
        "status": "ok",
        "source": "EODHD.economic-events",
        "country": str(country),
        "fetch_start": str(fetch_start_eff),
        "n_raw": int(len(raw)),
    }
    out = out.loc[(out.index >= sessions[0]) & (out.index <= sessions[-1])].copy()
    out.attrs["feature_counts"] = {"generated": int(len(out.columns))}
    out.attrs["provenance"] = {"source": "EODHD.economic-events", "leakage": "surprise shifted to next session"}
    return out
