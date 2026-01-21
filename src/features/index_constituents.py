"""Index constituents / membership features.

Builds per-symbol daily features from EODHD index components data.

Data source:
- Preferred: EODHD Fundamentals endpoint for index symbols (e.g. GSPC.INDX, DJI.INDX)
  which can include both current Components (with Weight) and HistoricalTickerComponents
  (with StartDate/EndDate).

Look-ahead policy:
- Treat StartDate/EndDate as known at end-of-day.
- Apply membership state changes and event pulses on the *next* trading session.

All outputs are aligned to the NYSE (XNYS) trading calendar via
`nyse_sessions_in_range` and session arithmetic helpers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from src.dcf_lab.utils.timealign import nyse_sessions_in_range
from src.dcf_lab.utils.trading_calendar import add_sessions, session_on_or_after, session_on_or_before


@dataclass(frozen=True)
class IndexMembershipEvent:
    index_symbol: str
    component_code: str
    start_date: pd.Timestamp
    end_date: Optional[pd.Timestamp]


def _norm_component_code(code: str) -> str:
    c = (code or "").strip().upper()
    if c.endswith(".US"):
        c = c[:-3]
    return c


def _index_key_from_symbol(index_symbol: str) -> str:
    # e.g. GSPC.INDX -> gspc
    token = (index_symbol or "").strip().upper()
    if token.endswith(".INDX"):
        token = token[:-5]
    return token.lower()


def _extract_index_membership_events(
    *,
    fundamentals: Dict[str, Any],
    index_symbol: str,
) -> List[IndexMembershipEvent]:
    hist = fundamentals.get("HistoricalTickerComponents") or {}
    if not isinstance(hist, dict):
        return []

    events: List[IndexMembershipEvent] = []
    for _, payload in hist.items():
        if not isinstance(payload, dict):
            continue
        code = payload.get("Code")
        if not code:
            continue
        start = payload.get("StartDate")
        if not start:
            continue
        end = payload.get("EndDate")
        start_ts = pd.to_datetime(start, errors="coerce")
        if pd.isna(start_ts):
            continue
        end_ts = pd.to_datetime(end, errors="coerce") if end else pd.NaT
        events.append(
            IndexMembershipEvent(
                index_symbol=index_symbol,
                component_code=str(code),
                start_date=pd.Timestamp(start_ts).normalize(),
                end_date=(pd.Timestamp(end_ts).normalize() if pd.notna(end_ts) else None),
            )
        )

    return events


def _event_to_sessions(
    *,
    start_date: pd.Timestamp,
    end_date: Optional[pd.Timestamp],
) -> Tuple[pd.Timestamp, Optional[pd.Timestamp], Optional[pd.Timestamp]]:
    """Map raw effective dates to (membership_start, membership_end, removal_pulse_session).

    - membership_start: next session after StartDate
    - membership_end: session on/before EndDate (if provided)
    - removal_pulse_session: next session after membership_end (if provided)
    """

    base_add = session_on_or_after(start_date)
    membership_start = add_sessions(base_add, 1)

    if end_date is None:
        return membership_start, None, None

    base_end = session_on_or_before(end_date)
    removal_pulse = add_sessions(base_end, 1)
    return membership_start, base_end, removal_pulse


def _days_since_pulse(pulse: pd.Series, *, never_value: float = 9999.0) -> pd.Series:
    values = pulse.values
    out = np.empty(len(values), dtype=float)
    last_idx = None
    for i, v in enumerate(values):
        if v and float(v) != 0.0:
            last_idx = i
            out[i] = 0.0
            continue
        if last_idx is None:
            out[i] = float(never_value)
        else:
            out[i] = float(i - last_idx)
    return pd.Series(out, index=pulse.index)


def fetch(
    symbol: str,
    start: str,
    end: str,
    *,
    indices: Optional[Sequence[str]] = None,
    include_current_weights: bool = False,
) -> Optional[pd.DataFrame]:
    """Fetch index constituents features for a single symbol.

    Args:
        symbol: equity ticker (e.g. AAPL)
        start/end: YYYY-MM-DD bounds for feature index
        indices: list of index symbols to track (e.g. ["GSPC.INDX", "DJI.INDX"]).

    Returns:
        DataFrame indexed by tz-naive NYSE sessions.
    """

    try:
        from src.data_sources.eodhd_provider import get_eodhd_provider  # type: ignore
    except Exception:
        return None

    tracked = list(indices) if indices is not None else ["GSPC.INDX", "DJI.INDX"]
    tracked = [str(x).strip().upper() for x in tracked if x and str(x).strip()]
    if not tracked:
        return None

    # True NYSE sessions; ensures holidays are excluded.
    sessions = nyse_sessions_in_range(start, end, tz_aware_utc=False)
    if len(sessions) == 0:
        return None

    provider = get_eodhd_provider()
    if not getattr(provider, "api_key", None):
        return None

    symbol_norm = _norm_component_code(symbol)

    out = pd.DataFrame(index=sessions)
    out["index_constituents_has_data"] = 0.0

    had_any_fundamentals = False
    had_any_hist = False

    add_cols: List[str] = []
    remove_cols: List[str] = []
    member_cols: List[str] = []

    # Build per-index membership/event features.
    for index_symbol in tracked:
        fundamentals = provider.get_fundamentals(index_symbol)
        if not isinstance(fundamentals, dict) or not fundamentals:
            # No access / unsupported index symbol. Keep schema stable and continue.
            key = _index_key_from_symbol(index_symbol)
            out[f"index_constituents_has_hist_{key}"] = 0.0
            out[f"index_constituents_member_{key}"] = 0.0
            out[f"index_constituents_added_{key}"] = 0.0
            out[f"index_constituents_removed_{key}"] = 0.0
            out[f"index_constituents_days_since_add_{key}"] = 9999.0
            out[f"index_constituents_days_since_remove_{key}"] = 9999.0
            out[f"index_constituents_weight_{key}"] = 0.0
            member_cols.append(f"index_constituents_member_{key}")
            add_cols.append(f"index_constituents_added_{key}")
            remove_cols.append(f"index_constituents_removed_{key}")
            continue

        had_any_fundamentals = True

        events = _extract_index_membership_events(
            fundamentals=fundamentals,
            index_symbol=index_symbol,
        )

        key = _index_key_from_symbol(index_symbol)
        has_hist = bool(isinstance(fundamentals.get("HistoricalTickerComponents"), dict) and fundamentals.get("HistoricalTickerComponents"))
        out[f"index_constituents_has_hist_{key}"] = 1.0 if has_hist else 0.0
        if has_hist:
            had_any_hist = True

        # Filter to events for this symbol.
        events = [
            ev
            for ev in events
            if _norm_component_code(ev.component_code) == symbol_norm
        ]

        member = pd.Series(0.0, index=sessions)
        added = pd.Series(0.0, index=sessions)
        removed = pd.Series(0.0, index=sessions)

        for ev in events:
            membership_start, membership_end, removal_pulse = _event_to_sessions(
                start_date=ev.start_date,
                end_date=ev.end_date,
            )

            # Clamp to our requested session range.
            if membership_end is None:
                active_end = sessions[-1]
            else:
                active_end = membership_end

            # Membership state (after EOD look-ahead shift).
            if membership_start <= sessions[-1] and active_end >= sessions[0]:
                lo = max(membership_start, sessions[0])
                hi = min(active_end, sessions[-1])
                member.loc[(member.index >= lo) & (member.index <= hi)] = 1.0

            # Event pulses (applied on the first session when the new composition is tradable).
            if membership_start in added.index:
                added.loc[membership_start] = 1.0
            if removal_pulse is not None and removal_pulse in removed.index:
                removed.loc[removal_pulse] = 1.0

        member_col = f"index_constituents_member_{key}"
        add_col = f"index_constituents_added_{key}"
        rem_col = f"index_constituents_removed_{key}"

        out[member_col] = member
        out[add_col] = added
        out[rem_col] = removed

        out[f"index_constituents_days_since_add_{key}"] = _days_since_pulse(added)
        out[f"index_constituents_days_since_remove_{key}"] = _days_since_pulse(removed)

        member_cols.append(member_col)
        add_cols.append(add_col)
        remove_cols.append(rem_col)

        # Optional: current weight snapshot (NOT historical). Do not forward-fill.
        # This is opt-in because it can be look-ahead when building historical panels.
        try:
            components = fundamentals.get("Components")
            weight = 0.0
            if isinstance(components, dict):
                for _, comp in components.items():
                    if not isinstance(comp, dict):
                        continue
                    code = _norm_component_code(str(comp.get("Code") or ""))
                    if code != symbol_norm:
                        continue
                    w = comp.get("Weight")
                    if w is None:
                        continue
                    wv = pd.to_numeric(w, errors="coerce")
                    if pd.notna(wv):
                        weight = float(wv)
                        break
            out[f"index_constituents_weight_{key}"] = 0.0
            if include_current_weights:
                # Assign weight only on the last available session in our window.
                out.loc[out.index[-1], f"index_constituents_weight_{key}"] = float(weight)
        except Exception:
            out[f"index_constituents_weight_{key}"] = 0.0

    out["index_constituents_has_data"] = 1.0 if had_any_fundamentals else 0.0

    # Global aggregations across tracked indices.
    if member_cols:
        out["index_constituents_num_indices"] = out[member_cols].sum(axis=1)
    else:
        out["index_constituents_num_indices"] = 0.0

    add_any = out[add_cols].sum(axis=1) if add_cols else pd.Series(0.0, index=sessions)
    rem_any = out[remove_cols].sum(axis=1) if remove_cols else pd.Series(0.0, index=sessions)

    out["index_constituents_add_flow_5d"] = add_any.rolling(5, min_periods=1).sum()
    out["index_constituents_add_flow_20d"] = add_any.rolling(20, min_periods=1).sum()
    out["index_constituents_remove_flow_5d"] = rem_any.rolling(5, min_periods=1).sum()
    out["index_constituents_remove_flow_20d"] = rem_any.rolling(20, min_periods=1).sum()

    out = out.replace([np.inf, -np.inf], np.nan)

    out.attrs["telemetry"] = {
        "status": "ok",
        "source": "EODHD.fundamentals.index_components",
        "indices": tracked,
        "include_current_weights": bool(include_current_weights),
        "had_any_hist": bool(had_any_hist),
        "generated_features": int(len(out.columns)),
    }
    out.attrs["feature_counts"] = {"generated": int(len(out.columns))}
    out.attrs["provenance"] = {
        "source": "EODHD.fundamentals",
        "indices": tracked,
        "note": "membership/events are shifted +1 session to avoid look-ahead",
    }

    return out
