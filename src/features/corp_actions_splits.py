"""Corporate actions: stock splits (+ post-split microstructure regime) feature family.

Family: corp_actions_splits

Raw source:
- EODHD Corporate Actions: Splits API
  https://eodhd.com/financial-apis/api-splits-dividends/

Leakage / timing policy:
- Treat split as known at split effective date end-of-day.
- Apply the event pulse and magnitude on the *next* NYSE trading session.

Expected sparsity:
- Most symbols have zero splits in typical windows; outputs are mostly zeros.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

from src.dcf_lab.utils.timealign import nyse_sessions_in_range
from src.dcf_lab.utils.trading_calendar import add_sessions, session_on_or_after


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


def _parse_split_ratio(row: Dict[str, Any]) -> float:
    """Parse split ratio into a numeric multiplier.

    Examples:
      - "4/1" => 4.0
      - "2.000000/1.000000" => 2.0
      - 4 => 4.0
    """

    for k in ("split", "ratio", "split_ratio", "Split", "SplitRatio"):
        if k in row and row.get(k) is not None:
            raw = row.get(k)
            if isinstance(raw, (int, float, np.number)):
                r = _to_float(raw)
                return r if r > 0 else 0.0
            s = str(raw).strip()
            if not s:
                continue
            if "/" in s:
                left, right = s.split("/", 1)
                num = _to_float(left)
                den = _to_float(right)
                if den > 0 and num > 0:
                    return float(num / den)
            # Some providers use "2:1" formatting
            if ":" in s:
                left, right = s.split(":", 1)
                num = _to_float(left)
                den = _to_float(right)
                if den > 0 and num > 0:
                    return float(num / den)
            r = _to_float(s)
            return r if r > 0 else 0.0

    # Fallback: numerator/denominator fields
    num = None
    den = None
    for k in ("numerator", "Numerator", "from", "From"):
        if k in row and row.get(k) is not None:
            num = _to_float(row.get(k))
            break
    for k in ("denominator", "Denominator", "to", "To"):
        if k in row and row.get(k) is not None:
            den = _to_float(row.get(k))
            break
    if num is not None and den is not None and den > 0 and num > 0:
        return float(num / den)

    return 0.0


def _days_since_last_event(pulse: pd.Series, *, never_value: float = 9999.0) -> pd.Series:
    values = pulse.values
    out = np.empty(len(values), dtype=float)
    last_idx = None
    for i, v in enumerate(values):
        if bool(v) and float(v) != 0.0:
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
    post_5d_window: int = 5,
    post_20d_window: int = 20,
    count_5y_days: int = 1825,
) -> Optional[pd.DataFrame]:
    """Fetch split-based features for one symbol."""

    try:
        from src.data_sources.eodhd_provider import get_eodhd_provider  # type: ignore
    except Exception:
        return None

    sessions = nyse_sessions_in_range(start, end, tz_aware_utc=False)
    if len(sessions) == 0:
        return None

    # Extend history to compute the rolling 5y count.
    fetch_start = (pd.Timestamp(start) - pd.Timedelta(days=max(800, int(count_5y_days) + 90))).strftime("%Y-%m-%d")
    sessions_ext = nyse_sessions_in_range(fetch_start, end, tz_aware_utc=False)
    if len(sessions_ext) == 0:
        return None

    provider = get_eodhd_provider()
    if not getattr(provider, "api_key", None):
        return None

    splits_df = provider.get_splits(symbol, start_date=fetch_start, end_date=end)
    if splits_df is None:
        return None

    out = pd.DataFrame(index=sessions_ext)
    out["corp_actions_splits_has_data"] = 1.0
    out["corp_actions_splits_flag"] = 0.0
    out["corp_actions_splits_ratio"] = 0.0
    out["corp_actions_splits_log_ratio"] = 0.0

    # Handle empty payloads cleanly (no splits in range).
    if splits_df.empty:
        out["corp_actions_splits_post_5d"] = 0.0
        out["corp_actions_splits_post_20d"] = 0.0
        out["corp_actions_splits_days_since"] = _days_since_last_event(out["corp_actions_splits_flag"], never_value=9999.0)
        out["corp_actions_splits_count_5y"] = 0.0

        out = out.loc[(out.index >= sessions[0]) & (out.index <= sessions[-1])].copy()
        out.attrs["telemetry"] = {"status": "ok", "source": "EODHD.splits", "note": "empty payload"}
        out.attrs["feature_counts"] = {"generated": int(len(out.columns))}
        out.attrs["provenance"] = {"source": "EODHD.splits", "leakage": "shift to next session"}
        return out

    # Normalize into event list.
    records = []
    for ts, row in splits_df.iterrows():
        if pd.isna(ts):
            continue
        try:
            d = pd.Timestamp(ts).normalize()
        except Exception:
            continue
        payload = row.to_dict() if hasattr(row, "to_dict") else {}
        ratio = _parse_split_ratio(payload)
        if ratio <= 0:
            continue

        # Effective session: known end-of-day of split date -> apply next trading session.
        base = session_on_or_after(d)
        eff = add_sessions(base, 1)

        ann = None
        for k in ("announcementDate", "AnnouncementDate", "declarationDate", "DeclarationDate"):
            if k in payload and payload.get(k) is not None:
                ann = pd.to_datetime(payload.get(k), errors="coerce")
                break
        records.append((eff, ratio, pd.Timestamp(ann).normalize() if ann is not None and pd.notna(ann) else None))

    if not records:
        # No parseable events.
        out["corp_actions_splits_post_5d"] = 0.0
        out["corp_actions_splits_post_20d"] = 0.0
        out["corp_actions_splits_days_since"] = _days_since_last_event(out["corp_actions_splits_flag"], never_value=9999.0)
        out["corp_actions_splits_count_5y"] = 0.0

        out = out.loc[(out.index >= sessions[0]) & (out.index <= sessions[-1])].copy()
        out.attrs["telemetry"] = {"status": "ok", "source": "EODHD.splits", "note": "no parseable events"}
        out.attrs["feature_counts"] = {"generated": int(len(out.columns))}
        out.attrs["provenance"] = {"source": "EODHD.splits", "leakage": "shift to next session"}
        return out

    events = pd.DataFrame(records, columns=["effective_session", "ratio", "announcement_date"]).sort_values("effective_session")

    # When multiple splits map to same effective session, combine multiplicatively.
    agg = events.groupby("effective_session", as_index=True)["ratio"].prod().sort_index()

    # Populate the pulses.
    for eff, ratio in agg.items():
        eff_ts = pd.Timestamp(eff).normalize()
        if eff_ts not in out.index:
            continue
        out.loc[eff_ts, "corp_actions_splits_flag"] = 1.0
        out.loc[eff_ts, "corp_actions_splits_ratio"] = float(ratio)
        out.loc[eff_ts, "corp_actions_splits_log_ratio"] = float(np.log(max(1e-12, float(ratio))))

    # Post-event regimes (inclusive window from effective session).
    days_since = _days_since_last_event(out["corp_actions_splits_flag"], never_value=9999.0)
    out["corp_actions_splits_days_since"] = days_since
    out["corp_actions_splits_post_5d"] = ((days_since >= 0.0) & (days_since <= float(post_5d_window))).astype(float)
    out["corp_actions_splits_post_20d"] = ((days_since >= 0.0) & (days_since <= float(post_20d_window))).astype(float)

    # Rolling 5y count of split events (time-based, session index is DatetimeIndex).
    out["corp_actions_splits_count_5y"] = (
        out["corp_actions_splits_flag"].rolling(f"{int(count_5y_days)}D", min_periods=1).sum().fillna(0.0)
    )

    out = out.loc[(out.index >= sessions[0]) & (out.index <= sessions[-1])].copy()
    out.attrs["telemetry"] = {"status": "ok", "source": "EODHD.splits"}
    out.attrs["feature_counts"] = {"generated": int(len(out.columns))}
    out.attrs["provenance"] = {"source": "EODHD.splits", "leakage": "shift to next session"}
    return out
