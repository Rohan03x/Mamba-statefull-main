from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, cast

import numpy as np
import pandas as pd

from .types import MergedParquetStore, ResearchRankingConfig


def _norm_symbol(sym: str) -> str:
    return (sym or "").strip().upper()


def _zscore_cross_section(x: pd.Series, *, clip: float) -> pd.Series:
    """Z-score across symbols for a single date (vector)."""
    vals = np.array([
        np.nan if v is None else float(v)
        for v in (_safe_float(v) for v in x.to_list())
    ], dtype=np.float64)
    finite = np.isfinite(vals)
    if not finite.any():
        z = np.zeros_like(vals)
    else:
        mu = float(np.nanmean(vals))
        sd = float(np.nanstd(vals, ddof=0))
        if sd <= 1e-12:
            z = np.zeros_like(vals)
        else:
            z = (vals - mu) / sd
            z[~np.isfinite(z)] = 0.0
    if clip > 0:
        z = np.clip(z, -float(clip), float(clip))
    return pd.Series(z, index=x.index)


def _safe_float(x: Any) -> float | None:
    try:
        if x is None:
            return None
        v = float(x)
        if not np.isfinite(v):
            return None
        return v
    except Exception:
        return None


@dataclass
class Tier2Result:
    asof: pd.Timestamp
    ranked: pd.DataFrame
    meta: Dict[str, Any]


@dataclass
class Tier2PanelResult:
    start: pd.Timestamp
    end: pd.Timestamp
    panel: pd.DataFrame
    meta: Dict[str, Any]


class ResearchRankingGate:
    """Tier-2 ranking gate (research-only).

    Inputs:
      - EligibleUniverse(t) from Tier 1
      - internal merged parquet panels for each symbol
      - Stage-A family weights

    Outputs:
      - per-date ranked universe snapshot

    Critical invariants:
      - Tier 2 only consumes data <= t (optionally shifted by sessions)
      - Cross-sectional normalization is per-date (never across time)
      - Raw merged features never leave this module
    """

    def __init__(
        self,
        *,
        store: MergedParquetStore,
        cfg: ResearchRankingConfig,
    ):
        self._store = store
        self._cfg = cfg

    def rank(
        self,
        *,
        eligible: Sequence[str],
        asof: str | pd.Timestamp,
        stage_a_weights: Mapping[str, float],
    ) -> Tier2Result:
        asof_ts = pd.Timestamp(asof).normalize()
        use_fams = list(self._cfg.families) if self._cfg.families else sorted(stage_a_weights.keys())
        weights = {str(k): float(v) for k, v in stage_a_weights.items() if str(k) in set(use_fams)}

        # Per-symbol, compute a scalar snapshot per family.
        # Architecture default: use family-level mean across that family's columns at (t - shift).
        # This respects "snapshots not histories" while still allowing a large family to contribute.
        shift = int(max(0, self._cfg.leakage_safe_shift_sessions))
        effective_ts = asof_ts

        # Load panels internally.
        fam_rows: List[Dict[str, Any]] = []
        missing_syms: List[str] = []

        for sym_raw in eligible:
            sym = _norm_symbol(sym_raw)
            if not sym:
                continue
            try:
                panel = self._store.load_panel(symbol=sym, horizon=int(self._cfg.horizon))
            except Exception:
                panel = None
            if panel is None or panel.empty:
                missing_syms.append(sym)
                continue

            idx = pd.DatetimeIndex(pd.to_datetime(panel.index, errors="coerce")).tz_localize(None)
            panel = panel.copy()
            panel.index = idx
            panel = panel.sort_index()

            # Find the last available session <= asof, then shift backwards by N sessions.
            avail = panel.index[panel.index <= asof_ts]
            if len(avail) == 0:
                missing_syms.append(sym)
                continue
            pos = len(avail) - 1
            eff_pos = max(0, pos - shift)
            effective_ts = pd.Timestamp(avail[eff_pos]).normalize()
            row = cast(pd.Series, panel.loc[avail[eff_pos]])

            row_out: Dict[str, Any] = {"date": asof_ts, "symbol": sym, "effective_date": effective_ts}

            non_nan = 0
            for fam in use_fams:
                fam = str(fam)
                cols = [c for c in panel.columns if str(c).startswith(f"{fam}_")]
                if not cols:
                    row_out[f"snap_{fam}"] = np.nan
                    continue
                vals = np.array([
                    np.nan if v is None else float(v)
                    for v in (_safe_float(row.get(c)) for c in cols)
                ], dtype=np.float64)
                score = float(np.nanmean(vals)) if np.isfinite(vals).any() else np.nan
                row_out[f"snap_{fam}"] = score
                if np.isfinite(float(score)):
                    non_nan += 1

            row_out["non_nan_families"] = int(non_nan)
            fam_rows.append(row_out)

        snap = pd.DataFrame(fam_rows)
        if snap.empty:
            return Tier2Result(
                asof=asof_ts,
                ranked=pd.DataFrame(columns=["date", "symbol", "score", "rank"]),
                meta={
                    "eligible_in": int(len(eligible)),
                    "eligible_ranked": 0,
                    "missing_panels": missing_syms,
                    "weights_used": dict(weights),
                },
            )

        # Apply min coverage.
        snap = snap[snap["non_nan_families"] >= int(self._cfg.min_non_nan_families)].copy()

        # Cross-sectional normalize per family.
        z_cols: List[str] = []
        for fam in use_fams:
            fam = str(fam)
            col = f"snap_{fam}"
            if col not in snap.columns:
                continue
            z = _zscore_cross_section(snap[col], clip=float(self._cfg.zscore_clip))
            z_col = f"z_{fam}"
            snap[z_col] = z
            z_cols.append(z_col)

        # Weighted sum score.
        def _w_for(z_col: str) -> float:
            fam = z_col.replace("z_", "", 1)
            return float(weights.get(fam, 0.0) or 0.0)

        if not z_cols:
            snap["score"] = 0.0
        else:
            w = np.array([_w_for(c) for c in z_cols], dtype=np.float64)
            zmat = snap[z_cols].to_numpy(dtype=np.float64)
            snap["score"] = (zmat * w.reshape(1, -1)).sum(axis=1)

        snap["rank"] = snap["score"].rank(ascending=False, method="dense").astype(int)
        snap = snap.sort_values(["rank", "symbol"]).reset_index(drop=True)

        meta: Dict[str, Any] = {
            "eligible_in": int(len(eligible)),
            "eligible_ranked": int(len(snap)),
            "missing_panels": sorted(set(missing_syms)),
            "weights_used": dict(weights),
            "families_used": list(use_fams),
            "effective_shift_sessions": int(shift),
        }

        # Output schema is intentionally lean; we keep z-scores (safe research summary)
        # but we do not include raw merged feature columns.
        keep = ["date", "symbol", "effective_date", "score", "rank", "non_nan_families"] + z_cols
        ranked_df = snap[keep].copy()
        return Tier2Result(asof=asof_ts, ranked=ranked_df, meta=meta)

    def score_panel(
        self,
        *,
        eligible: Sequence[str],
        start: str | pd.Timestamp,
        end: str | pd.Timestamp,
        stage_a_weights: Mapping[str, float],
    ) -> Tier2PanelResult:
        """Compute Tier-2 universe scores for every date in [start, end].

        Output is a (date, symbol) panel with cross-sectional scores and ranks.

        Leak safety:
        - For target date t, features are read from (t - shift_sessions) sessions.
        - Cross-sectional normalization is done per target date t.
        """

        start_ts = pd.Timestamp(start).normalize()
        end_ts = pd.Timestamp(end).normalize()
        if end_ts < start_ts:
            start_ts, end_ts = end_ts, start_ts

        use_fams = list(self._cfg.families) if self._cfg.families else sorted(stage_a_weights.keys())
        fam_set = set(str(f) for f in use_fams)
        weights = {str(k): float(v) for k, v in stage_a_weights.items() if str(k) in fam_set}

        shift = int(max(0, self._cfg.leakage_safe_shift_sessions))

        # Load panels once.
        panels: Dict[str, pd.DataFrame] = {}
        missing_syms: List[str] = []
        for sym_raw in eligible:
            sym = _norm_symbol(sym_raw)
            if not sym:
                continue
            try:
                panel = self._store.load_panel(symbol=sym, horizon=int(self._cfg.horizon))
            except Exception:
                panel = None
            if panel is None or panel.empty:
                missing_syms.append(sym)
                continue

            idx = pd.DatetimeIndex(pd.to_datetime(panel.index, errors="coerce")).tz_localize(None)
            panel = panel.copy()
            panel.index = idx
            panel = panel.sort_index()
            panel = panel[(panel.index >= start_ts) & (panel.index <= end_ts)]
            if panel.empty:
                missing_syms.append(sym)
                continue
            panels[sym] = panel

        # Establish the trading-date axis (union across symbols).
        all_dates: List[pd.Timestamp] = []
        for p in panels.values():
            all_dates.extend([pd.Timestamp(x).normalize() for x in p.index.to_list()])
        date_index = pd.DatetimeIndex(sorted(set(all_dates)))
        date_index = date_index[(date_index >= start_ts) & (date_index <= end_ts)]
        if len(date_index) == 0:
            return Tier2PanelResult(
                start=start_ts,
                end=end_ts,
                panel=pd.DataFrame(columns=["date", "symbol", "score", "rank"]),
                meta={
                    "start": start_ts.strftime("%Y-%m-%d"),
                    "end": end_ts.strftime("%Y-%m-%d"),
                    "eligible_in": int(len(eligible)),
                    "dates": 0,
                    "missing_panels": sorted(set(missing_syms)),
                    "weights_used": dict(weights),
                    "families_used": list(use_fams),
                    "effective_shift_sessions": int(shift),
                },
            )

        # Effective feature date is t-shift sessions on the master trading axis.
        if shift <= 0:
            effective_dates = pd.Series(date_index, index=date_index)
        else:
            eff = date_index.to_series().shift(shift)
            effective_dates = eff

        pieces: List[pd.DataFrame] = []
        for sym, panel in panels.items():
            # For each family, compute a per-date scalar from the merged panel.
            fam_cols: Dict[str, List[str]] = {
                str(fam): [c for c in panel.columns if str(c).startswith(f"{fam}_")] for fam in use_fams
            }

            sym_frame = pd.DataFrame(index=date_index)
            for fam in use_fams:
                fam = str(fam)
                cols = fam_cols.get(fam) or []
                if not cols:
                    sym_frame[f"snap_{fam}"] = np.nan
                    continue
                # Mean across that family's columns for each day.
                s = panel[cols].mean(axis=1, skipna=True)
                # Reindex to the master calendar and shift forward by N sessions so
                # value at date t is computed from (t-shift).
                s = s.reindex(date_index)
                if shift:
                    s = s.shift(shift)
                sym_frame[f"snap_{fam}"] = s

            sym_frame["date"] = pd.to_datetime(date_index, errors="coerce")
            sym_frame["symbol"] = sym
            sym_frame["effective_date"] = pd.to_datetime(effective_dates.values, errors="coerce")
            pieces.append(sym_frame.reset_index(drop=True))

        snap = pd.concat(pieces, axis=0, ignore_index=True)
        snap["date"] = pd.to_datetime(snap["date"], errors="coerce").dt.tz_localize(None)
        snap["effective_date"] = pd.to_datetime(snap["effective_date"], errors="coerce").dt.tz_localize(None)

        snap_cols = [f"snap_{str(f)}" for f in use_fams]
        snap["non_nan_families"] = (
            snap[snap_cols]
            .apply(lambda r: int(np.isfinite(r.to_numpy(dtype=np.float64)).sum()), axis=1)
            .astype(int)
        )

        # Cross-sectional normalize per date.
        z_cols: List[str] = []
        for fam in use_fams:
            fam = str(fam)
            col = f"snap_{fam}"
            if col not in snap.columns:
                continue
            z_col = f"z_{fam}"
            snap[z_col] = snap.groupby("date", sort=False)[col].transform(
                lambda s: _zscore_cross_section(s, clip=float(self._cfg.zscore_clip))
            )
            z_cols.append(z_col)

        # Weighted sum score.
        def _w_for(z_col: str) -> float:
            fam = z_col.replace("z_", "", 1)
            return float(weights.get(fam, 0.0) or 0.0)

        if not z_cols:
            snap["score"] = 0.0
        else:
            w = np.array([_w_for(c) for c in z_cols], dtype=np.float64)
            zmat = snap[z_cols].to_numpy(dtype=np.float64)
            snap["score"] = (zmat * w.reshape(1, -1)).sum(axis=1)

        # Mark rankable symbols and rank only those (others keep NaN rank/score).
        rankable = snap["non_nan_families"] >= int(self._cfg.min_non_nan_families)
        snap.loc[~rankable, "score"] = np.nan
        snap["rank"] = snap.groupby("date", sort=False)["score"].transform(
            lambda s: s.rank(ascending=False, method="dense")
        )

        keep = [
            "date",
            "symbol",
            "effective_date",
            "score",
            "rank",
            "non_nan_families",
        ] + z_cols

        out = snap[keep].copy()
        out = out.sort_values(["date", "rank", "symbol"], na_position="last").reset_index(drop=True)

        meta: Dict[str, Any] = {
            "start": start_ts.strftime("%Y-%m-%d"),
            "end": end_ts.strftime("%Y-%m-%d"),
            "eligible_in": int(len(eligible)),
            "dates": int(len(date_index)),
            "eligible_ranked_rows": int(out["score"].notna().sum()),
            "missing_panels": sorted(set(missing_syms)),
            "weights_used": dict(weights),
            "families_used": list(use_fams),
            "effective_shift_sessions": int(shift),
            "min_non_nan_families": int(self._cfg.min_non_nan_families),
        }

        return Tier2PanelResult(start=start_ts, end=end_ts, panel=out, meta=meta)
