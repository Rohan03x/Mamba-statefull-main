"""Role-aware portfolio context loader.

This reads the per-symbol provenance sidecar emitted by `tools/prep_families.py` and
uses it to route features into the correct portfolio roles/usages:
- predictive: direction (alpha)
- risk: risk scaling (sigma_exec modifiers)
- regime: exposure modulation
- hygiene: hard veto/gating

It is intentionally conservative:
- If a column has no explicit role in provenance, it falls back to family intent
  via the canonical family metadata registry.
- If allowed-usage maps are missing, it infers them using the same rules as
  `src.features.feature_roles`.

This module is designed for backtests / research runs; it subselects only the
columns required for risk/regime/hygiene overlays.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class RoleAwareDayContext:
    symbols: List[str]
    hygiene_ok: np.ndarray  # shape (n_assets,) bool
    risk_scale: np.ndarray  # shape (n_assets,) float in (0,1]
    regime_multiplier: np.ndarray  # shape (n_assets,) float in [0,1]
    # Quantile forecast blending (Option 1 from spec)
    quantile_z: np.ndarray  # shape (n_assets,) float, z_q = mu_q / sigma_q
    # Split stress for execution discipline: max(flag, post_5d, post_20d) * abs(log_ratio)
    split_stress: np.ndarray  # shape (n_assets,) float in [0, ~2]
    # Policy controller state features from portfolio families
    cboe_panic_premium: float = 0.0
    cboe_term_slope: float = 0.0
    cboe_vol_risk_premium_z: float = 0.0
    calibration_overall_score: float = 1.0
    online_trust_score: float = 1.0


def _safe_bool(v: object) -> bool:
    try:
        return bool(v)
    except Exception:
        return False


def _safe_float(v: object, default: float = 0.0) -> float:
    try:
        if v is None:
            return float(default)
        return float(v)
    except Exception:
        return float(default)


def _load_json(path: Path) -> Dict[str, object]:
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


# Removed: _load_provenance_columns_csv() - CSV provenance was redundant with JSON


def _as_dict(payload: Mapping[str, object], key: str) -> Dict[str, object]:
    v = payload.get(key, {})
    return v if isinstance(v, dict) else {}


def _first_existing(paths: Sequence[Path]) -> Optional[Path]:
    for p in paths:
        try:
            if p.exists():
                return p
        except Exception:
            continue
    return None


class RoleAwareContext:
    """Loads (features,index,provenance) per symbol and provides per-day overlays.
    
    This module reads the portfolio parquet (not Mamba parquet) and computes:
    - hygiene_ok: hard veto based on has_data / days_since_update (NaN also vetoes)
    - risk_scale: 1/(1+risk_agg) from RISK columns using robust normalization
    - regime_multiplier: 1-regime_agg from REGIME columns
    
    For quantile forecast blending, it also provides:
    - quantile_z: auxiliary z-score from quantile forecast (mu_q/sigma_q)
    
    This is used in phase2_stateful.py to apply overlays AFTER z is computed,
    BEFORE thresholding.
    """

    def __init__(
        self,
        *,
        symbols: Sequence[str],
        horizon: int,
        parquet_dir: Path,
        strict: bool = False,
        registry_path: Optional[Path] = None,
        data_source: str = "portfolio",  # portfolio|mamba|merged|auto|features
    ) -> None:
        self.symbols = [str(s).upper() for s in symbols]
        self.horizon = int(horizon)
        self.parquet_dir = Path(parquet_dir)
        self.strict = bool(strict)
        self.registry_path = Path(registry_path) if registry_path is not None else None
        self.data_source = str(data_source or "auto").strip().lower()

        # Family metadata is required for fallback intent + usage inference.
        # Load once (do not re-read the registry per column).
        try:
            from src.features.feature_roles import load_family_meta_from_registry  # type: ignore

            self._family_meta = load_family_meta_from_registry(self.registry_path)
        except Exception:
            self._family_meta = {}

        # Per symbol store:
        # - date_int -> row_id
        # - features_df subset (numpy)
        # - column lists for risk/regime/hygiene
        self._row_by_date: Dict[str, Dict[int, int]] = {}
        self._feat_cols: Dict[str, List[str]] = {}
        self._feat_mat: Dict[str, np.ndarray] = {}

        self._risk_cols: Dict[str, List[str]] = {}
        self._regime_cols: Dict[str, List[str]] = {}
        self._hygiene_cols: Dict[str, List[str]] = {}

        # Quantile forecast columns for z_q blending (Option 1 from spec)
        self._quantile_mu_col: Dict[str, Optional[str]] = {}  # q50 or q_median_50
        self._quantile_sigma_col: Dict[str, Optional[str]] = {}  # q_spread_95_5 or q_vol_forecast

        # Policy controller state columns (cross-sectional aggregates)
        self._cboe_cols: Dict[str, List[str]] = {}  # panic_premium, normalized_term_slope, etc.
        self._calibration_cols: Dict[str, List[str]] = {}  # overall_score, etc.
        self._online_cols: Dict[str, List[str]] = {}  # trust_score, etc.

        # Corp actions splits columns for split_stress computation
        self._splits_cols: Dict[str, List[str]] = {}  # flag, post_5d, post_20d, log_ratio, recency

        # Column → family mapping for per-family staleness thresholds
        self._col_family: Dict[str, Dict[str, str]] = {}

        # Optional per-column normalization stats (p05/p95) for risk/regime aggregation.
        self._col_norm: Dict[str, Dict[str, Tuple[float, float]]] = {}

        self._load_all()

    def _role_from_family_intent(self, family_id: str) -> str:
        # Map canonical intent to role string used in provenance.
        # mixed falls back to predictive.
        try:
            from src.features.feature_roles import FeatureRole  # type: ignore
        except Exception:
            return "predictive"

        meta = self._family_meta.get(str(family_id), None)
        if meta is None:
            return FeatureRole.PREDICTIVE.value
        return str(getattr(meta.primary_intent, "value", meta.primary_intent))

    def _infer_usages_for_column(self, *, role: str, family_id: str) -> Tuple[Dict[str, bool], Dict[str, str]]:
        try:
            from src.features.feature_roles import FeatureRole, infer_allowed_usages_with_reasons  # type: ignore
        except Exception:
            return {"risk_scale_ok": False, "gating_ok": False, "veto_ok": False}, {}

        fam_meta = self._family_meta.get(str(family_id))
        if fam_meta is None:
            # Conservative default.
            fam_meta = self._family_meta.get("<unknown>")
        if fam_meta is None:
            return {"risk_scale_ok": False, "gating_ok": False, "veto_ok": False}, {}
        try:
            fr = FeatureRole(str(role))
        except Exception:
            fr = FeatureRole.PREDICTIVE
        usages, reasons = infer_allowed_usages_with_reasons(feature_role=fr, family_meta=fam_meta)
        return usages, reasons

    def _load_all(self) -> None:
        for sym in self.symbols:
            h = int(self.horizon)

            # Be permissive about symbol casing (workspace has mixed conventions).
            # Try new .meta.json format first, then legacy .provenance.json
            prov_path = _first_existing(
                [
                    self.parquet_dir / f"{sym}_h{h}_merged.meta.json",
                    self.parquet_dir / f"{sym.lower()}_h{h}_merged.meta.json",
                    # Legacy paths for backward compatibility
                    self.parquet_dir / f"{sym}_h{h}_merged.provenance.json",
                    self.parquet_dir / f"{sym.lower()}_h{h}_merged.provenance.json",
                ]
            )

            # There are multiple filename conventions in this workspace; be permissive.
            index_path = _first_existing(
                [
                    self.parquet_dir / f"{sym}_{h}_index.parquet",
                    self.parquet_dir / f"{sym}_h{h}_index.parquet",
                    self.parquet_dir / f"{sym}_h{h}__index.parquet",
                ]
            )
            features_path = _first_existing(
                [
                    self.parquet_dir / f"{sym}_{h}_features.parquet",
                    self.parquet_dir / f"{sym}_h{h}_features.parquet",
                    self.parquet_dir / f"{sym}_h{h}__features.parquet",
                ]
            )
            merged_path = self.parquet_dir / f"{sym}_h{h}_merged.parquet"
            merged_mamba_path = self.parquet_dir / f"{sym}_h{h}_merged_mamba.parquet"
            merged_portfolio_path = self.parquet_dir / f"{sym}_h{h}_merged_portfolio.parquet"

            # Load provenance from JSON only (CSV was redundant and removed)
            prov = _load_json(prov_path) if prov_path is not None else {}

            # Extract maps directly from JSON
            role_map = dict(_as_dict(prov, "column_role_map") or {})
            family_map = dict(_as_dict(prov, "column_family_map") or {})

            # Usage maps from JSON
            risk_scale_ok_map = dict(_as_dict(prov, "column_risk_scale_ok_map") or {})
            gating_ok_map = dict(_as_dict(prov, "column_gating_ok_map") or {})
            veto_ok_map = dict(_as_dict(prov, "column_veto_ok_map") or {})

            if index_path is None or not index_path.exists():
                if self.strict:
                    raise FileNotFoundError(f"Missing index parquet for {sym} (h={h}) in {self.parquet_dir}")
                continue

            # Decide which data parquet to use for values.
            use_features = self.data_source in {"auto", "features"}
            use_merged = self.data_source in {"auto", "merged"}

            data_path: Optional[Path] = None
            if self.data_source == "features":
                data_path = features_path
            elif self.data_source == "merged":
                data_path = merged_path if merged_path.exists() else None
            elif self.data_source == "portfolio":
                if merged_portfolio_path.exists():
                    data_path = merged_portfolio_path
                elif merged_path.exists() and not self.strict:
                    data_path = merged_path
            elif self.data_source == "mamba":
                if merged_mamba_path.exists():
                    data_path = merged_mamba_path
                elif merged_path.exists() and not self.strict:
                    data_path = merged_path
            else:
                # auto: prefer session-aligned features parquet (it matches *_index.parquet rows).
                if features_path is not None and features_path.exists():
                    data_path = features_path
                elif merged_path.exists():
                    data_path = merged_path
                elif merged_portfolio_path.exists():
                    data_path = merged_portfolio_path

            if data_path is None or not data_path.exists():
                if self.strict:
                    raise FileNotFoundError(f"Missing data parquet for {sym} (h={h}) in {self.parquet_dir}")
                continue

            idx_df = pd.read_parquet(index_path)
            data_df = pd.read_parquet(data_path)

            # Build date->row map.
            date_int = pd.to_numeric(idx_df.get("date"), errors="coerce").fillna(-1).astype(int)
            row_id = pd.to_numeric(idx_df.get("row_id"), errors="coerce").fillna(-1).astype(int)

            # Map session YYYYMMDD -> row in the chosen data_df.
            # - If using *_features.parquet, row_id matches data_df row index.
            # - If using *_merged.parquet, it is calendar-aligned; map by date column.
            m: Dict[int, int] = {}
            if data_path.name.endswith("_merged.parquet"):
                merged_date_col = data_df.get("date")
                if merged_date_col is None:
                    merged_dates = pd.Series([-1] * len(data_df), dtype=int)
                else:
                    # Support either YYYYMMDD ints or datetime-like values.
                    try:
                        is_dt = pd.api.types.is_datetime64_any_dtype(merged_date_col)
                    except Exception:
                        is_dt = False

                    if is_dt:
                        dt = pd.to_datetime(merged_date_col, errors="coerce")
                        merged_dates = dt.dt.strftime("%Y%m%d").fillna("-1").astype(int)
                    else:
                        merged_dates_num = pd.to_numeric(merged_date_col, errors="coerce")
                        # Heuristic: YYYYMMDD ints live around 2e7; nanosecond epochs are ~1e18.
                        if merged_dates_num.notna().any() and float(merged_dates_num.dropna().median()) < 1e10:
                            merged_dates = merged_dates_num.fillna(-1).astype(int)
                        else:
                            dt = pd.to_datetime(merged_date_col, errors="coerce")
                            merged_dates = dt.dt.strftime("%Y%m%d").fillna("-1").astype(int)
                merged_row_by_date: Dict[int, int] = {}
                for j, d in enumerate(merged_dates.to_list()):
                    if int(d) > 0:
                        merged_row_by_date[int(d)] = int(j)
                for d in date_int.to_list():
                    di = int(d)
                    r = merged_row_by_date.get(di)
                    if di > 0 and r is not None and int(r) >= 0:
                        m[di] = int(r)
            else:
                for d, r in zip(date_int.to_list(), row_id.to_list()):
                    if int(d) <= 0 or int(r) < 0:
                        continue
                    m[int(d)] = int(r)

            self._row_by_date[sym] = m

            # Decide which columns we need for overlays.
            risk_cols: List[str] = []
            regime_cols: List[str] = []
            hygiene_cols: List[str] = []

            has_explicit_veto = len(veto_ok_map) > 0

            # Snapshot-only families (e.g. options) may legitimately have sparse coverage.
            # Treat their *_has_data / *_days_since_update signals as informative by default,
            # not as portfolio-wide hard vetoes.
            snapshot_optional_families = set(
                [
                    s.strip().lower()
                    for s in str(os.getenv("PORTFOLIO_OPTIONAL_SNAPSHOT_FAMILIES", "options")).split(",")
                    if s.strip()
                ]
            )
            snapshot_optional_wildcard = "*" in snapshot_optional_families

            optional_hygiene_families = set(
                [
                    s.strip().lower()
                    for s in str(
                        os.getenv(
                            "PORTFOLIO_OPTIONAL_HYGIENE_FAMILIES",
                            "options,finbert,short_interest,index_constituents,calibration,peer_screener_context",
                        )
                    ).split(",")
                    if s.strip()
                ]
            )

            def _is_optional_snapshot_family(family_id: str) -> bool:
                fid = str(family_id or "").strip().lower()
                if not fid:
                    return False
                if fid in snapshot_optional_families:
                    return True
                if not snapshot_optional_wildcard:
                    return False
                meta = self._family_meta.get(str(family_id))
                cadence = str(getattr(meta, "update_cadence", "")).strip().lower() if meta is not None else ""
                return cadence == "snapshot"

            allow_optional_snapshot_veto = str(os.getenv("PORTFOLIO_ALLOW_OPTIONAL_SNAPSHOT_VETO", "0")).strip() in {
                "1",
                "true",
                "yes",
            }

            def _is_optional_hygiene_family(family_id: str, col_name: str) -> bool:
                fid = str(family_id or "").strip().lower()
                if fid and fid in optional_hygiene_families:
                    return True
                # Treat HF / LLM / transcript-derived snapshot feeds as optional for portfolio-wide veto.
                if fid.endswith("_hf") or "_hf_" in str(col_name).lower():
                    return True
                return _is_optional_snapshot_family(family_id)

            def _is_vetoish_name(col_name: str) -> bool:
                n = str(col_name).lower()
                return (
                    ("has_data" in n)
                    or ("days_since" in n)
                    or ("stale" in n)
                    or ("missing" in n)
                    or ("delist" in n)
                    or ("eligible" in n)
                    or n.endswith("__days_since_update")
                )

            # Avoid accidentally pulling non-feature columns.
            cols_iter = [c for c in data_df.columns if str(c) != "date"]
            for col in cols_iter:
                col_s = str(col)
                role = str(role_map.get(col_s, "") or "")
                fam = str(family_map.get(col_s, "") or "")

                # Staleness weights are policy controls; treat as regime/gating.
                if "stale_weight" in col_s:
                    role = "regime"

                if not role:
                    role = self._role_from_family_intent(fam)

                # If allowed-usage maps are missing, infer usage flags.
                rs_ok = risk_scale_ok_map.get(col_s)
                g_ok = gating_ok_map.get(col_s)
                v_ok = veto_ok_map.get(col_s)
                if rs_ok is None and g_ok is None and v_ok is None:
                    usages, _reasons = self._infer_usages_for_column(role=role, family_id=fam)
                    rs_ok = usages.get("risk_scale_ok", False)
                    g_ok = usages.get("gating_ok", False)
                    v_ok = usages.get("veto_ok", False)

                if _safe_bool(rs_ok):
                    risk_cols.append(col_s)
                if role == "regime" or _safe_bool(g_ok):
                    regime_cols.append(col_s)

                # Hygiene veto should be hard and conservative:
                # - If explicit veto_ok exists in provenance, ONLY those columns can veto.
                # - Otherwise, allow hygiene-role columns to veto only if they look like gate signals.
                # - Snapshot-only families are NOT allowed to portfolio-veto by default.
                if has_explicit_veto:
                    if _safe_bool(v_ok):
                        if _is_optional_snapshot_family(fam) and not allow_optional_snapshot_veto:
                            continue
                        hygiene_cols.append(col_s)
                else:
                    if role == "hygiene" and _is_vetoish_name(col_s):
                        if _is_optional_hygiene_family(fam, col_s):
                            continue
                        hygiene_cols.append(col_s)

            # De-dup while preserving order.
            def _uniq(xs: List[str]) -> List[str]:
                seen: set[str] = set()
                out: List[str] = []
                for x in xs:
                    if x in seen:
                        continue
                    seen.add(x)
                    out.append(x)
                return out

            risk_cols = _uniq([c for c in risk_cols if c in data_df.columns])
            regime_cols = _uniq([c for c in regime_cols if c in data_df.columns])
            hygiene_cols = _uniq([c for c in hygiene_cols if c in data_df.columns])

            # Identify quantile forecast columns for z_q blending
            quantile_mu_col = None
            quantile_sigma_col = None
            for c in data_df.columns:
                c_lower = str(c).lower()
                if c_lower in {"quantile_forecast_q50", "quantile_forecast_q_median_50"}:
                    quantile_mu_col = str(c)
                if c_lower in {"quantile_forecast_q_spread_95_5", "quantile_forecast_q_vol_forecast"}:
                    quantile_sigma_col = str(c)
            self._quantile_mu_col[sym] = quantile_mu_col
            self._quantile_sigma_col[sym] = quantile_sigma_col

            # Identify CBOE term structure columns for policy state
            cboe_cols = [c for c in data_df.columns if str(c).lower().startswith("cboe_term_") or str(c).lower() in {
                "panic_premium", "normalized_term_slope", "vix_ratio_term", "vol_risk_premium_z",
                "vix_roll_yield", "vix_contango_strength"
            }]
            self._cboe_cols[sym] = cboe_cols

            # Identify calibration columns for policy state
            calibration_cols = [c for c in data_df.columns if str(c).lower().startswith("calibration_")]
            self._calibration_cols[sym] = calibration_cols

            # Identify online learning columns for policy state
            online_cols = [c for c in data_df.columns if str(c).lower().startswith("online_learning_")]
            self._online_cols[sym] = online_cols

            # Identify corp actions splits columns for split_stress
            # CRITICAL: Use bounded recency instead of raw days_since
            splits_cols = [c for c in data_df.columns if str(c).lower().startswith("corp_actions_splits_")]
            self._splits_cols[sym] = splits_cols

            # Subselect for memory - include quantile/cboe/calibration/online/splits for policy state
            extra_cols = []
            if quantile_mu_col:
                extra_cols.append(quantile_mu_col)
            if quantile_sigma_col:
                extra_cols.append(quantile_sigma_col)
            extra_cols.extend(cboe_cols)
            extra_cols.extend(calibration_cols)
            extra_cols.extend(online_cols)
            extra_cols.extend(splits_cols)

            use_cols = _uniq([*risk_cols, *regime_cols, *hygiene_cols, *extra_cols])
            if not use_cols:
                # Still keep a small empty matrix.
                self._feat_cols[sym] = []
                self._feat_mat[sym] = np.zeros((len(data_df), 0), dtype=np.float32)
                self._risk_cols[sym] = []
                self._regime_cols[sym] = []
                self._hygiene_cols[sym] = []
                continue

            use_cols = [c for c in use_cols if c in data_df.columns]
            sub = data_df[use_cols].copy()
            sub = sub.apply(pd.to_numeric, errors="coerce").fillna(0.0)

            # Precompute per-column robust normalization for risk/regime aggregation.
            # We only do this for columns that participate in risk/regime overlays.
            norm_stats: Dict[str, Tuple[float, float]] = {}
            for c in _uniq([*risk_cols, *regime_cols]):
                if c not in sub.columns:
                    continue
                try:
                    v = pd.to_numeric(sub[c], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
                    if v.empty:
                        continue
                    lo = float(v.quantile(0.05))
                    hi = float(v.quantile(0.95))
                    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
                        continue
                    norm_stats[str(c)] = (lo, hi)
                except Exception:
                    continue
            self._col_norm[sym] = norm_stats

            self._feat_cols[sym] = use_cols
            self._feat_mat[sym] = sub.to_numpy(dtype=np.float32, copy=True)
            self._risk_cols[sym] = risk_cols
            self._regime_cols[sym] = regime_cols
            self._hygiene_cols[sym] = hygiene_cols
            # Store column → family map for per-family staleness thresholds
            self._col_family[sym] = dict(family_map)

    # -------------------------------------------------------------------------
    # Per-family staleness thresholds (trading days)
    # Quarterly families (fin_g*) can go ~120 days between updates legitimately.
    # Monthly families (some macro) ~30 days. Daily families ~10 days.
    # -------------------------------------------------------------------------
    _FAMILY_MAX_DAYS_SINCE_UPDATE: Dict[str, float] = {
        # Quarterly fundamentals (fin_g1-7, options_anchoring, calibration)
        "fin_g1": 130.0,
        "fin_g2": 130.0,
        "fin_g3": 130.0,
        "fin_g4": 130.0,
        "fin_g5": 130.0,
        "fin_g6": 130.0,
        "fin_g7": 130.0,
        "options_anchoring": 130.0,
        "calibration": 130.0,
        # Monthly/semi-monthly
        "google_trends": 45.0,
        "macro_panel": 45.0,
        "short_interest": 30.0,
        "insider_form4": 60.0,
        # Snapshot families (less frequent, but valid)
        "dcf": 90.0,
        "earnings": 120.0,
        "dividends": 120.0,
        # Daily families (default)
        "ohlcv": 10.0,
        "candle_mechanics": 10.0,
        "garch_iv": 10.0,
        "microstructure_intraday": 10.0,
        "cboe_term": 10.0,
        "vix_futures": 10.0,
        "correlation": 10.0,
        "alternative_signals": 10.0,
        # Event-driven (can be stale for long periods legitimately)
        "corp_actions_splits": 365.0,
        "econ_events_calendar": 10.0,
        "exchange_calendar": 10.0,
    }

    def _get_staleness_threshold(self, sym: str, col_name: str) -> float:
        """Get the max_days_since_update threshold for a column based on its family."""
        default_max = float(os.getenv("PORTFOLIO_MAX_DAYS_SINCE_UPDATE", "10"))
        family = self._col_family.get(sym, {}).get(col_name, "")
        if not family:
            # Try to infer from column prefix
            col_lower = str(col_name).lower()
            for fam in self._FAMILY_MAX_DAYS_SINCE_UPDATE:
                if col_lower.startswith(f"{fam}_"):
                    family = fam
                    break
        if family and family in self._FAMILY_MAX_DAYS_SINCE_UPDATE:
            return self._FAMILY_MAX_DAYS_SINCE_UPDATE[family]
        # Check family metadata for cadence
        meta = self._family_meta.get(family)
        if meta is not None:
            cadence = str(getattr(meta, "update_cadence", "")).strip().lower()
            if cadence == "quarterly":
                return 130.0
            elif cadence == "monthly":
                return 45.0
            elif cadence in {"weekly", "daily"}:
                return 10.0
        return default_max

    def get_for_day(self, day: pd.Timestamp) -> RoleAwareDayContext:
        """Return per-symbol overlays for this session day.

        NOTE: day should match the session label used in the *_index.parquet (YYYYMMDD).
        """

        d_int = int(pd.Timestamp(day).strftime("%Y%m%d"))
        n = len(self.symbols)

        hygiene_ok = np.ones(n, dtype=bool)
        risk_scale = np.ones(n, dtype=float)
        regime_multiplier = np.ones(n, dtype=float)
        quantile_z = np.zeros(n, dtype=float)
        split_stress = np.zeros(n, dtype=float)  # Per-asset split stress for execution discipline

        # Policy state feature collectors (cross-sectional aggregation)
        cboe_panic_premium_vals: List[float] = []
        cboe_term_slope_vals: List[float] = []
        cboe_vol_risk_premium_vals: List[float] = []
        calibration_score_vals: List[float] = []
        online_trust_vals: List[float] = []

        # Tunables (conservative defaults).
        max_days_since_update = float(os.getenv("PORTFOLIO_MAX_DAYS_SINCE_UPDATE", "10"))

        hygiene_prob_threshold = float(os.getenv("PORTFOLIO_HYGIENE_PROB_THRESHOLD", "0.5"))

        # Risk aggregation.
        risk_agg_mode = str(os.getenv("PORTFOLIO_RISK_AGG", "mean")).strip().lower()  # mean|max
        regime_mode = str(os.getenv("PORTFOLIO_REGIME_MODE", "one_minus_mean")).strip().lower()  # mean|one_minus_mean

        def _robust_unit(sym: str, col: str, v: float) -> float:
            stats = self._col_norm.get(sym, {}).get(col)
            if stats is None:
                return float(np.clip(v, 0.0, 1.0))
            lo, hi = stats
            x = (float(v) - float(lo)) / (float(hi) - float(lo) + 1e-12)
            return float(np.clip(x, 0.0, 1.0))

        for i, sym in enumerate(self.symbols):
            row = self._row_by_date.get(sym, {}).get(d_int)
            if row is None:
                hygiene_ok[i] = False
                risk_scale[i] = 0.0
                regime_multiplier[i] = 0.0
                continue

            cols = self._feat_cols.get(sym, [])
            mat = self._feat_mat.get(sym)
            if mat is None or not cols:
                continue

            # Build quick col->idx map for this symbol.
            idx_map = {c: j for j, c in enumerate(cols)}

            # Hygiene: hard veto.
            h_cols = self._hygiene_cols.get(sym, [])
            ok = True
            for c in h_cols:
                j = idx_map.get(c)
                if j is None:
                    continue
                v = float(mat[row, j])
                name = str(c).lower()
                # Only treat explicit update-age columns as staleness.
                # Many families have feature names like *_days_since_add/remove which are NOT freshness.
                if "days_since_update" in name or name.endswith("__days_since_update"):
                    # Use per-family staleness threshold (quarterly families get ~130 days, daily ~10)
                    family_threshold = self._get_staleness_threshold(sym, c)
                    if v > family_threshold:
                        ok = False
                        break
                elif "has_data" in name:
                    if v < hygiene_prob_threshold:
                        ok = False
                        break
                else:
                    # Default: treat <0 as bad for gate-ish features, otherwise require non-zero.
                    if v != v:  # NaN
                        ok = False
                        break
            hygiene_ok[i] = bool(ok)

            # Risk scaling: downweight when risk is high.
            r_cols = self._risk_cols.get(sym, [])
            if r_cols:
                vals = []
                for c in r_cols:
                    j = idx_map.get(c)
                    if j is None:
                        continue
                    vals.append(_robust_unit(sym, c, float(mat[row, j])))
                if vals:
                    arr = np.asarray(vals, dtype=float)
                    risk_agg = float(np.mean(arr)) if risk_agg_mode != "max" else float(np.max(arr))
                    risk_scale[i] = float(1.0 / (1.0 + risk_agg))

            # Regime multiplier: smooth [0,1] modulation.
            g_cols = self._regime_cols.get(sym, [])
            if g_cols:
                vals = []
                for c in g_cols:
                    j = idx_map.get(c)
                    if j is None:
                        continue
                    vals.append(_robust_unit(sym, c, float(mat[row, j])))
                if vals:
                    arr = np.asarray(vals, dtype=float)
                    m = float(np.mean(arr))
                    regime_multiplier[i] = float(1.0 - m) if regime_mode == "one_minus_mean" else float(m)

            # Quantile forecast z_q (Option 1 from spec: blend z_q with z_mamba)
            mu_col = self._quantile_mu_col.get(sym)
            sigma_col = self._quantile_sigma_col.get(sym)
            if mu_col is not None and sigma_col is not None:
                mu_j = idx_map.get(mu_col)
                sigma_j = idx_map.get(sigma_col)
                if mu_j is not None and sigma_j is not None:
                    mu_q = float(mat[row, mu_j])
                    sigma_q = float(mat[row, sigma_j])
                    if np.isfinite(mu_q) and np.isfinite(sigma_q) and sigma_q > 1e-9:
                        quantile_z[i] = float(mu_q / sigma_q)

            # Policy state: CBOE panic premium
            for c in self._cboe_cols.get(sym, []):
                c_lower = str(c).lower()
                j = idx_map.get(c)
                if j is None:
                    continue
                v = float(mat[row, j])
                if "panic_premium" in c_lower and np.isfinite(v):
                    cboe_panic_premium_vals.append(v)
                if "normalized_term_slope" in c_lower and np.isfinite(v):
                    cboe_term_slope_vals.append(v)
                if "vol_risk_premium" in c_lower and np.isfinite(v):
                    cboe_vol_risk_premium_vals.append(v)

            # Policy state: calibration overall_score
            for c in self._calibration_cols.get(sym, []):
                c_lower = str(c).lower()
                j = idx_map.get(c)
                if j is None:
                    continue
                v = float(mat[row, j])
                if "overall_score" in c_lower and np.isfinite(v):
                    calibration_score_vals.append(v)

            # Policy state: online learning trust_score
            for c in self._online_cols.get(sym, []):
                c_lower = str(c).lower()
                j = idx_map.get(c)
                if j is None:
                    continue
                v = float(mat[row, j])
                if "trust_score" in c_lower and np.isfinite(v):
                    online_trust_vals.append(v)

            # Split stress: max(flag, post_5d, post_20d) * abs(log_ratio)
            # This provides a single scalar for explicit execution discipline:
            #   - inflate sigma: sigma_exec *= (1 + k * split_stress)
            #   - or shrink z: z *= (1 - k * split_stress)
            split_flag = 0.0
            split_post_5d = 0.0
            split_post_20d = 0.0
            split_log_ratio = 0.0
            split_recency = 0.0
            for c in self._splits_cols.get(sym, []):
                c_lower = str(c).lower()
                j = idx_map.get(c)
                if j is None:
                    continue
                v = float(mat[row, j])
                if not np.isfinite(v):
                    continue
                if c_lower.endswith("_flag"):
                    split_flag = v
                elif c_lower.endswith("_post_5d"):
                    split_post_5d = v
                elif c_lower.endswith("_post_20d"):
                    split_post_20d = v
                elif c_lower.endswith("_log_ratio"):
                    split_log_ratio = v
                elif c_lower.endswith("_recency"):
                    split_recency = v
            # Compute split_stress = max(flag, post_5d, post_20d, recency) * abs(log_ratio)
            split_event_strength = max(split_flag, split_post_5d, split_post_20d, split_recency)
            split_stress[i] = float(split_event_strength * abs(split_log_ratio))

        # Enforce hygiene last: if vetoed, kill exposure.
        risk_scale = np.where(hygiene_ok, risk_scale, 0.0)
        regime_multiplier = np.where(hygiene_ok, regime_multiplier, 0.0)
        quantile_z = np.where(hygiene_ok, quantile_z, 0.0)
        split_stress = np.where(hygiene_ok, split_stress, 0.0)

        # Aggregate policy state features (cross-sectional means)
        cboe_panic = float(np.mean(cboe_panic_premium_vals)) if cboe_panic_premium_vals else 0.0
        cboe_slope = float(np.mean(cboe_term_slope_vals)) if cboe_term_slope_vals else 0.0
        cboe_vrp = float(np.mean(cboe_vol_risk_premium_vals)) if cboe_vol_risk_premium_vals else 0.0
        calib_score = float(np.mean(calibration_score_vals)) if calibration_score_vals else 1.0
        online_trust = float(np.mean(online_trust_vals)) if online_trust_vals else 1.0

        return RoleAwareDayContext(
            symbols=list(self.symbols),
            hygiene_ok=hygiene_ok,
            risk_scale=risk_scale,
            regime_multiplier=regime_multiplier,
            quantile_z=quantile_z,
            split_stress=split_stress,
            cboe_panic_premium=cboe_panic,
            cboe_term_slope=cboe_slope,
            cboe_vol_risk_premium_z=cboe_vrp,
            calibration_overall_score=calib_score,
            online_trust_score=online_trust,
        )
