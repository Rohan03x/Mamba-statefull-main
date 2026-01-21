"""Cross-sectional peer context features (sector/industry ranks).

Family: peer_screener_context

Goal
- Provide cross-sectional ranks/percentiles for a symbol vs peers (same sector/industry)
  using already-cached per-symbol fundamental/valuation time series.

Data sources
- Cached per-symbol family panels (currently: fin_g6) from local_cache.
- Sector/industry metadata via EODHD fundamentals (General->Sector/Industry).

Leakage policy
- All peer metrics are shifted by +1 NYSE session before ranking to reduce same-day
  lookahead from filings/snapshots.

Notes
- This family is operationally heavier than per-symbol families. It uses a shared
  cross-symbol cache under the local cache root to avoid recomputing ranks for each symbol.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import erf, sqrt
import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from src.dcf_lab.utils.timealign import nyse_sessions_in_range


# Bump this to invalidate old shared caches when the rank/percentile logic changes.
PEER_SCREENER_CONTEXT_SHARED_CACHE_VERSION = 2


@dataclass(frozen=True)
class _Profile:
    sector: str
    industry: str


def _norm_symbol(sym: str) -> str:
    return (sym or "").strip().upper()


def _parse_universe_env() -> Optional[List[str]]:
    """Comma-separated peer universe override.

    If set, PEER_SCREENER_CONTEXT_UNIVERSE takes precedence over cache-root
    discovery and DEFAULT_CANDIDATE_UNIVERSE.
    """

    raw = os.getenv("PEER_SCREENER_CONTEXT_UNIVERSE", "")
    if not str(raw).strip():
        return None
    parts = [p.strip() for p in str(raw).split(",") if p.strip()]
    out: List[str] = []
    seen: set[str] = set()
    for p in parts:
        s = _norm_symbol(p)
        if not s or s in seen:
            continue
        seen.add(s)
        out.append(s)
    return out or None


def _discover_universe_from_cache(cache_root: Path, *, horizon: int) -> List[str]:
    """Infer a candidate peer universe from per-symbol cache directories."""

    cache_root = Path(cache_root)
    if not cache_root.exists():
        return []

    h = int(horizon)
    suffix = f"_h{h}".lower()
    out: List[str] = []
    seen: set[str] = set()

    try:
        for p in sorted(cache_root.glob(f"*{suffix}")):
            if not p.is_dir():
                continue
            name = str(p.name)
            if not name.lower().endswith(suffix):
                continue
            sym = _norm_symbol(name[: -len(suffix)])
            if not sym or sym in seen:
                continue
            seen.add(sym)
            out.append(sym)
    except Exception:
        return []

    return out


def _start_end_key(start: str, end: str) -> Tuple[str, str]:
    s = pd.to_datetime(start).strftime("%Y%m%d")
    e = pd.to_datetime(end).strftime("%Y%m%d")
    return s, e


def _shared_cache_path(cache_root: Path, horizon: int, start: str, end: str) -> Path:
    s, e = _start_end_key(start, end)
    return (
        cache_root
        / "_shared_cross_sectional"
        / f"h{int(horizon)}"
        / "peer_screener_context"
        / f"v{int(PEER_SCREENER_CONTEXT_SHARED_CACHE_VERSION)}"
        / f"{s}_{e}.parquet"
    )


def _shared_universe_stats_path(cache_root: Path, horizon: int, start: str, end: str) -> Path:
    s, e = _start_end_key(start, end)
    return (
        cache_root
        / "_shared_cross_sectional"
        / f"h{int(horizon)}"
        / "peer_screener_context"
        / f"v{int(PEER_SCREENER_CONTEXT_SHARED_CACHE_VERSION)}"
        / f"{s}_{e}_universe_stats.parquet"
    )


def _normalize_index(idx: pd.Index) -> pd.DatetimeIndex:
    out = pd.to_datetime(idx, errors="coerce")
    out = pd.DatetimeIndex(out)
    out = out.tz_localize(None)
    out = pd.DatetimeIndex([pd.Timestamp(x).normalize() for x in out if pd.notna(x)])
    return out


def _coerce_numeric(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    s = s.replace([np.inf, -np.inf], np.nan)
    return s


def _align_series_to_sessions(series: pd.Series, sessions: pd.DatetimeIndex, *, method: str = "ffill") -> pd.Series:
    if series is None or len(series) == 0:
        return pd.Series(np.nan, index=sessions)
    s = series.copy()
    s.index = _normalize_index(s.index)
    s = s[~s.index.duplicated(keep="last")].sort_index()
    s = _coerce_numeric(s)
    # Reindex onto sessions with forward-fill (fundamentals are slow moving).
    if str(method).lower() in {"ffill", "pad"}:
        s = s.reindex(sessions, method="ffill")
    else:
        s = s.reindex(sessions)
    return s


def _normal_cdf(z: pd.Series) -> pd.Series:
    """Approximate standard normal CDF without scipy."""
    # Phi(z) = 0.5 * (1 + erf(z / sqrt(2)))
    return 0.5 * (1.0 + z.apply(lambda x: erf(float(x) / sqrt(2.0)) if pd.notna(x) else 0.0))


def _safe_zscore(x: pd.Series, mean: pd.Series, std: pd.Series, *, eps: float = 1e-6) -> pd.Series:
    m = pd.to_numeric(mean, errors="coerce").fillna(0.0)
    s = pd.to_numeric(std, errors="coerce").fillna(0.0).abs() + float(eps)
    out = (pd.to_numeric(x, errors="coerce") - m) / s
    out = out.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    # Mild clip to prevent extreme outliers from dominating.
    return out.clip(-8.0, 8.0)


def _effective_min_peers(peer_count: pd.Series, *, min_peers_configured: int) -> int:
    """Choose a practical peer-count threshold for gating.

    The intent of min_peers is to prevent unstable cross-sectional stats when
    peer coverage is thin. In practice, peer_screener_context is often computed
    in partial-universe runs (e.g., only a few symbols were materialized into
    the local_cache). In that case, a fixed min_peers (default 20) would gate
    universe z/pct to all-zeros, even though the stats are still useful.

    Policy:
    - Keep the configured threshold as an upper bound.
    - Never gate more strictly than the maximum observed peer_count for this
      metric (so outputs can become non-zero once there is any meaningful peer
      set).
    - Enforce a floor of 3 peers for minimal stability.
    """

    try:
        mx = float(pd.to_numeric(peer_count, errors="coerce").fillna(0.0).max())
    except Exception:
        mx = 0.0

    floor = 3
    cap = int(max(floor, int(min_peers_configured)))
    eff = int(min(cap, max(floor, int(round(mx)))))
    return int(max(floor, eff))


def _load_series_from_cache(
    cache_root: Path,
    symbol: str,
    *,
    horizon: int,
    family: str,
    column: str,
) -> Optional[pd.Series]:
    frame = _load_family_frame(cache_root, symbol, horizon=horizon, family=family)
    if frame is None or frame.empty:
        return None
    if column not in frame.columns:
        return None
    return frame[column]


def _load_series_live_or_cache(
    cache_root: Path,
    symbol: str,
    *,
    horizon: int,
    family: str,
    column: str,
    start: str,
    end: str,
    _memo: Optional[Dict[Tuple[str, str], pd.DataFrame]] = None,
) -> Optional[pd.Series]:
    """Load a per-symbol series from local_cache, with a live fallback.

    peer_screener_context is cross-symbol: it often needs peer-series for symbols
    that were never explicitly materialized into data/local_cache (e.g. when
    prepping only AAPL/MSFT/NVDA). In that case we fall back to generating the
    needed family panel on-demand via build_panel.

    IMPORTANT: build_panel fills NaNs to 0 for most columns. For families that
    include a *_has_data flag, we re-mask the requested column to NaN when
    has_data==0 so those rows do not count as valid peers.
    """

    sym = _norm_symbol(symbol)
    fam = str(family).strip().lower()

    cached = _load_series_from_cache(Path(cache_root), sym, horizon=horizon, family=fam, column=column)
    if cached is not None:
        return cached

    # Guardrail: avoid surprising huge fetches.
    try:
        enabled = str(os.getenv("PEER_SCREENER_CONTEXT_LIVE_FETCH", "1")).strip() == "1"
    except Exception:
        enabled = True
    if not enabled:
        return None

    # Memoize per-(symbol,family) build_panel calls within a single stats build.
    if _memo is None:
        _memo = {}
    memo_key = (sym, fam)

    panel = _memo.get(memo_key)
    if panel is None:
        try:
            from src.features.aggregator_panel import build_panel  # type: ignore

            panel = build_panel(
                sym,
                str(start),
                str(end),
                families=[fam],
                cache_dir=None,
                horizon=int(horizon),
                view="raw",
            )
        except Exception:
            panel = None

        if not isinstance(panel, pd.DataFrame) or panel.empty:
            _memo[memo_key] = pd.DataFrame()
            return None

        _memo[memo_key] = panel

    if column not in panel.columns:
        return None

    series = panel[column].copy()
    try:
        series.index = _normalize_index(series.index)
    except Exception:
        pass

    # Re-mask provider-unavailable spans to NaN when a family reports *_has_data.
    has_data_col = f"{fam}_has_data"
    if has_data_col in panel.columns and column != has_data_col:
        try:
            has_data = pd.to_numeric(panel[has_data_col], errors="coerce").fillna(0.0)
            has_data.index = series.index
            series = series.where(has_data > 0.0)
        except Exception:
            pass

    return series


def _build_or_load_universe_stats(
    *,
    cache_root: Path,
    horizon: int,
    start: str,
    end: str,
    universe: Sequence[str],
    leakage_safe_shift_sessions: int,
    min_peers: int,
) -> pd.DataFrame:
    """Build small per-date universe mean/std/count for a few metrics.

    This avoids writing/reading enormous wide symbol|feature matrices.
    """

    sessions = nyse_sessions_in_range(start, end, tz_aware_utc=False)
    out = pd.DataFrame(index=sessions)

    stats_path = _shared_universe_stats_path(Path(cache_root), horizon, start, end)
    if stats_path.exists():
        try:
            cached = pd.read_parquet(stats_path)
            if cached is not None and not cached.empty:
                cached.index = pd.to_datetime(cached.index).tz_localize(None)
                cached = cached.reindex(sessions)
                return cached
        except Exception:
            pass

    # Metric specs: (family, column, transform)
    # Note: We transform valuation into a cheapness proxy (higher=cheaper).
    metric_specs: Dict[str, Tuple[str, str, str]] = {
        "momentum": ("dcf", "dcf_mom_3m", "log"),
        "valuation": ("fin_g6", "fin_g6_pe_ratio", "neg_log"),
        "options_iv": ("options", "options_atm_iv", "identity"),
        "options_skew": ("options", "options_iv_spread", "identity"),
        "short_interest": ("short_interest", "short_interest_float_short_pct", "identity"),
    }

    # Streaming aggregates per metric.
    sums: Dict[str, pd.Series] = {k: pd.Series(0.0, index=sessions) for k in metric_specs}
    sums_sq: Dict[str, pd.Series] = {k: pd.Series(0.0, index=sessions) for k in metric_specs}
    counts: Dict[str, pd.Series] = {k: pd.Series(0.0, index=sessions) for k in metric_specs}

    universe_syms = [_norm_symbol(s) for s in universe]
    universe_syms = [s for s in universe_syms if s]
    universe_syms = list(dict.fromkeys(universe_syms))

    live_memo: Dict[Tuple[str, str], pd.DataFrame] = {}

    for sym in universe_syms:
        for metric, (fam, col, transform) in metric_specs.items():
            series = _load_series_live_or_cache(
                Path(cache_root),
                sym,
                horizon=horizon,
                family=fam,
                column=col,
                start=start,
                end=end,
                _memo=live_memo,
            )
            if series is None:
                continue
            aligned = _align_series_to_sessions(series, sessions, method="ffill")
            if leakage_safe_shift_sessions:
                aligned = aligned.shift(int(leakage_safe_shift_sessions))

            x = pd.to_numeric(aligned, errors="coerce")
            if transform == "log":
                x = np.log(x.replace([np.inf, -np.inf], np.nan).clip(lower=1e-9))
            elif transform == "neg_log":
                # cheapness proxy (lower PE => higher value)
                pe = x.replace([np.inf, -np.inf], np.nan)
                pe = pe.where(pe > 0.0)
                x = -np.log(pe.clip(lower=1e-9))
            else:
                x = x.replace([np.inf, -np.inf], np.nan)

            valid = x.notna()
            if not bool(valid.any()):
                continue

            sums[metric] = sums[metric].where(~valid, sums[metric] + x.fillna(0.0))
            sums_sq[metric] = sums_sq[metric].where(~valid, sums_sq[metric] + (x.fillna(0.0) ** 2))
            counts[metric] = counts[metric].where(~valid, counts[metric] + 1.0)

    for metric in metric_specs:
        c = counts[metric]
        # Mean/std only meaningful when we have sufficient peers.
        mean = (sums[metric] / c.replace(0.0, np.nan)).fillna(0.0)
        var = (sums_sq[metric] / c.replace(0.0, np.nan)) - (mean ** 2)
        var = var.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        std = np.sqrt(var.clip(lower=0.0))

        out[f"peer_screener_context_universe_{metric}_peer_count"] = c.astype(float)
        out[f"peer_screener_context_universe_{metric}_mean"] = mean.astype(float)
        out[f"peer_screener_context_universe_{metric}_std"] = std.astype(float)

    # Persist small stats cache for reuse.
    try:
        stats_path.parent.mkdir(parents=True, exist_ok=True)
        out.to_parquet(stats_path)
    except Exception:
        pass

    return out


def _load_family_frame(
    cache_root: Path,
    symbol: str,
    *,
    horizon: int,
    family: str,
) -> Optional[pd.DataFrame]:
    """Best-effort loader for per-symbol cached family frames.

    Mirrors the on-disk naming convention used by prep_families.
    """

    sym = _norm_symbol(symbol)
    if not sym:
        return None

    cache_dir = Path(cache_root) / f"{sym.lower()}_h{int(horizon)}"
    if not cache_dir.exists():
        return None

    def _read_candidate(path: Path) -> Optional[pd.DataFrame]:
        if not path.exists():
            return None
        try:
            df = pd.read_parquet(path)
        except Exception:
            return None
        if df is None or df.empty:
            return None
        if "date" in df.columns:
            df = df.copy()
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            df = df.set_index("date")
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index, errors="coerce")
        df.index = df.index.tz_localize(None)
        df = df.sort_index()
        df = df[~df.index.duplicated(keep="last")]
        df = df.replace([np.inf, -np.inf], np.nan)
        return df

    frames: List[pd.DataFrame] = []
    for split in ("train", "valid"):
        # Prefer non-lagged feature caches.
        feature_path = cache_dir / f"{sym.lower()}_h{int(horizon)}_{family}_{split}_features.parquet"
        raw = _read_candidate(feature_path)
        if raw is not None:
            frames.append(raw)

    if not frames:
        return None

    merged = frames[0]
    for block in frames[1:]:
        overlap = merged.columns.intersection(block.columns)
        if len(overlap) > 0:
            block = block.drop(columns=overlap, errors="ignore")
        if not block.empty:
            merged = merged.join(block, how="outer")

    merged = merged.sort_index()
    return merged


def _get_profiles(symbols: Sequence[str]) -> Dict[str, _Profile]:
    try:
        from src.data_sources.eodhd_provider import get_eodhd_provider  # type: ignore
    except Exception:
        return {}

    provider = get_eodhd_provider()
    profiles: Dict[str, _Profile] = {}

    for sym in symbols:
        s = _norm_symbol(sym)
        if not s:
            continue
        try:
            prof = provider.get_profile(s)
        except Exception:
            prof = {}
        sector = str((prof or {}).get("sector") or "").strip()
        industry = str((prof or {}).get("industry") or "").strip()
        profiles[s] = _Profile(sector=sector or "Unknown", industry=industry or "Unknown")

    return profiles


def _percentile_rank(df: pd.DataFrame, *, higher_is_better: bool) -> pd.DataFrame:
    """Row-wise percentile ranks across columns."""

    if df is None or df.empty:
        return df

    ranked = df.rank(axis=1, pct=True, method="average", na_option="keep")
    if higher_is_better:
        return ranked
    return (1.0 - ranked)


def _stub(sessions: pd.DatetimeIndex) -> pd.DataFrame:
    out = pd.DataFrame(index=sessions)
    out["peer_screener_context_has_data"] = 0.0
    out["peer_screener_context_sector_peer_count"] = 0.0
    out["peer_screener_context_industry_peer_count"] = 0.0

    # Stable schema for the default metrics.
    for metric in ("pe_ratio", "pb_ratio", "ev_ebitda"):
        # Neutral default (0.5) avoids encoding missing/degenerate groups as extremes.
        out[f"peer_screener_context_sector_{metric}_pct"] = 0.5
        out[f"peer_screener_context_sector_{metric}_cheap_pct"] = 0.5
        out[f"peer_screener_context_industry_{metric}_pct"] = 0.5
        out[f"peer_screener_context_industry_{metric}_cheap_pct"] = 0.5

    # Candidate-universe context scalars (z-score + percentile).
    for metric in ("momentum", "valuation", "options_iv", "options_skew", "short_interest"):
        out[f"peer_screener_context_universe_{metric}_z"] = 0.0
        out[f"peer_screener_context_universe_{metric}_pct"] = 0.0
        out[f"peer_screener_context_universe_{metric}_peer_count"] = 0.0

    out.attrs["telemetry"] = {"status": "ok", "source": "stub"}
    out.attrs["provenance"] = {
        "source": "peer_screener_context",
        "note": "stub (missing cache_root or peer inputs)",
    }
    out.attrs["feature_counts"] = {"generated": int(len(out.columns))}
    return out


def fetch_peer_screener_context(
    symbol: str,
    start: str,
    end: str,
    *,
    cache_root: Optional[Path] = None,
    horizon: int = 63,
    universe: Optional[Sequence[str]] = None,
    base_family: str = "fin_g6",
    leakage_safe_shift_sessions: int = 1,
) -> Optional[pd.DataFrame]:
    """Compute cross-sectional peer percentiles for the requested symbol.

    Returns a NYSE-session-aligned frame indexed by session dates.

    cache_root must point at the parent folder that contains per-symbol cache dirs
    like <symbol>_h<horizon> (e.g. data/local_cache).
    """

    target = _norm_symbol(symbol)
    if not target:
        return None

    sessions = nyse_sessions_in_range(start, end, tz_aware_utc=False)
    if len(sessions) == 0:
        return None

    if cache_root is None:
        return _stub(sessions)

    # Legacy shared wide cache is optional; avoid depending on it.
    shared_path = _shared_cache_path(Path(cache_root), horizon, start, end)

    universe_source = "explicit"
    if universe is None:
        env_universe = _parse_universe_env()
        if env_universe is not None:
            universe = env_universe
            universe_source = "env:PEER_SCREENER_CONTEXT_UNIVERSE"
        else:
            # Phase2 default: use the fixed core+satellite candidate universe.
            # Do NOT infer from cache contents; the cache may be incomplete and
            # would make the universe depend on which symbols were prepared.
            try:
                from src.stage_b.universe_selector import DEFAULT_CANDIDATE_UNIVERSE  # type: ignore

                universe = list(DEFAULT_CANDIDATE_UNIVERSE)
                universe_source = "DEFAULT_CANDIDATE_UNIVERSE"
            except Exception:
                universe = [target]
                universe_source = "target_only"

    universe_syms = [_norm_symbol(s) for s in universe]
    universe_syms = [s for s in universe_syms if s]
    if target not in universe_syms:
        universe_syms = [target] + universe_syms

    min_peers = int(os.getenv("PEER_SCREENER_CONTEXT_MIN_PEERS", "20"))
    min_peers = int(max(3, min_peers))

    # Fetch only the target profile up-front. Fetching profiles for a large
    # discovered universe can be expensive even if the provider caches.
    target_profile = _get_profiles([target]).get(target)
    if target_profile is None:
        out = _stub(sessions)
        out.attrs.setdefault("provenance", {})
        try:
            out.attrs["provenance"].update({"universe_source": universe_source, "universe_size": int(len(universe_syms))})
        except Exception:
            pass
        return out

    # Fast-path: if we already built the shared cross-symbol cache for this
    # (start,end,horizon), reuse it to avoid recomputation per requested symbol.
    shared_cache_used = False
    shared_df: Optional[pd.DataFrame] = None
    force_recompute = str(os.getenv("PEER_SCREENER_CONTEXT_FORCE_RECOMPUTE", "0") or "0").strip().lower() in {"1", "true", "yes"}
    if (not force_recompute) and shared_path.exists():
        try:
            shared_df = pd.read_parquet(shared_path)
            if isinstance(shared_df, pd.DataFrame) and not shared_df.empty:
                shared_cache_used = True
        except Exception:
            shared_df = None
            shared_cache_used = False

    if shared_cache_used and shared_df is not None:
        try:
            # Align shared cache onto current session calendar.
            if not isinstance(shared_df.index, pd.DatetimeIndex):
                shared_df.index = pd.to_datetime(shared_df.index, errors="coerce")
            shared_df.index = pd.DatetimeIndex(shared_df.index).tz_localize(None)
            shared_df = shared_df.sort_index()
            shared_df = shared_df[~shared_df.index.duplicated(keep="last")]
            shared_df = shared_df.reindex(sessions)

            target_prefix = f"{target}|"
            out_cols = [c for c in shared_df.columns if str(c).startswith(target_prefix)]
            if out_cols:
                out = pd.DataFrame(
                    {str(c).split("|", 1)[1]: pd.to_numeric(shared_df[c], errors="coerce") for c in out_cols},
                    index=sessions,
                )

                pct_cols = [c for c in out.columns if str(c).endswith("_pct") or str(c).endswith("_cheap_pct")]
                if pct_cols:
                    out[pct_cols] = out[pct_cols].fillna(0.5).clip(0.0, 1.0)
                other_cols = [c for c in out.columns if c not in pct_cols]
                if other_cols:
                    out[other_cols] = out[other_cols].fillna(0.0)
            else:
                out = _stub(sessions)

            # Reconstruct available_syms count only for telemetry.
            available_syms = []
            for c in shared_df.columns:
                cs = str(c)
                if "|" not in cs:
                    continue
                sym = cs.split("|", 1)[0]
                if sym and sym not in available_syms:
                    available_syms.append(sym)
        except Exception:
            out = _stub(sessions)
            available_syms = []

        # Candidate-universe context remains symbol-specific and uses the stats cache.
        # Ensure min_peers_effective is always defined for telemetry.
        min_peers_effective = int(min_peers)
        max_peer_count_seen = 0.0

        try:
            stats = _build_or_load_universe_stats(
                cache_root=Path(cache_root),
                horizon=horizon,
                start=start,
                end=end,
                universe=universe_syms,
                leakage_safe_shift_sessions=int(leakage_safe_shift_sessions),
                min_peers=min_peers,
            )

            metric_defs = {
                "momentum": ("dcf", "dcf_mom_3m", "log"),
                "valuation": ("fin_g6", "fin_g6_pe_ratio", "neg_log"),
                "options_iv": ("options", "options_atm_iv", "identity"),
                "options_skew": ("options", "options_iv_spread", "identity"),
                "short_interest": ("short_interest", "short_interest_float_short_pct", "identity"),
            }

            live_memo: Dict[Tuple[str, str], pd.DataFrame] = {}
            for metric, (fam, col, transform) in metric_defs.items():
                series = _load_series_live_or_cache(
                    Path(cache_root),
                    target,
                    horizon=horizon,
                    family=fam,
                    column=col,
                    start=start,
                    end=end,
                    _memo=live_memo,
                )
                if series is None:
                    out[f"peer_screener_context_universe_{metric}_z"] = 0.0
                    out[f"peer_screener_context_universe_{metric}_pct"] = 0.0
                    out[f"peer_screener_context_universe_{metric}_peer_count"] = 0.0
                    continue

                aligned = _align_series_to_sessions(series, sessions, method="ffill")
                if leakage_safe_shift_sessions:
                    aligned = aligned.shift(int(leakage_safe_shift_sessions))

                x = pd.to_numeric(aligned, errors="coerce")
                if transform == "log":
                    x = np.log(x.replace([np.inf, -np.inf], np.nan).clip(lower=1e-9))
                elif transform == "neg_log":
                    pe = x.replace([np.inf, -np.inf], np.nan)
                    pe = pe.where(pe > 0.0)
                    x = -np.log(pe.clip(lower=1e-9))
                else:
                    x = x.replace([np.inf, -np.inf], np.nan)

                c = pd.to_numeric(stats.get(f"peer_screener_context_universe_{metric}_peer_count"), errors="coerce").fillna(0.0)
                try:
                    max_peer_count_seen = float(max(max_peer_count_seen, float(c.max())))
                except Exception:
                    pass
                mean = stats.get(f"peer_screener_context_universe_{metric}_mean")
                std = stats.get(f"peer_screener_context_universe_{metric}_std")

                z = _safe_zscore(x, mean, std)
                pct = _normal_cdf(z)
                pct = pct.replace([np.inf, -np.inf], np.nan).fillna(0.5).clip(0.0, 1.0)

                min_peers_eff = _effective_min_peers(c, min_peers_configured=int(min_peers))
                gate = (c >= float(min_peers_eff)).astype(float)
                out[f"peer_screener_context_universe_{metric}_peer_count"] = c.astype(float)
                out[f"peer_screener_context_universe_{metric}_z"] = (z * gate).astype(float)
                out[f"peer_screener_context_universe_{metric}_pct"] = (pct * gate).astype(float)
        except Exception:
            pass

        # Surface the effective gating threshold for debugging/audits.
        try:
            min_peers_effective = int(min(int(min_peers), max(3, int(round(float(max_peer_count_seen or 0.0))))))
        except Exception:
            min_peers_effective = int(min_peers)

        out.attrs["telemetry"] = {
            "status": "ok",
            "source": "shared_cache",
            "base_family": base_family,
            "universe_size": int(len(universe_syms)),
            "peers_loaded": int(len(available_syms)),
            "universe_source": universe_source,
            "shared_cache_used": True,
            "shared_cache_path": str(shared_path),
            "min_peers_configured": int(min_peers),
            "min_peers_effective": int(min_peers_effective),
            "sector": target_profile.sector,
            "industry": target_profile.industry,
        }
        out.attrs["provenance"] = {
            "source": "peer_screener_context",
            "base_family": base_family,
            "grouping": "sector+industry (EODHD fundamentals General->Sector/Industry)",
            "leakage": f"peer metrics shifted +{int(leakage_safe_shift_sessions)} NYSE session(s)",
            "notes": "reused shared cross-symbol cache for ranks; universe context computed per-symbol",
            "universe_source": universe_source,
            "universe_size": int(len(universe_syms)),
            "peers_loaded": int(len(available_syms)),
            "shared_cache_used": True,
            "shared_cache_path": str(shared_path),
            "min_peers_configured": int(min_peers),
            "min_peers_effective": int(min_peers_effective),
        }
        out.attrs["feature_counts"] = {"generated": int(len(out.columns))}
        return out

    # Load base metrics for all peers.
    metrics = {
        "pe_ratio": f"{base_family}_pe_ratio",
        "pb_ratio": f"{base_family}_pb_ratio",
        "ev_ebitda": f"{base_family}_ev_ebitda",
    }

    matrices: Dict[str, pd.DataFrame] = {}
    available_syms: List[str] = []

    rank_memo: Dict[Tuple[str, str], pd.DataFrame] = {}
    for sym in universe_syms:
        any_loaded = False
        for short_name, col in metrics.items():
            series = _load_series_live_or_cache(
                Path(cache_root),
                sym,
                horizon=horizon,
                family=base_family,
                column=col,
                start=start,
                end=end,
                _memo=rank_memo,
            )
            if series is None:
                continue
            aligned = _align_series_to_sessions(series, sessions)
            if leakage_safe_shift_sessions:
                aligned = aligned.shift(int(leakage_safe_shift_sessions))
            matrices.setdefault(short_name, pd.DataFrame(index=sessions))
            matrices[short_name][sym] = aligned
            any_loaded = True
        if any_loaded:
            available_syms.append(sym)

    if target not in available_syms:
        out = _stub(sessions)
        out.attrs.setdefault("provenance", {})
        try:
            out.attrs["provenance"].update({"universe_source": universe_source, "universe_size": int(len(universe_syms))})
        except Exception:
            pass
        return out

    # Now fetch profiles only for symbols that actually have base-family data.
    profiles = _get_profiles(available_syms)

    # Build grouping maps for all available symbols.
    sector_groups: Dict[str, List[str]] = {}
    industry_groups: Dict[str, List[str]] = {}
    for sym in available_syms:
        prof = profiles.get(sym)
        if prof is None:
            continue
        sector_groups.setdefault(prof.sector or "Unknown", []).append(sym)
        industry_groups.setdefault(prof.industry or "Unknown", []).append(sym)

    # Prepare shared outputs for ALL symbols in one pass.
    shared_cols: Dict[str, pd.Series] = {}
    pe_matrix = matrices.get("pe_ratio")

    def _emit_for_group(
        *,
        group_type: str,
        group_members: Dict[str, List[str]],
        mat: pd.DataFrame,
        metric_name: str,
    ) -> None:
        min_group_peers = int(os.getenv("PEER_SCREENER_CONTEXT_MIN_GROUP_PEERS", "3"))
        min_group_peers = int(max(2, min_group_peers))

        for _, members in group_members.items():
            if not members:
                continue
            cols = [m for m in members if m in mat.columns]
            if not cols:
                continue
            sub = mat[cols]

            # Require a minimum number of valid peers per date for the group percentile
            # to be meaningful. If the industry group is too small, fall back to the
            # sector distribution for that metric (same symbol, same date).
            counts = sub.notna().sum(axis=1).astype(float)
            gate = counts >= float(min_group_peers)

            pct_high = _percentile_rank(sub, higher_is_better=True)
            pct_low = _percentile_rank(sub, higher_is_better=False)
            for sym in cols:
                if sym not in pct_high.columns:
                    continue

                pct_s = _coerce_numeric(pct_high[sym]).where(gate, np.nan)
                cheap_s = _coerce_numeric(pct_low[sym]).where(gate, np.nan)

                if group_type == "industry":
                    sector_pct_key = f"{sym}|peer_screener_context_sector_{metric_name}_pct"
                    sector_cheap_key = f"{sym}|peer_screener_context_sector_{metric_name}_cheap_pct"
                    sector_pct = shared_cols.get(sector_pct_key)
                    sector_cheap = shared_cols.get(sector_cheap_key)
                    if isinstance(sector_pct, pd.Series):
                        pct_s = pct_s.fillna(sector_pct)
                    if isinstance(sector_cheap, pd.Series):
                        cheap_s = cheap_s.fillna(sector_cheap)

                # Neutral default for any remaining missing/degenerate rows.
                pct_s = pct_s.fillna(0.5).clip(0.0, 1.0)
                cheap_s = cheap_s.fillna(0.5).clip(0.0, 1.0)

                shared_cols[f"{sym}|peer_screener_context_{group_type}_{metric_name}_pct"] = pct_s
                shared_cols[f"{sym}|peer_screener_context_{group_type}_{metric_name}_cheap_pct"] = cheap_s

    # Counts and has_data gates (per-date, based on pe_ratio validity).
    if pe_matrix is not None and not pe_matrix.empty:
        for sector, members in sector_groups.items():
            if not members:
                continue
            cols = [m for m in members if m in pe_matrix.columns]
            if not cols:
                continue
            counts = pe_matrix[cols].notna().sum(axis=1).astype(float)
            has_data = (counts >= 3.0).astype(float)
            for sym in cols:
                shared_cols[f"{sym}|peer_screener_context_sector_peer_count"] = counts
                shared_cols[f"{sym}|peer_screener_context_has_data"] = has_data

        for industry, members in industry_groups.items():
            if not members:
                continue
            cols = [m for m in members if m in pe_matrix.columns]
            if not cols:
                continue
            counts = pe_matrix[cols].notna().sum(axis=1).astype(float)
            for sym in cols:
                shared_cols[f"{sym}|peer_screener_context_industry_peer_count"] = counts

    # Percentiles for each metric across sector and industry groups.
    for metric_name, mat in matrices.items():
        if mat is None or mat.empty:
            continue
        _emit_for_group(group_type="sector", group_members=sector_groups, mat=mat, metric_name=metric_name)
        _emit_for_group(group_type="industry", group_members=industry_groups, mat=mat, metric_name=metric_name)

    # Ensure stable schema columns exist for every available symbol (fill zeros).
    for sym in available_syms:
        shared_cols.setdefault(f"{sym}|peer_screener_context_has_data", pd.Series(0.0, index=sessions))
        shared_cols.setdefault(f"{sym}|peer_screener_context_sector_peer_count", pd.Series(0.0, index=sessions))
        shared_cols.setdefault(f"{sym}|peer_screener_context_industry_peer_count", pd.Series(0.0, index=sessions))
        for metric in ("pe_ratio", "pb_ratio", "ev_ebitda"):
            for scope in ("sector", "industry"):
                shared_cols.setdefault(
                    f"{sym}|peer_screener_context_{scope}_{metric}_pct",
                    pd.Series(0.5, index=sessions),
                )
                shared_cols.setdefault(
                    f"{sym}|peer_screener_context_{scope}_{metric}_cheap_pct",
                    pd.Series(0.5, index=sessions),
                )

    # Build target output view.
    out_cols = [k for k in shared_cols.keys() if k.startswith(f"{target}|")]
    out = pd.DataFrame({k.split("|", 1)[1]: shared_cols[k] for k in out_cols}, index=sessions)

    # ------------------------------------------------------------------
    # Candidate-universe context: z-scores/percentiles for a small set.
    # ------------------------------------------------------------------
    # These are computed vs the full candidate universe (not sector-only), using
    # cached per-symbol families and a small per-date stats cache.
    try:
        stats = _build_or_load_universe_stats(
            cache_root=Path(cache_root),
            horizon=horizon,
            start=start,
            end=end,
            universe=universe_syms,
            leakage_safe_shift_sessions=int(leakage_safe_shift_sessions),
            min_peers=min_peers,
        )

        metric_defs = {
            "momentum": ("dcf", "dcf_mom_3m", "log"),
            "valuation": ("fin_g6", "fin_g6_pe_ratio", "neg_log"),
            "options_iv": ("options", "options_atm_iv", "identity"),
            "options_skew": ("options", "options_iv_spread", "identity"),
            "short_interest": ("short_interest", "short_interest_float_short_pct", "identity"),
        }

        max_peer_count_seen = 0.0

        # For each metric, compute per-date z-score and approximate percentile.
        for metric, (fam, col, transform) in metric_defs.items():
            series = _load_series_live_or_cache(
                Path(cache_root),
                target,
                horizon=horizon,
                family=fam,
                column=col,
                start=start,
                end=end,
            )
            if series is None:
                out[f"peer_screener_context_universe_{metric}_z"] = 0.0
                out[f"peer_screener_context_universe_{metric}_pct"] = 0.0
                out[f"peer_screener_context_universe_{metric}_peer_count"] = 0.0
                continue

            aligned = _align_series_to_sessions(series, sessions, method="ffill")
            if leakage_safe_shift_sessions:
                aligned = aligned.shift(int(leakage_safe_shift_sessions))

            x = pd.to_numeric(aligned, errors="coerce")
            if transform == "log":
                x = np.log(x.replace([np.inf, -np.inf], np.nan).clip(lower=1e-9))
            elif transform == "neg_log":
                pe = x.replace([np.inf, -np.inf], np.nan)
                pe = pe.where(pe > 0.0)
                x = -np.log(pe.clip(lower=1e-9))
            else:
                x = x.replace([np.inf, -np.inf], np.nan)

            c = pd.to_numeric(stats.get(f"peer_screener_context_universe_{metric}_peer_count"), errors="coerce").fillna(0.0)
            try:
                max_peer_count_seen = float(max(max_peer_count_seen, float(c.max())))
            except Exception:
                pass
            mean = stats.get(f"peer_screener_context_universe_{metric}_mean")
            std = stats.get(f"peer_screener_context_universe_{metric}_std")

            z = _safe_zscore(x, mean, std)
            pct = _normal_cdf(z)
            pct = pct.replace([np.inf, -np.inf], np.nan).fillna(0.5).clip(0.0, 1.0)

            # Gate outputs when peer coverage is insufficient.
            min_peers_eff = _effective_min_peers(c, min_peers_configured=int(min_peers))
            gate = (c >= float(min_peers_eff)).astype(float)
            out[f"peer_screener_context_universe_{metric}_peer_count"] = c.astype(float)
            out[f"peer_screener_context_universe_{metric}_z"] = (z * gate).astype(float)
            out[f"peer_screener_context_universe_{metric}_pct"] = (pct * gate).astype(float)

        # Fold universe availability into the family has_data flag (keep legacy behavior too).
        try:
            min_peers_effective = int(min(int(min_peers), max(3, int(round(float(max_peer_count_seen or 0.0))))))
            universe_counts = [
                pd.to_numeric(out.get(f"peer_screener_context_universe_{m}_peer_count"), errors="coerce").fillna(0.0)
                for m in metric_defs.keys()
            ]
            if universe_counts:
                universe_peer_count = pd.concat(universe_counts, axis=1).max(axis=1).fillna(0.0)
                out["peer_screener_context_has_data"] = (
                    (pd.to_numeric(out.get("peer_screener_context_has_data"), errors="coerce").fillna(0.0) > 0.0)
                    | (universe_peer_count >= float(min_peers_effective))
                ).astype(float)
        except Exception:
            pass
    except Exception:
        pass

    out.attrs["telemetry"] = {
        "status": "ok",
        "source": "cross_sectional_rank",
        "base_family": base_family,
        "universe_size": int(len(universe_syms)),
        "peers_loaded": int(len(available_syms)),
        "universe_source": universe_source,
        "min_peers_configured": int(min_peers),
        "min_peers_effective": int(min_peers_effective) if "min_peers_effective" in locals() else int(min_peers),
        "sector": target_profile.sector,
        "industry": target_profile.industry,
    }
    out.attrs["provenance"] = {
        "source": "peer_screener_context",
        "base_family": base_family,
        "grouping": "sector+industry (EODHD fundamentals General->Sector/Industry)",
        "leakage": f"peer metrics shifted +{int(leakage_safe_shift_sessions)} NYSE session(s)",
        "notes": "ranks computed from cached per-symbol panels; sector/industry treated as static metadata",
        "universe_source": universe_source,
        "universe_size": int(len(universe_syms)),
        "peers_loaded": int(len(available_syms)),
        "min_peers_configured": int(min_peers),
        "min_peers_effective": int(min_peers_effective) if "min_peers_effective" in locals() else int(min_peers),
    }
    out.attrs["feature_counts"] = {"generated": int(len(out.columns))}

    # Persist shared cache for reuse (best-effort).
    try:
        shared_path.parent.mkdir(parents=True, exist_ok=True)
        if shared_cols and (force_recompute or (not shared_path.exists())):
            shared_df = pd.DataFrame(shared_cols, index=sessions)
            shared_df.to_parquet(shared_path)
    except Exception:
        pass

    return out


def build_peer_screener_context_snapshot(
    *,
    start: str,
    end: str,
    cache_root: Path,
    horizon: int = 63,
    universe: Sequence[str],
    base_family: str = "fin_g6",
    leakage_safe_shift_sessions: int = 1,
) -> pd.DataFrame:
    """Build a universe-wide peer context snapshot (date × symbol).

    Output schema:
    - Columns: date, symbol, and peer_screener_context_* feature columns.
    - One row per (date, symbol) NYSE session.

    This is the preferred artifact for universe selection / gating. It is not
    intended to be merged into per-symbol model features unless explicitly
    requested.
    """

    if cache_root is None:
        raise ValueError("cache_root is required")

    sessions = nyse_sessions_in_range(start, end, tz_aware_utc=False)
    if len(sessions) == 0:
        return pd.DataFrame(columns=["date", "symbol"])  # empty

    universe_syms = [_norm_symbol(s) for s in universe]
    universe_syms = [s for s in universe_syms if s]
    universe_syms = list(dict.fromkeys(universe_syms))
    if not universe_syms:
        return pd.DataFrame(columns=["date", "symbol"])  # empty

    profiles = _get_profiles(universe_syms)

    metrics = {
        "pe_ratio": f"{base_family}_pe_ratio",
        "pb_ratio": f"{base_family}_pb_ratio",
        "ev_ebitda": f"{base_family}_ev_ebitda",
    }

    matrices: Dict[str, pd.DataFrame] = {}
    available_syms: List[str] = []
    rank_memo: Dict[Tuple[str, str], pd.DataFrame] = {}
    for sym in universe_syms:
        any_loaded = False
        for short_name, col in metrics.items():
            series = _load_series_live_or_cache(
                Path(cache_root),
                sym,
                horizon=horizon,
                family=base_family,
                column=col,
                start=start,
                end=end,
                _memo=rank_memo,
            )
            if series is None:
                continue
            aligned = _align_series_to_sessions(series, sessions)
            if leakage_safe_shift_sessions:
                aligned = aligned.shift(int(leakage_safe_shift_sessions))
            matrices.setdefault(short_name, pd.DataFrame(index=sessions))
            matrices[short_name][sym] = aligned
            any_loaded = True
        if any_loaded:
            available_syms.append(sym)

    # Group members restricted to symbols we actually have data for.
    sector_groups: Dict[str, List[str]] = {}
    industry_groups: Dict[str, List[str]] = {}
    for sym in available_syms:
        prof = profiles.get(sym)
        if prof is None:
            continue
        sector_groups.setdefault(prof.sector or "Unknown", []).append(sym)
        industry_groups.setdefault(prof.industry or "Unknown", []).append(sym)

    shared_cols: Dict[str, pd.Series] = {}
    pe_matrix = matrices.get("pe_ratio")

    def _emit_for_group(
        *,
        group_type: str,
        group_members: Dict[str, List[str]],
        mat: pd.DataFrame,
        metric_name: str,
    ) -> None:
        for _, members in group_members.items():
            if not members:
                continue
            cols = [m for m in members if m in mat.columns]
            if not cols:
                continue
            sub = mat[cols]
            pct_high = _percentile_rank(sub, higher_is_better=True)
            pct_low = _percentile_rank(sub, higher_is_better=False)
            for sym in cols:
                if sym not in pct_high.columns:
                    continue
                shared_cols[f"{sym}|peer_screener_context_{group_type}_{metric_name}_pct"] = (
                    _coerce_numeric(pct_high[sym]).fillna(0.0).clip(0.0, 1.0)
                )
                shared_cols[f"{sym}|peer_screener_context_{group_type}_{metric_name}_cheap_pct"] = (
                    _coerce_numeric(pct_low[sym]).fillna(0.0).clip(0.0, 1.0)
                )

    # Counts and has_data gates (per-date, based on pe_ratio validity).
    if pe_matrix is not None and not pe_matrix.empty:
        for _, members in sector_groups.items():
            if not members:
                continue
            cols = [m for m in members if m in pe_matrix.columns]
            if not cols:
                continue
            counts = pe_matrix[cols].notna().sum(axis=1).astype(float)
            has_data = (counts >= 3.0).astype(float)
            for sym in cols:
                shared_cols[f"{sym}|peer_screener_context_sector_peer_count"] = counts
                shared_cols[f"{sym}|peer_screener_context_has_data"] = has_data

        for _, members in industry_groups.items():
            if not members:
                continue
            cols = [m for m in members if m in pe_matrix.columns]
            if not cols:
                continue
            counts = pe_matrix[cols].notna().sum(axis=1).astype(float)
            for sym in cols:
                shared_cols[f"{sym}|peer_screener_context_industry_peer_count"] = counts

    for metric_name, mat in matrices.items():
        if mat is None or mat.empty:
            continue
        _emit_for_group(group_type="sector", group_members=sector_groups, mat=mat, metric_name=metric_name)
        _emit_for_group(group_type="industry", group_members=industry_groups, mat=mat, metric_name=metric_name)

    # Ensure stable schema per available symbol.
    for sym in available_syms:
        shared_cols.setdefault(f"{sym}|peer_screener_context_has_data", pd.Series(0.0, index=sessions))
        shared_cols.setdefault(f"{sym}|peer_screener_context_sector_peer_count", pd.Series(0.0, index=sessions))
        shared_cols.setdefault(f"{sym}|peer_screener_context_industry_peer_count", pd.Series(0.0, index=sessions))
        for metric in ("pe_ratio", "pb_ratio", "ev_ebitda"):
            for scope in ("sector", "industry"):
                shared_cols.setdefault(
                    f"{sym}|peer_screener_context_{scope}_{metric}_pct",
                    pd.Series(0.0, index=sessions),
                )
                shared_cols.setdefault(
                    f"{sym}|peer_screener_context_{scope}_{metric}_cheap_pct",
                    pd.Series(0.0, index=sessions),
                )

    if not shared_cols:
        return pd.DataFrame(columns=["date", "symbol"])  # empty

    wide = pd.DataFrame(shared_cols, index=sessions)

    # Convert from wide columns "SYM|feature" to long (date, symbol, feature...).
    records: List[pd.DataFrame] = []
    for sym in available_syms:
        prefix = f"{sym}|"
        cols = [c for c in wide.columns if str(c).startswith(prefix)]
        if not cols:
            continue
        block = wide[cols].copy()
        block.columns = [str(c).split("|", 1)[1] for c in cols]
        block = block.reset_index(names="date")
        block.insert(1, "symbol", sym)
        records.append(block)

    if not records:
        return pd.DataFrame(columns=["date", "symbol"])  # empty

    out = pd.concat(records, axis=0, ignore_index=True)
    out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.tz_localize(None)
    out["symbol"] = out["symbol"].astype(str).str.upper()

    out.attrs["provenance"] = {
        "source": "peer_screener_context",
        "base_family": base_family,
        "grouping": "sector+industry (EODHD fundamentals General->Sector/Industry)",
        "leakage": f"peer metrics shifted +{int(leakage_safe_shift_sessions)} NYSE session(s)",
        "universe_requested": int(len(universe_syms)),
        "universe_loaded": int(len(available_syms)),
    }
    return out
