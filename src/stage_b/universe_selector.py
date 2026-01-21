from __future__ import annotations

# pyright: reportUnknownMemberType=false
# pyright: reportUnknownArgumentType=false
# pyright: reportUnknownVariableType=false
# pyright: reportUnknownParameterType=false

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

import numpy as np
import pandas as pd


# A stable CORE list meant to broaden regime/bear-market coverage while keeping
# the portfolio anchor set consistent.
DEFAULT_CORE_UNIVERSE: tuple[str, ...] = (
    # Core anchors (structural)
    "SPY",
    "AAPL",
    "MSFT",
    "NVDA",
    "AMZN",
    "GOOGL",
    "META",
    "JPM",
    "XOM",
    "UNH",
    "COST",
    # Financial structure
    "GS",
    "BAC",
    "BLK",
    # Defensive quality
    "PG",
    "KO",
    "JNJ",
    "WMT",
    # Macro transmitters
    "CVX",
    "CAT",
    "BA",
)


DEFAULT_SATELLITE_CANDIDATES: tuple[str, ...] = (
    # A. Earnings / narrative volatility
    "NFLX",
    "PYPL",
    "SHOP",
    "SQ",
    "ZM",
    # B. Macro / inflation cyclicals
    "FCX",
    "COP",
    "NEM",
    # C. Rate / liquidity stress
    "SCHW",
    "MS",
    # D. Consumer stress / recovery
    "HD",
    "LOW",
    "DIS",
    # Common episodic convexity names (kept as satellites)
    "TSLA",
    "AMD",
)


DEFAULT_CANDIDATE_UNIVERSE: tuple[str, ...] = (
    *DEFAULT_CORE_UNIVERSE,
    *DEFAULT_SATELLITE_CANDIDATES,
)


@dataclass(frozen=True)
class UniverseSelectorConfig:
    core_symbols: Sequence[str]
    candidate_symbols: Sequence[str]
    # Active SATELLITE sizing (will be clamped by total-universe bounds).
    satellite_min: int = 8
    satellite_max: int = 15

    # Selection uses a recent lookback window ending at `asof`.
    lookback_sessions: int = 63
    min_history_sessions: int = 63

    # Selection cadence (monthly aligned with U=21). If invoked more frequently
    # than this cadence, selector returns the prior state universe unchanged.
    rebalance_every_sessions: int = 21

    # Minimum stay in sessions for any newly-added satellite.
    min_stay_sessions: int = 21
    max_stay_sessions: int = 63

    # Removal rule: only remove after min-stay and when the event score decays.
    # A satellite is eligible for removal if:
    #   score_now <= max(remove_abs_threshold, peak_score * decay_fraction)
    decay_fraction: float = 0.5
    remove_abs_threshold: float = 0.25

    # Optional liquidity filter (best-effort; skipped if volume/close not present).
    min_adv_usd: Optional[float] = None

    # Optional universe-level peer context snapshot (date×symbol) used to gate candidates.
    # This is produced as a standalone parquet (not merged into model features).
    peer_snapshot_path: Optional[str] = None
    peer_require_has_data: bool = False
    peer_min_sector_peers: int = 3
    peer_min_industry_peers: int = 3

    # Event score weights (not currently exposed in compute_event_score; kept for future use).
    w_earnings: float = 0.20
    w_transcript: float = 0.20
    w_news: float = 0.20
    w_vol: float = 0.20
    w_beta: float = 0.20


def _panel_trading_sessions(panel_dir: Path, *, horizon: int) -> Optional[pd.DatetimeIndex]:
    """Use SPY TrackC panel as a proxy for session calendar (best-effort)."""

    spy_path = panel_dir / f"SPY_h{int(horizon)}_trackc.parquet"
    if not spy_path.exists():
        return None
    try:
        spy = _read_trackc_panel(spy_path)
        if spy.empty:
            return None
        idx = pd.DatetimeIndex(spy.index)
        unique = sorted({pd.Timestamp(x).normalize() for x in idx if pd.notna(x)})
        return pd.DatetimeIndex(unique)
    except Exception:
        return None


def _count_sessions_between(
    sessions: Optional[pd.DatetimeIndex],
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> Optional[int]:
    if sessions is None or len(sessions) == 0:
        return None
    s = pd.Timestamp(start).normalize()
    e = pd.Timestamp(end).normalize()
    if e < s:
        return 0
    mask = (sessions >= s) & (sessions <= e)
    return int(mask.sum())


def _load_state(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        with open(path, "r") as handle:
            data = json.load(handle)
        if not isinstance(data, dict):
            return {}
        # Normalize keys to strings for stable downstream typing.
        return {str(k): v for k, v in data.items()}
    except Exception:
        return {}


def _save_state(path: Path, state: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as handle:
        json.dump(state, handle, indent=2, sort_keys=True)
    tmp.replace(path)


def _normalize_symbols(symbols: Iterable[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for sym in symbols:
        s = (sym or "").strip().upper()
        if not s or s in seen:
            continue
        seen.add(s)
        out.append(s)
    return out


def _find_first_existing_column(df: pd.DataFrame, candidates: Sequence[str]) -> Optional[str]:
    for col in candidates:
        if col in df.columns:
            return col
    return None


def _coerce_numeric(series: pd.Series) -> pd.Series:
    out = pd.to_numeric(series, errors="coerce")
    out = out.replace([np.inf, -np.inf], np.nan)
    return out


def _clip_fill(series: pd.Series, *, lower: Optional[float] = None, upper: Optional[float] = None) -> pd.Series:
    s = _coerce_numeric(series)
    s = s.fillna(0.0)
    if lower is not None or upper is not None:
        s = s.clip(lower=lower, upper=upper)
    return s


def _get_series(df: pd.DataFrame, col: str, default: float = 0.0) -> pd.Series:
    if col in df.columns:
        return df[col]
    return pd.Series(float(default), index=df.index)


def compute_event_score(panel: pd.DataFrame, *, beta_col: Optional[str] = None) -> pd.Series:
    """Compute a best-effort event/regime score time series.

    Expected TrackC columns (after build_panel prefixing):
    Primary event drivers (cross-sectional discrimination):
    - earnings_* (event_decay, days_to_next_earnings, surprise/revisions)
    - earnings_transcript_hf_* (score deltas / risk / uncertainty × conf)
    - alternative_signals_* (news_volume_z, volume_zscore, liquidity_stress)
    - options_* (iv/skew/put-call metrics; normalized via rolling z)
    - short_interest_* (squeeze risk / changes)
    - microstructure_* (intraday liquidity stress proxies)

    Regime/context modifiers (mostly conditioning; may be weak cross-sectionally):
    - garch_iv_* (vol shock / zscore)
    - cboe_term_* (term structure stress)
    - correlation_* (decoupling z / trend / vol)
    - cross_asset_* (beta change / sign flips)
    - macro_tst_hf_* (macro stress proxies; used as a light multiplier)

    Returns a non-negative numeric series aligned to panel index.
    """

    if panel.empty:
        return pd.Series(dtype=float)

    df = panel

    def rolling_z(s: pd.Series, *, window: int = 252, min_periods: int = 60, clip: float = 3.0) -> pd.Series:
        x = _coerce_numeric(s).ffill()
        mu = x.rolling(window=window, min_periods=min_periods).mean()
        sd = x.rolling(window=window, min_periods=min_periods).std(ddof=0)
        z = (x - mu) / (sd + 1e-9)
        z = z.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        return z.clip(-clip, clip)

    # ------------------------------------------------------------------
    # 1) Earnings: post-event drift + pre-event anticipation + surprise/revisions
    # ------------------------------------------------------------------
    earnings_decay = _clip_fill(_get_series(df, "earnings_event_decay", 0.0), lower=0.0, upper=1.0)
    days_to_next = _clip_fill(_get_series(df, "earnings_days_to_next_earnings", np.nan))
    # Convert days_to_next into a smooth proximity score in [0,1].
    # 0 when far away; approaches 1 as days_to_next -> 0.
    pre_event = (1.0 / (1.0 + days_to_next.clip(lower=0.0))).clip(lower=0.0, upper=1.0)

    eps_surprise = _clip_fill(_get_series(df, "earnings_eps_surprise_pct", 0.0)).abs().clip(upper=3.0)
    rev_surprise = _clip_fill(_get_series(df, "earnings_revenue_surprise_pct", 0.0)).abs().clip(upper=3.0)
    surprise_vol = _clip_fill(_get_series(df, "earnings_surprise_volatility", 0.0)).clip(lower=0.0, upper=3.0)
    revision_breadth = _clip_fill(_get_series(df, "earnings_revision_breadth", 0.0)).abs().clip(upper=3.0)
    estimate_disp = _clip_fill(_get_series(df, "earnings_estimate_dispersion", 0.0)).clip(lower=0.0, upper=3.0)

    # Earnings intensity is higher (a) immediately after earnings, and (b) shortly before next earnings.
    earnings_intensity = (
        1.2 * earnings_decay
        + 0.8 * pre_event
        + earnings_decay * (0.6 * eps_surprise + 0.4 * rev_surprise + 0.5 * surprise_vol + 0.5 * revision_breadth)
        + pre_event * (0.3 * estimate_disp)
    ).clip(lower=0.0, upper=3.0)

    # ------------------------------------------------------------------
    # 2) Transcript HF: tone/uncertainty/risk deltas × confidence
    # ------------------------------------------------------------------
    transcript_conf = _clip_fill(_get_series(df, "earnings_transcript_hf_conf", 0.0), lower=0.0, upper=1.0)
    transcript_delta_qoq = _clip_fill(_get_series(df, "earnings_transcript_hf_score_delta_qoq", 0.0)).abs()
    transcript_delta_yoy = _clip_fill(_get_series(df, "earnings_transcript_hf_score_delta_yoy", 0.0)).abs()
    transcript_risk = _clip_fill(_get_series(df, "earnings_transcript_hf_risk_score", 0.0)).clip(lower=0.0, upper=3.0)
    transcript_unc = _clip_fill(_get_series(df, "earnings_transcript_hf_uncertainty_score", 0.0)).clip(lower=0.0, upper=3.0)
    prepared = _clip_fill(_get_series(df, "earnings_transcript_hf_score_prepared", 0.0))
    qa = _clip_fill(_get_series(df, "earnings_transcript_hf_score_qa", 0.0))
    prepared_qa_gap = (prepared - qa).abs().clip(upper=3.0)

    transcript_intensity = (
        transcript_conf
        * (0.8 * transcript_delta_qoq + 0.4 * transcript_delta_yoy + 0.6 * transcript_risk + 0.6 * transcript_unc + 0.4 * prepared_qa_gap)
    ).clip(lower=0.0, upper=3.0)

    # ------------------------------------------------------------------
    # 3) Alternative signals: news/volume/liquidity stress
    # ------------------------------------------------------------------
    news_z = _clip_fill(_get_series(df, "alternative_signals_news_volume_z", 0.0))
    news_change = _clip_fill(_get_series(df, "alternative_signals_news_volume_change", 0.0))
    vol_zscore = _clip_fill(_get_series(df, "alternative_signals_volume_zscore", 0.0)).abs()
    liq_stress = _clip_fill(_get_series(df, "alternative_signals_liquidity_stress", 0.0)).clip(lower=0.0, upper=3.0)
    overnight_z = _clip_fill(_get_series(df, "alternative_signals_overnight_return_z", 0.0)).abs().clip(upper=3.0)

    alt_intensity = (
        news_z.clip(lower=0.0).clip(upper=3.0)
        + 0.3 * rolling_z(news_change).clip(lower=0.0)
        + 0.5 * vol_zscore.clip(upper=3.0)
        + 0.5 * liq_stress
        + 0.3 * overnight_z
    ).clip(lower=0.0, upper=3.0)

    # ------------------------------------------------------------------
    # 4) Derivatives/regime (symbol-level): garch_iv, cboe_term, options
    # ------------------------------------------------------------------
    garch_z = _clip_fill(_get_series(df, "garch_iv_garch_zscore", 0.0))
    garch_spike = _clip_fill(_get_series(df, "garch_iv_garch_spike_flag", 0.0), lower=0.0, upper=1.0)
    garch_shock = _clip_fill(_get_series(df, "garch_iv_garch_shock_indicator", 0.0), lower=0.0, upper=1.0)
    garch_volvol = rolling_z(_get_series(df, "garch_iv_garch_vol_of_vol", 0.0)).abs()
    garch_intensity = (garch_z.clip(lower=0.0) + garch_spike + garch_shock + 0.4 * garch_volvol).clip(0.0, 3.0)

    cboe_panic = _clip_fill(_get_series(df, "cboe_term_panic_premium", 0.0))
    cboe_frontback = _clip_fill(_get_series(df, "cboe_term_front_back_spread", 0.0))
    cboe_slope_chg = rolling_z(_get_series(df, "cboe_term_vix_term_slope_change_1d", 0.0)).abs()
    cboe_intensity = (cboe_panic.clip(lower=0.0) + cboe_frontback.abs() + 0.5 * cboe_slope_chg).clip(0.0, 3.0)

    opt_atm_iv = rolling_z(_get_series(df, "options_atm_iv", 0.0)).clip(lower=0.0)
    opt_iv_spread = rolling_z(_get_series(df, "options_iv_spread", 0.0)).abs()
    opt_pcr_vol = _clip_fill(_get_series(df, "options_put_call_volume_ratio", np.nan))
    opt_pcr_oi = _clip_fill(_get_series(df, "options_put_call_oi_ratio", np.nan))
    opt_pcr_vol_z = rolling_z(np.log(opt_pcr_vol.clip(lower=1e-6))).abs()
    opt_pcr_oi_z = rolling_z(np.log(opt_pcr_oi.clip(lower=1e-6))).abs()
    opt_intensity = (0.6 * opt_atm_iv + 0.5 * opt_iv_spread + 0.4 * opt_pcr_vol_z + 0.4 * opt_pcr_oi_z).clip(0.0, 3.0)

    # ------------------------------------------------------------------
    # 5) Correlation/beta change-rate: decoupling + beta instability
    # ------------------------------------------------------------------
    corr_decouple = _clip_fill(_get_series(df, "correlation_corr_decoupling_z", 0.0)).abs().clip(upper=3.0)
    corr_spy_trend = _clip_fill(_get_series(df, "correlation_corr_20_spy_trend", 0.0)).abs().clip(upper=3.0)
    corr_vxx_vol = rolling_z(_get_series(df, "correlation_corr_20_vxx_vol", 0.0)).abs().clip(upper=3.0)
    corr_intensity = (0.6 * corr_decouple + 0.3 * corr_spy_trend + 0.3 * corr_vxx_vol).clip(0.0, 3.0)

    if beta_col is None:
        beta_col = _find_first_existing_column(
            df,
            (
                "cross_asset_beta_change_rate",
                "cross_asset_beta_volatility_change",
                "alternative_signals_beta_change_rate",
            ),
        )
    beta_shift = _clip_fill(_get_series(df, beta_col, 0.0)).abs() if beta_col else pd.Series(0.0, index=df.index)
    beta_flip = _clip_fill(_get_series(df, "cross_asset_beta_sign_flip_flag", 0.0), lower=0.0, upper=1.0)
    beta_intensity = (beta_shift.clip(upper=3.0) + 0.5 * beta_flip).clip(0.0, 3.0)

    # ------------------------------------------------------------------
    # 6) Positioning + microstructure
    # ------------------------------------------------------------------
    squeeze = _clip_fill(_get_series(df, "short_interest_short_squeeze_risk", 0.0)).clip(lower=0.0, upper=3.0)
    dtc = _clip_fill(_get_series(df, "short_interest_days_to_cover_zscore", 0.0)).clip(lower=0.0, upper=3.0)
    short_intensity = (0.7 * squeeze + 0.4 * dtc).clip(0.0, 3.0)

    ofi = rolling_z(_get_series(df, "microstructure_ofi_proxy", 0.0)).abs()
    imbalance = _clip_fill(_get_series(df, "microstructure_updown_imbalance", 0.0)).abs().clip(upper=3.0)
    liq_slope = rolling_z(_get_series(df, "microstructure_micro_liq_slope", 0.0)).clip(lower=0.0)
    stalled = _clip_fill(_get_series(df, "microstructure_stalled_prints_ratio", 0.0)).clip(lower=0.0, upper=3.0)
    micro_intensity = (0.4 * ofi + 0.4 * imbalance + 0.5 * liq_slope + 0.3 * stalled).clip(0.0, 3.0)

    # ------------------------------------------------------------------
    # 7) Novelty HF (narrative regime breaks)
    # ------------------------------------------------------------------
    novelty_score = _clip_fill(_get_series(df, "doc_embedding_novelty_hf_score", 0.0)).abs().clip(upper=3.0)
    novelty_conf = _clip_fill(_get_series(df, "doc_embedding_novelty_hf_conf", 0.0), lower=0.0, upper=1.0)
    novelty_spike = _clip_fill(_get_series(df, "doc_embedding_novelty_hf_novelty_spike_flag", 0.0), lower=0.0, upper=1.0)
    novelty_persist = _clip_fill(_get_series(df, "doc_embedding_novelty_hf_novelty_persistence_5d", 0.0)).clip(lower=0.0, upper=3.0)
    novelty_intensity = (novelty_conf * novelty_score + 0.6 * novelty_spike + 0.4 * novelty_persist).clip(0.0, 3.0)

    # ------------------------------------------------------------------
    # 8) Macro context (light multiplier; avoids dominating cross-section)
    # ------------------------------------------------------------------
    macro_vix = rolling_z(_get_series(df, "macro_tst_hf_macro_vix", 0.0)).clip(lower=0.0)
    macro_credit = rolling_z(_get_series(df, "macro_tst_hf_macro_credit_spread", 0.0)).clip(lower=0.0)
    macro_recession = _clip_fill(_get_series(df, "macro_tst_hf_macro_recession_risk", 0.0)).clip(lower=0.0, upper=3.0)
    macro_stress = (0.5 * macro_vix + 0.5 * macro_credit + 0.5 * macro_recession).clip(0.0, 3.0)
    macro_multiplier = (1.0 + 0.15 * macro_stress).clip(1.0, 1.5)

    # Final: non-negative, clipped, rule-based.
    base = (
        0.9 * earnings_intensity
        + 0.8 * transcript_intensity
        + 0.7 * alt_intensity
        + 0.6 * novelty_intensity
        + 0.7 * garch_intensity
        + 0.6 * cboe_intensity
        + 0.6 * opt_intensity
        + 0.5 * corr_intensity
        + 0.6 * beta_intensity
        + 0.4 * short_intensity
        + 0.4 * micro_intensity
    )
    score = (base * macro_multiplier).clip(lower=0.0)
    return _clip_fill(score, lower=0.0)


def _read_trackc_panel(panel_path: Path) -> pd.DataFrame:
    df = pd.read_parquet(panel_path)
    if "date" in df.columns:
        df = df.copy()
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date"]).sort_values("date")
        df = df.set_index("date")
    else:
        # Best-effort: assume already indexed by date-ish.
        try:
            df = df.copy()
            df.index = pd.to_datetime(df.index, errors="coerce")
            df = df.dropna(axis=0, how="any")
        except Exception:
            pass
    return df.sort_index()


def _estimate_adv_usd(panel: pd.DataFrame, *, window: int = 20) -> Optional[float]:
    if panel.empty:
        return None

    close_col = _find_first_existing_column(panel, ("close", "price_close", "adj_close"))
    vol_col = _find_first_existing_column(panel, ("volume", "price_volume"))
    if close_col is None or vol_col is None:
        return None

    close = _coerce_numeric(panel[close_col]).ffill()
    vol = _coerce_numeric(panel[vol_col]).ffill()
    dollar_vol = (close * vol).replace([np.inf, -np.inf], np.nan)
    adv = dollar_vol.rolling(window=window, min_periods=max(5, window // 4)).mean()
    if adv.empty:
        return None
    try:
        return float(adv.dropna().iloc[-1])
    except Exception:
        return None


def _load_peer_snapshot_eligible_symbols(
    *,
    snapshot_path: Path,
    asof_ts: pd.Timestamp,
    require_has_data: bool,
    min_sector_peers: int,
    min_industry_peers: int,
) -> Optional[set[str]]:
    """Return eligible symbols for the given asof date based on peer snapshot.

    Best-effort: if the snapshot can't be read, return None to indicate "no gating".
    """

    if not snapshot_path.exists():
        return None

    cols = [
        "date",
        "symbol",
        "peer_screener_context_has_data",
        "peer_screener_context_sector_peer_count",
        "peer_screener_context_industry_peer_count",
    ]

    df: Optional[pd.DataFrame] = None
    try:
        df = pd.read_parquet(
            snapshot_path,
            columns=cols,
            filters=[("date", "==", pd.Timestamp(asof_ts).normalize())],
        )
    except Exception:
        df = None

    if df is None or df.empty:
        try:
            df = pd.read_parquet(snapshot_path, columns=cols)
        except Exception:
            return None

    if df is None or df.empty:
        return None

    if "date" in df.columns:
        df = df.copy()
        df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.tz_localize(None)
        df = df[df["date"] == pd.Timestamp(asof_ts).normalize()]
    if df.empty:
        return None

    df["symbol"] = df["symbol"].astype(str).str.upper()

    sector_count = pd.to_numeric(df.get("peer_screener_context_sector_peer_count"), errors="coerce").fillna(0.0)
    industry_count = pd.to_numeric(df.get("peer_screener_context_industry_peer_count"), errors="coerce").fillna(0.0)

    mask = (sector_count >= float(min_sector_peers)) & (industry_count >= float(min_industry_peers))
    if require_has_data:
        has_data = pd.to_numeric(df.get("peer_screener_context_has_data"), errors="coerce").fillna(0.0)
        mask = mask & (has_data >= 1.0)

    eligible = set(df.loc[mask, "symbol"].astype(str).str.upper().tolist())
    return eligible


def select_universe(
    *,
    panel_dir: Path,
    horizon: int,
    asof: str | pd.Timestamp,
    cfg: UniverseSelectorConfig,
    state_path: Optional[Path] = None,
) -> list[str]:
    """Select CORE ∪ SATELLITE based on recent TrackC event score.

    Behavior constraints:
    - CORE is fixed (caller-controlled) and never changes intra-year.
    - SATELLITE updates are monthly-ish: only recompute when >= U sessions since last rebalance.
    - SATELLITE membership has a minimum stay and is removed only on score decay.
    """

    core = _normalize_symbols(cfg.core_symbols)
    candidates = _normalize_symbols(cfg.candidate_symbols)

    asof_ts = pd.to_datetime(asof).normalize()

    # Enforce CORE size bounds (never fewer than 15, never more than 25).
    if len(core) < 15:
        raise ValueError(f"CORE too small: {len(core)} < 15")
    if len(core) > 25:
        raise ValueError(f"CORE too large: {len(core)} > 25")

    # Determine satellite min/max to keep total universe in [30, 40].
    sat_min = max(int(cfg.satellite_min), max(0, 30 - len(core)))
    sat_max = min(int(cfg.satellite_max), max(0, 40 - len(core)))
    if sat_max < sat_min:
        # Fall back to a reasonable clamp; keep within [30,40] if possible.
        sat_max = sat_min

    sessions = _panel_trading_sessions(panel_dir, horizon=int(horizon))

    # Optional peer snapshot gating (applies to satellites/candidates only).
    if cfg.peer_snapshot_path and str(cfg.peer_snapshot_path).strip():
        try:
            eligible = _load_peer_snapshot_eligible_symbols(
                snapshot_path=Path(str(cfg.peer_snapshot_path)),
                asof_ts=asof_ts,
                require_has_data=bool(cfg.peer_require_has_data),
                min_sector_peers=int(cfg.peer_min_sector_peers),
                min_industry_peers=int(cfg.peer_min_industry_peers),
            )
        except Exception:
            eligible = None
        if eligible is not None and len(eligible) > 0:
            candidates = [s for s in candidates if s in core or s in eligible]

    state: dict[str, Any] = {}
    if state_path is not None:
        state = _load_state(state_path)

    # State schema (best-effort):
    # {
    #   "last_rebalance": "YYYY-MM-DD",
    #   "satellites": {"SYM": {"added": "YYYY-MM-DD", "peak": 0.0}}
    # }
    last_rebalance_raw = state.get("last_rebalance")
    last_rebalance = pd.to_datetime(str(last_rebalance_raw), errors="coerce") if last_rebalance_raw else pd.NaT
    if pd.notna(last_rebalance):
        last_rebalance = pd.Timestamp(last_rebalance).normalize()
    else:
        last_rebalance = None

    if last_rebalance is not None:
        n_since = _count_sessions_between(sessions, last_rebalance, asof_ts)
        if n_since is not None and n_since < int(cfg.rebalance_every_sessions):
            # Too soon: return previous universe unchanged (but validate core).
            prev_sat = []
            sat_state_prev_obj = state.get("satellites")
            sat_state_prev = sat_state_prev_obj if isinstance(sat_state_prev_obj, dict) else {}
            for sym in sat_state_prev:
                if isinstance(sym, str):
                    prev_sat.append(sym.upper())
            prev_universe = _normalize_symbols([*core, *prev_sat])
            return prev_universe

    sat_state_obj = state.get("satellites")
    sat_state = sat_state_obj if isinstance(sat_state_obj, dict) else {}
    active_satellites = _normalize_symbols(list(sat_state.keys()))

    # Score all candidates as-of.
    scores: dict[str, float] = {}

    def score_symbol(sym: str) -> Optional[float]:
        panel_path = panel_dir / f"{sym}_h{int(horizon)}_trackc.parquet"
        if not panel_path.exists():
            return None
        panel = _read_trackc_panel(panel_path)
        if panel.empty:
            return None
        panel = panel.loc[panel.index <= asof_ts]
        if panel.empty:
            return None
        tail = panel.tail(int(cfg.lookback_sessions))
        if len(tail) < int(cfg.min_history_sessions):
            return None
        if cfg.min_adv_usd is not None:
            adv_usd = _estimate_adv_usd(tail)
            if adv_usd is None or adv_usd < float(cfg.min_adv_usd):
                return None
        event_score = compute_event_score(tail)
        if event_score.empty:
            return None
        return float(event_score.mean())

    for sym in candidates:
        if sym in core:
            continue
        sc = score_symbol(sym)
        if sc is None:
            continue
        scores[sym] = float(sc)

    # Update peaks for active satellites.
    for sym in active_satellites:
        cur = scores.get(sym)
        if cur is None:
            continue
        entry = sat_state.get(sym, {}) if isinstance(sat_state.get(sym), dict) else {}
        peak = float(entry.get("peak", 0.0) or 0.0)
        if cur > peak:
            entry["peak"] = float(cur)
            sat_state[sym] = entry

    # Removal pass: only remove after min-stay and only on decay.
    kept: list[str] = []
    removed: list[str] = []

    for sym in active_satellites:
        entry = sat_state.get(sym, {}) if isinstance(sat_state.get(sym), dict) else {}
        added_raw = entry.get("added")
        added_ts_parsed = pd.to_datetime(str(added_raw), errors="coerce") if added_raw else pd.NaT
        added_ts = pd.Timestamp(added_ts_parsed).normalize() if pd.notna(added_ts_parsed) else None

        cur = float(scores.get(sym, 0.0) or 0.0)
        peak = float(entry.get("peak", cur) or cur)
        decay_gate = max(float(cfg.remove_abs_threshold), float(peak) * float(cfg.decay_fraction))

        # Compute age in sessions if we have a session calendar.
        age_sessions = None
        if added_ts is not None:
            age_sessions = _count_sessions_between(sessions, added_ts, asof_ts)

        must_keep = False
        if age_sessions is None:
            # No reliable calendar; be conservative: keep unless clearly decayed.
            must_keep = False
        else:
            if age_sessions < int(cfg.min_stay_sessions):
                must_keep = True

        if must_keep:
            kept.append(sym)
            continue

        # Eligible for removal: only remove when score has decayed.
        if cur <= decay_gate:
            removed.append(sym)
        else:
            kept.append(sym)

    # Drop removed from state.
    for sym in removed:
        sat_state.pop(sym, None)

    # Add pass: top up to sat_min..sat_max with highest scores.
    kept_set = set(kept)
    ranked = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
    additions: list[str] = []
    for sym, _ in ranked:
        if sym in kept_set:
            continue
        if sym in core:
            continue
        additions.append(sym)
        kept_set.add(sym)
        if len(kept_set) >= sat_max:
            break

    # If still below sat_min, keep adding (even if scores are weak) from ranked list.
    if len(kept_set) < sat_min:
        for sym, _ in ranked:
            if sym in kept_set or sym in core:
                continue
            kept_set.add(sym)
            additions.append(sym)
            if len(kept_set) >= sat_min:
                break

    satellites = _normalize_symbols([*kept, *additions])

    # Update state entries for new additions.
    for sym in additions:
        cur = float(scores.get(sym, 0.0) or 0.0)
        sat_state[sym] = {
            "added": asof_ts.strftime("%Y-%m-%d"),
            "peak": float(cur),
        }

    # Persist state.
    if state_path is not None:
        state_out: dict[str, Any] = {
            "last_rebalance": asof_ts.strftime("%Y-%m-%d"),
            "satellites": sat_state,
        }
        _save_state(state_path, state_out)

    universe = _normalize_symbols([*core, *satellites])
    return universe
