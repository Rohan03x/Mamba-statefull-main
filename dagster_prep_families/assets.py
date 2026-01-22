from pathlib import Path
from typing import Any, Mapping, Optional, cast

from dagster import (
    AssetMaterialization,
    AssetCheckResult,
    Config,
    Failure,
    asset,
    asset_check,
)

from .partitions import get_partitions_def
from .quality import QualityThresholds, validate_panel_by_known_families

# Import centralized cache paths
try:
    from src.cache_paths import MERGED_CACHE_ROOT, SYMBOLS_CACHE_ROOT, DOC_EMBEDDING_SHARED_DIR
    FEATURE_PANEL_DIR = MERGED_CACHE_ROOT  # Final merged parquets go here
except ImportError:
    FEATURE_PANEL_DIR = Path("cache/merged")


def _serialize_result(value: Any) -> Any:
    if value is None:
        return None
    try:
        # pandas Timestamp
        import pandas as pd

        if isinstance(value, pd.Timestamp):
            return value.isoformat()
    except Exception:
        pass

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, (str, int, float, bool)):
        return value

    if isinstance(value, dict):
        d = cast(Mapping[Any, Any], value)
        return {str(k): _serialize_result(v) for k, v in d.items()}

    if isinstance(value, (list, tuple)):
        return [_serialize_result(v) for v in value]

    return str(value)


class PrepFamiliesConfig(Config):
    symbol: str = "AAPL"
    horizon: int = 63

    # Walk-forward
    wf_start: Optional[str] = None
    wf_end: Optional[str] = None
    wf_train_years: Optional[int] = None
    wf_step_years: Optional[int] = None
    wf_step_days: Optional[int] = None

    # Safety: Dagster UI often reuses the last run config. To avoid accidentally
    # truncating merged panels, wf_start/wf_end overrides are ignored unless this
    # flag (or the env var) is explicitly enabled.
    allow_wf_override: bool = False

    # Selection + execution
    families: str = "all"
    strict: bool = False
    workers: int = 1
    hf_workers: Optional[int] = None
    mode: str = "stage-b"
    
    # Sequential execution mode (default: True)
    # When True, all tasks run sequentially within a symbol run:
    #   1. Horizon-linked families (quantile_forecast → calibration → online_learning)
    #   2. Base families (symbol-only)
    #   3. HF blocks
    # No process/thread pools are spawned. This is the preferred mode for Dagster.
    sequential_mode: bool = True

    # Feast merged parquet
    write_merged: bool = True
    merged_out: Optional[str] = None
    merged_service: Optional[str] = None

    # Cache reuse
    reuse_symbol_only_cache: bool = True

    # Preprocessing defaults (kept explicit here so Dagster runs are deterministic
    # even when workers reuse process-level environment variables).
    # These mirror the defaults in tools/prep_families.py.
    preprocess_role_normalize: bool = True
    preprocess_timing_rules: bool = True
    preprocess_timing_shift_days: int = 1
    preprocess_timing_ffill: bool = True
    preprocess_timing_decay: bool = True

    # Optional numeric-only hint channels (off by default).
    preprocess_role_hint_channels: bool = False
    preprocess_days_since_update_hints: bool = False

    # Role-aware split outputs (mamba vs portfolio)
    preprocess_write_role_splits: bool = True
    alt_signals_staleness_k: Optional[float] = None

    mamba_optional_all: bool = False

    # Source policy: enforce "real" data at generation time.
    # Set to False to allow proxy/fallback data sources like garch_iv
    enforce_no_proxy_sources: bool = False
    enforce_no_live_fallback: bool = False
    enforce_eodhd_only: bool = False

    # Quality thresholds (post-merge)
    quality_min_variance_threshold: float = 1e-10
    quality_max_zero_percentage: float = 0.95
    quality_max_null_percentage: float = 0.90
    quality_min_has_data_mean: float = 0.05
    quality_max_proxy_flag_mean: float = 0.05
    quality_max_rows: Optional[int] = 100_000

    quality_has_data_tail_rows: int = 252
    quality_min_has_data_nonzero_tail: int = 3

    # Coverage policy: by default, require each requested family to have some columns
    # in the merged panel (unless explicitly allowed). This catches silent cache misses.
    quality_fail_on_missing_families: bool = True
    quality_allowed_missing_families: str = "calibration,peer_screener_context,fx,commodities,crypto"

    # If enabled, the asset itself raises Failure when quality checks fail.
    # (Asset checks will still run, but this makes failures impossible to ignore.)
    fail_on_quality_checks: bool = False

    # Some families are designed to be "present but not confident" (shape-stable)
    # and legitimately emit has_data ~ 0 for long periods (e.g. peer coverage < N).
    # These should not hard-fail strict Dagster runs.
    quality_allowed_dormant_families: str = "calibration,peer_screener_context,fx,commodities,crypto"


class UniverseDelistingRegistryConfig(Config):
    """Config for building the universe-level delisting/listing-status registry."""

    exchange_code: str = "US"
    cache_path: Optional[str] = None
    timeout_seconds: int = 60
    stale_sessions: int = 25
    vol_bad_fraction: float = 0.8


class UniverseRegistryAssetConfig(Config):
    """Config for building the authoritative universe registry parquet."""

    exchange_code: str = "US"
    out_path: Optional[str] = None
    # If None, uses today's date (UTC) as an as-of.
    asof_date: Optional[str] = None

    # Which symbol set to include.
    symbols_mode: str = "default_candidate_universe"  # {default_candidate_universe, dagster_symbols}

    # OHLCV-based staleness policy.
    stale_sessions: int = 25
    vol_bad_fraction: float = 0.8
    prices_lookback_days: int = 420

    # Optional: use the delisting registry parquet as an input cache.
    delisting_cache_path: Optional[str] = None


class PeerContextSnapshotConfig(Config):
    """Config for building the universe-wide peer context snapshot parquet."""

    horizon: int = 63
    start: Optional[str] = None
    end: Optional[str] = None
    cache_root: str = "cache/symbols"
    out_path: Optional[str] = None
    base_family: str = "fin_g6"
    leakage_safe_shift_sessions: int = 1

    # Optional: if provided, restrict peer snapshot to symbols eligible under this universe registry.
    universe_registry_path: Optional[str] = None


def _partition_symbol_horizon(context: Any) -> tuple[Optional[str], Optional[int]]:
    """Extract symbol/horizon from a Dagster multi-partition key if present."""

    pk = getattr(context, "partition_key", None)
    if not pk:
        return None, None

    keys_by_dim = getattr(pk, "keys_by_dimension", None)
    if isinstance(keys_by_dim, dict):
        sym = keys_by_dim.get("symbol")
        hor = keys_by_dim.get("horizon")
        try:
            hor_i = int(hor) if hor is not None else None
        except Exception:
            hor_i = None
        return (str(sym).upper() if sym else None), hor_i

    # Fallback: partition_key might be a plain string in some contexts
    try:
        s = str(pk)
        if "|" in s:
            parts = dict(p.split("=") for p in s.split("|"))
            sym = parts.get("symbol")
            hor = parts.get("horizon")
            return (sym.upper() if sym else None), (int(hor) if hor else None)
    except Exception:
        pass

    return None, None


@asset(
    name="prep_families_manifest",
    config_schema=PrepFamiliesConfig.to_config_schema(),
    partitions_def=get_partitions_def(),
    description=(
        "Runs tools/prep_families.prepare_families for one symbol/horizon and returns the completeness manifest path."
    ),
)
def prep_families_manifest(context) -> str:
    """Materializes a run of prep_families as a Dagster asset.

    Notes:
    - This wraps the in-repo orchestration function, not a subprocess.
    - In stage-b mode with write_merged=true, it writes ONE unified, all-dates parquet under cache/features.
    """

    # Local import so dagster can load without importing the whole repo at module import time.
    from tools.prep_families import (
        DEFAULT_WF_END,
        DEFAULT_WF_START,
        DEFAULT_WF_STEP_DAYS,
        DEFAULT_WF_TRAIN_YEARS,
        prepare_families,
    )

    prepare_families_any = cast(Any, prepare_families)

    cfg = PrepFamiliesConfig(**(getattr(context, "op_config", None) or {}))

    p_symbol, p_horizon = _partition_symbol_horizon(context)
    run_symbol = p_symbol or cfg.symbol
    run_horizon = int(p_horizon) if p_horizon is not None else int(cfg.horizon)

    # Enforce strict "real data" sourcing in-process.
    # These are already supported by src.features.aggregator_panel + universal fetchers.
    import os

    if cfg.enforce_no_proxy_sources:
        os.environ["STAGE_B_NO_PROXY_SOURCES"] = "1"
    if cfg.enforce_no_live_fallback:
        os.environ["STAGE_B_NO_LIVE_FALLBACK"] = "1"
    if cfg.enforce_eodhd_only:
        os.environ["STAGE_B_EODHD_ONLY"] = "1"

    # Make preprocessing behavior deterministic in Dagster by setting these per-run,
    # rather than inheriting whatever happens to be in the worker's environment.
    os.environ["PREP_FAMILIES_ROLE_NORMALIZE"] = "1" if bool(cfg.preprocess_role_normalize) else "0"
    os.environ["PREP_FAMILIES_TIMING_RULES"] = "1" if bool(cfg.preprocess_timing_rules) else "0"
    os.environ["PREP_FAMILIES_TIMING_SHIFT_DAYS"] = str(int(cfg.preprocess_timing_shift_days))
    os.environ["PREP_FAMILIES_TIMING_FFILL"] = "1" if bool(cfg.preprocess_timing_ffill) else "0"
    os.environ["PREP_FAMILIES_TIMING_DECAY"] = "1" if bool(cfg.preprocess_timing_decay) else "0"
    os.environ["PREP_FAMILIES_ROLE_HINT_CHANNELS"] = "1" if bool(cfg.preprocess_role_hint_channels) else "0"
    os.environ["PREP_FAMILIES_DAYS_SINCE_UPDATE_HINTS"] = "1" if bool(cfg.preprocess_days_since_update_hints) else "0"
    os.environ["PREP_FAMILIES_WRITE_ROLE_SPLITS"] = "1" if bool(cfg.preprocess_write_role_splits) else "0"
    if cfg.alt_signals_staleness_k is not None:
        os.environ["ALT_SIGNALS_STALENESS_K"] = str(cfg.alt_signals_staleness_k)
    if bool(cfg.mamba_optional_all):
        os.environ["ALT_SIGNALS_MAMBA_OPTIONAL"] = "all"
        os.environ["ARIMA_FORECAST_MAMBA_OPTIONAL"] = "all"
        os.environ["QUANTILE_FORECAST_MAMBA_STACKING"] = "all"
        os.environ["CANDLE_MECHANICS_MAMBA_OPTIONAL"] = "all"

    allow_wf_override = bool(cfg.allow_wf_override) or (os.environ.get("DAGSTER_PREP_ALLOW_WF_OVERRIDE", "0") == "1")

    # Fill defaults that live in prep_families.py.
    # By default, ignore Dagster-provided overrides unless explicitly allowed.
    wf_start = (cfg.wf_start if (allow_wf_override and cfg.wf_start) else DEFAULT_WF_START)
    wf_end = (cfg.wf_end if (allow_wf_override and cfg.wf_end) else DEFAULT_WF_END)
    wf_train_years = int(cfg.wf_train_years) if cfg.wf_train_years is not None else int(DEFAULT_WF_TRAIN_YEARS)
    wf_step_years = int(cfg.wf_step_years) if cfg.wf_step_years is not None else None
    wf_step_days = int(cfg.wf_step_days) if cfg.wf_step_days is not None else None
    if wf_step_years is None and wf_step_days is None:
        wf_step_days = int(DEFAULT_WF_STEP_DAYS)
    if wf_step_years is not None and wf_step_days is not None:
        raise ValueError("Specify exactly one of wf_step_years or wf_step_days")

    result = prepare_families_any(
        symbol=run_symbol,
        horizon=int(run_horizon),
        wf_start=wf_start,
        wf_end=wf_end,
        wf_train_years=wf_train_years,
        wf_step_years=wf_step_years,
        wf_step_days=wf_step_days,
        families=cfg.families,
        strict=bool(cfg.strict),
        workers=int(cfg.workers),
        hf_workers=cfg.hf_workers,
        mode=cfg.mode,
        write_merged=bool(cfg.write_merged),
        merged_out=(Path(cfg.merged_out) if cfg.merged_out else None),
        merged_service=cfg.merged_service,
        reuse_symbol_only_cache=bool(cfg.reuse_symbol_only_cache),
        sequential_mode=bool(cfg.sequential_mode),
    )

    manifest = str(result.completeness_manifest_path) if result.completeness_manifest_path else ""

    context.log_event(
        AssetMaterialization(
            asset_key="prep_families_manifest",
            description=f"prep_families completed for {run_symbol} h{int(run_horizon)}",
            metadata={
                "symbol": run_symbol,
                "horizon": int(run_horizon),
                "wf_start": str(wf_start),
                "wf_end": str(wf_end),
                "allow_wf_override": bool(allow_wf_override),
                "success": bool(result.success),
                "failed_tasks": int(result.failed_tasks),
                "duration_seconds": float(result.duration_seconds),
                "manifest_path": manifest,
                "result": _serialize_result(result.__dict__),
            },
        )
    )

    if not bool(result.success):
        raise Failure(
            description=(
                f"prep_families failed for {run_symbol} h{int(run_horizon)} "
                f"(failed_tasks={int(result.failed_tasks)})"
            ),
            metadata={
                "symbol": run_symbol,
                "horizon": int(run_horizon),
                "failed_tasks": int(result.failed_tasks),
                "failed_families": ",".join(list(getattr(result, "failed_families", []) or []))
                if getattr(result, "failed_families", None) is not None
                else "",
                "manifest_path": manifest or "",
            },
        )

    if not manifest:
        raise RuntimeError("prep_families did not produce a completeness manifest path")

    # Optional: enforce post-merge quality/coverage inside the asset.
    # Dagster asset checks already report this, but raising here ensures the run fails
    # in strict workflows.
    if bool(cfg.fail_on_quality_checks) and bool(cfg.write_merged):
        merged_path = (
            Path(cfg.merged_out)
            if cfg.merged_out
            else FEATURE_PANEL_DIR / f"{run_symbol}_h{int(run_horizon)}_merged.parquet"
        )
        if merged_path.exists():
            import pandas as pd

            thresholds = QualityThresholds(
                min_variance_threshold=float(cfg.quality_min_variance_threshold),
                max_zero_percentage=float(cfg.quality_max_zero_percentage),
                max_null_percentage=float(cfg.quality_max_null_percentage),
                min_has_data_mean=float(cfg.quality_min_has_data_mean),
                max_proxy_flag_mean=float(cfg.quality_max_proxy_flag_mean),
                has_data_tail_rows=int(getattr(cfg, "quality_has_data_tail_rows", 252)),
                min_has_data_nonzero_tail=int(getattr(cfg, "quality_min_has_data_nonzero_tail", 3)),
                max_rows=cfg.quality_max_rows,
            )

            allowed_dormant_families = [
                t.strip()
                for t in str(getattr(cfg, "quality_allowed_dormant_families", "") or "").split(",")
                if t.strip()
            ]
            allowed_missing_families = [
                t.strip()
                for t in str(getattr(cfg, "quality_allowed_missing_families", "") or "").split(",")
                if t.strip()
            ]

            # Resolve expected families based on selection. For the common "all" path,
            # validate stage-B families (base + HF + META).
            families_to_check = None
            fam_sel = str(cfg.families or "all").strip().lower()
            if fam_sel in {"all", "stage-b", "stage_b", "b"}:
                from src.features.family_spec import families_for_stage

                families_to_check = families_for_stage("B")
            elif fam_sel not in {"none", ""}:
                families_to_check = [t.strip() for t in str(cfg.families).split(",") if t.strip()]

            df = pd.read_parquet(merged_path)
            passed, results = validate_panel_by_known_families(
                panel=df,
                thresholds=thresholds,
                families=families_to_check,
                allowed_dormant_families=allowed_dormant_families,
                fail_on_missing=bool(cfg.quality_fail_on_missing_families),
                allowed_missing_families=allowed_missing_families,
            )
            if not passed:
                failed = [r for r in results if not r.passed]
                worst = failed[:25]
                raise Failure(
                    description=(
                        f"Merged panel failed quality checks for {run_symbol} h{int(run_horizon)} "
                        f"({len(failed)}/{len(results)} families failed)"
                    ),
                    metadata={
                        "symbol": run_symbol,
                        "horizon": int(run_horizon),
                        "merged_path": str(merged_path),
                        "families_failed": int(len(failed)),
                        "failed_families": [f"{r.family}: {r.reason}" for r in worst],
                        "quality_fail_on_missing_families": bool(cfg.quality_fail_on_missing_families),
                        "quality_allowed_missing_families": allowed_missing_families,
                        "quality_allowed_dormant_families": allowed_dormant_families,
                    },
                )

    return manifest


@asset(
    name="universe_delisting_registry",
    config_schema=UniverseDelistingRegistryConfig.to_config_schema(),
    description="Builds/updates the per-symbol delisting/listing-status proxy registry parquet used for survivorship gating.",
    group_name="universe",
)
def universe_delisting_registry(context) -> str:
    from dagster_prep_families.partitions import get_symbol_partitions_def

    from src.data_sources.eodhd_provider import get_eodhd_provider
    from src.stage_b.delisting_meta import refresh_delisting_registry_for_symbols

    cfg = UniverseDelistingRegistryConfig(**(getattr(context, "op_config", None) or {}))

    symbols = list(get_symbol_partitions_def().get_partition_keys())
    provider = get_eodhd_provider()
    df = refresh_delisting_registry_for_symbols(
        provider,
        symbols=[str(s).upper() for s in symbols],
        exchange_code=str(cfg.exchange_code or "US").upper(),
        cache_path=str(cfg.cache_path) if cfg.cache_path else None,
        timeout_seconds=int(cfg.timeout_seconds),
        stale_sessions=int(cfg.stale_sessions),
        vol_bad_fraction=float(cfg.vol_bad_fraction),
    )

    # The helper writes to cache_path/default; expose that path.
    implied = (
        str(cfg.cache_path)
        if cfg.cache_path
        else str(Path("data") / "cache" / "eodhd" / f"delisted_companies_{str(cfg.exchange_code or 'US').upper()}.parquet")
    )

    context.log_event(
        AssetMaterialization(
            asset_key="universe_delisting_registry",
            description="Universe delisting registry updated",
            metadata={
                "exchange_code": str(cfg.exchange_code or "US").upper(),
                "symbols": len(symbols),
                "rows": int(len(df)) if df is not None else 0,
                "path": implied,
            },
        )
    )

    return implied


@asset(
    name="universe_registry",
    config_schema=UniverseRegistryAssetConfig.to_config_schema(),
    description=(
        "Builds the authoritative universe_registry parquet used to prevent zombie symbols from entering "
        "panel build, training updates, inference, and portfolio weights."
    ),
    group_name="universe",
)
def universe_registry(context) -> str:
    from dagster_prep_families.partitions import get_symbol_partitions_def

    from src.stage_b.universe_registry import UniverseRegistryConfig, build_universe_registry
    from src.stage_b.universe_selector import DEFAULT_CANDIDATE_UNIVERSE

    import pandas as pd

    cfg = UniverseRegistryAssetConfig(**(getattr(context, "op_config", None) or {}))

    mode = str(cfg.symbols_mode or "default_candidate_universe").lower().strip()
    if mode == "dagster_symbols":
        symbols = list(get_symbol_partitions_def().get_partition_keys())
    else:
        symbols = list(DEFAULT_CANDIDATE_UNIVERSE)

    out_path = (
        Path(str(cfg.out_path))
        if cfg.out_path
        else Path("data") / "cache" / "universe" / "universe_registry.parquet"
    )

    asof = cfg.asof_date
    if not asof:
        asof = pd.Timestamp.utcnow().strftime("%Y-%m-%d")

    reg_cfg = UniverseRegistryConfig(
        exchange_code=str(cfg.exchange_code or "US").upper(),
        stale_sessions=int(cfg.stale_sessions),
        vol_bad_fraction=float(cfg.vol_bad_fraction),
        prices_lookback_days=int(cfg.prices_lookback_days),
    )

    df = build_universe_registry(
        symbols=[str(s).upper() for s in symbols],
        asof_date=str(asof),
        out_path=out_path,
        cfg=reg_cfg,
        delisting_cache_path=str(cfg.delisting_cache_path) if cfg.delisting_cache_path else None,
    )

    context.log_event(
        AssetMaterialization(
            asset_key="universe_registry",
            description="Universe registry written",
            metadata={
                "exchange_code": str(cfg.exchange_code or "US").upper(),
                "asof_date": str(asof),
                "symbols": int(len(symbols)),
                "rows": int(len(df)) if df is not None else 0,
                "eligible_today": int(df["is_eligible_today"].sum()) if df is not None and "is_eligible_today" in df.columns else 0,
                "path": str(out_path),
            },
        )
    )

    return str(out_path)


@asset(
    name="peer_context_snapshot",
    config_schema=PeerContextSnapshotConfig.to_config_schema(),
    description="Builds a universe-wide peer context snapshot parquet (date×symbol) for universe selection/gating.",
    deps=["universe_registry"],
    group_name="universe",
)
def peer_context_snapshot(context) -> str:
    from dagster_prep_families.partitions import get_symbol_partitions_def
    from tools.prep_families import DEFAULT_WF_END, DEFAULT_WF_START

    from src.features.peer_screener_context import build_peer_screener_context_snapshot

    cfg = PeerContextSnapshotConfig(**(getattr(context, "op_config", None) or {}))

    start = cfg.start or str(DEFAULT_WF_START)
    end = cfg.end or str(DEFAULT_WF_END)

    symbols = list(get_symbol_partitions_def().get_partition_keys())
    cache_root = Path(str(cfg.cache_root))

    out_path = (
        Path(str(cfg.out_path))
        if cfg.out_path
        else Path("data") / "cache" / "universe" / f"peer_context_snapshot_h{int(cfg.horizon)}.parquet"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Optional: restrict snapshot to eligible symbols.
    universe = [str(s).upper() for s in symbols]
    if cfg.universe_registry_path:
        try:
            from src.stage_b.universe_registry import eligible_symbols_for_date, load_universe_registry

            reg = load_universe_registry(str(cfg.universe_registry_path))
            eligible = eligible_symbols_for_date(reg, asof_date=str(end), symbols=universe)
            universe = [s for s in universe if s in eligible]
        except Exception:
            pass

    df = build_peer_screener_context_snapshot(
        start=start,
        end=end,
        cache_root=cache_root,
        horizon=int(cfg.horizon),
        universe=universe,
        base_family=str(cfg.base_family or "fin_g6"),
        leakage_safe_shift_sessions=int(cfg.leakage_safe_shift_sessions),
    )
    df.to_parquet(out_path, index=False)

    context.log_event(
        AssetMaterialization(
            asset_key="peer_context_snapshot",
            description="Peer context snapshot written",
            metadata={
                "horizon": int(cfg.horizon),
                "start": str(start),
                "end": str(end),
                "symbols": len(symbols),
                "rows": int(len(df)) if df is not None else 0,
                "path": str(out_path),
            },
        )
    )
    return str(out_path)


@asset_check(
    asset=prep_families_manifest,
    name="merged_parquet_valid",
    description="Verifies cache/features/<SYMBOL>_h<H>_merged.parquet exists and has unique date.",
)
def merged_parquet_valid(context) -> AssetCheckResult:
    cfg = PrepFamiliesConfig(**(getattr(context, "op_config", None) or {}))
    p_symbol, p_horizon = _partition_symbol_horizon(context)
    run_symbol = p_symbol or cfg.symbol
    run_horizon = int(p_horizon) if p_horizon is not None else int(cfg.horizon)

    if cfg.write_merged is False:
        return AssetCheckResult(passed=True, metadata={"skipped": True, "reason": "write_merged=false"})

    merged_path = Path(cfg.merged_out) if cfg.merged_out else FEATURE_PANEL_DIR / f"{run_symbol}_h{int(run_horizon)}_merged.parquet"

    if not merged_path.exists():
        return AssetCheckResult(
            passed=False,
            metadata={
                "merged_path": str(merged_path),
                "exists": False,
            },
        )

    import pandas as pd

    df = pd.read_parquet(merged_path, columns=["date"])
    if "date" not in df.columns:
        return AssetCheckResult(
            passed=False,
            metadata={
                "merged_path": str(merged_path),
                "exists": True,
                "reason": "missing date column",
            },
        )

    dates = pd.to_datetime(df["date"], errors="coerce")
    dup_count = int(dates.duplicated().sum())
    null_count = int(dates.isna().sum())

    return AssetCheckResult(
        passed=(dup_count == 0),
        metadata={
            "merged_path": str(merged_path),
            "rows": int(len(df)),
            "null_date": null_count,
            "duplicate_date": dup_count,
        },
    )


@asset_check(
    asset=prep_families_manifest,
    name="merged_parquet_real_data_quality",
    description=(
        "Checks merged parquet for proxy/stub usage (has_data/proxy flags) and for degenerate real features "
        "(all-zero / all-null / zero-variance) per family."
    ),
)
def merged_parquet_real_data_quality(context) -> AssetCheckResult:
    cfg = PrepFamiliesConfig(**(getattr(context, "op_config", None) or {}))
    p_symbol, p_horizon = _partition_symbol_horizon(context)
    if p_symbol:
        cfg.symbol = p_symbol
    if p_horizon is not None:
        cfg.horizon = p_horizon

    if cfg.write_merged is False:
        return AssetCheckResult(passed=True, metadata={"skipped": True, "reason": "write_merged=false"})

    merged_path = Path(cfg.merged_out) if cfg.merged_out else FEATURE_PANEL_DIR / f"{cfg.symbol}_h{int(cfg.horizon)}_merged.parquet"
    if not merged_path.exists():
        return AssetCheckResult(passed=False, metadata={"merged_path": str(merged_path), "exists": False})

    thresholds = QualityThresholds(
        min_variance_threshold=float(cfg.quality_min_variance_threshold),
        max_zero_percentage=float(cfg.quality_max_zero_percentage),
        max_null_percentage=float(cfg.quality_max_null_percentage),
        min_has_data_mean=float(cfg.quality_min_has_data_mean),
        max_proxy_flag_mean=float(cfg.quality_max_proxy_flag_mean),
        has_data_tail_rows=int(getattr(cfg, "quality_has_data_tail_rows", 252)),
        min_has_data_nonzero_tail=int(getattr(cfg, "quality_min_has_data_nonzero_tail", 3)),
        max_rows=cfg.quality_max_rows,
    )

    allowed_dormant_families = [
        t.strip()
        for t in str(getattr(cfg, "quality_allowed_dormant_families", "") or "").split(",")
        if t.strip()
    ]

    allowed_missing_families = [
        t.strip()
        for t in str(getattr(cfg, "quality_allowed_missing_families", "") or "").split(",")
        if t.strip()
    ]

    import pandas as pd

    df = pd.read_parquet(merged_path)
    passed, results = validate_panel_by_known_families(
        panel=df,
        thresholds=thresholds,
        allowed_dormant_families=allowed_dormant_families,
        fail_on_missing=bool(getattr(cfg, "quality_fail_on_missing_families", False)),
        allowed_missing_families=allowed_missing_families,
    )

    failed = [r for r in results if not r.passed]
    worst = failed[:25]

    dormant_allowed = [r for r in results if r.passed and str(r.reason).startswith("allowed dormant")]
    dormant_preview = dormant_allowed[:25]

    metadata = {
        "merged_path": str(merged_path),
        "rows": int(len(df)),
        "cols": int(df.shape[1]),
        "families_checked": int(len(results)),
        "families_failed": int(len(failed)),
        "failed_families": [f"{r.family}: {r.reason}" for r in worst],
        "allowed_dormant_families": allowed_dormant_families,
        "allowed_missing_families": allowed_missing_families,
        "dormant_allowed_hits": int(len(dormant_allowed)),
        "dormant_allowed_examples": [f"{r.family}: {r.reason}" for r in dormant_preview],
        "thresholds": {
            "min_variance_threshold": thresholds.min_variance_threshold,
            "max_zero_percentage": thresholds.max_zero_percentage,
            "max_null_percentage": thresholds.max_null_percentage,
            "min_has_data_mean": thresholds.min_has_data_mean,
            "max_proxy_flag_mean": thresholds.max_proxy_flag_mean,
            "max_rows": thresholds.max_rows,
        },
    }

    return AssetCheckResult(passed=passed, metadata=metadata)
