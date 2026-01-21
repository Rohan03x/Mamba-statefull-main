from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from dagster import (
    AssetCheckResult,
    AssetDep,
    AssetKey,
    AssetMaterialization,
    Config,
    MultiToSingleDimensionPartitionMapping,
    asset,
    asset_check,
)

from .partitions import get_symbol_horizon_partitions_def, get_symbol_partitions_def


HORIZON_BOUND_FAMILIES: set[str] = {"quantile_forecast", "calibration", "online_learning"}


class FamilyRunConfig(Config):
    # Walk-forward-ish range used for consolidated caches (stage-a mode).
    wf_start: Optional[str] = None
    wf_end: Optional[str] = None

    # Safety: ignore wf overrides unless explicitly enabled.
    allow_wf_override: bool = False

    # Required by prepare_families signature; used for windows in stage-b, but harmless in stage-a.
    wf_train_years: Optional[int] = None
    wf_step_years: Optional[int] = None
    wf_step_days: Optional[int] = None

    workers: int = 1
    hf_workers: Optional[int] = None
    strict: bool = False

    # For symbol-only assets, we still need a horizon value for compute-time
    # config (some generators accept a horizon even if the cache path is
    # symbol-only).
    horizon: int = 63

    # Validation thresholds
    min_variance_threshold: float = 1e-10
    max_zero_percentage: float = 0.95
    max_null_percentage: float = 0.90

    # Extra fallback/proxy enforcement
    fail_on_fallback_columns: bool = True


def _partition_symbol(context) -> Optional[str]:
    pk = getattr(context, "partition_key", None)
    if not pk:
        return None
    if isinstance(pk, str):
        return pk.upper()
    try:
        return str(pk).upper()
    except Exception:
        return None


def _partition_symbol_horizon(context) -> tuple[Optional[str], Optional[int]]:
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

    return None, None


def _cache_path(symbol: str, horizon: int, family: str, split: str = "") -> Path:
    """Generate cache file path for individual family caches.
    
    - Horizon-linked families: cache/symbols/<SYMBOL>/h<H>/<family>.parquet
    - Symbol-only HF blocks: cache/symbols/<SYMBOL>/hf/<family>.parquet
    - doc_embedding_novelty_hf: cache/shared/doc_embedding/doc_embedding_novelty_hf.parquet (shared across all symbols)
    
    Split parameter is deprecated - ignored for clean caching structure.
    """
    from src.features.family_spec import SYMBOL_ONLY_HF_BLOCKS

    sym_upper = symbol.upper()
    
    # doc_embedding is symbol-independent - use shared cache
    if family == "doc_embedding_novelty_hf":
        return Path("cache/shared/doc_embedding") / "doc_embedding_novelty_hf.parquet"

    if family in set(SYMBOL_ONLY_HF_BLOCKS):
        # Symbol-only HF families: no horizon, under symbols/<SYM>/hf/
        return Path("cache/symbols") / sym_upper / "hf" / f"{family}.parquet"

    # Standard horizon-linked families: under symbols/<SYM>/h<H>/
    return Path("cache/symbols") / sym_upper / f"h{int(horizon)}" / f"{family}.parquet"


def _list_families_for_dagster() -> List[str]:
    # Keep this lightweight: reuse the existing family resolver.
    from tools.prep_families import DEFAULT_EXCLUDE_FAMILIES, resolve_families

    base_families, hf_modules, hf_blocks, meta_families = resolve_families(
        "all",
        exclude_families=list(DEFAULT_EXCLUDE_FAMILIES),
    )

    # Meta families are derived. We still expose hf_agg as a Dagster asset so its
    # dependencies are explicit in the asset graph.
    families = list(dict.fromkeys(base_families + hf_modules + hf_blocks + meta_families))

    # Dagster-only exclusions (do not expose as assets).
    dagster_exclude = {
        "news_sentiment_hf",
        # Universe-wide / cross-sectional snapshot; not a per-symbol family cache.
        "peer_screener_context",
    }

    out: List[str] = []
    for f in families:
        if not f or not str(f).strip():
            continue
        name = str(f).strip()
        if name in dagster_exclude:
            continue
        out.append(name)
    return out


def _default_range() -> tuple[str, str, int, Optional[int], Optional[int]]:
    from tools.prep_families import DEFAULT_WF_END, DEFAULT_WF_START, DEFAULT_WF_STEP_DAYS, DEFAULT_WF_TRAIN_YEARS

    return (
        str(DEFAULT_WF_START),
        str(DEFAULT_WF_END),
        int(DEFAULT_WF_TRAIN_YEARS),
        None,
        int(DEFAULT_WF_STEP_DAYS),
    )


def build_family_assets() -> List[Any]:
    symbol_partitions_def = get_symbol_partitions_def()
    symbol_horizon_partitions_def = get_symbol_horizon_partitions_def()
    families = _list_families_for_dagster()
    out: List[Any] = []

    families_set = set(families)

    def _sanitize_family_key(name: str) -> str:
        return "".join([c if (c.isalnum() or c == "_") else "_" for c in name.lower()])

    def _asset_key_for_family(name: str) -> AssetKey:
        return AssetKey([f"family_{_sanitize_family_key(name)}"])

    # HF blocks should have explicit Dagster deps so they only run after their prerequisites.
    # We derive these deps from the same source of truth used by prep_families.
    from tools import prep_families as _pf

    hf_blocks_set = set(getattr(_pf, "DEFAULT_HF_BLOCKS", ()))
    _family_dependencies = getattr(_pf, "family_dependencies", None)
    if _family_dependencies is None:
        raise RuntimeError("tools.prep_families.family_dependencies is missing; cannot derive HF block prerequisites")

    def _is_horizon_bound(name: str) -> bool:
        # Base horizon-bound families are always horizon-partitioned.
        if name in HORIZON_BOUND_FAMILIES:
            return True

        # META families (hf_agg) are horizon-bound by construction.
        if name == "hf_agg":
            return True

        # HF blocks: only forecast_hf remains horizon-bound; the rest are symbol-only.
        if name in hf_blocks_set:
            from src.features.family_spec import SYMBOL_ONLY_HF_BLOCKS

            return name not in set(SYMBOL_ONLY_HF_BLOCKS)

        return False

    def _make_family_defs(fam: str) -> List[Any]:
        fam_key = "family_" + _sanitize_family_key(fam)

        horizon_bound = _is_horizon_bound(fam)
        partitions_def = symbol_horizon_partitions_def if horizon_bound else symbol_partitions_def

        deps: List[Any] = []
        for dep in list(_family_dependencies(fam)):
            if dep not in families_set:
                continue
            dep_horizon_bound = _is_horizon_bound(dep)

            # Same-shape deps can wire directly.
            if dep_horizon_bound == horizon_bound:
                deps.append(_asset_key_for_family(dep))
                continue

            # Special case: hf_agg is (symbol,horizon) and depends on symbol-only HF blocks.
            if fam == "hf_agg" and horizon_bound and (not dep_horizon_bound):
                deps.append(
                    AssetDep(
                        asset=_asset_key_for_family(dep),
                        partition_mapping=MultiToSingleDimensionPartitionMapping(partition_dimension_name="symbol"),
                    )
                )

        if fam == "hf_agg":
            @asset(
                name=fam_key,
                partitions_def=symbol_horizon_partitions_def,
                config_schema=FamilyRunConfig.to_config_schema(),
                deps=deps,
                group_name="families_horizon",
                description=(
                    "Materialize derived META family 'hf_agg' after all HF blocks are ready. "
                    "Writes a cached parquet for debugging/inspection and makes Dagster deps explicit."
                ),
            )
            def _hf_agg_asset(context) -> Dict[str, str]:
                import pandas as pd

                from src.features.aggregator_panel import build_panel

                cfg = FamilyRunConfig(**(getattr(context, "op_config", None) or {}))
                p_symbol, p_horizon = _partition_symbol_horizon(context)
                if not p_symbol or p_horizon is None:
                    raise RuntimeError("hf_agg asset requires (symbol,horizon) partitions")

                run_symbol = p_symbol
                run_horizon = int(p_horizon)
                default_start, default_end, *_ = _default_range()
                allow_wf_override = bool(cfg.allow_wf_override) or (
                    os.environ.get("DAGSTER_PREP_ALLOW_WF_OVERRIDE", "0") == "1"
                )
                wf_start = (cfg.wf_start if (allow_wf_override and cfg.wf_start) else default_start)
                wf_end = (cfg.wf_end if (allow_wf_override and cfg.wf_end) else default_end)

                sym_lower = run_symbol.lower()
                sym_upper = run_symbol.upper()
                cache_dir = Path("cache/symbols") / sym_upper / f"h{run_horizon}"
                cache_dir.mkdir(parents=True, exist_ok=True)

                panel = build_panel(
                    symbol=run_symbol,
                    start=str(wf_start),
                    end=str(wf_end),
                    families=["hf_agg"],
                    cache_dir=cache_dir,
                    horizon=int(run_horizon),
                    stage="B",
                    view="both",
                    window_idx=None,
                )

                if panel is None or panel.empty:
                    raise RuntimeError(f"hf_agg build returned empty for {run_symbol} h{int(run_horizon)}")

                df = panel.copy()
                if "date" not in df.columns:
                    df = df.reset_index().rename(columns={"index": "date"})
                df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()
                df = df.dropna(subset=["date"]).sort_values("date")
                df = df.drop_duplicates(subset=["date"], keep="last")

                # Expand to calendar-daily index and forward-fill (avoids weekend/holiday NaNs).
                date_index = pd.date_range(df["date"].min(), df["date"].max(), freq="D")
                df = df.set_index("date").reindex(date_index).ffill().bfill().reset_index().rename(columns={"index": "date"})

                train_path = _cache_path(run_symbol, int(run_horizon), "hf_agg", "train")
                valid_path = _cache_path(run_symbol, int(run_horizon), "hf_agg", "valid")
                df.to_parquet(train_path, index=False)
                df.to_parquet(valid_path, index=False)

                context.log_event(
                    AssetMaterialization(
                        asset_key=fam_key,
                        description=f"hf_agg cached for {run_symbol} h{int(run_horizon)}",
                        metadata={
                            "symbol": run_symbol,
                            "horizon": int(run_horizon),
                            "wf_start": str(wf_start),
                            "wf_end": str(wf_end),
                            "allow_wf_override": bool(allow_wf_override),
                            "train_cache": str(train_path),
                            "valid_cache": str(valid_path),
                            "rows": int(len(df)),
                        },
                    )
                )

                return {"train": str(train_path), "valid": str(valid_path)}

            @asset_check(
                asset=_hf_agg_asset,
                name="quality",
                description="Sanity check: hf_agg governance columns exist and are not mostly-null.",
            )
            def _hf_agg_quality_check(context) -> AssetCheckResult:
                import pandas as pd

                cfg = FamilyRunConfig(**(getattr(context, "op_config", None) or {}))
                p_symbol, p_horizon = _partition_symbol_horizon(context)
                if not p_symbol or p_horizon is None:
                    return AssetCheckResult(passed=False, metadata={"reason": "missing partition key"})

                run_symbol = p_symbol
                run_horizon = int(p_horizon)
                train_path = _cache_path(run_symbol, int(run_horizon), "hf_agg", "train")
                valid_path = _cache_path(run_symbol, int(run_horizon), "hf_agg", "valid")

                if not train_path.exists():
                    return AssetCheckResult(passed=False, metadata={"reason": "missing train cache", "path": str(train_path)})

                df = pd.read_parquet(train_path)
                # Check for governance columns instead of score/conf
                required = ["hf_agg_has_data", "hf_agg_activity", "hf_agg_days_since_update"]
                missing = [c for c in required if c not in df.columns]
                if missing:
                    return AssetCheckResult(passed=False, metadata={"reason": f"missing cols: {missing}"})

                nan_frac_has_data = float(df["hf_agg_has_data"].isna().mean())
                nan_frac_activity = float(df["hf_agg_activity"].isna().mean())
                passed = (nan_frac_has_data <= float(cfg.max_null_percentage)) and (nan_frac_activity <= float(cfg.max_null_percentage))
                return AssetCheckResult(
                    passed=passed,
                    metadata={
                        "train_cache": str(train_path),
                        "valid_cache": str(valid_path),
                        "nan_frac_has_data": nan_frac_has_data,
                        "nan_frac_activity": nan_frac_activity,
                    },
                )

            return [_hf_agg_asset, _hf_agg_quality_check]

        @asset(
            name=fam_key,
            partitions_def=partitions_def,
            config_schema=FamilyRunConfig.to_config_schema(),
            deps=deps,
            group_name="families_horizon" if horizon_bound else "families",
            description=f"Materialize consolidated caches for family '{fam}' via tools/prep_families.prepare_families (stage-a, no Feast).",
        )
        def _family_asset(context) -> Dict[str, str]:
            from tools.prep_families import prepare_families

            cfg = FamilyRunConfig(**(getattr(context, "op_config", None) or {}))
            if horizon_bound:
                p_symbol, p_horizon = _partition_symbol_horizon(context)
                if not p_symbol or p_horizon is None:
                    raise RuntimeError("This asset requires (symbol,horizon) partitions")
                run_symbol = p_symbol
                run_horizon = int(p_horizon)
            else:
                run_symbol = _partition_symbol(context)
                if not run_symbol:
                    raise RuntimeError("This asset requires symbol partitions")
                run_horizon = int(cfg.horizon)

            default_start, default_end, default_train_years, default_step_years, default_step_days = _default_range()

            allow_wf_override = bool(cfg.allow_wf_override) or (os.environ.get("DAGSTER_PREP_ALLOW_WF_OVERRIDE", "0") == "1")

            wf_start = (cfg.wf_start if (allow_wf_override and cfg.wf_start) else default_start)
            wf_end = (cfg.wf_end if (allow_wf_override and cfg.wf_end) else default_end)
            wf_train_years = int(cfg.wf_train_years) if cfg.wf_train_years is not None else int(default_train_years)
            wf_step_years = int(cfg.wf_step_years) if cfg.wf_step_years is not None else default_step_years
            wf_step_days = int(cfg.wf_step_days) if cfg.wf_step_days is not None else default_step_days

            os.environ.setdefault("PREP_FAMILIES_STRICT_TRACKC", "0")

            result = prepare_families(
                symbol=run_symbol,
                horizon=int(run_horizon),
                wf_start=wf_start,
                wf_end=wf_end,
                wf_train_years=wf_train_years,
                wf_step_years=wf_step_years,
                wf_step_days=wf_step_days,
                families=fam,
                strict=bool(cfg.strict),
                workers=int(cfg.workers),
                hf_workers=cfg.hf_workers,
                mode="stage-a",
                write_merged=False,
                merged_out=None,
                merged_service=None,
                manifest_scope="all",
                manifest_exclude_families=["news_sentiment_hf"],
                sequential_mode=True,  # Per-symbol sequential execution (no spawns)
            )

            train_path = _cache_path(run_symbol, int(run_horizon), fam, "train")
            valid_path = _cache_path(run_symbol, int(run_horizon), fam, "valid")

            context.log_event(
                AssetMaterialization(
                    asset_key=fam_key,
                    description=f"family '{fam}' cached for {run_symbol} h{int(run_horizon)}",
                    metadata={
                        "symbol": run_symbol,
                        "horizon": int(run_horizon),
                        "family": fam,
                        "wf_start": str(wf_start),
                        "wf_end": str(wf_end),
                        "allow_wf_override": bool(allow_wf_override),
                        "success": bool(result.success),
                        "train_cache": str(train_path),
                        "valid_cache": str(valid_path),
                        "manifest": str(result.completeness_manifest_path) if result.completeness_manifest_path else "",
                    },
                )
            )

            if not result.success:
                raise RuntimeError(f"prepare_families failed for family={fam} {run_symbol} h{int(run_horizon)}")

            return {"train": str(train_path), "valid": str(valid_path)}

        @asset_check(
            asset=_family_asset,
            name="quality",
            description="Validates non-proxy features have variance and are not mostly zeros/nulls; flags fallback/proxy columns.",
        )
        def _family_quality_check(context) -> AssetCheckResult:
            import pandas as pd

            from tools.prep_families import validate_family_quality

            cfg = FamilyRunConfig(**(getattr(context, "op_config", None) or {}))
            if horizon_bound:
                p_symbol, p_horizon = _partition_symbol_horizon(context)
                if not p_symbol or p_horizon is None:
                    return AssetCheckResult(passed=False, metadata={"reason": "missing partition key"})
                run_symbol = p_symbol
                run_horizon = int(p_horizon)
            else:
                run_symbol = _partition_symbol(context)
                if not run_symbol:
                    return AssetCheckResult(passed=False, metadata={"reason": "missing partition key"})
                run_horizon = int(cfg.horizon)

            train_path = _cache_path(run_symbol, int(run_horizon), fam, "train")
            valid_path = _cache_path(run_symbol, int(run_horizon), fam, "valid")

            ok_train, reason_train = validate_family_quality(
                feature_cache_path=train_path,
                family=fam,
                min_variance_threshold=float(cfg.min_variance_threshold),
                max_zero_percentage=float(cfg.max_zero_percentage),
                max_null_percentage=float(cfg.max_null_percentage),
            )
            ok_valid, reason_valid = validate_family_quality(
                feature_cache_path=valid_path,
                family=fam,
                min_variance_threshold=float(cfg.min_variance_threshold),
                max_zero_percentage=float(cfg.max_zero_percentage),
                max_null_percentage=float(cfg.max_null_percentage),
            )

            fallback_hits: Dict[str, int] = {}
            if cfg.fail_on_fallback_columns:
                for split, path in [("train", train_path), ("valid", valid_path)]:
                    if not path.exists():
                        fallback_hits[split] = -1
                        continue
                    df = pd.read_parquet(path)
                    cols = [str(c) for c in df.columns]
                    fb_cols = [c for c in cols if "fallback" in c.lower() or "proxy" in c.lower()]
                    hit = 0
                    for c in fb_cols:
                        s = df[c]
                        if s.dtype == bool:
                            hit += int(bool(s.fillna(False).any()))
                        else:
                            try:
                                hit += int((pd.to_numeric(s, errors="coerce").fillna(0.0) != 0.0).any())
                            except Exception:
                                continue
                    fallback_hits[split] = hit

            passed = bool(ok_train and ok_valid)
            if cfg.fail_on_fallback_columns and any(v and v > 0 for v in fallback_hits.values()):
                passed = False

            return AssetCheckResult(
                passed=passed,
                metadata={
                    "family": fam,
                    "train": reason_train,
                    "valid": reason_valid,
                    "train_cache": str(train_path),
                    "valid_cache": str(valid_path),
                    "fallback_proxy_hits": fallback_hits,
                },
            )

        return [_family_asset, _family_quality_check]

    for fam in families:
        out.extend(_make_family_defs(fam))

    return out


# Materialize at import time so Dagster can discover these assets.
FAMILY_ASSETS = build_family_assets()
