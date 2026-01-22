from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Iterable

from dagster import (
    AssetSelection,
    DefaultSensorStatus,
    Definitions,
    RunRequest,
    define_asset_job,
    load_assets_from_modules,
    schedule,
    sensor,
)

from . import assets as assets_module
from . import family_assets as family_assets_module
from .partitions import get_symbol_horizon_partitions_def, get_symbol_partitions_def


symbol_partitions_def = get_symbol_partitions_def()
symbol_horizon_partitions_def = get_symbol_horizon_partitions_def()


def _day_utc() -> str:
    return datetime.now(timezone.utc).date().isoformat()


def _automation_mode() -> str:
    # Default to sensor-driven automation (preferred).
    return (os.getenv("DAGSTER_PREP_AUTOMATION_MODE") or "sensor").strip().lower()


AUTOMATION_MODE = _automation_mode()


def _parse_symbol_horizon_from_partition_key(partition_key: str) -> tuple[str | None, str | None]:
    """Best-effort parse of (symbol, horizon) from a multi-partition key.

    Dagster multi-partition keys are commonly rendered like:
      "horizon=63|symbol=AAPL" (order may vary)

    Some internal tooling may use a compact form like:
      "63|AAPL" or "AAPL|63"
    """

    s = (partition_key or "").strip()
    if not s:
        return None, None

    sym: str | None = None
    hor: str | None = None

    # Canonical Dagster form: "dim=value|dim=value".
    if "=" in s:
        parts: dict[str, str] = {}
        for seg in s.split("|"):
            if "=" not in seg:
                continue
            k, v = seg.split("=", 1)
            parts[k.strip()] = v.strip()
        sym = parts.get("symbol")
        hor = parts.get("horizon")
    else:
        # Compact fallback form: "63|AAPL" or "AAPL|63".
        toks = [t.strip() for t in s.split("|") if t.strip()]
        if len(toks) == 2:
            a, b = toks
            if a.isdigit() and not b.isdigit():
                hor, sym = a, b
            elif b.isdigit() and not a.isdigit():
                hor, sym = b, a

    if sym is not None:
        sym = str(sym).upper().strip() or None
    if hor is not None:
        try:
            hor = str(int(hor))
        except Exception:
            hor = None

    return sym, hor


def _symbol_horizon_tags(partition_key: str) -> dict[str, str]:
    sym, hor = _parse_symbol_horizon_from_partition_key(str(partition_key))
    tags: dict[str, str] = {}
    if sym:
        tags["dagster/symbol"] = sym
    if hor:
        tags["dagster/horizon"] = hor
    return tags

prep_families_job = define_asset_job(
    name="prep_families_job",
    selection=(
        AssetSelection.assets("prep_families_manifest")
        | AssetSelection.checks_for_assets("prep_families_manifest")
    ),
    partitions_def=symbol_horizon_partitions_def,
)


prep_families_families_symbol_job = define_asset_job(
    name="prep_families_families_symbol_job",
    selection="group:families",
    partitions_def=symbol_partitions_def,
)


prep_families_families_horizon_job = define_asset_job(
    name="prep_families_families_horizon_job",
    selection="group:families_horizon",
    partitions_def=symbol_horizon_partitions_def,
)


prep_universe_prereqs_job = define_asset_job(
    name="prep_universe_prereqs_job",
    selection="group:universe",
)


def _schedule_run_config() -> dict:
    # Optional overrides via env vars.
    cfg: dict = {}
    ops_cfg: dict = {}

    for k in [
        "wf_start",
        "wf_end",
        "wf_train_years",
        "wf_step_years",
        "wf_step_days",
        "families",
        "workers",
        "hf_workers",
        "mode",
        "write_merged",
        "merged_out",
        "merged_service",
        "strict",
        "enforce_no_proxy_sources",
        "enforce_no_live_fallback",
        "enforce_eodhd_only",
        "quality_min_variance_threshold",
        "quality_max_zero_percentage",
        "quality_max_null_percentage",
        "quality_min_has_data_mean",
        "quality_max_proxy_flag_mean",
        "quality_max_rows",
        "reuse_symbol_only_cache",
        "preprocess_write_role_splits",
        "alt_signals_staleness_k",
        "mamba_optional_all",
    ]:
        env_key = f"DAGSTER_PREP_{k.upper()}"
        v = os.getenv(env_key)
        if v is None or str(v).strip() == "":
            continue
        # Keep strings; prep_families_manifest will coerce where needed.
        ops_cfg[k] = v

    if ops_cfg:
        cfg["ops"] = {"prep_families_manifest": {"config": ops_cfg}}
    return cfg


@schedule(job=prep_families_job, cron_schedule="0 2 * * *", execution_timezone="UTC")
def prep_families_universe_daily() -> Iterable[RunRequest]:
    """Kicks off one run per (symbol,horizon) partition daily.

    Control universe via:
    - DAGSTER_SYMBOLS="AAPL,MSFT,..." (or DAGSTER_SYMBOLS_FILE)
    - DAGSTER_HORIZONS="63,126"
    """

    run_config = _schedule_run_config()
    day = _day_utc()
    for pk in symbol_horizon_partitions_def.get_partition_keys():
        yield RunRequest(
            run_key=f"prep_families|{day}|{pk}",
            partition_key=pk,
            run_config=run_config,
            tags={"dagster/source": "schedule", "dagster/partition": pk, **_symbol_horizon_tags(pk)},
        )


@schedule(job=prep_families_families_symbol_job, cron_schedule="30 2 * * *", execution_timezone="UTC")
def prep_families_families_symbol_daily() -> Iterable[RunRequest]:
    """Runs symbol-only family assets (no Feast) per symbol partition daily."""

    day = _day_utc()
    for pk in symbol_partitions_def.get_partition_keys():
        yield RunRequest(
            run_key=f"prep_families_families_symbol|{day}|{pk}",
            partition_key=pk,
            run_config={},
            tags={"dagster/source": "schedule", "dagster/partition": pk, "dagster/job": "families_symbol", "dagster/symbol": str(pk).upper()},
        )


@schedule(job=prep_families_families_horizon_job, cron_schedule="45 2 * * *", execution_timezone="UTC")
def prep_families_families_horizon_daily() -> Iterable[RunRequest]:
    """Runs horizon-bound family assets (no Feast) per (symbol,horizon) partition daily."""

    day = _day_utc()
    for pk in symbol_horizon_partitions_def.get_partition_keys():
        yield RunRequest(
            run_key=f"prep_families_families_horizon|{day}|{pk}",
            partition_key=pk,
            run_config={},
            tags={"dagster/source": "schedule", "dagster/partition": pk, "dagster/job": "families_horizon", **_symbol_horizon_tags(pk)},
        )


@schedule(job=prep_universe_prereqs_job, cron_schedule="15 2 * * *", execution_timezone="UTC")
def prep_universe_prereqs_daily() -> Iterable[RunRequest]:
    """Materialize universe registry/snapshot prerequisites daily."""

    yield RunRequest(
        run_key=f"prep_universe_prereqs|{_day_utc()}",
        run_config={},
        tags={"dagster/source": "schedule", "dagster/job": "universe_prereqs"},
    )


@sensor(
    name="prep_families_all_families_sensor",
    minimum_interval_seconds=60,
    default_status=(DefaultSensorStatus.STOPPED if AUTOMATION_MODE == "schedule" else DefaultSensorStatus.RUNNING),
    jobs=[
        prep_families_job,
        prep_families_families_symbol_job,
        prep_families_families_horizon_job,
    ],
)
def prep_families_all_families_sensor():
    """Launch family preparation runs.

    Default behavior is to launch the single `prep_families_job` (one run per
    (symbol,horizon) partition). This path writes the unified merged parquet and
    contains the cache-reuse logic.

    Set `DAGSTER_PREP_SENSOR_MODE=families` to restore the legacy two-job flow:
    - symbol-only families (partitioned by symbol)
    - horizon-bound families (partitioned by (horizon|symbol))
    """

    mode = (os.getenv("DAGSTER_PREP_SENSOR_MODE") or "manifest").strip().lower()

    # Use a stable day string to dedupe via run_key.
    day = datetime.now(timezone.utc).date().isoformat()

    if mode != "families":
        run_config = _schedule_run_config()
        for pk in symbol_horizon_partitions_def.get_partition_keys():
            yield RunRequest(
                job_name=prep_families_job.name,
                run_key=f"prep_families_all|manifest|{day}|{pk}",
                partition_key=pk,
                run_config=run_config,
                tags={"dagster/source": "sensor", "dagster/job": "manifest", "dagster/partition": pk, **_symbol_horizon_tags(pk)},
            )
        return

    # Legacy: two-job flow (caches only; does not write merged parquet).
    for symbol in symbol_partitions_def.get_partition_keys():
        yield RunRequest(
            job_name=prep_families_families_symbol_job.name,
            run_key=f"prep_families_all|families_symbol|{day}|{symbol}",
            partition_key=symbol,
            run_config={},
            tags={"dagster/source": "sensor", "dagster/job": "families_symbol", "dagster/symbol": symbol},
        )

    for pk in symbol_horizon_partitions_def.get_partition_keys():
        yield RunRequest(
            job_name=prep_families_families_horizon_job.name,
            run_key=f"prep_families_all|families_horizon|{day}|{pk}",
            partition_key=pk,
            run_config={},
            tags={"dagster/source": "sensor", "dagster/job": "families_horizon", "dagster/partition": pk, **_symbol_horizon_tags(pk)},
        )


defs = Definitions(
    assets=load_assets_from_modules([assets_module, family_assets_module]),
    jobs=[prep_families_job, prep_families_families_symbol_job, prep_families_families_horizon_job, prep_universe_prereqs_job],
    schedules=(
        [
            prep_families_universe_daily,
            prep_families_families_symbol_daily,
            prep_families_families_horizon_daily,
            prep_universe_prereqs_daily,
        ]
        if AUTOMATION_MODE == "schedule"
        else [prep_universe_prereqs_daily]
    ),
    sensors=([prep_families_all_families_sensor] if AUTOMATION_MODE != "schedule" else []),
)
