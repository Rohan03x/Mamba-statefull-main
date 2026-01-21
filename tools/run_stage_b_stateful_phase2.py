from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
import time


# Allow running as: `python tools/run_stage_b_stateful_phase2.py ...`
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from src.stage_b_stateful.phase2_stateful import (
    DEFAULT_PHASE2_HOLDOUT_END,
    DEFAULT_PHASE2_HOLDOUT_START,
    DEFAULT_PHASE2_OOS_END,
    DEFAULT_PHASE2_OOS_START,
    DEFAULT_PHASE2_TRAIN_END,
    DEFAULT_PHASE2_TRAIN_START,
    run_phase2_stateful_optuna,
)
from src.stage_b.pipeline import GLOBAL_OPTUNA_SYMBOLS
from src.stage_b.universe_selector import DEFAULT_CANDIDATE_UNIVERSE, DEFAULT_CORE_UNIVERSE
from src.stage_b_stateful.group_map import build_symbol_group_map_from_eodhd_cache


DEFAULT_BEST_TRIAL_JSON = Path(
    "artifacts/optuna_studies/BEST_TRIAL_trial9_stage_b_global_mamba_h63_1510621193_reset_5ae41193_m_avg_score_focus_80b7b9ff.json"
)


def _auto_default_phase2_family_weights_path() -> Path | None:
    """Best-effort default Stage-A weights artifact for Phase 2.

    If the user does not pass --phase2-family-weights-path, we prefer the latest
    GLOBAL13 real Stage-A artifact under artifacts/debug.
    """

    base = Path("artifacts/debug")
    if not base.exists():
        return None

    # Prefer the latest run directory by name (timestamp embedded).
    candidates = sorted(base.glob("stage_a_global13_real_*/family_weights_best.json"))
    if not candidates:
        return None
    try:
        # Use parent directory name sorting (lexicographic timestamp) as primary signal.
        candidates.sort(key=lambda p: str(p.parent.name))
    except Exception:
        pass
    chosen = candidates[-1]
    return chosen if chosen.exists() else None


def _build_runtime_overrides_for_preset(
    *,
    preset: str,
    symbols: list[str],
    group_map_path: Path | None,
    capital_usd: float | None,
) -> dict[str, object]:
    p = str(preset or "").lower().strip()
    if p in {"", "none"}:
        return {}

    # Shared HF-discipline posture.
    cfg: dict[str, object] = {
        # Enforced posture (should not be overridden per-trial).
        "phase2_engine": "walkforward_v2",
        "phase2_require_v2": True,
        "phase2_safety_overlays": True,
        # Safety overlay defaults (HF discipline).
        "phase2_dd_throttle_1": 0.05,
        "phase2_dd_throttle_2": 0.10,
        "phase2_dd_kill": 0.25,
        "phase2_vol_throttle_mult": 2.0,
        "phase2_vol_kill_mult": 3.0,

        # Survivorship control: delisting metadata (NOT model features).
        # Default posture: enabled + strict. Provide a local parquet cache to avoid
        # relying on slow/unstable network calls during runs.
        "phase2_delisting_meta_enabled": True,
        "phase2_delisting_meta_strict": True,
        "phase2_delisting_meta_exchange": "US",
        # Default to refresh (bounded to run symbols) to ensure strict mode doesn't
        # silently run without a registry when the cache hasn't been built yet.
        "phase2_delisting_meta_refresh": True,
        "phase2_delisting_meta_cache_path": str(Path("data") / "cache" / "eodhd" / "delisted_companies_US.parquet"),
        "phase2_delisting_meta_timeout_seconds": 60,
    }

    # Presets currently only differ by naming; keep behavior identical unless
    # explicitly overridden by a trial_cfg.
    if p not in {"research", "production"}:
        raise ValueError(f"unknown preset: {preset}")

    # If the user explicitly provides a group map path, pass it through.
    if group_map_path is not None:
        cfg["phase2_group_map_path"] = str(group_map_path)

    # Liquidity constraints require a capital base and explicit limits.
    # We only pass through the capital base here; per-trial limits are tuned.
    if capital_usd is not None and float(capital_usd) > 0:
        cfg["phase2_capital_usd"] = float(capital_usd)

    return cfg


def main() -> None:
    ap = argparse.ArgumentParser(description="Run Stage-B Phase-2 stateful Optuna (refinement or full search).")
    ap.add_argument(
        "--search-mode",
        default="full",
        choices=["refinement", "full"],
        help="Search mode: refinement (legacy Phase-2) or full (broad Stage-B space evaluated statefully).",
    )
    ap.add_argument(
        "--best-trial-json",
        default=None,
        type=Path,
        help="Path to exported Stage-B best-trial bundle JSON (used only in --search-mode refinement).",
    )
    ap.add_argument(
        "--symbol",
        default="AAPL",
        help="Legacy primary symbol (kept for compatibility). If --symbols is omitted, defaults to the fixed GLOBAL13 universe.",
    )
    ap.add_argument(
        "--symbols",
        default=None,
        help="Comma-separated symbols for portfolio eval. If omitted, uses the fixed GLOBAL13 universe from src.stage_b.pipeline.",
    )

    ap.add_argument(
        "--universe-mode",
        default="full",
        choices=["fixed_global13", "full", "core_satellite"],
        help=(
            "Universe mode: fixed_global13 (trade the fixed GLOBAL13 list), "
            "full (default; trade the full candidate universe daily), "
            "or core_satellite (rotate CORE ∪ SATELLITE monthly-ish by event score)."
        ),
    )
    ap.add_argument(
        "--universe-core",
        default=None,
        help="Optional comma-separated CORE symbols (overrides default CORE).",
    )
    ap.add_argument(
        "--universe-candidates",
        default=None,
        help="Optional comma-separated candidate symbols (overrides default candidates).",
    )
    ap.add_argument(
        "--universe-satellite-k",
        default=12,
        type=int,
        help="(Deprecated) Kept for compatibility; selector now uses satellite min/max + stateful sizing.",
    )
    ap.add_argument(
        "--universe-lookback",
        default=63,
        type=int,
        help="Lookback sessions for event score averaging. Default: 63.",
    )
    ap.add_argument(
        "--universe-min-history",
        default=63,
        type=int,
        help="Minimum sessions of history required for a candidate to be eligible. Default: 63.",
    )
    ap.add_argument(
        "--universe-rebalance-every",
        default=21,
        type=int,
        help="Universe selection cadence in sessions (monthly aligned with U=21). Default: 21.",
    )
    ap.add_argument(
        "--universe-min-stay",
        default=21,
        type=int,
        help="Minimum satellite stay in sessions. Default: 21.",
    )
    ap.add_argument(
        "--universe-max-stay",
        default=63,
        type=int,
        help="Maximum satellite stay in sessions (guardrail; currently not forced-remove). Default: 63.",
    )
    ap.add_argument(
        "--universe-min-adv-usd",
        default=None,
        type=float,
        help="Optional minimum ADV$ (best-effort; only applied if close/volume are present in panels).",
    )
    ap.add_argument(
        "--universe-panel-dir",
        default="artifacts/prediction_tapes/feature_panels",
        help="Directory containing consolidated TrackC panels from tools/prep_families.py.",
    )
    ap.add_argument(
        "--universe-state-path",
        default=None,
        help="Optional path to persist satellite membership state (recommended). Default: artifacts/meta_optimizer/universe_selector_state_h<H>.json",
    )
    ap.add_argument("--horizon", default=63, type=int)

    ap.add_argument(
        "--phase2-label-id",
        default="base",
        choices=["base", "voladj"],
        help="Phase-2 label variant (study identity; not Optuna-sampled).",
    )

    ap.add_argument(
        "--phase2-family-weights-path",
        default=None,
        type=Path,
        help="Optional Stage-A weights JSON to use in Phase2 (supports time-varying schedule). If provided, overrides any default Stage-A artifact discovery.",
    )

    ap.add_argument("--train-start", default=DEFAULT_PHASE2_TRAIN_START)
    ap.add_argument("--train-end", default=DEFAULT_PHASE2_TRAIN_END)
    ap.add_argument("--oos-start", default=DEFAULT_PHASE2_OOS_START)
    ap.add_argument("--oos-end", default=DEFAULT_PHASE2_OOS_END)

    ap.add_argument("--holdout-start", default=DEFAULT_PHASE2_HOLDOUT_START)
    ap.add_argument("--holdout-end", default=DEFAULT_PHASE2_HOLDOUT_END)

    ap.add_argument("--n-trials", default=100, type=int)
    # Default study name is intentionally stable for research discipline.
    # The core driver will suffix the label id as: __label_<id>.
    ap.add_argument("--study-name", default="stage_b_stateful_phase2")
    ap.add_argument("--study-db", default=None, type=Path)

    ap.add_argument(
        "--no-prune",
        action="store_true",
        help="Disable all pruning; every trial runs full evaluation (step=2 objective).",
    )
    ap.add_argument(
        "--prune-update-sessions",
        default=21,
        type=int,
        help="Fold size for intermediate reporting/pruning (in trading sessions). Default: 21.",
    )
    ap.add_argument(
        "--prune-warmup-folds",
        default=12,
        type=int,
        help="Do not allow Hyperband pruning until this many folds have been reported. Default: 12.",
    )
    ap.add_argument(
        "--safety-prune-after-folds",
        default=12,
        type=int,
        help="Optional safety prune: after N folds, prune if Sharpe < floor AND MaxDD > ceiling. Set 0 to disable. Default: 12.",
    )
    ap.add_argument(
        "--safety-prune-sharpe-floor",
        default=0.0,
        type=float,
        help="Safety prune Sharpe floor (prune if Sharpe < floor). Default: 0.0.",
    )
    ap.add_argument(
        "--safety-prune-maxdd-ceil",
        default=0.12,
        type=float,
        help="Safety prune MaxDD ceiling (prune if MaxDD > ceiling). Default: 0.12.",
    )
    ap.add_argument(
        "--tpe-top-frac",
        default=0.2,
        type=float,
        help="TPE good-set fraction (approx top X%% of trials). Default: 0.2.",
    )
    ap.add_argument(
        "--deterministic-all",
        action="store_true",
        help="Force deterministic training for all trials (seed derived from STAGE_B_PHASE2_SEED + trial.number).",
    )

    ap.add_argument("--aggregation-rule", default="equal_weight", choices=["equal_weight", "explicit_weights"])
    ap.add_argument(
        "--portfolio-weights",
        default=None,
        help="Optional explicit weights like AAPL=0.5,MSFT=0.5 (requires --aggregation-rule explicit_weights)",
    )
    ap.add_argument(
        "--turnover-penalty",
        default=None,
        type=float,
        help="Objective turnover regularization λ_turn. If omitted: full-mode defaults to 0.2; refinement defaults to 0.0.",
    )

    ap.add_argument(
        "--preset",
        default="research",
        choices=["none", "research", "production"],
        help="Preset defaults for Phase2 runtime controls. research (default) enables v2-only + safety overlays; production tightens caps.",
    )
    ap.add_argument(
        "--group-map-path",
        default=None,
        type=Path,
        help="Optional symbol->group CSV/JSON for group caps. If omitted for research/production, auto-generates from local EODHD cache when available.",
    )
    ap.add_argument(
        "--capital-usd",
        default=100000,
        type=float,
        help="Capital base in USD for ADV-based liquidity constraints. Default: $100K.",
    )

    args = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if args.search_mode == "refinement" and args.best_trial_json is None:
        args.best_trial_json = DEFAULT_BEST_TRIAL_JSON

    # Default behavior (no flags): stable study naming for the control experiment.
    # - study_name becomes: phase2_v16_<mode>_h<H> (v16 = hedge fund governance with vol deployment penalty)
    # - Optuna DB defaults to artifacts/optuna_studies/<study_name>__label_<id>.db (set inside core)
    # NOTE: We intentionally do NOT auto-timestamp the study/db here, because that
    # breaks the discipline of "one label per study name". Users can still pass
    # --study-db / --study-name explicitly for isolated runs.
    if args.study_name == "stage_b_stateful_phase2":
        mode = str(args.search_mode).lower().strip()
        # v16 = hedge fund governance: fixed max_gross=1.75, vol underutil penalty, looser beta constraint
        args.study_name = f"phase2_v16_{mode}_h{int(args.horizon)}"

    print(
        f"search_mode={args.search_mode} n_trials={args.n_trials} "
        f"phase2_label_id={args.phase2_label_id} study_name={args.study_name} study_db={args.study_db}"
    )

    symbols: list[str]
    universe_runtime_overrides: dict[str, object] = {}
    universe_mode = str(args.universe_mode).lower().strip()

    if args.symbols:
        symbols = [s.strip().upper() for s in str(args.symbols).split(",") if s.strip()]
    else:
        # Default: run the broader candidate universe (36) so training can be pooled.
        # - full: trade the full eligible set daily.
        # - core_satellite: trade CORE ∪ SATELLITE rotation.
        if universe_mode in {"full", "core_satellite"}:
            symbols = sorted({str(s).upper() for s in DEFAULT_CANDIDATE_UNIVERSE})
        else:
            symbols = list(GLOBAL_OPTUNA_SYMBOLS)

    # Optional dynamic universe gating (runtime, Phase2 walkforward_v2).
    # Only applies when the user did NOT explicitly provide --symbols.
    if not args.symbols and universe_mode == "core_satellite":
        core = (
            [s.strip().upper() for s in str(args.universe_core).split(",") if s.strip()]
            if args.universe_core
            else list(DEFAULT_CORE_UNIVERSE)
        )
        candidates = (
            [s.strip().upper() for s in str(args.universe_candidates).split(",") if s.strip()]
            if args.universe_candidates
            else list(DEFAULT_CANDIDATE_UNIVERSE)
        )
        # Training stays pooled across the full candidate set; trading will be
        # gated by a piecewise-constant universe schedule inside Phase2.
        symbols = sorted({s.strip().upper() for s in ([*core, *candidates]) if s.strip()})

        # Pass runtime gating params to Phase2.
        state_path = (
            Path(str(args.universe_state_path))
            if args.universe_state_path
            else Path("artifacts/meta_optimizer") / f"universe_selector_state_h{int(args.horizon)}.json"
        )
        universe_runtime_overrides = {
            "phase2_runtime_universe_mode": "core_satellite",
            "phase2_runtime_universe_core_symbols": core,
            "phase2_runtime_universe_candidate_symbols": candidates,
            "phase2_runtime_universe_lookback_sessions": int(args.universe_lookback),
            "phase2_runtime_universe_min_history_sessions": int(args.universe_min_history),
            "phase2_runtime_universe_rebalance_every_sessions": int(args.universe_rebalance_every),
            "phase2_runtime_universe_min_stay_sessions": int(args.universe_min_stay),
            "phase2_runtime_universe_max_stay_sessions": int(args.universe_max_stay),
            "phase2_runtime_universe_decay_fraction": 0.5,
            "phase2_runtime_universe_remove_abs_threshold": 0.25,
            "phase2_runtime_universe_min_adv_usd": (float(args.universe_min_adv_usd) if args.universe_min_adv_usd is not None else None),
            "phase2_runtime_universe_state_path": str(state_path),
        }

    # Full-universe gating (runtime, Phase2 walkforward_v2): trade the full eligible set daily.
    # Only applies when the user did NOT explicitly provide --symbols.
    if not args.symbols and universe_mode == "full":
        universe_runtime_overrides = {
            "phase2_runtime_universe_mode": "full",
        }

    portfolio_weights = None
    if args.portfolio_weights:
        portfolio_weights = {}
        for part in str(args.portfolio_weights).split(","):
            part = part.strip()
            if not part:
                continue
            k, v = part.split("=", 1)
            portfolio_weights[k.strip().upper()] = float(v)

    from src.stage_b_stateful.phase2_stateful import Phase2ObjectiveSpec, Phase2SearchSpec

    # Preset wiring: generate a group map from local EODHD cache if needed.
    preset = str(args.preset).lower().strip()
    group_map_path: Path | None = args.group_map_path
    if preset in {"research", "production"} and group_map_path is None:
        default_map = Path("artifacts/meta_optimizer/phase2_symbol_group_map_gic_sector.csv")
        if default_map.exists():
            group_map_path = default_map
        else:
            try:
                # Use centralized cache paths
                try:
                    from src.cache_paths import resolve_eodhd_cache_dir
                    eodhd_cache = resolve_eodhd_cache_dir()
                except ImportError:
                    eodhd_cache = Path("data/cache/eodhd")
                res = build_symbol_group_map_from_eodhd_cache(
                    cache_dir=eodhd_cache,
                    symbols=symbols,
                    out_path=default_map,
                    group_key_preference=("GicSector", "Sector"),
                )
                if res.group_by_symbol:
                    group_map_path = default_map
            except Exception:
                group_map_path = None

    runtime_overrides = _build_runtime_overrides_for_preset(
        preset=preset,
        symbols=symbols,
        group_map_path=group_map_path,
        capital_usd=(float(args.capital_usd) if args.capital_usd is not None else None),
    )

    if universe_runtime_overrides:
        runtime_overrides.update(universe_runtime_overrides)

    # Label choice is a study identity (one label per study). The core Optuna driver
    # will suffix the study_name and validate mismatch if user pre-suffixed.
    runtime_overrides["phase2_label_id"] = str(args.phase2_label_id).lower().strip()

    if args.phase2_family_weights_path is not None:
        runtime_overrides["phase2_family_weights_path"] = str(args.phase2_family_weights_path)
    else:
        auto_path = _auto_default_phase2_family_weights_path()
        if auto_path is not None:
            runtime_overrides["phase2_family_weights_path"] = str(auto_path)
            logging.getLogger(__name__).info(
                "Auto-defaulting Phase2 Stage-A weights: %s (override with --phase2-family-weights-path)",
                str(auto_path),
            )

    # Semantic run mode wiring (do not rely on manual toggles).
    # - full: insight > speed
    # - refine: keep compute lighter
    mode = str(args.search_mode).lower().strip()
    if mode == "full":
        runtime_overrides["phase2_run_mode"] = "full"
        runtime_overrides["phase2_diag_pre_post_update_eval"] = True
        runtime_overrides["phase2_write_diagnostics"] = True
        runtime_overrides["phase2_enable_diagnostics"] = True
        runtime_overrides["phase2_prune_guards"] = True
        # Objective defaults for full-mode studies.
        runtime_overrides.setdefault("phase2_obj_lambda_flat", 0.05)
    else:
        runtime_overrides["phase2_run_mode"] = "refine"
        runtime_overrides["phase2_diag_pre_post_update_eval"] = False
        runtime_overrides["phase2_write_diagnostics"] = False
        runtime_overrides["phase2_enable_diagnostics"] = True
        runtime_overrides["phase2_prune_guards"] = False

    # Default λ_turn depends on mode when not explicitly provided.
    turnover_penalty = float(args.turnover_penalty) if args.turnover_penalty is not None else (0.2 if mode == "full" else 0.0)

    result = run_phase2_stateful_optuna(
        best_trial_json=args.best_trial_json,
        symbol=args.symbol,
        symbols=symbols,
        horizon=args.horizon,
        train_start=args.train_start,
        train_end=args.train_end,
        oos_start=args.oos_start,
        oos_end=args.oos_end,
        holdout_start=args.holdout_start,
        holdout_end=args.holdout_end,
        n_trials=args.n_trials,
        study_db_path=args.study_db,
        study_name=args.study_name,
        aggregation_rule=args.aggregation_rule,
        portfolio_weights=portfolio_weights,
        search_spec=Phase2SearchSpec(mode=str(args.search_mode)),
        objective_spec=Phase2ObjectiveSpec(turnover_penalty=float(turnover_penalty)),
        no_prune=bool(args.no_prune),
        deterministic_all=bool(args.deterministic_all),
        prune_update_sessions=int(args.prune_update_sessions),
        prune_warmup_folds=int(args.prune_warmup_folds),
        safety_prune_after_folds=int(args.safety_prune_after_folds),
        safety_prune_sharpe_floor=float(args.safety_prune_sharpe_floor),
        safety_prune_maxdd_ceiling=float(args.safety_prune_maxdd_ceil),
        tpe_top_fraction=float(args.tpe_top_frac),
        runtime_overrides=runtime_overrides,
    )

    print(result)


if __name__ == "__main__":
    main()
