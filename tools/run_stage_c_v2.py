#!/usr/bin/env python3
"""
Stage C Runner v2
=================

Complete threshold optimization and trading policy system.

Usage:
    # Full analysis (static + walk-forward + ML)
    python tools/run_stage_c_v2.py --symbol AAPL --horizon 63 --full
    
    # Static threshold search only
    python tools/run_stage_c_v2.py --symbol AAPL --horizon 63 --static
    
    # Walk-forward threshold training
    python tools/run_stage_c_v2.py --symbol AAPL --horizon 63 --walkforward
    
    # ML meta-learner
    python tools/run_stage_c_v2.py --symbol AAPL --horizon 63 --ml
    
    # Custom threshold grid
    python tools/run_stage_c_v2.py --symbol AAPL --horizon 63 --static --thresholds 0.05,0.10,0.15,0.20
"""

import argparse
import json
import logging
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from src.stage_c.stage_c_policy_v2 import (
    load_prediction_tape,
    static_threshold_search,
    walkforward_threshold_training,
    build_meta_learner_dataset,
    train_meta_learner,
    run_dynamic_policy,
    sweep_confidence_thresholds,
    plot_threshold_sweep,
    plot_walkforward_equity,
    run_full_stage_c_analysis,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("stage_c_v2")


def get_tape_path(symbol: str, horizon: int) -> Path:
    """Get default prediction tape path."""
    return PROJECT_ROOT / "artifacts" / "prediction_tapes" / f"{symbol.upper()}_h{horizon}_predictions.parquet"


def print_separator(title: str = "", char: str = "=", width: int = 80):
    """Print a separator line."""
    if title:
        padding = (width - len(title) - 2) // 2
        print(f"\n{char * padding} {title} {char * padding}")
    else:
        print(char * width)


def print_table(df: pd.DataFrame, title: str = ""):
    """Print DataFrame as formatted table."""
    if title:
        print(f"\n{title}")
        print("-" * 80)
    print(df.to_string(index=False))


def main():
    parser = argparse.ArgumentParser(
        description="Stage C v2: Complete Threshold Optimization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    # Required arguments
    parser.add_argument("--symbol", required=True, help="Stock symbol (e.g., AAPL)")
    parser.add_argument("--horizon", type=int, required=True, help="Forecast horizon (e.g., 63)")
    
    # Mode selection
    parser.add_argument("--full", action="store_true", help="Run full analysis (static + WF + ML)")
    parser.add_argument("--static", action="store_true", default=True, help="Run static threshold search (default: ON)")
    parser.add_argument("--walkforward", "--wf", action="store_true", default=True, help="Run walk-forward training (default: ON)")
    parser.add_argument("--ml", action="store_true", help="Run ML meta-learner")
    parser.add_argument("--no-static", action="store_true", help="Disable static threshold search")
    parser.add_argument("--no-walkforward", "--no-wf", action="store_true", help="Disable walk-forward training")
    
    # Parameters
    parser.add_argument("--thresholds", type=str, help="Comma-separated thresholds (e.g., 0.05,0.10,0.15)")
    parser.add_argument("--alpha", type=float, default=0.3, help="Coverage penalty alpha (default: 0.3)")
    parser.add_argument("--min-coverage", type=float, default=0.10, help="Minimum coverage (default: 0.10)")
    parser.add_argument("--max-dd", type=float, default=-0.50, help="Maximum drawdown (default: -0.50)")
    parser.add_argument("--min-train-folds", type=int, default=5, help="Min folds for WF training (default: 5)")
    
    # ML options
    parser.add_argument("--model-type", type=str, default="logistic", 
                        choices=["logistic", "lightgbm"], help="ML model type (default: logistic)")
    
    # Output
    parser.add_argument("--save", action="store_true", default=True, help="Save results and plots (default: ON)")
    parser.add_argument("--no-save", action="store_true", help="Disable saving results")
    parser.add_argument("--tape-path", type=str, help="Custom path to prediction tape")
    
    args = parser.parse_args()
    
    # Determine output directory
    output_dir = PROJECT_ROOT / "artifacts" / "stage_c_reports" / f"{args.symbol}_h{args.horizon}"
    
    # Load prediction tape
    tape_path = Path(args.tape_path) if args.tape_path else get_tape_path(args.symbol, args.horizon)
    
    if not tape_path.exists():
        logger.error(f"❌ Prediction tape not found: {tape_path}")
        logger.info("Run Stage B first:")
        logger.info(f"  python tools/run_stage_b.py --symbols {args.symbol} --horizons {args.horizon}")
        sys.exit(1)
    
    tape = load_prediction_tape(tape_path)
    
    # Header
    print_separator(f"STAGE C v2: {args.symbol} H{args.horizon}")
    print(f"Prediction tape: {len(tape)} rows, {tape['fold_id'].nunique()} folds")
    print(f"Date range: {tape['timestamp'].min().date()} to {tape['timestamp'].max().date()}")
    print(f"Score range: [{tape['score'].min():.4f}, {tape['score'].max():.4f}]")
    
    # Handle --no-* flags
    if args.no_static:
        args.static = False
    if args.no_walkforward:
        args.walkforward = False
    if args.no_save:
        args.save = False
    
    # --full enables all modes
    if args.full:
        args.static = True
        args.walkforward = True
        args.ml = True
    
    # Parse thresholds
    if args.thresholds:
        t_values = [float(t.strip()) for t in args.thresholds.split(",")]
    else:
        t_values = np.arange(0.02, 0.52, 0.02).tolist()
    
    # Periods per year for the horizon (252 trading days / horizon)
    periods_per_year = 252 / args.horizon
    
    # =========================================================================
    # 1. STATIC THRESHOLD SEARCH
    # =========================================================================
    if args.static:
        print_separator("1. STATIC THRESHOLD SEARCH")
        
        sweep_df, best_result = static_threshold_search(
            tape,
            t_values=t_values,
            periods_per_year=periods_per_year,
            coverage_penalty_alpha=args.alpha,
            min_coverage=args.min_coverage,
            max_dd_limit=args.max_dd,
        )
        
        # Print sweep table
        display_df = sweep_df[["T", "Coverage", "Sharpe", "MaxDD", "HitRate", "AnnReturn", "Objective"]].copy()
        display_df["Coverage"] = display_df["Coverage"].apply(lambda x: f"{x:.1%}")
        display_df["MaxDD"] = display_df["MaxDD"].apply(lambda x: f"{x:.1%}")
        display_df["HitRate"] = display_df["HitRate"].apply(lambda x: f"{x:.1%}")
        display_df["AnnReturn"] = display_df["AnnReturn"].apply(lambda x: f"{x:.1%}")
        display_df["Sharpe"] = display_df["Sharpe"].apply(lambda x: f"{x:.3f}")
        display_df["Objective"] = display_df["Objective"].apply(lambda x: f"{x:.3f}")
        print_table(display_df, "Threshold Sweep Results:")
        
        # Best result
        print_separator("Best Static Threshold", char="-")
        print(f"  T*           = {best_result.threshold:.3f}")
        print(f"  Sharpe       = {best_result.sharpe:.3f}")
        print(f"  Coverage     = {best_result.coverage:.1%}")
        print(f"  MaxDD        = {best_result.maxdd:.1%}")
        print(f"  HitRate      = {best_result.hitrate:.1%}")
        print(f"  Ann Return   = {best_result.ann_return:.1%}")
        print(f"  Objective    = {best_result.objective:.3f}")
        print(f"  Long/Short   = {best_result.n_long}/{best_result.n_short}")
        
        if args.save:
            output_dir.mkdir(parents=True, exist_ok=True)
            sweep_df.to_csv(output_dir / "static_sweep.csv", index=False)
            plot_threshold_sweep(sweep_df, output_dir / "static_sweep.png")
            print(f"\n📊 Saved: {output_dir}/static_sweep.csv, static_sweep.png")
    
    # =========================================================================
    # 2. WALK-FORWARD THRESHOLD TRAINING
    # =========================================================================
    if args.walkforward:
        print_separator("2. WALK-FORWARD THRESHOLD TRAINING")
        
        wf_result = walkforward_threshold_training(
            tape,
            t_values=t_values,
            min_train_folds=args.min_train_folds,
            periods_per_year=periods_per_year,
            coverage_penalty_alpha=args.alpha,
        )
        
        print_separator("Walk-Forward Results", char="-")
        print(f"  Sharpe       = {wf_result.aggregate_sharpe:.3f}")
        print(f"  Ann Return   = {wf_result.aggregate_return:.1%}")
        print(f"  MaxDD        = {wf_result.aggregate_maxdd:.1%}")
        print(f"  HitRate      = {wf_result.aggregate_hitrate:.1%}")
        print(f"  Folds        = {len(wf_result.fold_thresholds)}")
        print(f"  T* Range     = [{min(wf_result.fold_thresholds):.2f}, {max(wf_result.fold_thresholds):.2f}]")
        print(f"  Mean T*      = {np.mean(wf_result.fold_thresholds):.3f}")
        
        # Show threshold evolution
        print("\n  Threshold by Fold (last 10):")
        for i, (fold_t, fold_ret) in enumerate(zip(wf_result.fold_thresholds[-10:], wf_result.fold_returns[-10:])):
            fold_idx = len(wf_result.fold_thresholds) - 10 + i
            sign = "+" if fold_ret > 0 else ""
            print(f"    Fold {fold_idx:2d}: T*={fold_t:.2f}, ret={sign}{fold_ret:.1%}")
        
        if args.save:
            output_dir.mkdir(parents=True, exist_ok=True)
            wf_result.trades_df.to_csv(output_dir / "walkforward_trades.csv", index=False)
            plot_walkforward_equity(wf_result, output_dir / "walkforward_equity.png")
            print(f"\n📊 Saved: {output_dir}/walkforward_trades.csv, walkforward_equity.png")
    
    # =========================================================================
    # 3. ML META-LEARNER (with proper OOS evaluation)
    # =========================================================================
    if args.ml:
        print_separator("3. ML META-LEARNER")
        
        # Build dataset
        dataset = build_meta_learner_dataset(tape)
        
        # Split folds for training vs OOS evaluation
        all_folds = sorted(dataset["fold_id"].unique())
        n_folds = len(all_folds)
        n_train = max(5, n_folds // 2)  # Train on first 50% of folds
        train_folds = all_folds[:n_train]
        eval_folds = all_folds[n_train:]
        
        logger.info(f"🔀 Train/Eval split: {len(train_folds)} train folds, {len(eval_folds)} eval folds")
        
        # Train model on ONLY train folds
        train_data = dataset[dataset["fold_id"].isin(train_folds)]
        model, train_metrics = train_meta_learner(
            train_data, 
            model_type=args.model_type,
            task="classification"
        )
        
        print_separator("Training Results (in-sample)", char="-")
        for k, v in train_metrics.items():
            print(f"  {k:<15} = {v:.4f}")
        
        # Sweep confidence thresholds on OOS ONLY
        print(f"\n🧪 Evaluating on {len(eval_folds)} OOS folds (folds {min(eval_folds)}-{max(eval_folds)})")
        conf_sweep = sweep_confidence_thresholds(
            dataset, model, 
            thresholds=np.arange(0.50, 0.90, 0.05).tolist(),
            eval_folds=eval_folds,
        )
        
        print("\nConfidence Threshold Sweep (OOS):")
        display_conf = conf_sweep[["threshold", "coverage", "sharpe", "maxdd", "hitrate"]].copy()
        display_conf["coverage"] = display_conf["coverage"].apply(lambda x: f"{x:.1%}")
        display_conf["maxdd"] = display_conf["maxdd"].apply(lambda x: f"{x:.1%}")
        display_conf["hitrate"] = display_conf["hitrate"].apply(lambda x: f"{x:.1%}")
        display_conf["sharpe"] = display_conf["sharpe"].apply(lambda x: f"{x:.3f}")
        print(display_conf.to_string(index=False))
        
        # Best confidence
        best_conf = conf_sweep.loc[conf_sweep["sharpe"].idxmax(), "threshold"]
        _, ml_metrics = run_dynamic_policy(
            dataset, model, confidence_threshold=best_conf, eval_folds=eval_folds
        )
        
        print_separator("Best ML Policy (OOS)", char="-")
        print(f"  Confidence   = {best_conf:.2f}")
        print(f"  Sharpe       = {ml_metrics['sharpe']:.3f}")
        print(f"  Coverage     = {ml_metrics['coverage']:.1%}")
        print(f"  MaxDD        = {ml_metrics['maxdd']:.1%}")
        print(f"  HitRate      = {ml_metrics['hitrate']:.1%}")
        print(f"  Ann Return   = {ml_metrics['ann_return']:.1%}")
        print(f"  Eval Folds   = {len(eval_folds)}")
        
        if args.save:
            output_dir.mkdir(parents=True, exist_ok=True)
            conf_sweep.to_csv(output_dir / "ml_confidence_sweep.csv", index=False)
            print(f"\n📊 Saved: {output_dir}/ml_confidence_sweep.csv")
    
    # =========================================================================
    # SUMMARY
    # =========================================================================
    print_separator("SUMMARY")
    
    summary = []
    if args.static:
        summary.append(f"  Static T*={best_result.threshold:.2f}: Sharpe={best_result.sharpe:.3f}, Coverage={best_result.coverage:.1%}")
    if args.walkforward:
        summary.append(f"  Walk-Forward: Sharpe={wf_result.aggregate_sharpe:.3f}, HitRate={wf_result.aggregate_hitrate:.1%}")
    if args.ml:
        summary.append(f"  ML (conf={best_conf:.2f}): Sharpe={ml_metrics['sharpe']:.3f}, Coverage={ml_metrics['coverage']:.1%}")
    
    for s in summary:
        print(s)
    
    print_separator()


if __name__ == "__main__":
    main()
