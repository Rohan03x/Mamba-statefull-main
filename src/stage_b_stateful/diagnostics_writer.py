"""
Phase-2 Diagnostics Writer.

Writes structured diagnostic artifacts to disk for post-run analysis:
- phase2_trace_<run_id>.parquet: Daily signal chain metrics
- phase2_events_<run_id>.jsonl: Structured event log
- phase2_shadow_<run_id>.parquet: Shadow/ablation returns (backtest only)
- phase2_summary_<run_id>.json: Final metrics + event counts

Copyright 2024-2026. All rights reserved.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Run ID Generation
# ─────────────────────────────────────────────────────────────────────────────

def generate_run_id(
    symbols: List[str],
    horizon: int,
    trial_number: Optional[int] = None,
) -> str:
    """Generate a unique run ID based on symbols, horizon, and timestamp.
    
    Format: h{horizon}_{sym_hash}_{timestamp}[_t{trial}]
    """
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    sym_hash = hashlib.sha256(",".join(sorted(symbols)).encode()).hexdigest()[:8]
    
    run_id = f"h{horizon}_{sym_hash}_{ts}"
    if trial_number is not None:
        run_id = f"{run_id}_t{trial_number}"
    
    return run_id


# ─────────────────────────────────────────────────────────────────────────────
# Trace Parquet Writer
# ─────────────────────────────────────────────────────────────────────────────

def write_trace_parquet(
    traces: List[Dict[str, Any]],
    run_id: str,
    output_dir: Optional[Path] = None,
) -> Optional[Path]:
    """Write daily traces to parquet file.
    
    Args:
        traces: List of DailyTracePayload dicts (from trace.to_dict())
        run_id: Unique run identifier
        output_dir: Output directory (default: cache/debugging/)
    
    Returns:
        Path to written file, or None on failure
    """
    if not traces:
        logger.warning("[diagnostics] No traces to write")
        return None
    
    if output_dir is None:
        output_dir = Path("cache/debugging")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Convert traces to DataFrame
        # Filter out complex nested fields that don't serialize well to parquet
        scalar_traces = []
        for t in traces:
            scalar_t = {}
            for k, v in t.items():
                if isinstance(v, (int, float, str, bool, type(None))):
                    scalar_t[k] = v
                elif isinstance(v, (list, tuple)) and len(v) <= 10:
                    # Small lists can be serialized as strings
                    scalar_t[k] = json.dumps(v)
                elif isinstance(v, np.ndarray) and v.size <= 10:
                    scalar_t[k] = json.dumps(v.tolist())
                # Skip large arrays
            scalar_traces.append(scalar_t)
        
        df = pd.DataFrame(scalar_traces)
        
        # Convert date column to datetime if present
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
            df = df.set_index("date")
        
        output_path = output_dir / f"phase2_trace_{run_id}.parquet"
        df.to_parquet(output_path, compression="snappy")
        
        logger.info("[diagnostics] 💾 Wrote trace parquet: %s (%d rows)", output_path, len(df))
        return output_path
    
    except Exception as e:
        logger.error("[diagnostics] Failed to write trace parquet: %s", e)
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Events JSONL Writer
# ─────────────────────────────────────────────────────────────────────────────

def write_events_jsonl(
    events: List[Dict[str, Any]],
    run_id: str,
    output_dir: Optional[Path] = None,
) -> Optional[Path]:
    """Write events to JSONL file (one event per line).
    
    Args:
        events: List of StructuredEvent dicts
        run_id: Unique run identifier
        output_dir: Output directory (default: cache/debugging/)
    
    Returns:
        Path to written file, or None on failure
    """
    if not events:
        logger.warning("[diagnostics] No events to write")
        return None
    
    if output_dir is None:
        output_dir = Path("cache/debugging")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        output_path = output_dir / f"phase2_events_{run_id}.jsonl"
        
        with open(output_path, "w", encoding="utf-8") as f:
            for event in events:
                # Ensure serializable
                event_clean = {}
                for k, v in event.items():
                    if isinstance(v, np.ndarray):
                        event_clean[k] = v.tolist()
                    elif isinstance(v, (np.integer, np.floating)):
                        event_clean[k] = float(v)
                    else:
                        event_clean[k] = v
                f.write(json.dumps(event_clean, default=str) + "\n")
        
        logger.info("[diagnostics] 💾 Wrote events JSONL: %s (%d events)", output_path, len(events))
        return output_path
    
    except Exception as e:
        logger.error("[diagnostics] Failed to write events JSONL: %s", e)
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Shadow/Ablation Parquet Writer
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ShadowRun:
    """Shadow/ablation run results for comparison."""
    name: str  # e.g., "no_overlay", "no_policy", "half_gross"
    returns: pd.Series  # Daily returns
    weights: Optional[pd.DataFrame] = None  # Optional daily weights
    description: str = ""


def write_shadow_parquet(
    shadows: List[ShadowRun],
    baseline_returns: pd.Series,
    run_id: str,
    output_dir: Optional[Path] = None,
) -> Optional[Path]:
    """Write shadow/ablation returns to parquet for comparison.
    
    Args:
        shadows: List of ShadowRun objects
        baseline_returns: Primary strategy returns
        run_id: Unique run identifier
        output_dir: Output directory (default: cache/debugging/)
    
    Returns:
        Path to written file, or None on failure
    """
    if not shadows:
        logger.warning("[diagnostics] No shadow runs to write")
        return None
    
    if output_dir is None:
        output_dir = Path("cache/debugging")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Build combined DataFrame
        df = pd.DataFrame({"baseline": baseline_returns})
        
        for shadow in shadows:
            # Align to baseline index
            aligned = shadow.returns.reindex(baseline_returns.index).fillna(0.0)
            df[shadow.name] = aligned
        
        output_path = output_dir / f"phase2_shadow_{run_id}.parquet"
        df.to_parquet(output_path, compression="snappy")
        
        logger.info("[diagnostics] 💾 Wrote shadow parquet: %s (%d shadows)", output_path, len(shadows))
        return output_path
    
    except Exception as e:
        logger.error("[diagnostics] Failed to write shadow parquet: %s", e)
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Summary JSON Writer
# ─────────────────────────────────────────────────────────────────────────────

def write_summary_json(
    portfolio_metrics: Dict[str, float],
    event_counts: Dict[str, int],
    run_id: str,
    symbols: List[str],
    horizon: int,
    config: Optional[Dict[str, Any]] = None,
    output_dir: Optional[Path] = None,
) -> Optional[Path]:
    """Write summary JSON with final metrics and event counts.
    
    Args:
        portfolio_metrics: Dict with sharpe, max_drawdown, turnover, etc.
        event_counts: Dict mapping event code to count
        run_id: Unique run identifier
        symbols: List of symbols in the run
        horizon: Forecast horizon
        config: Optional config dict to include
        output_dir: Output directory (default: cache/debugging/)
    
    Returns:
        Path to written file, or None on failure
    """
    if output_dir is None:
        output_dir = Path("cache/debugging")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        summary = {
            "run_id": run_id,
            "timestamp": datetime.now().isoformat(),
            "symbols": symbols,
            "horizon": horizon,
            "metrics": {
                k: float(v) if isinstance(v, (int, float, np.integer, np.floating)) else v
                for k, v in portfolio_metrics.items()
            },
            "event_counts": event_counts,
            "total_events": sum(event_counts.values()),
        }
        
        # Add config subset (exclude large/sensitive values)
        if config:
            safe_config = {}
            for k, v in config.items():
                if isinstance(v, (int, float, str, bool)):
                    safe_config[k] = v
            summary["config"] = safe_config
        
        output_path = output_dir / f"phase2_summary_{run_id}.json"
        
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, default=str)
        
        logger.info("[diagnostics] 💾 Wrote summary JSON: %s", output_path)
        return output_path
    
    except Exception as e:
        logger.error("[diagnostics] Failed to write summary JSON: %s", e)
        return None


# ─────────────────────────────────────────────────────────────────────────────
# All-in-One Writer
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class DiagnosticsBundle:
    """All diagnostic artifacts from a Phase-2 run."""
    run_id: str
    trace_path: Optional[Path] = None
    events_path: Optional[Path] = None
    shadow_path: Optional[Path] = None
    summary_path: Optional[Path] = None
    z_explainer_path: Optional[Path] = None  # Daily z-explainer logs


# ─────────────────────────────────────────────────────────────────────────────
# Z-Explainer Logs Writer
# ─────────────────────────────────────────────────────────────────────────────

def write_z_explainer_logs(
    z_explainer_logs: List[Dict[str, Any]],
    run_id: str,
    output_dir: Optional[Path] = None,
) -> Optional[Path]:
    """Write z-explainer daily logs to JSON file.
    
    Args:
        z_explainer_logs: List of daily z-explainer dicts with fidelity metrics
        run_id: Unique run identifier
        output_dir: Output directory (default: cache/debugging/)
    
    Returns:
        Path to written file, or None on failure
    """
    if not z_explainer_logs:
        logger.warning("[diagnostics] No z-explainer logs to write")
        return None
    
    if output_dir is None:
        output_dir = Path("cache/debugging")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Compute summary statistics
        reliable_days = [d for d in z_explainer_logs if d.get("fidelity", {}).get("is_reliable", False)]
        suppressed_days = [d for d in z_explainer_logs if d.get("explanation_suppressed", False)]
        
        correlations = [d.get("fidelity", {}).get("correlation", 0.0) for d in z_explainer_logs]
        dir_agrees = [d.get("fidelity", {}).get("directional_agreement", 0.0) for d in z_explainer_logs]
        
        summary = {
            "total_days": len(z_explainer_logs),
            "reliable_days": len(reliable_days),
            "suppressed_days": len(suppressed_days),
            "mean_correlation": np.mean(correlations) if correlations else 0.0,
            "mean_directional_agreement": np.mean(dir_agrees) if dir_agrees else 0.0,
            "fidelity_threshold_corr": 0.3,
            "fidelity_threshold_dir_agree": 0.55,
        }
        
        # Serialize logs (handle numpy arrays)
        serializable_logs = []
        for log_entry in z_explainer_logs:
            clean_entry = {}
            for k, v in log_entry.items():
                if isinstance(v, np.ndarray):
                    clean_entry[k] = v.tolist()
                elif isinstance(v, (np.integer, np.floating)):
                    clean_entry[k] = float(v)
                elif isinstance(v, dict):
                    clean_entry[k] = {
                        sk: float(sv) if isinstance(sv, (np.integer, np.floating)) else sv
                        for sk, sv in v.items()
                    }
                else:
                    clean_entry[k] = v
            serializable_logs.append(clean_entry)
        
        output = {
            "run_id": run_id,
            "summary": summary,
            "daily_logs": serializable_logs,
        }
        
        output_path = output_dir / f"phase2_z_explainer_{run_id}.json"
        
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, default=str)
        
        logger.info(
            "[diagnostics] 💾 Wrote z-explainer logs: %s (%d days, %.0f%% reliable)",
            output_path,
            len(z_explainer_logs),
            100 * len(reliable_days) / max(1, len(z_explainer_logs)),
        )
        return output_path
    
    except Exception as e:
        logger.error("[diagnostics] Failed to write z-explainer logs: %s", e)
        return None


def write_all_diagnostics(
    *,
    run_id: str,
    traces: Optional[List[Dict[str, Any]]] = None,
    events: Optional[List[Dict[str, Any]]] = None,
    event_counts: Optional[Dict[str, int]] = None,
    portfolio_metrics: Optional[Dict[str, float]] = None,
    shadows: Optional[List[ShadowRun]] = None,
    baseline_returns: Optional[pd.Series] = None,
    symbols: Optional[List[str]] = None,
    horizon: int = 63,
    config: Optional[Dict[str, Any]] = None,
    z_explainer_logs: Optional[List[Dict[str, Any]]] = None,
    output_dir: Optional[Path] = None,
) -> DiagnosticsBundle:
    """Write all diagnostic artifacts in one call.
    
    Args:
        run_id: Unique run identifier
        traces: List of daily trace dicts
        events: List of event dicts
        event_counts: Dict mapping event code to count
        portfolio_metrics: Final portfolio metrics
        shadows: List of ShadowRun objects for ablation comparison
        baseline_returns: Primary strategy returns
        symbols: List of symbols
        horizon: Forecast horizon
        config: Config dict
        z_explainer_logs: List of daily z-explainer logs with fidelity metrics
        output_dir: Output directory
    
    Returns:
        DiagnosticsBundle with paths to all written files
    """
    if output_dir is None:
        output_dir = Path("cache/debugging")
    
    bundle = DiagnosticsBundle(run_id=run_id)
    
    # Write traces
    if traces:
        bundle.trace_path = write_trace_parquet(traces, run_id, output_dir)
    
    # Write events
    if events:
        bundle.events_path = write_events_jsonl(events, run_id, output_dir)
    
    # Write shadows
    if shadows and baseline_returns is not None:
        bundle.shadow_path = write_shadow_parquet(shadows, baseline_returns, run_id, output_dir)
    
    # Write z-explainer logs
    if z_explainer_logs:
        bundle.z_explainer_path = write_z_explainer_logs(z_explainer_logs, run_id, output_dir)
    
    # Write summary
    if portfolio_metrics is not None:
        bundle.summary_path = write_summary_json(
            portfolio_metrics=portfolio_metrics,
            event_counts=event_counts or {},
            run_id=run_id,
            symbols=symbols or [],
            horizon=horizon,
            config=config,
            output_dir=output_dir,
        )
    
    return bundle


# ─────────────────────────────────────────────────────────────────────────────
# Readers (for dashboard)
# ─────────────────────────────────────────────────────────────────────────────

def load_trace_parquet(path: Path) -> pd.DataFrame:
    """Load trace parquet file."""
    return pd.read_parquet(path)


def load_events_jsonl(path: Path) -> List[Dict[str, Any]]:
    """Load events from JSONL file."""
    events = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                events.append(json.loads(line))
    return events


def load_shadow_parquet(path: Path) -> pd.DataFrame:
    """Load shadow parquet file."""
    return pd.read_parquet(path)


def load_summary_json(path: Path) -> Dict[str, Any]:
    """Load summary JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def find_latest_run(output_dir: Optional[Path] = None) -> Optional[str]:
    """Find the most recent run_id in the output directory."""
    if output_dir is None:
        output_dir = Path("cache/debugging")
    
    if not output_dir.exists():
        return None
    
    # Find all summary files and extract run_ids
    summaries = list(output_dir.glob("phase2_summary_*.json"))
    if not summaries:
        return None
    
    # Sort by modification time (most recent first)
    summaries.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    
    # Extract run_id from filename
    latest = summaries[0].stem.replace("phase2_summary_", "")
    return latest
