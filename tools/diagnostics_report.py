#!/usr/bin/env python3
"""
Phase-2 Diagnostics Report Generator.

Generates HTML or Markdown reports from Phase-2 diagnostic artifacts:
- Equity curve + drawdown plot
- Rolling Sharpe, rolling turnover, rolling costs
- Calibration score timeline + correlation to returns
- Histogram of z_raw vs z_after_overlays vs z_thr
- Event timeline (stacked bar: veto/kill/throttle counts)
- Policy action usage frequency + performance conditional on action

Usage:
    python diagnostics_report.py                     # Latest run
    python diagnostics_report.py --run-id <run_id>  # Specific run
    python diagnostics_report.py --output html      # HTML output (default)
    python diagnostics_report.py --output markdown  # Markdown output

Copyright 2024-2026. All rights reserved.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import logging
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Plotting imports (with fallback)
try:
    import matplotlib
    matplotlib.use("Agg")  # Non-interactive backend
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from matplotlib.figure import Figure
    _HAS_MPL = True
except ImportError:
    _HAS_MPL = False
    plt = None  # type: ignore
    mdates = None  # type: ignore
    Figure = None  # type: ignore

HAS_MATPLOTLIB = _HAS_MPL

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_OUTPUT_DIR = Path("cache/debugging")
ROLLING_WINDOW = 63  # ~3 months for rolling metrics


# ─────────────────────────────────────────────────────────────────────────────
# Data Loaders
# ─────────────────────────────────────────────────────────────────────────────

def load_trace_df(run_id: str, output_dir: Path) -> Optional[pd.DataFrame]:
    """Load trace parquet for a run."""
    path = output_dir / f"phase2_trace_{run_id}.parquet"
    if path.exists():
        return pd.read_parquet(path)
    return None


def load_events(run_id: str, output_dir: Path) -> List[Dict[str, Any]]:
    """Load events JSONL for a run."""
    path = output_dir / f"phase2_events_{run_id}.jsonl"
    if not path.exists():
        return []
    
    events = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                events.append(json.loads(line))
    return events


def load_shadow_df(run_id: str, output_dir: Path) -> Optional[pd.DataFrame]:
    """Load shadow parquet for a run."""
    path = output_dir / f"phase2_shadow_{run_id}.parquet"
    if path.exists():
        return pd.read_parquet(path)
    return None


def load_summary(run_id: str, output_dir: Path) -> Optional[Dict[str, Any]]:
    """Load summary JSON for a run."""
    path = output_dir / f"phase2_summary_{run_id}.json"
    if not path.exists():
        return None
    
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def find_latest_run(output_dir: Path) -> Optional[str]:
    """Find the most recent run_id in the output directory."""
    if not output_dir.exists():
        return None
    
    summaries = list(output_dir.glob("phase2_summary_*.json"))
    if not summaries:
        # Try traces
        traces = list(output_dir.glob("phase2_trace_*.parquet"))
        if not traces:
            return None
        traces.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return traces[0].stem.replace("phase2_trace_", "")
    
    summaries.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return summaries[0].stem.replace("phase2_summary_", "")


# ─────────────────────────────────────────────────────────────────────────────
# Metric Calculators
# ─────────────────────────────────────────────────────────────────────────────

def compute_rolling_sharpe(returns: pd.Series, window: int = ROLLING_WINDOW) -> pd.Series:
    """Compute rolling annualized Sharpe ratio."""
    roll_mean = returns.rolling(window).mean()
    roll_std = returns.rolling(window).std()
    sharpe = (roll_mean / roll_std.replace(0, np.nan)) * np.sqrt(252)
    return sharpe


def compute_drawdown(equity: pd.Series) -> pd.Series:
    """Compute drawdown series from equity curve."""
    peak = equity.expanding().max()
    dd = (peak - equity) / peak
    return dd


def compute_equity_from_returns(returns: pd.Series) -> pd.Series:
    """Compute equity curve from returns."""
    return (1 + returns).cumprod()


# ─────────────────────────────────────────────────────────────────────────────
# Plot Generators
# ─────────────────────────────────────────────────────────────────────────────

def fig_to_base64(fig: Figure) -> str:
    """Convert matplotlib figure to base64 PNG string."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)
    return b64


def plot_equity_and_drawdown(trace_df: pd.DataFrame) -> Optional[str]:
    """Plot equity curve and drawdown."""
    if not HAS_MATPLOTLIB or trace_df is None:
        return None
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    
    # Equity curve
    if "equity" in trace_df.columns:
        ax1 = axes[0]
        ax1.plot(trace_df.index, trace_df["equity"], color="blue", linewidth=1.5, label="Equity")
        ax1.set_ylabel("Equity")
        ax1.set_title("Equity Curve")
        ax1.legend(loc="upper left")
        ax1.grid(True, alpha=0.3)
        
        # Drawdown
        ax2 = axes[1]
        dd = compute_drawdown(trace_df["equity"])
        ax2.fill_between(trace_df.index, 0, -dd * 100, color="red", alpha=0.5, label="Drawdown")
        ax2.set_ylabel("Drawdown (%)")
        ax2.set_xlabel("Date")
        ax2.legend(loc="lower left")
        ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig_to_base64(fig)


def plot_rolling_metrics(trace_df: pd.DataFrame) -> Optional[str]:
    """Plot rolling Sharpe, turnover, and costs."""
    if not HAS_MATPLOTLIB or trace_df is None:
        return None
    
    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    
    # Rolling Sharpe
    if "pnl_net" in trace_df.columns:
        returns = trace_df["pnl_net"]
        rolling_sharpe = compute_rolling_sharpe(returns)
        axes[0].plot(trace_df.index, rolling_sharpe, color="green", linewidth=1.2)
        axes[0].axhline(y=0, color="gray", linestyle="--", alpha=0.5)
        axes[0].set_ylabel("Rolling Sharpe (63d)")
        axes[0].set_title("Rolling Metrics")
        axes[0].grid(True, alpha=0.3)
    
    # Rolling Turnover
    if "turnover_exec" in trace_df.columns:
        rolling_turn = trace_df["turnover_exec"].rolling(ROLLING_WINDOW).mean()
        axes[1].plot(trace_df.index, rolling_turn, color="orange", linewidth=1.2)
        axes[1].set_ylabel("Avg Turnover (63d)")
        axes[1].grid(True, alpha=0.3)
    elif "turnover_intent" in trace_df.columns:
        rolling_turn = trace_df["turnover_intent"].rolling(ROLLING_WINDOW).mean()
        axes[1].plot(trace_df.index, rolling_turn, color="orange", linewidth=1.2)
        axes[1].set_ylabel("Avg Turnover (63d)")
        axes[1].grid(True, alpha=0.3)
    
    # Rolling Costs
    if "cost_total" in trace_df.columns:
        rolling_cost = trace_df["cost_total"].rolling(ROLLING_WINDOW).mean() * 10000  # bps
        axes[2].plot(trace_df.index, rolling_cost, color="red", linewidth=1.2)
        axes[2].set_ylabel("Avg Cost (bps, 63d)")
        axes[2].set_xlabel("Date")
        axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig_to_base64(fig)


def plot_calibration_timeline(trace_df: pd.DataFrame) -> Optional[str]:
    """Plot calibration score timeline and its correlation to returns."""
    if not HAS_MATPLOTLIB or trace_df is None:
        return None
    
    # Look for calibration-related columns
    calib_col = None
    for col in ["calibration_score", "calib_score", "mamba_calib_score"]:
        if col in trace_df.columns:
            calib_col = col
            break
    
    if calib_col is None:
        return None
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    
    # Calibration score timeline
    ax1 = axes[0]
    ax1.plot(trace_df.index, trace_df[calib_col], color="purple", linewidth=1.2)
    ax1.set_ylabel("Calibration Score")
    ax1.set_title("Mamba Calibration Timeline")
    ax1.grid(True, alpha=0.3)
    
    # Rolling correlation between calibration and returns
    ax2 = axes[1]
    if "pnl_net" in trace_df.columns:
        rolling_corr = trace_df[calib_col].rolling(ROLLING_WINDOW).corr(trace_df["pnl_net"])
        ax2.plot(trace_df.index, rolling_corr, color="teal", linewidth=1.2)
        ax2.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
        ax2.set_ylabel("Corr(Calib, Returns)")
        ax2.set_xlabel("Date")
        ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig_to_base64(fig)


def plot_z_histograms(trace_df: pd.DataFrame) -> Optional[str]:
    """Plot histograms of z_raw, z_after_overlays, z_thr."""
    if not HAS_MATPLOTLIB or trace_df is None:
        return None
    
    # Look for z-related columns
    z_cols = {}
    for prefix, label in [("z_raw", "z_raw"), ("z_overlay", "z_after_overlays"), 
                          ("z_thr", "z_thr"), ("z_thresholded", "z_thr")]:
        # Check for summary stats
        for suffix in ["_mean", "_median", "_max", "_min", ""]:
            col = f"{prefix}{suffix}"
            if col in trace_df.columns:
                z_cols[label] = col
                break
    
    if not z_cols:
        # Try to find any z-related columns
        for col in trace_df.columns:
            if "z_raw" in col.lower():
                z_cols["z_raw"] = col
            elif "z_overlay" in col.lower() or "z_after" in col.lower():
                z_cols["z_after_overlays"] = col
            elif "z_thr" in col.lower():
                z_cols["z_thr"] = col
    
    if not z_cols:
        return None
    
    fig, ax = plt.subplots(figsize=(10, 5))
    
    colors = {"z_raw": "blue", "z_after_overlays": "orange", "z_thr": "green"}
    for label, col in z_cols.items():
        data = trace_df[col].dropna()
        if len(data) > 0:
            ax.hist(data, bins=50, alpha=0.5, label=label, color=colors.get(label, "gray"))
    
    ax.set_xlabel("Z-score Value")
    ax.set_ylabel("Frequency")
    ax.set_title("Distribution of Z-scores (raw → overlays → thresholded)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig_to_base64(fig)


def plot_event_timeline(events: List[Dict[str, Any]]) -> Optional[str]:
    """Plot stacked bar chart of event counts by day."""
    if not HAS_MATPLOTLIB or not events:
        return None
    
    # Group events by date and code
    date_code_counts = defaultdict(lambda: defaultdict(int))
    for event in events:
        date = event.get("date", "unknown")
        code = event.get("code", "UNKNOWN")
        date_code_counts[date][code] += 1
    
    if not date_code_counts:
        return None
    
    # Convert to DataFrame
    df = pd.DataFrame(date_code_counts).T.fillna(0)
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    
    # Filter to interesting event codes (veto, kill, throttle)
    interesting_codes = [c for c in df.columns if any(
        kw in c.upper() for kw in ["VETO", "KILL", "THROTTLE", "COOLDOWN", "ERROR", "CRITICAL"]
    )]
    
    if interesting_codes:
        df = df[interesting_codes]
    
    if df.empty or df.sum().sum() == 0:
        return None
    
    fig, ax = plt.subplots(figsize=(12, 5))
    
    df.plot(kind="bar", stacked=True, ax=ax, width=0.8)
    ax.set_xlabel("Date")
    ax.set_ylabel("Event Count")
    ax.set_title("Event Timeline (Kill/Throttle/Veto Events)")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")
    
    # Reduce x-tick density
    if len(df) > 30:
        step = len(df) // 15
        ax.set_xticks(ax.get_xticks()[::step])
    
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    return fig_to_base64(fig)


def plot_policy_actions(trace_df: pd.DataFrame) -> Optional[str]:
    """Plot policy action usage frequency and performance by action."""
    if not HAS_MATPLOTLIB or trace_df is None:
        return None
    
    # Look for policy action column
    action_col = None
    for col in ["policy_action_idx", "action_idx", "policy_action"]:
        if col in trace_df.columns:
            action_col = col
            break
    
    if action_col is None:
        return None
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Action frequency
    ax1 = axes[0]
    action_counts = trace_df[action_col].value_counts().sort_index()
    ax1.bar(action_counts.index.astype(str), action_counts.values, color="steelblue")
    ax1.set_xlabel("Action Index")
    ax1.set_ylabel("Frequency")
    ax1.set_title("Policy Action Usage")
    ax1.grid(True, alpha=0.3, axis="y")
    
    # Performance by action
    ax2 = axes[1]
    if "pnl_net" in trace_df.columns:
        action_perf = trace_df.groupby(action_col)["pnl_net"].mean() * 252 * 100  # Annualized %
        colors = ["green" if v > 0 else "red" for v in action_perf.values]
        ax2.bar(action_perf.index.astype(str), action_perf.values, color=colors)
        ax2.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
        ax2.set_xlabel("Action Index")
        ax2.set_ylabel("Avg Return (ann. %)")
        ax2.set_title("Performance by Action")
        ax2.grid(True, alpha=0.3, axis="y")
    
    plt.tight_layout()
    return fig_to_base64(fig)


def plot_shadow_comparison(shadow_df: pd.DataFrame) -> Optional[str]:
    """Plot equity curves for baseline vs shadow/ablation runs."""
    if not HAS_MATPLOTLIB or shadow_df is None:
        return None
    
    fig, ax = plt.subplots(figsize=(12, 5))
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(shadow_df.columns)))
    
    for i, col in enumerate(shadow_df.columns):
        equity = compute_equity_from_returns(shadow_df[col])
        linewidth = 2.0 if col == "baseline" else 1.0
        linestyle = "-" if col == "baseline" else "--"
        ax.plot(shadow_df.index, equity, label=col, color=colors[i], 
                linewidth=linewidth, linestyle=linestyle)
    
    ax.set_xlabel("Date")
    ax.set_ylabel("Equity")
    ax.set_title("Baseline vs Shadow/Ablation Runs")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig_to_base64(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Report Generators
# ─────────────────────────────────────────────────────────────────────────────

def generate_html_report(
    run_id: str,
    summary: Optional[Dict[str, Any]],
    trace_df: Optional[pd.DataFrame],
    events: List[Dict[str, Any]],
    shadow_df: Optional[pd.DataFrame],
) -> str:
    """Generate HTML report."""
    
    # Generate plots
    plots = {}
    if trace_df is not None:
        plots["equity_dd"] = plot_equity_and_drawdown(trace_df)
        plots["rolling"] = plot_rolling_metrics(trace_df)
        plots["calibration"] = plot_calibration_timeline(trace_df)
        plots["z_hist"] = plot_z_histograms(trace_df)
        plots["policy"] = plot_policy_actions(trace_df)
    
    if events:
        plots["events"] = plot_event_timeline(events)
    
    if shadow_df is not None:
        plots["shadow"] = plot_shadow_comparison(shadow_df)
    
    # Build HTML
    html_parts = [
        "<!DOCTYPE html>",
        "<html>",
        "<head>",
        f"<title>Phase-2 Diagnostics: {run_id}</title>",
        "<style>",
        "body { font-family: Arial, sans-serif; max-width: 1200px; margin: 0 auto; padding: 20px; }",
        "h1 { color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 10px; }",
        "h2 { color: #34495e; margin-top: 30px; }",
        ".metrics-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 15px; margin: 20px 0; }",
        ".metric-card { background: #f8f9fa; padding: 15px; border-radius: 8px; text-align: center; }",
        ".metric-value { font-size: 24px; font-weight: bold; color: #2980b9; }",
        ".metric-label { font-size: 12px; color: #7f8c8d; margin-top: 5px; }",
        ".event-summary { background: #fff3cd; padding: 15px; border-radius: 8px; margin: 20px 0; }",
        ".event-count { display: inline-block; margin-right: 15px; }",
        "img { max-width: 100%; height: auto; margin: 10px 0; }",
        "table { border-collapse: collapse; width: 100%; margin: 15px 0; }",
        "th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }",
        "th { background: #3498db; color: white; }",
        "tr:nth-child(even) { background: #f2f2f2; }",
        ".timestamp { color: #7f8c8d; font-size: 12px; }",
        "</style>",
        "</head>",
        "<body>",
        f"<h1>Phase-2 Diagnostics Report</h1>",
        f"<p class='timestamp'>Run ID: {run_id} | Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>",
    ]
    
    # Summary metrics
    if summary:
        metrics = summary.get("metrics", {})
        html_parts.append("<h2>📊 Summary Metrics</h2>")
        html_parts.append("<div class='metrics-grid'>")
        
        metric_display = [
            ("Sharpe", metrics.get("sharpe", "N/A"), "{:.2f}"),
            ("Max DD", metrics.get("max_drawdown", "N/A"), "{:.1%}"),
            ("Turnover", metrics.get("turnover", "N/A"), "{:.3f}"),
            ("Score", metrics.get("score", "N/A"), "{:.3f}"),
            ("Vol", metrics.get("realized_vol", "N/A"), "{:.1%}"),
            ("Target Vol", metrics.get("target_vol", "N/A"), "{:.1%}"),
            ("Flat Rate", metrics.get("flat_rate", "N/A"), "{:.1%}"),
            ("Turn Drift", metrics.get("turnover_drift", "N/A"), "{:.4f}"),
        ]
        
        for label, value, fmt in metric_display:
            if isinstance(value, (int, float)):
                display = fmt.format(value)
            else:
                display = str(value)
            html_parts.append(f"<div class='metric-card'><div class='metric-value'>{display}</div><div class='metric-label'>{label}</div></div>")
        
        html_parts.append("</div>")
        
        # Event counts
        event_counts = summary.get("event_counts", {})
        if event_counts:
            html_parts.append("<div class='event-summary'>")
            html_parts.append("<strong>Event Counts:</strong> ")
            for code, count in sorted(event_counts.items()):
                html_parts.append(f"<span class='event-count'>{code}: {count}</span>")
            html_parts.append("</div>")
    
    # Plots
    if plots.get("equity_dd"):
        html_parts.append("<h2>📈 Equity Curve & Drawdown</h2>")
        html_parts.append(f"<img src='data:image/png;base64,{plots['equity_dd']}' />")
    
    if plots.get("rolling"):
        html_parts.append("<h2>📉 Rolling Metrics (63-day)</h2>")
        html_parts.append(f"<img src='data:image/png;base64,{plots['rolling']}' />")
    
    if plots.get("calibration"):
        html_parts.append("<h2>🎯 Calibration Timeline</h2>")
        html_parts.append(f"<img src='data:image/png;base64,{plots['calibration']}' />")
    
    if plots.get("z_hist"):
        html_parts.append("<h2>📊 Z-Score Distributions</h2>")
        html_parts.append(f"<img src='data:image/png;base64,{plots['z_hist']}' />")
    
    if plots.get("events"):
        html_parts.append("<h2>⚠️ Event Timeline</h2>")
        html_parts.append(f"<img src='data:image/png;base64,{plots['events']}' />")
    
    if plots.get("policy"):
        html_parts.append("<h2>🎮 Policy Actions</h2>")
        html_parts.append(f"<img src='data:image/png;base64,{plots['policy']}' />")
    
    if plots.get("shadow"):
        html_parts.append("<h2>🔬 Shadow/Ablation Comparison</h2>")
        html_parts.append(f"<img src='data:image/png;base64,{plots['shadow']}' />")
    
    # Event details table (top 20)
    if events:
        html_parts.append("<h2>📋 Recent Events (Top 20)</h2>")
        html_parts.append("<table>")
        html_parts.append("<tr><th>Date</th><th>Day</th><th>Severity</th><th>Code</th><th>Message</th></tr>")
        
        # Sort by severity (ERROR > WARNING > INFO)
        severity_order = {"CRITICAL": 0, "ERROR": 1, "WARNING": 2, "INFO": 3, "DEBUG": 4}
        sorted_events = sorted(events, key=lambda e: (severity_order.get(e.get("severity", "INFO"), 4), -e.get("day_idx", 0)))
        
        for event in sorted_events[:20]:
            severity = event.get("severity", "INFO")
            color = {"CRITICAL": "#e74c3c", "ERROR": "#e74c3c", "WARNING": "#f39c12", "INFO": "#3498db"}.get(severity, "#95a5a6")
            html_parts.append(f"<tr><td>{event.get('date', 'N/A')}</td><td>{event.get('day_idx', 'N/A')}</td><td style='color:{color}'>{severity}</td><td>{event.get('code', 'N/A')}</td><td>{event.get('message', '')[:100]}</td></tr>")
        
        html_parts.append("</table>")
    
    # Footer
    html_parts.extend([
        "<hr>",
        f"<p class='timestamp'>Symbols: {summary.get('symbols', 'N/A') if summary else 'N/A'} | Horizon: {summary.get('horizon', 'N/A') if summary else 'N/A'}</p>",
        "</body>",
        "</html>",
    ])
    
    return "\n".join(html_parts)


def generate_markdown_report(
    run_id: str,
    summary: Optional[Dict[str, Any]],
    trace_df: Optional[pd.DataFrame],
    events: List[Dict[str, Any]],
    shadow_df: Optional[pd.DataFrame],
    output_dir: Path,
) -> str:
    """Generate Markdown report with embedded images saved to disk."""
    
    md_parts = [
        f"# Phase-2 Diagnostics Report",
        f"",
        f"**Run ID:** `{run_id}`",
        f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
    ]
    
    # Summary metrics
    if summary:
        metrics = summary.get("metrics", {})
        md_parts.extend([
            "## 📊 Summary Metrics",
            "",
            "| Metric | Value |",
            "|--------|-------|",
            f"| Sharpe | {metrics.get('sharpe', 'N/A'):.2f} |" if isinstance(metrics.get('sharpe'), (int, float)) else f"| Sharpe | N/A |",
            f"| Max Drawdown | {metrics.get('max_drawdown', 0):.1%} |" if isinstance(metrics.get('max_drawdown'), (int, float)) else f"| Max Drawdown | N/A |",
            f"| Avg Turnover | {metrics.get('turnover', 0):.4f} |" if isinstance(metrics.get('turnover'), (int, float)) else f"| Avg Turnover | N/A |",
            f"| Score | {metrics.get('score', 0):.3f} |" if isinstance(metrics.get('score'), (int, float)) else f"| Score | N/A |",
            f"| Realized Vol | {metrics.get('realized_vol', 0):.1%} |" if isinstance(metrics.get('realized_vol'), (int, float)) else f"| Realized Vol | N/A |",
            f"| Turnover Drift | {metrics.get('turnover_drift', 0):.4f} |" if isinstance(metrics.get('turnover_drift'), (int, float)) else f"| Turnover Drift | N/A |",
            "",
        ])
        
        # Event counts
        event_counts = summary.get("event_counts", {})
        if event_counts:
            md_parts.append("### Event Counts")
            md_parts.append("")
            for code, count in sorted(event_counts.items()):
                md_parts.append(f"- **{code}**: {count}")
            md_parts.append("")
    
    # Save plots to disk and reference them
    if HAS_MATPLOTLIB:
        plot_dir = output_dir / "plots"
        plot_dir.mkdir(parents=True, exist_ok=True)
        
        if trace_df is not None:
            # Equity plot
            fig_data = plot_equity_and_drawdown(trace_df)
            if fig_data:
                plot_path = plot_dir / f"equity_dd_{run_id}.png"
                with open(plot_path, "wb") as f:
                    f.write(base64.b64decode(fig_data))
                md_parts.extend([
                    "## 📈 Equity Curve & Drawdown",
                    f"![Equity and Drawdown](plots/equity_dd_{run_id}.png)",
                    "",
                ])
            
            # Rolling metrics
            fig_data = plot_rolling_metrics(trace_df)
            if fig_data:
                plot_path = plot_dir / f"rolling_{run_id}.png"
                with open(plot_path, "wb") as f:
                    f.write(base64.b64decode(fig_data))
                md_parts.extend([
                    "## 📉 Rolling Metrics (63-day)",
                    f"![Rolling Metrics](plots/rolling_{run_id}.png)",
                    "",
                ])
    
    # Event summary
    if events:
        md_parts.extend([
            "## ⚠️ Key Events",
            "",
            "| Date | Severity | Code | Message |",
            "|------|----------|------|---------|",
        ])
        
        severity_order = {"CRITICAL": 0, "ERROR": 1, "WARNING": 2, "INFO": 3, "DEBUG": 4}
        sorted_events = sorted(events, key=lambda e: (severity_order.get(e.get("severity", "INFO"), 4), -e.get("day_idx", 0)))
        
        for event in sorted_events[:15]:
            md_parts.append(f"| {event.get('date', 'N/A')} | {event.get('severity', 'INFO')} | {event.get('code', 'N/A')} | {event.get('message', '')[:60]} |")
        
        md_parts.append("")
    
    # Footer
    if summary:
        md_parts.extend([
            "---",
            f"**Symbols:** {', '.join(summary.get('symbols', [])) or 'N/A'}",
            f"**Horizon:** {summary.get('horizon', 'N/A')}",
        ])
    
    return "\n".join(md_parts)


# ─────────────────────────────────────────────────────────────────────────────
# Main Entry Point
# ─────────────────────────────────────────────────────────────────────────────

def generate_report(
    run_id: Optional[str] = None,
    output_format: str = "html",
    output_dir: Optional[Path] = None,
) -> Optional[Path]:
    """Generate diagnostic report for a Phase-2 run.
    
    Args:
        run_id: Run identifier (None = latest)
        output_format: "html" or "markdown"
        output_dir: Output directory (default: cache/debugging/)
    
    Returns:
        Path to generated report, or None on failure
    """
    if output_dir is None:
        output_dir = DEFAULT_OUTPUT_DIR
    output_dir = Path(output_dir)
    
    # Find run_id
    if run_id is None:
        run_id = find_latest_run(output_dir)
        if run_id is None:
            logger.error("No runs found in %s", output_dir)
            return None
        logger.info("Using latest run: %s", run_id)
    
    # Load data
    summary = load_summary(run_id, output_dir)
    trace_df = load_trace_df(run_id, output_dir)
    events = load_events(run_id, output_dir)
    shadow_df = load_shadow_df(run_id, output_dir)
    
    if summary is None and trace_df is None and not events:
        logger.error("No data found for run %s", run_id)
        return None
    
    # Generate report
    if output_format == "html":
        content = generate_html_report(run_id, summary, trace_df, events, shadow_df)
        report_path = output_dir / f"phase2_report_{run_id}.html"
    else:
        content = generate_markdown_report(run_id, summary, trace_df, events, shadow_df, output_dir)
        report_path = output_dir / f"phase2_report_{run_id}.md"
    
    # Write report
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(content)
    
    logger.info("📄 Generated report: %s", report_path)
    return report_path


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Generate Phase-2 diagnostics report",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python diagnostics_report.py                      # Latest run, HTML
  python diagnostics_report.py --run-id h63_abc123  # Specific run
  python diagnostics_report.py --output markdown    # Markdown output
  python diagnostics_report.py --dir /path/to/data  # Custom directory
        """,
    )
    parser.add_argument(
        "--run-id", "-r",
        help="Run ID (default: latest)",
    )
    parser.add_argument(
        "--output", "-o",
        choices=["html", "markdown"],
        default="html",
        help="Output format (default: html)",
    )
    parser.add_argument(
        "--dir", "-d",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Data directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose logging",
    )
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    
    # Generate report
    report_path = generate_report(
        run_id=args.run_id,
        output_format=args.output,
        output_dir=args.dir,
    )
    
    if report_path:
        print(f"\n✅ Report generated: {report_path}")
        print(f"   Open in browser: file://{report_path.absolute()}")
    else:
        print("\n❌ Failed to generate report")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
