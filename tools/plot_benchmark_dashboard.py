#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def _as_date_index(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if not isinstance(out.index, pd.DatetimeIndex):
        for col in ("date", "dt", "timestamp"):
            if col in out.columns:
                out[col] = pd.to_datetime(out[col], errors="coerce")
                out = out.set_index(col)
                break
    if not isinstance(out.index, pd.DatetimeIndex):
        raise ValueError("Expected a DatetimeIndex")
    out = out.sort_index()
    out.index = pd.to_datetime(out.index, errors="coerce").tz_localize(None).normalize()
    out = out[~out.index.duplicated(keep="first")]
    return out


def _cum_nav(r: pd.Series) -> pd.Series:
    return (1.0 + pd.to_numeric(r, errors="coerce").fillna(0.0)).cumprod()


def _drawdown_from_nav(nav: pd.Series) -> pd.Series:
    peak = nav.cummax().replace(0.0, np.nan)
    return (nav / peak - 1.0).replace([np.inf, -np.inf], np.nan)


def main() -> int:
    ap = argparse.ArgumentParser(description="Plot benchmark diagnostics dashboards from bt_benchmark_timeseries.parquet")
    ap.add_argument("--ts", required=True, help="Path to bt_benchmark_timeseries.parquet")
    ap.add_argument("--out-dir", default=None, help="Output directory for PNGs (default: alongside ts)")
    ap.add_argument("--window", type=int, default=63, help="Primary window to plot for beta/alpha/IR (default: 63)")
    ap.add_argument("--title", default=None, help="Optional plot title prefix")

    args = ap.parse_args()
    ts_path = Path(args.ts)
    if not ts_path.exists():
        raise SystemExit(f"Missing: {ts_path}")

    out_dir = Path(args.out_dir) if args.out_dir else ts_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        raise SystemExit(f"matplotlib required for plotting: {exc}")

    df = pd.read_parquet(ts_path)
    df = _as_date_index(df)

    if "rp" not in df.columns or "rb" not in df.columns:
        raise SystemExit(f"Missing required columns in timeseries: cols={list(df.columns)}")

    w = int(args.window)
    beta_col = f"beta_{w}"
    alpha_col = f"alpha_{w}"
    ir_col = f"ir_{w}"

    nav_p = _cum_nav(df["rp"])
    nav_b = _cum_nav(df["rb"])
    # Active NAV = cumulative active return approximation.
    nav_active = _cum_nav((df["rp"] - df["rb"]).rename("active"))

    title_prefix = (str(args.title).strip() + " — ") if args.title else ""

    # A) Cumulative return vs benchmark (+ alpha curve proxy)
    plt.figure(figsize=(12, 6))
    plt.plot(nav_p.index, nav_p.values, label="Portfolio NAV")
    plt.plot(nav_b.index, nav_b.values, label="Benchmark NAV")
    plt.plot(nav_active.index, nav_active.values, label="Active NAV (Rp-Rb)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.title(f"{title_prefix}Cumulative NAV vs Benchmark")
    plt.tight_layout()
    plt.savefig(out_dir / "nav_vs_benchmark.png", dpi=160)
    plt.close()

    # B) Rolling beta
    if beta_col in df.columns:
        plt.figure(figsize=(12, 4))
        plt.plot(df.index, df[beta_col].values, label=f"Rolling beta ({w}d)")
        plt.axhline(1.0, color="black", lw=1.0, alpha=0.6)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.title(f"{title_prefix}Rolling Beta")
        plt.tight_layout()
        plt.savefig(out_dir / "rolling_beta.png", dpi=160)
        plt.close()

    # C) Rolling alpha & IR
    fig, ax = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    if alpha_col in df.columns:
        ax[0].plot(df.index, df[alpha_col].values, label=f"Daily Jensen alpha ({w}d beta)")
        ax[0].plot(df.index, df.get(f"alpha_roll_{w}", pd.Series(index=df.index)).values, label=f"Rolling alpha mean ({w}d)")
        ax[0].legend()
        ax[0].grid(True, alpha=0.3)
    ax[0].set_title(f"{title_prefix}Rolling Alpha")

    if ir_col in df.columns:
        ax[1].plot(df.index, df[ir_col].values, label=f"Rolling IR ({w}d)")
        ax[1].legend()
        ax[1].grid(True, alpha=0.3)
    ax[1].set_title(f"{title_prefix}Rolling IR")

    plt.tight_layout()
    plt.savefig(out_dir / "rolling_alpha_ir.png", dpi=160)
    plt.close()

    # D) Scatter: portfolio vs benchmark daily returns
    plt.figure(figsize=(6.5, 6.5))
    plt.scatter(df["rb"].values, df["rp"].values, s=8, alpha=0.35)
    plt.xlabel("Benchmark daily return")
    plt.ylabel("Portfolio daily return")
    plt.grid(True, alpha=0.3)
    plt.title(f"{title_prefix}Returns Scatter (slope≈beta)")
    plt.tight_layout()
    plt.savefig(out_dir / "scatter_rp_rb.png", dpi=160)
    plt.close()

    # E) Drawdown comparison
    dd_p = _drawdown_from_nav(nav_p)
    dd_b = _drawdown_from_nav(nav_b)
    plt.figure(figsize=(12, 4))
    plt.plot(dd_p.index, dd_p.values, label="Portfolio drawdown")
    plt.plot(dd_b.index, dd_b.values, label="Benchmark drawdown")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.title(f"{title_prefix}Drawdown Comparison")
    plt.tight_layout()
    plt.savefig(out_dir / "drawdown_compare.png", dpi=160)
    plt.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
