#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

# Allow running as a script without installing the package.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _as_date_index(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if not isinstance(out.index, pd.DatetimeIndex):
        for col in ("date", "dt", "timestamp"):
            if col in out.columns:
                out[col] = pd.to_datetime(out[col], errors="coerce")
                out = out.set_index(col)
                break
    if not isinstance(out.index, pd.DatetimeIndex):
        raise ValueError("Expected DatetimeIndex")

    out = out.sort_index()
    out.index = pd.to_datetime(out.index, errors="coerce").tz_localize(None).normalize()
    out = out[~out.index.duplicated(keep="first")]
    return out


def _cum_nav(r: pd.Series) -> pd.Series:
    return (1.0 + pd.to_numeric(r, errors="coerce").fillna(0.0)).cumprod()


def _summarize_one(ts: pd.DataFrame, *, window: int) -> Dict[str, float]:
    out: Dict[str, float] = {}

    if "rp" in ts.columns:
        nav = _cum_nav(ts["rp"])
        out["nav_final"] = float(nav.iloc[-1]) if len(nav) else float("nan")
        out["ret_total"] = float(nav.iloc[-1] - 1.0) if len(nav) else float("nan")

    if "rb" in ts.columns:
        navb = _cum_nav(ts["rb"])
        out["bench_ret_total"] = float(navb.iloc[-1] - 1.0) if len(navb) else float("nan")

    if "active" in ts.columns:
        nav_a = _cum_nav(ts["active"])
        out["active_total"] = float(nav_a.iloc[-1] - 1.0) if len(nav_a) else float("nan")

    for col in (f"beta_{window}", f"ir_{window}", f"te_{window}"):
        if col in ts.columns:
            v = pd.to_numeric(ts[col], errors="coerce").dropna()
            out[col] = float(v.iloc[-1]) if len(v) else float("nan")

    for col in (f"alpha_{window}", f"alpha_roll_{window}"):
        if col in ts.columns:
            v = pd.to_numeric(ts[col], errors="coerce").dropna()
            out[col] = float(v.mean()) if len(v) else float("nan")

    if "drawdown" in ts.columns:
        dd = pd.to_numeric(ts["drawdown"], errors="coerce").dropna()
        out["max_drawdown"] = float(dd.min()) if len(dd) else float("nan")

    # Overlay means (if present)
    for col in ("step_g_scale", "role_risk_scale_mean", "role_regime_mult_mean", "role_hygiene_ok_frac"):
        if col in ts.columns:
            v = pd.to_numeric(ts[col], errors="coerce").dropna()
            out[f"mean_{col}"] = float(v.mean()) if len(v) else float("nan")

    return out


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Compare bt_benchmark_timeseries.parquet across horizons (e.g. h20 vs h63)."
    )
    ap.add_argument(
        "--portfolio-dir",
        default="artifacts/backtests/PORTFOLIO",
        help="Base directory containing h*/bt_benchmark_timeseries.parquet",
    )
    ap.add_argument(
        "--horizons",
        default="20,63",
        help="Comma list of horizons to compare (default: 20,63)",
    )
    ap.add_argument(
        "--window",
        type=int,
        default=63,
        help="Primary rolling window to summarize (default: 63)",
    )
    ap.add_argument(
        "--out-dir",
        default=None,
        help="Output directory (default: <portfolio-dir>)",
    )
    ap.add_argument(
        "--plot",
        action="store_true",
        help="Also write comparison PNG plots (requires matplotlib)",
    )

    args = ap.parse_args()
    base = Path(args.portfolio_dir)
    horizons = [int(x.strip()) for x in str(args.horizons).split(",") if x.strip()]
    if not horizons:
        raise SystemExit("No horizons provided")

    out_dir = Path(args.out_dir) if args.out_dir else base
    out_dir.mkdir(parents=True, exist_ok=True)

    loaded: Dict[int, pd.DataFrame] = {}
    summaries: List[Dict[str, object]] = []

    for h in horizons:
        p = base / f"h{h}" / "bt_benchmark_timeseries.parquet"
        if not p.exists():
            print(f"[warn] missing: {p}")
            continue
        ts = pd.read_parquet(p)
        ts = _as_date_index(ts)
        loaded[h] = ts

        s = _summarize_one(ts, window=int(args.window))
        s_out: Dict[str, object] = {"horizon": int(h), "path": str(p)}
        s_out.update(s)
        summaries.append(s_out)

    if not loaded:
        raise SystemExit("No horizon timeseries found")

    # Summary table
    summary_df = pd.DataFrame(summaries).sort_values("horizon").reset_index(drop=True)
    summary_path = out_dir / "bt_benchmark_horizon_compare.parquet"
    summary_df.to_parquet(summary_path, index=False)

    summary_json = out_dir / "bt_benchmark_horizon_compare.json"
    summary_json.write_text(json.dumps(summaries, indent=2, sort_keys=True, default=str) + "\n")

    print(f"Wrote: {summary_path}")
    print(f"Wrote: {summary_json}")

    if args.plot:
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except Exception as exc:
            raise SystemExit(f"matplotlib required for plotting: {exc}")

        # Align dates across horizons
        all_idx = sorted(set().union(*[set(ts.index) for ts in loaded.values()]))
        idx = pd.DatetimeIndex(all_idx)

        # NAV compare
        plt.figure(figsize=(12, 6))
        bench_done = False
        for h, ts in sorted(loaded.items()):
            if "rp" in ts.columns:
                nav = _cum_nav(ts.reindex(idx)["rp"].fillna(0.0))
                plt.plot(nav.index, nav.values, label=f"Portfolio h{h}")
            if not bench_done and "rb" in ts.columns:
                navb = _cum_nav(ts.reindex(idx)["rb"].fillna(0.0))
                plt.plot(navb.index, navb.values, label="Benchmark", color="black", alpha=0.6)
                bench_done = True
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.title("NAV comparison across horizons")
        plt.tight_layout()
        nav_path = out_dir / "nav_compare_horizons.png"
        plt.savefig(nav_path, dpi=160)
        plt.close()
        print(f"Wrote: {nav_path}")

        # Rolling IR compare
        ir_col = f"ir_{int(args.window)}"
        if any(ir_col in ts.columns for ts in loaded.values()):
            plt.figure(figsize=(12, 4))
            for h, ts in sorted(loaded.items()):
                if ir_col in ts.columns:
                    s = pd.to_numeric(ts.reindex(idx)[ir_col], errors="coerce")
                    plt.plot(idx, s.values, label=f"IR h{h}")
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.title(f"Rolling IR comparison ({args.window}d)")
            plt.tight_layout()
            ir_path = out_dir / "rolling_ir_compare_horizons.png"
            plt.savefig(ir_path, dpi=160)
            plt.close()
            print(f"Wrote: {ir_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
