#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Tuple

import pandas as pd

# Allow running as a script without installing the package.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.analytics.benchmarking import (  # noqa: E402
    BenchmarkFramework,
    compute_benchmark_timeseries,
    summarize_benchmark,
)
from src.analytics.eodhd_benchmark_data import (  # noqa: E402
    EODHDBenchmarkSpec,
    equal_weight_benchmark_returns_eodhd,
    fetch_eodhd_adjusted_close,
    prices_to_returns,
)


def _read_bt_equity(bt_equity_path: Path) -> pd.DataFrame:
    df = pd.read_parquet(bt_equity_path)
    if not isinstance(df.index, pd.DatetimeIndex):
        # Try common columns
        for col in ("date", "dt", "timestamp"):
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors="coerce")
                df = df.set_index(col)
                break
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError(f"bt_equity index must be DatetimeIndex: {bt_equity_path}")
    # Standardize to date index (tz-naive) to align with Yahoo daily bars.
    df = df.sort_index()
    df.index = pd.to_datetime(df.index, errors="coerce").tz_localize(None).normalize()
    df = df[~df.index.duplicated(keep="first")]
    return df


def _parse_members(members: str) -> Tuple[str, ...]:
    p = Path(members)
    if p.exists():
        txt = p.read_text().strip()
        items = [line.strip() for line in txt.splitlines() if line.strip()]
    else:
        items = [x.strip() for x in str(members).split(",") if x.strip()]
    out = tuple(dict.fromkeys([str(x).upper() for x in items]))
    if not out:
        raise ValueError("No benchmark members provided")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Benchmark a Stage B backtest vs SPY (and compute committee metrics).")
    ap.add_argument("--bt-equity", required=True, help="Path to bt_equity.parquet")
    ap.add_argument("--benchmark", default="SPY", help="Benchmark ticker (default: SPY)")
    ap.add_argument("--benchmark-exchange-suffix", default=".US", help="EODHD exchange suffix (default: .US)")
    ap.add_argument(
        "--benchmark-use-adjusted",
        action="store_true",
        help="Use adjusted close from EODHD when available (recommended).",
    )
    ap.add_argument(
        "--benchmark-members",
        default=None,
        help="Optional: build an equal-weight benchmark from members (csv list or path to file).",
    )
    ap.add_argument("--out-dir", default=None, help="Output directory (default: alongside bt_equity)")
    ap.add_argument("--write-csv", action="store_true", help="Also write bt_benchmark_timeseries.csv")
    ap.add_argument("--rf-annual", type=float, default=0.0, help="Annual risk-free rate (default: 0.0)")
    ap.add_argument("--windows", default="20,63,126", help="Comma windows for rolling metrics")
    ap.add_argument("--target-vol", type=float, default=None, help="Target annual vol for vol-target error")
    ap.add_argument("--cvar-alpha", type=float, default=0.05, help="CVaR tail probability (default: 0.05)")
    ap.add_argument(
        "--use-excess-alpha",
        action="store_true",
        help="Compute Jensen alpha on excess returns (Rp-Rf) vs (Rb-Rf)",
    )

    args = ap.parse_args()
    bt_equity_path = Path(args.bt_equity)
    if not bt_equity_path.exists():
        raise SystemExit(f"Missing: {bt_equity_path}")

    windows = tuple(int(x.strip()) for x in str(args.windows).split(",") if x.strip())

    eq = _read_bt_equity(bt_equity_path)
    if "net_return" not in eq.columns:
        raise SystemExit(f"bt_equity missing net_return: cols={list(eq.columns)}")

    rp = pd.to_numeric(eq["net_return"], errors="coerce").dropna().sort_index()
    start = rp.index.min()
    end = rp.index.max()

    if args.benchmark_members:
        members = _parse_members(args.benchmark_members)
        rb = equal_weight_benchmark_returns_eodhd(
            members,
            start=start,
            end=end,
            exchange_suffix=str(args.benchmark_exchange_suffix),
            use_adjusted_close=bool(args.benchmark_use_adjusted),
        )
        bench_name = "EQUAL_WEIGHT"
    else:
        spec = EODHDBenchmarkSpec(
            ticker=str(args.benchmark),
            exchange_suffix=str(args.benchmark_exchange_suffix),
            use_adjusted_close=bool(args.benchmark_use_adjusted),
        )
        bench_prices = fetch_eodhd_adjusted_close(
            ticker=spec.ticker,
            start=start,
            end=end,
            exchange_suffix=spec.exchange_suffix,
            use_adjusted_close=spec.use_adjusted_close,
        )
        rb = prices_to_returns(bench_prices)
        bench_name = str(spec.ticker).upper()

    ts = compute_benchmark_timeseries(
        portfolio_returns=rp,
        benchmark_returns=rb,
        windows=windows,
        risk_free=float(args.rf_annual),
        trading_days=252,
        cvar_alpha=float(args.cvar_alpha),
        target_vol_annual=args.target_vol,
        use_excess_alpha=bool(args.use_excess_alpha),
    )

    # Join any extra audit columns from bt_equity onto the benchmark timeseries.
    # (e.g. step_g_scale from Phase2 v2)
    try:
        extra_cols = [c for c in eq.columns if c not in {"net_return", "equity_curve"}]
        if extra_cols:
            extras = eq[extra_cols].copy()
            extras.index = pd.to_datetime(extras.index, errors="coerce").tz_localize(None).normalize()
            ts = ts.join(extras, how="left")
    except Exception:
        pass

    summary = summarize_benchmark(ts=ts, windows=windows)

    out_dir = Path(args.out_dir) if args.out_dir else bt_equity_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    framework = BenchmarkFramework(primary=str(bench_name), secondary=(), trading_days=252)

    ts_path = out_dir / "bt_benchmark_timeseries.parquet"
    ts_csv_path = out_dir / "bt_benchmark_timeseries.csv"
    summary_path = out_dir / "bt_benchmark_summary.json"
    framework_path = out_dir / "bt_benchmark_framework.json"

    ts.to_parquet(ts_path)
    if args.write_csv:
        ts.to_csv(ts_csv_path)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    framework_path.write_text(json.dumps(framework.__dict__, indent=2, sort_keys=True) + "\n")

    print(f"Wrote: {ts_path}")
    if args.write_csv:
        print(f"Wrote: {ts_csv_path}")
    print(f"Wrote: {summary_path}")
    print(f"Wrote: {framework_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
