#!/usr/bin/env python3
# pyright: reportUnknownMemberType=false
# pyright: reportUnknownVariableType=false
# pyright: reportUnknownArgumentType=false
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd

# Allow running as: `python tools/realtime_benchmark_tracker.py ...`
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.analytics.benchmarking import BenchmarkFramework, compute_benchmark_timeseries, summarize_benchmark  # noqa: E402
from src.analytics.eodhd_benchmark_data import (  # noqa: E402
    EODHDBenchmarkSpec,
    fetch_eodhd_adjusted_close,
    prices_to_returns,
)


def _cum_nav(r: pd.Series) -> pd.Series:
    return (1.0 + pd.to_numeric(r, errors="coerce").fillna(0.0)).cumprod()


def _drawdown_from_nav(nav: pd.Series) -> pd.Series:
    peak = nav.cummax().replace(0.0, pd.NA)
    return (nav / peak - 1.0).replace([float("inf"), float("-inf")], pd.NA)


def _try_plot_dashboard(*, ts: pd.DataFrame, out_dir: Path, window: int, title: Optional[str]) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"[tracker] plotting skipped (matplotlib missing): {exc}")
        return

    if ts.empty or ("rp" not in ts.columns) or ("rb" not in ts.columns):
        return

    df = ts.sort_index()
    df.index = pd.to_datetime(df.index, errors="coerce").tz_localize(None).normalize()
    df = df[~df.index.duplicated(keep="first")]

    w = int(window)
    beta_col = f"beta_{w}"
    alpha_col = f"alpha_{w}"
    ir_col = f"ir_{w}"

    nav_p = _cum_nav(df["rp"])
    nav_b = _cum_nav(df["rb"])
    nav_active = _cum_nav((df["rp"] - df["rb"]).rename("active"))

    title_prefix = (str(title).strip() + " — ") if title else ""

    # A) NAV vs benchmark (+ active NAV proxy)
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
        plt.plot(df.index, pd.to_numeric(df[beta_col], errors="coerce").values, label=f"Rolling beta ({w}d)")
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
        ax[0].plot(df.index, pd.to_numeric(df[alpha_col], errors="coerce").values, label=f"Daily Jensen alpha ({w}d beta)")
        if f"alpha_roll_{w}" in df.columns:
            ax[0].plot(df.index, pd.to_numeric(df[f"alpha_roll_{w}"], errors="coerce").values, label=f"Rolling alpha mean ({w}d)")
        ax[0].legend()
        ax[0].grid(True, alpha=0.3)
    ax[0].set_title(f"{title_prefix}Rolling Alpha")

    if ir_col in df.columns:
        ax[1].plot(df.index, pd.to_numeric(df[ir_col], errors="coerce").values, label=f"Rolling IR ({w}d)")
        ax[1].legend()
        ax[1].grid(True, alpha=0.3)
    ax[1].set_title(f"{title_prefix}Rolling IR")

    plt.tight_layout()
    plt.savefig(out_dir / "rolling_alpha_ir.png", dpi=160)
    plt.close()

    # D) Scatter
    plt.figure(figsize=(6.5, 6.5))
    plt.scatter(pd.to_numeric(df["rb"], errors="coerce").values, pd.to_numeric(df["rp"], errors="coerce").values, s=8, alpha=0.35)
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
    plt.plot(dd_p.index, pd.to_numeric(dd_p, errors="coerce").values, label="Portfolio drawdown")
    plt.plot(dd_b.index, pd.to_numeric(dd_b, errors="coerce").values, label="Benchmark drawdown")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.title(f"{title_prefix}Drawdown Comparison")
    plt.tight_layout()
    plt.savefig(out_dir / "drawdown_compare.png", dpi=160)
    plt.close()


def _read_bt_equity(bt_equity_path: Path) -> pd.DataFrame:
    # Concurrent writer safety: if Phase2/Optuna is writing the parquet at the same
    # time we read it, we may see transient EOF/corruption. Retry a few times.
    last_exc: Optional[Exception] = None
    for _ in range(5):
        try:
            df = pd.read_parquet(bt_equity_path)
            break
        except Exception as exc:
            last_exc = exc
            time.sleep(0.2)
    else:
        raise last_exc or RuntimeError("Failed to read bt_equity")

    if not isinstance(df.index, pd.DatetimeIndex):
        for col in ("date", "dt", "timestamp"):
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors="coerce")
                df = df.set_index(col)
                break
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError(f"bt_equity index must be DatetimeIndex: {bt_equity_path}")

    df = df.sort_index()
    df.index = pd.to_datetime(df.index, errors="coerce").tz_localize(None).normalize()
    df = df[~df.index.duplicated(keep="first")]
    return df


def _parse_windows(w: str) -> Tuple[int, ...]:
    return tuple(int(x.strip()) for x in str(w).split(",") if x.strip())


def _default_out_dir(bt_equity_path: Path) -> Path:
    return bt_equity_path.parent


def _emit_status(ts: pd.DataFrame, windows: Tuple[int, ...]) -> None:
    if ts.empty:
        print("[tracker] empty timeseries")
        return

    last_dt = ts.index.max()
    parts = [f"dt={last_dt.date()}"]
    for w in windows:
        for col in (f"beta_{w}", f"ir_{w}", f"te_{w}"):
            if col in ts.columns:
                v = ts[col].dropna()
                if len(v):
                    parts.append(f"{col}={float(v.iloc[-1]):.3f}")
    if "drawdown" in ts.columns:
        dd = ts["drawdown"].dropna()
        if len(dd):
            parts.append(f"dd={float(dd.iloc[-1]):.3f}")

    print("[tracker] " + " ".join(parts))


def main() -> int:
    ap = argparse.ArgumentParser(description="Real-time benchmark tracker for bt_equity.parquet (EODHD-backed benchmarks).")
    ap.add_argument("--bt-equity", required=True, help="Path to bt_equity.parquet")
    ap.add_argument("--benchmark", default="SPY", help="Benchmark ticker (default: SPY)")
    ap.add_argument("--benchmark-exchange-suffix", default=".US", help="EODHD exchange suffix (default: .US)")
    ap.add_argument("--benchmark-use-adjusted", action="store_true", help="Use adjusted close when available")
    ap.add_argument("--out-dir", default=None, help="Output dir (default: alongside bt_equity)")
    ap.add_argument("--poll-seconds", type=float, default=10.0, help="Polling interval in seconds")
    ap.add_argument("--windows", default="20,63,126", help="Comma windows for rolling metrics")
    ap.add_argument("--rf-annual", type=float, default=0.0, help="Annual risk-free rate")
    ap.add_argument("--cvar-alpha", type=float, default=0.05, help="CVaR tail probability")
    ap.add_argument("--target-vol", type=float, default=None, help="Target annual vol for vol-target error")
    ap.add_argument("--use-excess-alpha", action="store_true", help="Compute Jensen alpha on excess returns")
    ap.add_argument("--write-csv", action="store_true", help="Also write bt_benchmark_timeseries.csv")
    ap.add_argument("--plot", action="store_true", help="Also refresh dashboard PNGs on each update")
    ap.add_argument("--plot-window", type=int, default=63, help="Window to plot for beta/alpha/IR (default: 63)")
    ap.add_argument("--plot-title", default=None, help="Optional title prefix for plots")

    args = ap.parse_args()

    bt_equity_path = Path(args.bt_equity)
    if not bt_equity_path.exists():
        raise SystemExit(f"Missing: {bt_equity_path}")

    windows = _parse_windows(args.windows)
    out_dir = Path(args.out_dir) if args.out_dir else _default_out_dir(bt_equity_path)
    out_dir.mkdir(parents=True, exist_ok=True)

    framework = BenchmarkFramework(primary=str(args.benchmark).upper(), secondary=(), trading_days=252)

    ts_path = out_dir / "bt_benchmark_timeseries.parquet"
    ts_csv_path = out_dir / "bt_benchmark_timeseries.csv"
    summary_path = out_dir / "bt_benchmark_summary.json"
    framework_path = out_dir / "bt_benchmark_framework.json"

    last_mtime: Optional[float] = None
    last_rows: int = -1

    print(f"[tracker] watching: {bt_equity_path}")
    print(f"[tracker] writing:  {out_dir}")

    while True:
        try:
            st = bt_equity_path.stat()
        except FileNotFoundError:
            time.sleep(float(args.poll_seconds))
            continue

        if last_mtime is not None and st.st_mtime <= last_mtime:
            time.sleep(float(args.poll_seconds))
            continue

        eq = _read_bt_equity(bt_equity_path)
        if "net_return" not in eq.columns:
            raise SystemExit(f"bt_equity missing net_return: cols={list(eq.columns)}")

        rp = pd.to_numeric(eq["net_return"], errors="coerce").dropna().sort_index()
        if rp.empty:
            time.sleep(float(args.poll_seconds))
            continue

        # Avoid recompute if file touched but data length unchanged.
        if last_rows >= 0 and len(rp) == last_rows:
            last_mtime = float(st.st_mtime)
            time.sleep(float(args.poll_seconds))
            continue

        start = rp.index.min()
        end = rp.index.max()

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
        try:
            extra_cols = [c for c in eq.columns if c not in {"net_return", "equity_curve"}]
            if extra_cols:
                extras = eq[extra_cols].copy()
                extras.index = pd.to_datetime(extras.index, errors="coerce").tz_localize(None).normalize()
                ts = ts.join(extras, how="left")
        except Exception:
            pass
        summary = summarize_benchmark(ts=ts, windows=windows)

        ts.to_parquet(ts_path)
        if args.write_csv:
            ts.to_csv(ts_csv_path)
        summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
        framework_path.write_text(json.dumps(framework.__dict__, indent=2, sort_keys=True) + "\n")

        if args.plot:
            try:
                _try_plot_dashboard(ts=ts, out_dir=out_dir, window=int(args.plot_window), title=args.plot_title)
            except Exception as exc:
                print(f"[tracker] plot failed (non-fatal): {exc}")

        _emit_status(ts, windows)

        last_mtime = float(st.st_mtime)
        last_rows = int(len(rp))

        time.sleep(float(args.poll_seconds))


if __name__ == "__main__":
    raise SystemExit(main())
