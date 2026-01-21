from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd

from src.dcf_lab.utils.trading_calendar import add_sessions
from src.data_sources.eodhd_provider import get_eodhd_provider
from src.stage_b.delisting_meta import (
    _detect_last_reliable_trade_date_from_prices,
    get_delisted_date_map,
    load_delisting_registry,
)


@dataclass(frozen=True)
class UniverseRegistryConfig:
    exchange_code: str = "US"
    stale_sessions: int = 25
    vol_bad_fraction: float = 0.8
    prices_lookback_days: int = 420


def _norm_symbol(sym: str) -> str:
    return (sym or "").strip().upper()


def build_universe_registry(
    *,
    symbols: Iterable[str],
    asof_date: str | pd.Timestamp,
    out_path: Path,
    cfg: UniverseRegistryConfig = UniverseRegistryConfig(),
    delisting_cache_path: Optional[str | Path] = None,
) -> pd.DataFrame:
    """Build the authoritative universe registry parquet.

    Schema (minimum):
      - symbol
      - stop_date
      - is_eligible_today (as-of `asof_date`)
      - reason

    stop_date semantics:
      - eligibility rule is: eligible on date D iff D <= stop_date
      - we set stop_date = add_sessions(last_reliable_trade_date, stale_sessions)
        so a symbol can be missing/illiquid for up to `stale_sessions` before
        becoming ineligible.

    Notes:
      - Uses OHLCV (EOD prices) as the primary signal.
      - Uses locally-cached delisting registry as a secondary hard-stop.
    """

    asof_ts = pd.Timestamp(asof_date).normalize()
    ex = str(cfg.exchange_code or "US").upper()

    # Secondary signal: delisted date mapping (best-effort; empty if missing).
    delisting_df = load_delisting_registry(exchange_code=ex, cache_path=delisting_cache_path)
    delisted_date_by_symbol = get_delisted_date_map(delisting_df)

    provider = get_eodhd_provider()

    lookback_start = (asof_ts - pd.Timedelta(days=int(cfg.prices_lookback_days))).strftime("%Y-%m-%d")
    lookback_end = asof_ts.strftime("%Y-%m-%d")

    rows: list[dict[str, object]] = []
    for raw in symbols:
        sym = _norm_symbol(str(raw))
        if not sym:
            continue

        delisted_date = delisted_date_by_symbol.get(sym)

        prices = None
        try:
            prices = provider.get_eod_prices(sym, start_date=lookback_start, end_date=lookback_end, period="d")
        except Exception:
            prices = None

        if prices is None or not isinstance(prices, pd.DataFrame) or prices.empty:
            stop_date = pd.Timestamp("1900-01-01")
            reason = "no_price"
            last_trade_date = pd.NaT
            days_stale = None
            stale_flag = True
        else:
            # Normalize index.
            try:
                prices = prices.copy()
                prices.index = pd.DatetimeIndex(prices.index).tz_localize(None)
            except Exception:
                pass

            # Primary last seen date.
            last_trade_date = pd.Timestamp(prices.index.max()).normalize() if len(prices.index) else pd.NaT

            # More conservative "last reliable" date to catch constant-close + zero-volume zombies.
            last_reliable = _detect_last_reliable_trade_date_from_prices(
                prices,
                stale_sessions=int(cfg.stale_sessions),
                vol_bad_fraction=float(cfg.vol_bad_fraction),
            )
            if last_reliable is None or pd.isna(last_reliable):
                last_reliable = last_trade_date if pd.notna(last_trade_date) else pd.NaT

            if pd.isna(last_reliable):
                stop_date = pd.Timestamp("1900-01-01")
                reason = "no_price"
                days_stale = None
                stale_flag = True
            else:
                stop_date = add_sessions(pd.Timestamp(last_reliable), int(cfg.stale_sessions))
                stale_flag = bool(asof_ts > pd.Timestamp(stop_date))
                try:
                    days_stale = int((asof_ts - pd.Timestamp(last_reliable)).days)
                except Exception:
                    days_stale = None
                reason = "stale" if stale_flag else "ok"

        # Secondary hard-stop: delisted date.
        if delisted_date is not None and pd.notna(delisted_date):
            delisted_stop = pd.Timestamp(delisted_date).normalize()
            if pd.isna(stop_date) or delisted_stop < pd.Timestamp(stop_date):
                stop_date = delisted_stop
                reason = "delisted"

        is_eligible_today = bool(pd.Timestamp(asof_ts) <= pd.Timestamp(stop_date))

        rows.append(
            {
                "symbol": sym,
                "asof_date": asof_ts,
                "stop_date": pd.Timestamp(stop_date).normalize() if pd.notna(stop_date) else pd.NaT,
                "is_eligible_today": bool(is_eligible_today),
                "reason": str(reason),
                "last_trade_date": pd.Timestamp(last_trade_date).normalize() if pd.notna(last_trade_date) else pd.NaT,
                "delisted_date": pd.Timestamp(delisted_date).normalize() if delisted_date is not None and pd.notna(delisted_date) else pd.NaT,
                "days_stale": days_stale,
                "stale_flag": bool(not is_eligible_today) if str(reason) in {"stale", "no_price"} else False,
                "exchange_code": ex,
            }
        )

    df = pd.DataFrame(rows)
    if df.empty:
        df = pd.DataFrame(
            columns=[
                "symbol",
                "asof_date",
                "stop_date",
                "is_eligible_today",
                "reason",
                "last_trade_date",
                "delisted_date",
                "days_stale",
                "stale_flag",
                "exchange_code",
            ]
        )

    df["symbol"] = df["symbol"].astype(str).str.upper()
    df["asof_date"] = pd.to_datetime(df["asof_date"], errors="coerce").dt.tz_localize(None)
    df["stop_date"] = pd.to_datetime(df["stop_date"], errors="coerce").dt.tz_localize(None)
    df["last_trade_date"] = pd.to_datetime(df["last_trade_date"], errors="coerce").dt.tz_localize(None)
    df["delisted_date"] = pd.to_datetime(df["delisted_date"], errors="coerce").dt.tz_localize(None)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df = df.sort_values(["symbol"]).drop_duplicates(subset=["symbol"], keep="last")
    df.to_parquet(out_path, index=False)
    return df


def load_universe_registry(path: str | Path) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        return pd.DataFrame(columns=["symbol", "stop_date", "is_eligible_today", "reason"])
    df = pd.read_parquet(p)
    if "symbol" in df.columns:
        df["symbol"] = df["symbol"].astype(str).str.upper()
    if "stop_date" in df.columns:
        df["stop_date"] = pd.to_datetime(df["stop_date"], errors="coerce").dt.tz_localize(None)
    return df


def eligible_symbols_for_date(df: pd.DataFrame, *, asof_date: str | pd.Timestamp, symbols: Iterable[str]) -> set[str]:
    if df is None or df.empty:
        return set(_norm_symbol(s) for s in symbols if _norm_symbol(str(s)))

    asof_ts = pd.Timestamp(asof_date).normalize()
    tmp = df
    if "symbol" not in tmp.columns or "stop_date" not in tmp.columns:
        return set(_norm_symbol(s) for s in symbols if _norm_symbol(str(s)))

    symbol_set = {_norm_symbol(str(s)) for s in symbols if _norm_symbol(str(s))}
    tmp = tmp[tmp["symbol"].isin(symbol_set)].copy()

    tmp["stop_date"] = pd.to_datetime(tmp["stop_date"], errors="coerce").dt.tz_localize(None)
    ok = tmp["stop_date"].notna() & (asof_ts <= tmp["stop_date"])
    return set(tmp.loc[ok, "symbol"].astype(str).str.upper().tolist())
