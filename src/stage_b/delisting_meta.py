from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd

from src.data_sources.eodhd_provider import EODHDProvider


@dataclass(frozen=True)
class DelistingRecord:
    symbol: str
    delisted_date: pd.Timestamp
    exchange: Optional[str] = None
    reason: Optional[str] = None


def _detect_last_reliable_trade_date_from_prices(
    prices: pd.DataFrame,
    *,
    stale_sessions: int = 25,
    vol_bad_fraction: float = 0.8,
) -> Optional[pd.Timestamp]:
    """Detect a likely "inactive" condition near the end of a price series.

    Rule: if the *tail* has >= N sessions of unchanged close and volume is 0/NA
    for at least `vol_bad_fraction` of those sessions, treat the last reliable
    trade date as the session immediately before that stale run.

    Returns:
        last_reliable_trade_date (normalized) or None.
    """

    if prices is None or not isinstance(prices, pd.DataFrame) or prices.empty:
        return None
    if "Close" not in prices.columns or "Volume" not in prices.columns:
        return None

    df = prices[["Close", "Volume"]].copy()
    try:
        df.index = pd.DatetimeIndex(df.index).tz_localize(None)
    except Exception:
        df.index = pd.DatetimeIndex(df.index)
    df = df.dropna(subset=["Close"], how="any")
    if df.empty:
        return None

    closes = pd.to_numeric(df["Close"], errors="coerce")
    vols = pd.to_numeric(df["Volume"], errors="coerce")
    if closes.isna().all():
        return None

    last_close = closes.iloc[-1]
    if pd.isna(last_close):
        return None

    # Backward scan over the trailing run of constant close.
    i = len(df) - 1
    while i >= 0 and not pd.isna(closes.iloc[i]) and float(closes.iloc[i]) == float(last_close):
        i -= 1
    run_start = i + 1
    run_len = len(df) - run_start
    if run_len < int(stale_sessions):
        return None

    run_vol = vols.iloc[run_start:]
    vol_bad = run_vol.isna() | (run_vol <= 0)
    frac_bad = float(vol_bad.mean()) if len(vol_bad) else 0.0
    if frac_bad < float(vol_bad_fraction):
        return None

    # last reliable date is the session immediately before the stale run.
    if run_start <= 0:
        return pd.Timestamp(df.index[0]).normalize()
    return pd.Timestamp(df.index[run_start - 1]).normalize()


def _default_cache_path(exchange_code: str) -> Path:
    ex = (exchange_code or "US").upper()
    return Path("data") / "cache" / "eodhd" / f"delisted_companies_{ex}.parquet"


def load_delisting_registry(
    *,
    exchange_code: str = "US",
    cache_path: Optional[str | Path] = None,
) -> pd.DataFrame:
    """Load the locally cached delisting registry.

    Returns a DataFrame with (at minimum) columns:
    - symbol (upper)
    - delisted_date (Timestamp)

    If no cache exists, returns an empty DataFrame.
    """

    path = Path(cache_path) if cache_path is not None else _default_cache_path(exchange_code)
    if not path.exists():
        return pd.DataFrame(columns=["symbol", "delisted_date", "exchange", "reason"])

    df = pd.read_parquet(path)
    if df.empty:
        return pd.DataFrame(columns=["symbol", "delisted_date", "exchange", "reason"])

    if "symbol" in df.columns:
        df["symbol"] = df["symbol"].astype(str).str.upper()
    if "delisted_date" in df.columns:
        df["delisted_date"] = pd.to_datetime(df["delisted_date"], errors="coerce")

    keep = [
        c
        for c in (
            "symbol",
            "delisted_date",
            "exchange",
            "reason",
            "source",
            "renamed_to",
            "type",
            "country",
        )
        if c in df.columns
    ]
    df = df[keep].copy()
    df = df.dropna(subset=["symbol", "delisted_date"]).drop_duplicates(subset=["symbol"], keep="last")
    return df


def refresh_delisting_registry(
    provider: EODHDProvider,
    *,
    exchange_code: str = "US",
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    cache_path: Optional[str | Path] = None,
    timeout_seconds: int = 120,
    chunk_days: int = 7,
) -> pd.DataFrame:
    """Fetch delisted companies from EODHD and upsert into a local parquet cache.

    This is best-effort: if EODHD returns nothing (timeouts/unavailable), the existing
    cache (if any) is returned.
    """

    path = Path(cache_path) if cache_path is not None else _default_cache_path(exchange_code)
    path.parent.mkdir(parents=True, exist_ok=True)

    existing = load_delisting_registry(exchange_code=exchange_code, cache_path=path)

    fetched = provider.get_delisted_companies(
        exchange_code=exchange_code,
        start_date=start_date,
        end_date=end_date,
        timeout_seconds=timeout_seconds,
        chunk_days=chunk_days,
    )

    if fetched is None or fetched.empty:
        if path.exists():
            return existing
        return pd.DataFrame(columns=["symbol", "delisted_date", "exchange", "reason"])

    # Normalize columns.
    df = fetched.copy()

    symbol_col = None
    for c in ("Code", "code", "Symbol", "symbol", "ticker", "Ticker"):
        if c in df.columns:
            symbol_col = c
            break
    date_col = None
    for c in ("DelistedDate", "delistedDate", "delisted_date", "Date", "date"):
        if c in df.columns:
            date_col = c
            break

    if symbol_col is None or date_col is None:
        # Can't normalize; don't clobber existing cache.
        return existing

    out = pd.DataFrame(
        {
            "symbol": df[symbol_col].astype(str).str.upper(),
            "delisted_date": pd.to_datetime(df[date_col], errors="coerce"),
        }
    )

    ex_col = None
    for c in ("Exchange", "exchange", "Ex", "ex"):
        if c in df.columns:
            ex_col = c
            break
    if ex_col is not None:
        out["exchange"] = df[ex_col].astype(str)
    else:
        out["exchange"] = exchange_code

    reason_col = None
    for c in ("Reason", "reason", "DelistedReason", "delistedReason"):
        if c in df.columns:
            reason_col = c
            break
    out["reason"] = df[reason_col].astype(str) if reason_col is not None else None

    out = out.dropna(subset=["symbol", "delisted_date"])
    if out.empty:
        return existing

    merged = pd.concat([existing, out], axis=0, ignore_index=True)
    merged = merged.dropna(subset=["symbol", "delisted_date"])
    merged = merged.sort_values("delisted_date").drop_duplicates(subset=["symbol"], keep="last")

    merged.to_parquet(path, index=False)
    return merged


def refresh_delisting_registry_for_symbols(
    provider: EODHDProvider,
    *,
    symbols: list[str],
    exchange_code: str = "US",
    cache_path: Optional[str | Path] = None,
    timeout_seconds: int = 60,
    stale_sessions: int = 25,
    vol_bad_fraction: float = 0.8,
) -> pd.DataFrame:
    """Build/update delisting registry for a specific symbol list.

    Uses the fundamentals endpoint (General.IsDelisted + General.DelistedDate when present).
    If a symbol is marked delisted but no delisted-date is present, falls back to the last
    available EOD price date as a proxy.

    This is the recommended path for Phase-2 stateful runs because it is bounded to the
    trial universe and avoids large, plan-gated delisted-company endpoints.
    """

    path = Path(cache_path) if cache_path is not None else _default_cache_path(exchange_code)
    path.parent.mkdir(parents=True, exist_ok=True)

    existing = load_delisting_registry(exchange_code=exchange_code, cache_path=path)
    existing_map = get_delisted_date_map(existing)

    # Best-effort symbol rename mapping (prevents treating a rename as a delisting).
    rename_map: dict[str, tuple[str, pd.Timestamp]] = {}
    try:
        sch = provider.get_symbol_change_history()
        if isinstance(sch, pd.DataFrame) and not sch.empty:
            cols = set(sch.columns)
            if {"old_symbol", "new_symbol", "effective"}.issubset(cols):
                tmp = sch[["old_symbol", "new_symbol", "effective"]].copy()
                tmp["old_symbol"] = tmp["old_symbol"].astype(str).str.upper()
                tmp["new_symbol"] = tmp["new_symbol"].astype(str).str.upper()
                tmp["effective"] = pd.to_datetime(tmp["effective"], errors="coerce")
                tmp = tmp.dropna(subset=["old_symbol", "new_symbol", "effective"])
                for _, r in tmp.iterrows():
                    old = str(r["old_symbol"]).upper()
                    new = str(r["new_symbol"]).upper()
                    eff = pd.Timestamp(r["effective"]).normalize()
                    if old and new and not pd.isna(eff):
                        rename_map[old] = (new, eff)
    except Exception:
        rename_map = {}

    out_rows: list[dict[str, object]] = []
    ex = str(exchange_code or "US").upper()
    for sym in [str(s).upper().strip() for s in (symbols or []) if str(s).strip()]:
        # Skip if we already have an entry (keep existing unless we can improve it).
        if sym in existing_map:
            continue

        # If this symbol is known to have been renamed, treat it as inactive after
        # the effective date (the entity continues under renamed_to).
        if sym in rename_map:
            new_sym, effective_dt = rename_map[sym]
            if effective_dt is not None and not pd.isna(effective_dt):
                stop_dt = pd.Timestamp(effective_dt).normalize() - pd.Timedelta(days=1)
                out_rows.append(
                    {
                        "symbol": sym,
                        "delisted_date": pd.Timestamp(stop_dt).normalize(),
                        "exchange": ex,
                        "reason": "symbol_change",
                        "source": "symbol_change_history",
                        "renamed_to": new_sym,
                    }
                )
                continue

        try:
            fd = provider.get_fundamentals(sym, timeout_seconds=int(timeout_seconds))
        except Exception:
            fd = None
        if not isinstance(fd, dict):
            fd = {}
        general = fd.get("General") if isinstance(fd, dict) else None
        general = general if isinstance(general, dict) else {}

        is_delisted = general.get("IsDelisted")
        is_delisted = bool(is_delisted) if isinstance(is_delisted, bool) else str(is_delisted).strip().lower() in {"1", "true", "yes"}
        type_val = general.get("Type") if isinstance(general, dict) else None
        country_val = general.get("CountryName") if isinstance(general, dict) else None

        delisted_date_val = None
        for k in ("DelistedDate", "DelistingDate", "DelistedAt", "DelistedDateTime"):
            if k in general and general.get(k):
                delisted_date_val = general.get(k)
                break

        stop_dt = pd.to_datetime(delisted_date_val, errors="coerce") if delisted_date_val is not None else pd.NaT
        source = None
        reason = None

        if is_delisted:
            source = "fundamentals"
            reason = general.get("DelistedReason") or general.get("Reason")
            if pd.isna(stop_dt):
                # Best-effort proxy: last available EOD price date.
                try:
                    prices = provider.get_eod_prices(sym)
                except Exception:
                    prices = None
                if isinstance(prices, pd.DataFrame) and not prices.empty:
                    try:
                        stop_dt = pd.Timestamp(prices.index.max()).normalize()
                        source = "fundamentals+last_eod"
                    except Exception:
                        stop_dt = pd.NaT

        if pd.isna(stop_dt):
            # Market-data staleness rule (primary non-fundamental inactivity signal).
            try:
                end_dt = pd.Timestamp.utcnow().normalize()
                start_dt = end_dt - pd.Timedelta(days=400)
                prices = provider.get_eod_prices(sym, start_date=str(start_dt.date()), end_date=str(end_dt.date()))
            except Exception:
                prices = None
            if isinstance(prices, pd.DataFrame) and not prices.empty:
                inferred = _detect_last_reliable_trade_date_from_prices(
                    prices,
                    stale_sessions=int(stale_sessions),
                    vol_bad_fraction=float(vol_bad_fraction),
                )
                if inferred is not None and not pd.isna(inferred):
                    stop_dt = pd.Timestamp(inferred).normalize()
                    source = "eod_staleness"
                    reason = "stale_price_volume"

        if pd.isna(stop_dt):
            continue

        out_rows.append(
            {
                "symbol": sym,
                "delisted_date": pd.Timestamp(stop_dt).normalize(),
                "exchange": (general.get("Exchange") if isinstance(general, dict) else None) or ex,
                "reason": reason,
                "source": source,
                "renamed_to": None,
                "type": type_val,
                "country": country_val,
            }
        )

    if not out_rows:
        # If strict mode relies on cache existence, ensure we materialize an empty cache.
        if not path.exists():
            existing.to_parquet(path, index=False)
        return existing

    new_df = pd.DataFrame(out_rows)
    if existing is None or existing.empty:
        merged = new_df
    else:
        merged = pd.concat([existing, new_df], axis=0, ignore_index=True)
    merged = merged.dropna(subset=["symbol", "delisted_date"])
    merged = merged.sort_values("delisted_date").drop_duplicates(subset=["symbol"], keep="last")
    merged.to_parquet(path, index=False)
    return merged


def get_delisted_date_map(registry_df: pd.DataFrame) -> dict[str, pd.Timestamp]:
    if registry_df is None or registry_df.empty:
        return {}
    if "symbol" not in registry_df.columns or "delisted_date" not in registry_df.columns:
        return {}

    out: dict[str, pd.Timestamp] = {}
    for _, row in registry_df[["symbol", "delisted_date"]].dropna().iterrows():
        sym = str(row["symbol"]).upper()
        dt = pd.to_datetime(row["delisted_date"], errors="coerce")
        if sym and not pd.isna(dt):
            out[sym] = pd.Timestamp(dt).normalize()
    return out
