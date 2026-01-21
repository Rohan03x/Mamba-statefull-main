from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import pandas as pd

from src.data_sources.eodhd_provider import EODHDProvider


@dataclass(frozen=True)
class EODHDBenchmarkSpec:
    """Spec for fetching a benchmark series from EODHD."""

    ticker: str = "SPY"
    exchange_suffix: str = ".US"
    use_adjusted_close: bool = True


def _normalize_symbol(symbol: str, *, exchange_suffix: str) -> str:
    s = str(symbol).strip().upper()
    if not s:
        raise ValueError("empty symbol")
    # If user already passed AAPL.US / SPY.US, keep it.
    if "." in s:
        return s
    # Default to US listings.
    return f"{s}{exchange_suffix}"


def fetch_eodhd_adjusted_close(
    *,
    ticker: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    exchange_suffix: str = ".US",
    use_adjusted_close: bool = True,
    provider: Optional[EODHDProvider] = None,
) -> pd.Series:
    """Fetch a daily (adjusted) close series via EODHD.

    Returns a tz-naive date-indexed Series.
    """

    provider = provider or EODHDProvider()
    if not getattr(provider, "api_key", None):
        raise RuntimeError("EODHD_API_KEY not set (required for benchmark fetching)")

    sym = _normalize_symbol(ticker, exchange_suffix=exchange_suffix)
    df = provider.get_eod_prices(
        sym,
        start_date=pd.Timestamp(start).strftime("%Y-%m-%d"),
        end_date=pd.Timestamp(end).strftime("%Y-%m-%d"),
        period="d",
    )
    if df is None or df.empty:
        raise RuntimeError(f"EODHD returned no price data for {sym}")

    col = None
    if bool(use_adjusted_close) and "Adj Close" in df.columns:
        col = "Adj Close"
    elif "Close" in df.columns:
        col = "Close"
    elif df.shape[1] == 1:
        col = df.columns[0]
    else:
        raise RuntimeError(f"EODHD price frame missing close columns for {sym}: cols={list(df.columns)}")

    s = pd.to_numeric(df[col], errors="coerce").dropna().sort_index()
    s.index = pd.to_datetime(s.index, errors="coerce").tz_localize(None).normalize()
    s = s[~s.index.duplicated(keep="first")]
    s.name = str(ticker).upper()

    s = s[(s.index >= pd.Timestamp(start).normalize()) & (s.index <= pd.Timestamp(end).normalize())]
    if s.empty:
        raise RuntimeError(f"No benchmark prices after date filtering for {sym}")

    return s


def prices_to_returns(prices: pd.Series) -> pd.Series:
    prices = prices.sort_index()
    r = prices.pct_change().dropna()
    r.name = f"r_{prices.name}"
    return r


def equal_weight_benchmark_returns_eodhd(
    members: Sequence[str],
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
    exchange_suffix: str = ".US",
    use_adjusted_close: bool = True,
    provider: Optional[EODHDProvider] = None,
) -> pd.Series:
    provider = provider or EODHDProvider()

    rets = []
    for m in members:
        px = fetch_eodhd_adjusted_close(
            ticker=m,
            start=start,
            end=end,
            exchange_suffix=exchange_suffix,
            use_adjusted_close=use_adjusted_close,
            provider=provider,
        )
        rets.append(prices_to_returns(px).rename(str(m).upper()))

    df = pd.concat(rets, axis=1).sort_index()
    b = df.mean(axis=1, skipna=True)
    b.name = "EW_BENCH"
    return b.dropna()
