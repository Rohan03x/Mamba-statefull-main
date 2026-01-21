from __future__ import annotations

from dataclasses import dataclass
import logging
import os
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, cast

import numpy as np
import pandas as pd

from .types import EligibilityConfig


logger = logging.getLogger(__name__)


@dataclass
class EligibilityResult:
    symbol: str
    eligible: bool
    flags: Dict[str, Any]


def _norm_symbol(sym: str) -> str:
    return (sym or "").strip().upper()


def _as_set(xs: Optional[Sequence[str]]) -> Optional[set[str]]:
    if not xs:
        return None
    out = {str(x).strip() for x in xs if str(x).strip()}
    return out or None


def _norm_country_code(x: str) -> str:
    s = (x or "").strip().upper()
    if s in {"USA", "UNITED STATES", "UNITED STATES OF AMERICA"}:
        return "US"
    return s


def _safe_float(x: object) -> Optional[float]:
    try:
        if x is None:
            return None
        v = float(cast(Any, x))
        if not np.isfinite(v):
            return None
        return v
    except Exception:
        return None


def _truthy_flag(x: object) -> bool:
    """Best-effort conversion for provider boolean-ish fields.

    EODHD sometimes returns flags as strings like "0"/"1" or "false"/"true".
    Using Python's plain bool("0") would be wrong.
    """

    if x is None:
        return False
    if isinstance(x, bool):
        return x
    if isinstance(x, (int, float)):
        try:
            return float(x) != 0.0
        except Exception:
            return False
    s = str(x).strip().lower()
    if s in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if s in {"0", "false", "f", "no", "n", "off", "", "none", "nan"}:
        return False
    # Fallback: any other non-empty string treated as truthy.
    return True


class EODHDEligibilityGate:
    """Tier-1 hard gate.

    Architecture notes:
    - This gate must be provider-only (EODHD / exchange metadata).
    - It must be time-aware (only data <= asof).
    - It must not perform ranking.

    Implementation status:
    - The repo has a rich EODHD provider, but not a single canonical "screener" API.
      This gate therefore supports both:
        (A) provider-native screener endpoint (to be plugged in later)
        (B) composition of EODHD primitives (profile + prices + marketcap)

    The goal here is the *architecture*; the exact upstream endpoint wiring can
    be replaced without changing Tier-2.
    """

    def __init__(self, provider: Any, cfg: EligibilityConfig):
        self._p = provider
        self._cfg = cfg

    def _should_use_screener(self, *, asof_ts: pd.Timestamp) -> bool:
        if not hasattr(self._p, "get_stock_screener"):
            return False
        use = str(os.getenv("TIER1_USE_SCREENER", "1") or "1").strip().lower() in {"1", "true", "yes", "on"}
        if not use:
            return False

        allow_historical = str(os.getenv("TIER1_SCREENER_ALLOW_HISTORICAL", "0") or "0").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        if allow_historical:
            return True

        # Screener API is typically "latest snapshot" (non-historical). To keep Tier-1 leak-free,
        # only enable screener-derived metrics when asof is near "today".
        now = pd.Timestamp.utcnow()
        # pandas returns tz-aware UTC for utcnow(); our `asof_ts` is tz-naive.
        # Normalize both sides to tz-naive so date-diff math is reliable.
        if getattr(now, "tzinfo", None) is not None:
            now = now.tz_localize(None)
        now = now.normalize()
        try:
            return abs(int((now - asof_ts).days)) <= 2
        except Exception:
            return False

    def _fetch_screener_universe(
        self,
        *,
        allow_sector: Optional[set[str]],
        block_sector: Optional[set[str]],
    ) -> pd.DataFrame:
        """Fetch a RawUniverse via screener, applying only hard constraints.

        Notes:
        - Screener filters do not support OR across multiple sector values; we apply
          multi-sector allow/block client-side after paging.
        - Paging controls are env-configurable.
        """

        cfg = self._cfg

        def _op_for_text(v: str) -> str:
            # EODHD docs suggest "match" for multi-word strings.
            vv = str(v or "")
            if " " in vv or "*" in vv:
                return "match"
            return "="

        filters: list[list[object]] = []
        if cfg.exchange_code:
            code = str(cfg.exchange_code).strip()
            if code.upper() in {"US", "USA"}:
                filters.append(["exchange", "=", "us"])
            else:
                filters.append(["exchange", "=", code])

        if cfg.min_market_cap is not None:
            filters.append(["market_capitalization", ">", float(cfg.min_market_cap)])
        if cfg.min_avg_volume is not None:
            # Screener provides avgvol_200d; use as a liquidity proxy for Tier-1.
            filters.append(["avgvol_200d", ">", float(cfg.min_avg_volume)])
        if cfg.min_price is not None:
            filters.append(["adjusted_close", ">", float(cfg.min_price)])

        # Single-sector allow-list can be pushed down; multi-sector is client-side.
        if allow_sector is not None and len(allow_sector) == 1:
            sector = next(iter(allow_sector))
            filters.append(["sector", _op_for_text(sector), sector])

        signals_env = str(os.getenv("TIER1_SCREENER_SIGNALS", "") or "").strip()
        signals: Optional[list[str]] = None
        if signals_env:
            signals = [s.strip() for s in signals_env.split(",") if s.strip()]

        try:
            limit = int(os.getenv("TIER1_SCREENER_LIMIT", "100") or "100")
        except Exception:
            limit = 100
        try:
            max_pages = int(os.getenv("TIER1_SCREENER_MAX_PAGES", "20") or "20")
        except Exception:
            max_pages = 20
        try:
            max_offset = int(os.getenv("TIER1_SCREENER_MAX_OFFSET", "5000") or "5000")
        except Exception:
            max_offset = 5000

        pieces: List[pd.DataFrame] = []
        offset = 0
        pages = 0
        while pages < max_pages and offset <= max_offset:
            df = self._p.get_stock_screener(
                filters=filters,
                signals=signals,
                sort="market_capitalization.desc",
                limit=int(limit),
                offset=int(offset),
                timeout_seconds=60,
            )
            if df is None or df.empty:
                break
            pieces.append(df)
            if len(df) < int(limit):
                break
            offset += int(limit)
            pages += 1

        if not pieces:
            return pd.DataFrame()
        out = pd.concat(pieces, axis=0, ignore_index=True)

        # Normalize symbol column.
        sym_col = None
        for c in ("code", "Code", "symbol", "Symbol", "ticker", "Ticker"):
            if c in out.columns:
                sym_col = c
                break
        if sym_col is None:
            return pd.DataFrame()

        out[sym_col] = out[sym_col].astype(str).str.upper().str.strip()
        out = out[out[sym_col].astype(str).str.len() > 0].copy()
        out = out.drop_duplicates(subset=[sym_col])

        # Client-side sector filtering for multi-sector allow/block.
        if "sector" in out.columns:
            sec = out["sector"].astype(str).str.strip()
            if allow_sector is not None and len(allow_sector) > 1:
                allow_list: List[str] = sorted([str(x) for x in allow_sector])
                out = out[sec.isin(allow_list)].copy()
            if block_sector is not None:
                block_list: List[str] = sorted([str(x) for x in block_sector])
                out = out[~sec.isin(block_list)].copy()

        return out

    def evaluate(self, *, symbols: Optional[Iterable[str]] = None, asof: str | pd.Timestamp) -> pd.DataFrame:
        asof_ts = pd.Timestamp(asof).normalize()
        allow_country_raw = _as_set(self._cfg.country_allow)
        allow_country = None if allow_country_raw is None else {_norm_country_code(x) for x in allow_country_raw}
        allow_sector = _as_set(self._cfg.sector_allow)
        block_sector = _as_set(self._cfg.sector_block)

        screener_df: Optional[pd.DataFrame] = None

        # Default candidate universe: mirror Phase2 mamba stateful defaults.
        if symbols is None:
            source = str(os.getenv("TIER1_SYMBOL_SOURCE", "phase2") or "phase2").strip().lower()
            if source in {"screener", "eodhd_screener"} and self._should_use_screener(asof_ts=asof_ts):
                try:
                    screener_df = self._fetch_screener_universe(allow_sector=allow_sector, block_sector=block_sector)
                except Exception as exc:
                    logger.warning("Tier1 screener universe fetch failed; falling back: %s", exc)
                    screener_df = None

                if screener_df is not None and not screener_df.empty:
                    sym_col = "code" if "code" in screener_df.columns else None
                    if sym_col is None:
                        for c in ("Code", "symbol", "Symbol", "ticker", "Ticker"):
                            if c in screener_df.columns:
                                sym_col = c
                                break
                    if sym_col is not None:
                        symbols = screener_df[sym_col].astype(str).str.upper().tolist()
                    else:
                        symbols = []
                else:
                    symbols = []
            else:
                from src.stage_b.universe_selector import DEFAULT_CANDIDATE_UNIVERSE

                symbols = DEFAULT_CANDIDATE_UNIVERSE

        screener_by_symbol: Dict[str, Dict[str, Any]] = {}
        if screener_df is not None and not screener_df.empty:
            sym_col = None
            for c in ("code", "Code", "symbol", "Symbol", "ticker", "Ticker"):
                if c in screener_df.columns:
                    sym_col = c
                    break
            if sym_col is not None:
                for _, r in screener_df.iterrows():
                    sym = _norm_symbol(str(r.get(sym_col) or ""))
                    if sym:
                        screener_by_symbol[sym] = dict(r)

        # Formal invariant: for every candidate symbol at date t, emit exactly one row.
        # Normalize + de-duplicate to enforce (date, symbol) uniqueness.
        seen: set[str] = set()
        normed_symbols: List[str] = []
        for s in symbols:
            sym = _norm_symbol(s)
            if not sym or sym in seen:
                continue
            seen.add(sym)
            normed_symbols.append(sym)

        rows: List[Dict[str, Any]] = []
        for sym in normed_symbols:

            flags: Dict[str, Any] = {
                "exchange_ok": True,
                "country_ok": True,
                "sector_ok": True,
                # Instrument classification flags are coerced to strict booleans
                # so the Tier-1 ledger is easy to aggregate/analyze.
                "is_etf": False,
                "is_adr": False,
                "is_fund": False,
                "market_cap_ok": True,
                "avg_volume_ok": True,
                "price_ok": True,
                "not_suspended": True,
                "not_delisted": True,
                # Optional screener signals (informational / sanity indicators only).
                "200d_new_hi": False,
                "200d_new_lo": False,
                "bookvalue_neg": False,
                "bookvalue_pos": False,
                "wallstreet_hi": False,
                "wallstreet_lo": False,
            }

            screener_row = screener_by_symbol.get(sym)

            # Pull profile metadata (best-effort).
            profile: Mapping[str, Any] = {}
            try:
                profile = self._p.get_profile(sym) or {}
            except Exception:
                profile = {}

            # Use screener identity fields when present; fall back to profile.
            country = str((screener_row or {}).get("country") or profile.get("country") or "").strip()
            sector = str((screener_row or {}).get("sector") or profile.get("sector") or "").strip()
            exchange = str((screener_row or {}).get("exchange") or profile.get("exchange") or "").strip()

            if allow_country is not None:
                flags["country_ok"] = _norm_country_code(country) in allow_country
            if allow_sector is not None:
                flags["sector_ok"] = sector in allow_sector
            if block_sector is not None and sector:
                flags["sector_ok"] = bool(flags["sector_ok"]) and (sector not in block_sector)

            # Exchange gate is config-level (provider may return different strings).
            # We treat missing exchange as unknown (keep by default; strict mode can be added later).
            if self._cfg.exchange_code and exchange:
                code = str(self._cfg.exchange_code).strip().upper()
                ex = exchange.strip().upper()
                if code in {"US", "USA"}:
                    # EODHD typically returns strings like "NASDAQ", "NYSE", etc.
                    # Treat these as valid US exchanges under the default "US" setting.
                    us_tokens = {"NASDAQ", "NYSE", "AMEX", "ARCA", "BATS", "IEX", "OTC"}
                    if ex in {"US", "USA", "UNITED STATES"}:
                        flags["exchange_ok"] = True
                    else:
                        flags["exchange_ok"] = any(tok in ex for tok in us_tokens)
                else:
                    flags["exchange_ok"] = code in ex

            # Asset type flags: EODHD may expose these as booleans/strings.
            # We coerce to strict booleans and treat missing as False.
            is_etf = _truthy_flag(profile.get("is_etf"))
            is_adr = _truthy_flag(profile.get("is_adr"))
            is_fund = _truthy_flag(profile.get("is_fund"))
            flags["is_etf"] = bool(is_etf)
            flags["is_adr"] = bool(is_adr)
            flags["is_fund"] = bool(is_fund)

            instrument_type = "COMMON_STOCK"
            if is_etf:
                instrument_type = "ETF"
            elif is_adr:
                instrument_type = "ADR"
            elif is_fund:
                instrument_type = "FUND"

            # Screener-derived asof metrics (when available) avoid per-symbol EOD calls.
            last_close = None
            last_vol = None
            avg_vol_20 = None
            market_cap = None

            if screener_row is not None:
                # Value columns per EODHD docs.
                last_close = _safe_float(screener_row.get("adjusted_close"))
                last_vol = _safe_float(screener_row.get("avgvol_1d"))
                avgvol_200d = _safe_float(screener_row.get("avgvol_200d"))
                # We record avgvol_200d into asof_adv20 as a generic ADV proxy.
                # Tier-1 uses it only as a binary liquidity gate.
                avg_vol_20 = avgvol_200d
                market_cap = _safe_float(screener_row.get("market_capitalization"))

                # Optional signals (informational)
                for k in (
                    "200d_new_hi",
                    "200d_new_lo",
                    "bookvalue_neg",
                    "bookvalue_pos",
                    "wallstreet_hi",
                    "wallstreet_lo",
                ):
                    if k in screener_row:
                        flags[k] = bool(_truthy_flag(screener_row.get(k)))

            # Prices/volume (as-of) gate.
            if last_close is None or last_vol is None or avg_vol_20 is None:
                try:
                    # Use a small lookback ending at asof to compute ADV and last price.
                    start = (asof_ts - pd.Timedelta(days=35)).strftime("%Y-%m-%d")
                    end = asof_ts.strftime("%Y-%m-%d")
                    px = self._p.get_eod_prices(sym, start_date=start, end_date=end, period="d")
                    if px is not None and not px.empty:
                        # standardize column names: provider uses 'Close'/'Volume'
                        close_col = "Close" if "Close" in px.columns else ("close" if "close" in px.columns else None)
                        vol_col = "Volume" if "Volume" in px.columns else ("volume" if "volume" in px.columns else None)
                        if close_col and last_close is None:
                            last_close = _safe_float(px[close_col].iloc[-1])
                        if vol_col and (last_vol is None or avg_vol_20 is None):
                            vol_raw = cast(pd.Series, px[vol_col])
                            vol_vals = [_safe_float(v) if v is not None else None for v in vol_raw.tolist()]
                            vol_arr = np.array([np.nan if v is None else float(v) for v in vol_vals], dtype=np.float64)
                            if vol_arr.size:
                                if last_vol is None:
                                    last_vol = _safe_float(vol_arr[-1])
                                if avg_vol_20 is None:
                                    tail = vol_arr[-20:] if vol_arr.size >= 20 else vol_arr
                                    tail_mean = float(np.nanmean(tail)) if tail.size else np.nan
                                    avg_vol_20 = _safe_float(tail_mean)
                except Exception:
                    pass

            if self._cfg.min_price is not None:
                flags["price_ok"] = (last_close is not None) and (last_close >= float(self._cfg.min_price))

            if self._cfg.min_avg_volume is not None:
                flags["avg_volume_ok"] = (avg_vol_20 is not None) and (avg_vol_20 >= float(self._cfg.min_avg_volume))

            # Market cap (best-effort): weekly series, take last <= asof.
            if market_cap is None:
                try:
                    cap_df = self._p.get_historical_market_cap(sym, start_date=None, end_date=asof_ts.strftime("%Y-%m-%d"))
                    if cap_df is not None and not cap_df.empty:
                        # normalize index/col
                        if "market_cap" in cap_df.columns:
                            cap_raw = cast(pd.Series, cap_df["market_cap"])  # provider-dependent dtype
                            cap_vals = [_safe_float(v) if v is not None else None for v in cap_raw.tolist()]
                            cap_arr = np.array([np.nan if v is None else float(v) for v in cap_vals], dtype=np.float64)
                            cap_arr = cap_arr[np.isfinite(cap_arr)]
                            if cap_arr.size:
                                market_cap = _safe_float(cap_arr[-1])
                except Exception:
                    pass

            if self._cfg.min_market_cap is not None:
                flags["market_cap_ok"] = (market_cap is not None) and (market_cap >= float(self._cfg.min_market_cap))

            # Suspension / delisting: architecture hook; this can later be replaced by a dedicated screener endpoint.
            if self._cfg.require_not_suspended:
                # default True unless provider supplies something explicit
                flags["not_suspended"] = not _truthy_flag(profile.get("suspended")) and _truthy_flag(
                    profile.get("not_suspended", True)
                )
            if self._cfg.require_not_delisted:
                flags["not_delisted"] = not _truthy_flag(profile.get("delisted")) and _truthy_flag(
                    profile.get("not_delisted", True)
                )

            eligible = bool(
                flags["exchange_ok"]
                and flags["country_ok"]
                and flags["sector_ok"]
                and (not bool(flags["is_etf"]))
                and (not bool(flags["is_adr"]))
                and (not bool(flags["is_fund"]))
                and flags["market_cap_ok"]
                and flags["avg_volume_ok"]
                and flags["price_ok"]
                and flags["not_suspended"]
                and flags["not_delisted"]
            )

            # Optional but powerful: a single reason code for why a symbol is excluded.
            # This is derived from the first failing gate in a stable priority order.
            exclusion_reason: Optional[str] = None
            if not eligible:
                if self._cfg.exclude_etf and is_etf:
                    exclusion_reason = "IS_ETF"
                elif self._cfg.exclude_adr and is_adr:
                    exclusion_reason = "IS_ADR"
                elif self._cfg.exclude_fund and is_fund:
                    exclusion_reason = "IS_FUND"
                elif not bool(flags["exchange_ok"]):
                    exclusion_reason = "EXCHANGE"
                elif not bool(flags["country_ok"]):
                    exclusion_reason = "COUNTRY"
                elif not bool(flags["sector_ok"]):
                    exclusion_reason = "SECTOR"
                elif not bool(flags["market_cap_ok"]):
                    exclusion_reason = "MARKET_CAP"
                elif not bool(flags["avg_volume_ok"]):
                    exclusion_reason = "AVG_VOLUME"
                elif not bool(flags["price_ok"]):
                    exclusion_reason = "PRICE"
                elif not bool(flags["not_suspended"]):
                    exclusion_reason = "SUSPENDED"
                elif not bool(flags["not_delisted"]):
                    exclusion_reason = "DELISTED"
                else:
                    exclusion_reason = "OTHER"

            rows.append(
                {
                    "date": asof_ts,
                    "symbol": sym,
                    "eligible": eligible,
                    "instrument_type": instrument_type,
                    "exclusion_reason": exclusion_reason,
                    **{f"flag_{k}": v for k, v in flags.items()},
                    "asof_close": last_close,
                    "asof_volume": last_vol,
                    "asof_adv20": avg_vol_20,
                    "asof_market_cap": market_cap,
                    "meta_country": country or None,
                    "meta_sector": sector or None,
                    "meta_exchange": exchange or None,
                }
            )

        df = pd.DataFrame(rows)
        if df.empty:
            return pd.DataFrame(
                columns=[
                    "date",
                    "symbol",
                    "eligible",
                ]
            )
        df["symbol"] = df["symbol"].astype(str).str.upper()
        df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.tz_localize(None)
        return df.sort_values(["symbol"]).reset_index(drop=True)
