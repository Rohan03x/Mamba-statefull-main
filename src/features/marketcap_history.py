"""Historical market cap / float / size dynamics (EODHD).

Family: marketcap_history

Source:
- EODHD Historical Market Capitalization (Marketcaps) API
  https://eodhd.com/financial-apis/historical-market-capitalization-api/

Notes:
- EODHD marketcap history is typically weekly points. We align to NYSE sessions and
  forward-fill to a daily panel.
- Leakage policy: market cap is computed from EOD price/shares. As a conservative
  policy, we shift published market cap points forward by 1 NYSE session.

Emitted features:
- marketcap_history_mcap
- marketcap_history_log_mcap
- marketcap_history_mcap_chg_{20d,63d,252d}
- marketcap_history_mcap_vol_63d (rolling std of 1d log diff)
- marketcap_history_float_turnover (volume / shares_float, if shares_float exists)
- marketcap_history_turnover_z_252d
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from src.dcf_lab.utils.timealign import nyse_sessions_in_range
from src.dcf_lab.utils.trading_calendar import add_sessions, session_on_or_before


def _safe_log(x: pd.Series) -> pd.Series:
    return np.log(np.maximum(x.astype(float), 1.0))


def _rolling_zscore(x: pd.Series, win: int, *, min_periods: int = 30) -> pd.Series:
    mu = x.rolling(win, min_periods=min_periods).mean()
    sd = x.rolling(win, min_periods=min_periods).std(ddof=0).replace(0.0, np.nan)
    z = (x - mu) / sd
    return z.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def fetch(
    symbol: str,
    start: str,
    end: str,
    *,
    leakage_safe_shift_sessions: int = 1,
    ffill_limit: int = 10,
) -> Optional[pd.DataFrame]:
    try:
        from src.data_sources.eodhd_provider import get_eodhd_provider  # type: ignore
    except Exception:
        return None

    sessions = nyse_sessions_in_range(start, end, tz_aware_utc=False)
    if len(sessions) == 0:
        return None

    # Extend history for rolling stats (252d turnover zscore, 63d vol).
    lookback_days = 1200
    fetch_start = (pd.Timestamp(start) - pd.Timedelta(days=int(lookback_days))).strftime("%Y-%m-%d")
    sessions_ext = nyse_sessions_in_range(fetch_start, end, tz_aware_utc=False)
    if len(sessions_ext) == 0:
        return None

    provider = get_eodhd_provider()
    if not getattr(provider, "api_key", None):
        return None

    def _fallback_market_cap_from_fundamentals_and_price() -> Optional[tuple[pd.Series, pd.Series]]:
        """Fallback market cap proxy = Close * shares_outstanding (quarterly, forward-filled).

        Rationale: EODHD historical-market-cap often starts late (~2021 for many tickers).
        This fallback provides a non-zero, leakage-safe size signal across the full WF span.

        Returns:
            (mcap_series, shares_series) on sessions_ext index, or None.
        """

        try:
            fund = provider.get_fundamentals(symbol)
        except Exception:
            fund = None
        if not fund:
            return None

        financials = (fund or {}).get("Financials", {})
        balance_q = (financials.get("Balance_Sheet", {}) or {}).get("quarterly", {})
        balance_y = (financials.get("Balance_Sheet", {}) or {}).get("yearly", {})

        shares_points = pd.Series(index=sessions_ext, dtype=float)

        def _ingest_balance(balance_dict: dict) -> None:
            for date_str, payload in (balance_dict or {}).items():
                if not isinstance(payload, dict):
                    continue
                val = (
                    payload.get("commonStockSharesOutstanding")
                    or payload.get("commonStockShares")
                    or payload.get("sharesOutstanding")
                    or payload.get("commonStockSharesIssued")
                )
                if val is None:
                    continue
                try:
                    shares = float(val)
                except Exception:
                    continue
                if not np.isfinite(shares) or shares <= 0:
                    continue

                try:
                    asof = session_on_or_before(pd.to_datetime(date_str, errors="coerce").tz_localize(None).normalize())
                except Exception:
                    continue
                if leakage_safe_shift_sessions:
                    asof = add_sessions(asof, int(leakage_safe_shift_sessions))
                if asof in shares_points.index:
                    shares_points.loc[asof] = shares

        _ingest_balance(balance_y)
        _ingest_balance(balance_q)

        shares = pd.to_numeric(shares_points, errors="coerce").ffill().fillna(0.0).clip(lower=0.0)
        if float(shares.max() or 0.0) <= 0.0:
            return None

        try:
            px = provider.get_eod_prices(symbol, start_date=fetch_start, end_date=end, period="d")
        except Exception:
            px = None
        if px is None or px.empty:
            return None

        close_col = "Close" if "Close" in px.columns else ("close" if "close" in px.columns else None)
        if close_col is None:
            return None

        close = px[close_col].copy()
        close.index = pd.to_datetime(close.index, errors="coerce").tz_localize(None).normalize()
        close = pd.to_numeric(close, errors="coerce").fillna(0.0)
        close = close.reindex(sessions_ext).ffill().fillna(0.0)

        mcap = (close * shares).replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(lower=0.0)
        return mcap.astype(float), shares.astype(float)

    raw = provider.get_historical_market_cap(symbol, start_date=fetch_start, end_date=end)

    # Place raw points onto NYSE sessions; apply conservative +1 session shift.
    mcap_points = pd.Series(index=sessions_ext, dtype=float)
    shares_float_points = pd.Series(index=sessions_ext, dtype=float)

    if raw is not None and not raw.empty and "market_cap" in raw.columns:
        for ts, row in raw.iterrows():
            try:
                asof = session_on_or_before(pd.Timestamp(ts).normalize())
            except Exception:
                continue
            if leakage_safe_shift_sessions:
                asof = add_sessions(asof, int(leakage_safe_shift_sessions))
            if asof not in mcap_points.index:
                continue

            try:
                mcap_points.loc[asof] = float(row.get("market_cap"))
            except Exception:
                pass

            if "shares_float" in raw.columns:
                try:
                    val = row.get("shares_float")
                    if val is not None and np.isfinite(float(val)):
                        shares_float_points.loc[asof] = float(val)
                except Exception:
                    pass

    mcap = mcap_points.ffill(limit=int(ffill_limit))
    mcap = pd.to_numeric(mcap, errors="coerce").fillna(0.0).clip(lower=0.0)

    shares_float = shares_float_points.ffill(limit=int(ffill_limit))
    shares_float = pd.to_numeric(shares_float, errors="coerce").fillna(0.0).clip(lower=0.0)

    fallback_used = False
    fallback = _fallback_market_cap_from_fundamentals_and_price()
    if fallback is not None:
        fallback_mcap, fallback_shares = fallback
        # Fill gaps (or missing full history) from the fallback proxy.
        mcap = mcap.where(mcap > 0.0, fallback_mcap)
        # shares_float is optional; use shares outstanding as a proxy for turnover denom if float is missing.
        shares_float = shares_float.where(shares_float > 0.0, fallback_shares)
        fallback_used = True

    if float(mcap.max() or 0.0) <= 0.0:
        return None

    out = pd.DataFrame(index=sessions_ext)
    out["marketcap_history_has_data"] = (mcap > 0.0).astype(float)

    out["marketcap_history_mcap"] = mcap.astype(float)
    out["marketcap_history_log_mcap"] = _safe_log(mcap)

    log_mcap = out["marketcap_history_log_mcap"]
    out["marketcap_history_mcap_chg_20d"] = (log_mcap - log_mcap.shift(20)).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    out["marketcap_history_mcap_chg_63d"] = (log_mcap - log_mcap.shift(63)).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    out["marketcap_history_mcap_chg_252d"] = (log_mcap - log_mcap.shift(252)).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    chg_1d = log_mcap.diff().replace([np.inf, -np.inf], np.nan).fillna(0.0)
    out["marketcap_history_mcap_vol_63d"] = (
        chg_1d.rolling(63, min_periods=20).std(ddof=0).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    )

    # Turnover proxy if float shares exists (requires OHLCV volume).
    float_turnover = pd.Series(0.0, index=sessions_ext)
    try:
        price_df = provider.get_eod_prices(symbol, start_date=fetch_start, end_date=end, period="d")
        if price_df is not None and not price_df.empty and "Volume" in price_df.columns:
            vol = price_df["Volume"].copy()
            vol.index = pd.to_datetime(vol.index, errors="coerce").tz_localize(None).normalize()
            vol = pd.to_numeric(vol, errors="coerce").fillna(0.0)
            vol = vol.reindex(sessions_ext).fillna(0.0)
            denom = shares_float.replace(0.0, np.nan)
            float_turnover = (vol / denom).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    except Exception:
        pass

    out["marketcap_history_float_turnover"] = float_turnover.astype(float)
    out["marketcap_history_turnover_z_252d"] = _rolling_zscore(out["marketcap_history_float_turnover"], 252)

    out.attrs["telemetry"] = {
        "status": "ok",
        "source": "EODHD.historical-market-cap" if (raw is not None and not raw.empty) else "fallback:fundamentals_x_price",
        "fallback_used": bool(fallback_used),
    }
    out.attrs["provenance"] = {
        "source": "EODHD.historical-market-cap" if (raw is not None and not raw.empty) else "fallback:fundamentals_x_price",
        "leakage": f"market cap points shifted +{int(leakage_safe_shift_sessions)} NYSE session(s)",
        "notes": "market cap history is often weekly; forward-filled to sessions. If missing, uses Close * shares_outstanding fallback.",
    }
    out.attrs["feature_counts"] = {"generated": int(len(out.columns))}

    out = out.loc[(out.index >= sessions[0]) & (out.index <= sessions[-1])].copy()
    return out
