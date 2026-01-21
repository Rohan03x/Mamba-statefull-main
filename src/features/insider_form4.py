"""SEC Form 4 insider transactions (EODHD) feature family.

Data source:
- EODHD Insider Transactions API (SEC "Form 4")
  https://eodhd.com/financial-apis/insider-transactions-api/

Core policy (no-lookahead):
- Aggregate raw transactions by *filing_date*.
- Apply the aggregated signal on the first tradable NYSE session *after* the filing date:
  - If filing_date is a trading session: effective_session = filing_date + 1 session
  - If filing_date is not a trading session: effective_session = first session after filing_date

Outputs are aligned to the NYSE (XNYS) trading calendar sessions.

Notes:
- This family is sparse by nature; many days will be exactly zero.
- Field names in EODHD payloads can vary; parsing is defensive.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from src.dcf_lab.utils.timealign import nyse_sessions_in_range
from src.dcf_lab.utils.trading_calendar import add_sessions, session_on_or_after, session_on_or_before


def _norm_symbol(symbol: str) -> str:
    s = (symbol or "").strip().upper()
    if not s:
        return s
    if "." not in s:
        return f"{s}.US"
    return s


def _pick_key(row: Dict[str, Any], candidates: Sequence[str]) -> Optional[str]:
    for k in candidates:
        if k in row and row.get(k) is not None:
            return k
    return None


def _to_float(v: Any) -> float:
    if v is None:
        return 0.0
    try:
        out = float(v)
        if np.isfinite(out):
            return out
        return 0.0
    except Exception:
        try:
            out = float(pd.to_numeric(v, errors="coerce"))
            if np.isfinite(out):
                return out
        except Exception:
            return 0.0
    return 0.0


def _to_str(v: Any) -> str:
    if v is None:
        return ""
    try:
        return str(v)
    except Exception:
        return ""


def _is_buy(code: str) -> bool:
    c = (code or "").strip().upper()
    return c == "P" or c.startswith("PURCHASE")


def _is_sell(code: str) -> bool:
    c = (code or "").strip().upper()
    return c == "S" or c.startswith("SALE")


def _is_exec_role(title: str) -> bool:
    t = (title or "").strip().lower()
    if not t:
        return False
    # Conservative: CEO / Chief Executive Officer variants.
    return ("chief executive" in t) or (t == "ceo") or (" ceo" in t) or ("ceo " in t)


def _days_since_event(pulse: pd.Series, *, never_value: float = 9999.0) -> pd.Series:
    values = pulse.values
    out = np.empty(len(values), dtype=float)
    last_idx = None
    for i, v in enumerate(values):
        if bool(v) and float(v) != 0.0:
            last_idx = i
            out[i] = 0.0
            continue
        if last_idx is None:
            out[i] = float(never_value)
        else:
            out[i] = float(i - last_idx)
    return pd.Series(out, index=pulse.index)


def _effective_session_from_filing_date(filing_date: pd.Timestamp) -> pd.Timestamp:
    """Map filing_date -> effective session under the no-lookahead rule."""
    filing_date = pd.Timestamp(filing_date).normalize()

    before = session_on_or_before(filing_date)
    after = session_on_or_after(filing_date)

    # filing_date is a session iff session_before == session_after == filing_date.
    is_session = (before == after) and (after == filing_date)
    if is_session:
        return add_sessions(after, 1)
    return after


def _rolling_zscore(x: pd.Series, window: int, *, min_periods: int = 20) -> pd.Series:
    mean = x.rolling(window, min_periods=min_periods).mean()
    std = x.rolling(window, min_periods=min_periods).std(ddof=0)
    z = (x - mean) / std.replace(0.0, np.nan)
    return z.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def fetch(
    symbol: str,
    start: str,
    end: str,
    *,
    rolling_windows: Sequence[int] = (5, 20, 63),
    zscore_window: int = 252,
    include_market_cap: bool = True,
) -> Optional[pd.DataFrame]:
    """Fetch insider Form 4 features for one symbol.

    Args:
        symbol: equity ticker (e.g. AAPL)
        start/end: YYYY-MM-DD bounds (panel bounds)
        rolling_windows: windows (in sessions) for intensities
        zscore_window: sessions window for z-score of net_value_20d
        include_market_cap: if True, attempts to fetch EODHD historical market cap series (weekly).

    Returns:
        DataFrame indexed by tz-naive NYSE sessions.
    """

    try:
        from src.data_sources.eodhd_provider import get_eodhd_provider  # type: ignore
    except Exception:
        return None

    sessions = nyse_sessions_in_range(start, end, tz_aware_utc=False)
    if len(sessions) == 0:
        return None

    # Extend history to support rolling and z-score windows.
    max_roll = int(max(list(rolling_windows) + [zscore_window, 20]))
    # Convert session count to approximate calendar-day lookback.
    fetch_start = (pd.Timestamp(start) - pd.Timedelta(days=int(max(420, max_roll * 2)))).strftime("%Y-%m-%d")

    provider = get_eodhd_provider()
    if not getattr(provider, "api_key", None):
        return None

    code = _norm_symbol(symbol)
    raw = provider.get_insider_transactions(code, start_date=fetch_start, end_date=end, limit=1000)
    if raw is None:
        return None

    # Extended session index (needed for rolling windows).
    sessions_ext = nyse_sessions_in_range(fetch_start, end, tz_aware_utc=False)
    if len(sessions_ext) == 0:
        return None

    out = pd.DataFrame(index=sessions_ext)
    out["insider_form4_has_data"] = 1.0

    # Initialize core daily aggregates.
    cols_1d = [
        "insider_form4_net_shares_1d",
        "insider_form4_net_value_1d",
        "insider_form4_buy_shares_1d",
        "insider_form4_sell_shares_1d",
        "insider_form4_buy_value_1d",
        "insider_form4_sell_value_1d",
        "insider_form4_num_trades_1d",
        "insider_form4_num_buy_trades_1d",
        "insider_form4_num_sell_trades_1d",
        "insider_form4_num_unique_insiders_1d",
        "insider_form4_num_unique_buy_insiders_1d",
        "insider_form4_num_unique_sell_insiders_1d",
        "insider_form4_exec_net_value_1d",
        "insider_form4_exec_buy_value_1d",
        "insider_form4_exec_sell_value_1d",
    ]
    for c in cols_1d:
        out[c] = 0.0

    if len(raw) == 0:
        # No filings in fetch window; keep has_data=1 to indicate provider access.
        out.attrs["telemetry"] = {
            "status": "ok",
            "source": "EODHD.insider-transactions",
            "note": "empty payload",
        }
        out = out.loc[(out.index >= sessions[0]) & (out.index <= sessions[-1])].copy()
        out.attrs["feature_counts"] = {"generated": int(len(out.columns))}
        out.attrs["provenance"] = {"source": "EODHD.insider-transactions", "leakage": "shift to next session"}
        return out

    # Parse into a DataFrame for aggregation.
    df = pd.DataFrame([x for x in raw if isinstance(x, dict)])
    if df.empty:
        return None

    # EODHD payloads commonly include:
    # - reportDate: SEC report/filing date
    # - date: filing date (provider-level)
    # - transactionDate: transaction date
    # We treat reportDate as the preferred "filing date" field for no-lookahead.
    filing_key = _pick_key(
        df.iloc[0].to_dict(),
        [
            "filing_date",
            "filingDate",
            "FilingDate",
            "reportDate",
            "report_date",
            "reportingDate",
            "ReportingDate",
            "filedAt",
            "date",
            "Date",
        ],
    )
    if filing_key is None:
        # Try scan columns if first row lacked it.
        for candidate in [
            "filing_date",
            "filingDate",
            "FilingDate",
            "reportDate",
            "report_date",
            "ReportingDate",
            "reportingDate",
            "date",
            "Date",
        ]:
            if candidate in df.columns:
                filing_key = candidate
                break
    if filing_key is None:
        return None

    # Normalize filing_date.
    df["_filing_date"] = pd.to_datetime(df[filing_key], errors="coerce").dt.normalize()
    df = df.dropna(subset=["_filing_date"])
    if df.empty:
        return None

    # Identify core columns.
    code_candidates = [
        "transaction_code",
        "transactionCode",
        "transaction_type",
        "transactionType",
        "transaction",
        "type",
        "TransactionCode",
    ]
    shares_candidates = [
        "shares",
        "transactionShares",
        "transaction_shares",
        "transactionAmount",
        "transaction_amount",
        "Shares",
        "amount",
        "Amount",
    ]
    price_candidates = [
        "price",
        "transactionPrice",
        "transaction_price",
        "Price",
    ]
    value_candidates = [
        "value",
        "transactionValue",
        "transaction_value",
        "Value",
    ]
    name_candidates = [
        "insiderName",
        "insider_name",
        "name",
        "ownerName",
        "OwnerName",
        "insider",
        "Insider",
    ]
    title_candidates = [
        "title",
        "insiderTitle",
        "insider_title",
        "officerTitle",
        "OfficerTitle",
        "position",
        "Position",
    ]

    tx_code_key = None
    for k in code_candidates:
        if k in df.columns:
            tx_code_key = k
            break

    # Build per-row numeric fields.
    def _row_tx_code(row: pd.Series) -> str:
        if tx_code_key is None:
            return ""
        return _to_str(row.get(tx_code_key))

    def _row_shares(row: pd.Series) -> float:
        for k in shares_candidates:
            if k in row.index:
                return _to_float(row.get(k))
        return 0.0

    def _row_price(row: pd.Series) -> float:
        for k in price_candidates:
            if k in row.index:
                return _to_float(row.get(k))
        return 0.0

    def _row_value(row: pd.Series) -> float:
        for k in value_candidates:
            if k in row.index:
                v = _to_float(row.get(k))
                if v != 0.0:
                    return v
        # Fallback: shares * price.
        return _row_shares(row) * _row_price(row)

    def _row_name(row: pd.Series) -> str:
        for k in name_candidates:
            if k in row.index:
                s = _to_str(row.get(k)).strip()
                if s:
                    return s
        return ""

    def _row_title(row: pd.Series) -> str:
        for k in title_candidates:
            if k in row.index:
                s = _to_str(row.get(k)).strip()
                if s:
                    return s
        return ""

    df["_tx_code"] = df.apply(_row_tx_code, axis=1)
    df["_shares"] = df.apply(_row_shares, axis=1)
    df["_value"] = df.apply(_row_value, axis=1)
    df["_insider_name"] = df.apply(_row_name, axis=1)
    df["_insider_title"] = df.apply(_row_title, axis=1)

    df["_is_buy"] = df["_tx_code"].map(_is_buy)
    df["_is_sell"] = df["_tx_code"].map(_is_sell)
    df["_is_exec"] = df["_insider_title"].map(_is_exec_role)

    # Aggregate by filing date.
    grp = df.groupby("_filing_date", dropna=True)

    daily_rows: List[Dict[str, Any]] = []
    for filing_date, g in grp:
        buy = g[g["_is_buy"]]
        sell = g[g["_is_sell"]]

        buy_shares = float(buy["_shares"].sum()) if not buy.empty else 0.0
        sell_shares = float(sell["_shares"].sum()) if not sell.empty else 0.0
        buy_value = float(buy["_value"].sum()) if not buy.empty else 0.0
        sell_value = float(sell["_value"].sum()) if not sell.empty else 0.0

        exec_buy = buy[buy["_is_exec"]]
        exec_sell = sell[sell["_is_exec"]]
        exec_buy_value = float(exec_buy["_value"].sum()) if not exec_buy.empty else 0.0
        exec_sell_value = float(exec_sell["_value"].sum()) if not exec_sell.empty else 0.0

        unique_insiders = int(g["_insider_name"].nunique()) if "_insider_name" in g.columns else 0
        unique_buy_insiders = int(buy["_insider_name"].nunique()) if not buy.empty else 0
        unique_sell_insiders = int(sell["_insider_name"].nunique()) if not sell.empty else 0

        daily_rows.append(
            {
                "filing_date": pd.Timestamp(filing_date).normalize(),
                "num_trades": float(len(g)),
                "num_buy_trades": float(len(buy)),
                "num_sell_trades": float(len(sell)),
                "buy_shares": buy_shares,
                "sell_shares": sell_shares,
                "buy_value": buy_value,
                "sell_value": sell_value,
                "net_shares": buy_shares - sell_shares,
                "net_value": buy_value - sell_value,
                "unique_insiders": float(unique_insiders),
                "unique_buy_insiders": float(unique_buy_insiders),
                "unique_sell_insiders": float(unique_sell_insiders),
                "exec_buy_value": exec_buy_value,
                "exec_sell_value": exec_sell_value,
                "exec_net_value": exec_buy_value - exec_sell_value,
            }
        )

    daily = pd.DataFrame(daily_rows)
    if daily.empty:
        return None

    # Shift to effective sessions.
    daily["effective_session"] = daily["filing_date"].map(_effective_session_from_filing_date)
    daily = daily.dropna(subset=["effective_session"])

    daily = daily.groupby("effective_session", dropna=True).sum(numeric_only=True).sort_index()

    # Write into output frame.
    for idx, row in daily.iterrows():
        if idx not in out.index:
            continue
        out.loc[idx, "insider_form4_net_shares_1d"] = float(row.get("net_shares", 0.0))
        out.loc[idx, "insider_form4_net_value_1d"] = float(row.get("net_value", 0.0))
        out.loc[idx, "insider_form4_buy_shares_1d"] = float(row.get("buy_shares", 0.0))
        out.loc[idx, "insider_form4_sell_shares_1d"] = float(row.get("sell_shares", 0.0))
        out.loc[idx, "insider_form4_buy_value_1d"] = float(row.get("buy_value", 0.0))
        out.loc[idx, "insider_form4_sell_value_1d"] = float(row.get("sell_value", 0.0))
        out.loc[idx, "insider_form4_num_trades_1d"] = float(row.get("num_trades", 0.0))
        out.loc[idx, "insider_form4_num_buy_trades_1d"] = float(row.get("num_buy_trades", 0.0))
        out.loc[idx, "insider_form4_num_sell_trades_1d"] = float(row.get("num_sell_trades", 0.0))
        out.loc[idx, "insider_form4_num_unique_insiders_1d"] = float(row.get("unique_insiders", 0.0))
        out.loc[idx, "insider_form4_num_unique_buy_insiders_1d"] = float(row.get("unique_buy_insiders", 0.0))
        out.loc[idx, "insider_form4_num_unique_sell_insiders_1d"] = float(row.get("unique_sell_insiders", 0.0))
        out.loc[idx, "insider_form4_exec_buy_value_1d"] = float(row.get("exec_buy_value", 0.0))
        out.loc[idx, "insider_form4_exec_sell_value_1d"] = float(row.get("exec_sell_value", 0.0))
        out.loc[idx, "insider_form4_exec_net_value_1d"] = float(row.get("exec_net_value", 0.0))

    # Cluster buy signal: multiple distinct buyers on the same effective day.
    out["insider_form4_cluster_buy_1d"] = (
        (out["insider_form4_buy_value_1d"] > 0.0)
        & (
            (out["insider_form4_num_unique_buy_insiders_1d"] >= 3.0)
            | (out["insider_form4_num_buy_trades_1d"] >= 3.0)
        )
    ).astype(float)

    # Days-since signals.
    buy_pulse = (out["insider_form4_buy_value_1d"] > 0.0).astype(float)
    sell_pulse = (out["insider_form4_sell_value_1d"] > 0.0).astype(float)
    out["insider_form4_days_since_buy"] = _days_since_event(buy_pulse)
    out["insider_form4_days_since_sell"] = _days_since_event(sell_pulse)

    # Rolling intensities.
    for w in rolling_windows:
        w = int(w)
        out[f"insider_form4_net_value_{w}d"] = out["insider_form4_net_value_1d"].rolling(w, min_periods=1).sum()
        out[f"insider_form4_buy_value_{w}d"] = out["insider_form4_buy_value_1d"].rolling(w, min_periods=1).sum()
        out[f"insider_form4_sell_value_{w}d"] = out["insider_form4_sell_value_1d"].rolling(w, min_periods=1).sum()
        out[f"insider_form4_num_trades_{w}d"] = out["insider_form4_num_trades_1d"].rolling(w, min_periods=1).sum()

    # Z-score on net_value_20d.
    out["insider_form4_net_value_20d_z252"] = _rolling_zscore(out["insider_form4_net_value_20d"], int(zscore_window), min_periods=20)

    # Market cap normalization (best-effort).
    out["insider_form4_has_mcap"] = 0.0
    out["insider_form4_net_value_pct_mcap_1d"] = 0.0
    if include_market_cap:
        try:
            mcap = provider.get_historical_market_cap(code, start_date=fetch_start, end_date=end)
        except Exception:
            mcap = None
        if mcap is not None and not mcap.empty and "market_cap" in mcap.columns:
            mcap_series = mcap["market_cap"].copy()
            mcap_series.index = pd.to_datetime(mcap_series.index).normalize()
            mcap_series = mcap_series[~mcap_series.index.duplicated(keep="last")].sort_index()
            # Forward-fill weekly points to sessions; do not back-fill.
            aligned = mcap_series.reindex(out.index, method="ffill")
            has = aligned.notna() & (aligned > 0)
            out.loc[has, "insider_form4_has_mcap"] = 1.0
            pct = (out["insider_form4_net_value_1d"] / aligned.where(has, np.nan)).replace([np.inf, -np.inf], np.nan)
            out["insider_form4_net_value_pct_mcap_1d"] = pct.fillna(0.0)

    out = out.replace([np.inf, -np.inf], np.nan)

    # Trim to requested sessions.
    out = out.loc[(out.index >= sessions[0]) & (out.index <= sessions[-1])].copy()

    out.attrs["telemetry"] = {
        "status": "ok",
        "source": "EODHD.insider-transactions",
        "include_market_cap": bool(include_market_cap),
        "generated_features": int(len(out.columns)),
    }
    out.attrs["feature_counts"] = {"generated": int(len(out.columns))}
    out.attrs["provenance"] = {
        "source": "EODHD.insider-transactions",
        "aggregation": "filing_date",
        "leakage": "shift to next session",
    }

    return out
