"""HF block generator for event risk management (earnings/macro).

This family provides explicit, controllable event risk features for the
portfolio parquet. These are NOT for Mamba prediction - they're for the
portfolio engine's risk management layer.

Features:
    - event_risk_hf_earnings_next_1d: 0/1 flag if earnings within 1 trading day
    - event_risk_hf_earnings_next_3d: 0/1 flag if earnings within 3 trading days
    - event_risk_hf_macro_next_1d: 0/1 flag if major macro event within 1 trading day
    - event_risk_hf_score: 0..1 composite event risk score
    - event_risk_hf_earnings_imminent: continuous 0..1 (decay as days increase)
    - event_risk_hf_macro_imminent: continuous 0..1 for macro events

Data Sources:
    - Earnings: EODHD fundamentals (raw data: Earnings.History)
    - Macro: EODHD economic-events API (raw data: economic-events endpoint)

Routing:
    - ALL columns → RISK role → Portfolio parquet ONLY
    - NOT for Mamba input (event timing is known, not predictive signal)
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ._hf_block_common import (
    prepare_block_payload,
    should_update_block_cache,
    update_block_cache,
)

LOGGER = logging.getLogger(__name__)
BLOCK_NAME = "event_risk_hf"

# Major macro event types from EODHD economic-events
MAJOR_MACRO_EVENTS = ["fomc", "cpi", "nfp", "gdp", "pce"]

# Pattern matching for macro event types (case-insensitive substring matching)
_MACRO_EVENT_PATTERNS: Dict[str, Tuple[str, ...]] = {
    "fomc": ("fomc", "federal open market", "interest rate decision"),
    "cpi": ("consumer price index", "cpi"),
    "nfp": ("nonfarm payroll", "non-farm payroll", "employment situation"),
    "gdp": ("gross domestic product", "gdp"),
    "pce": ("personal consumption expenditure", "pce"),
}


def _fetch_earnings_dates_from_eodhd(
    symbol: str,
    start_date: str,
    end_date: str,
) -> Optional[List[pd.Timestamp]]:
    """Fetch earnings dates directly from EODHD fundamentals (raw data source).
    
    Returns list of earnings announcement dates for the symbol.
    """
    try:
        from src.data_sources.eodhd_provider import get_eodhd_provider
        
        eodhd = get_eodhd_provider()
        fundamentals = eodhd.get_fundamentals(symbol)
        
        if not fundamentals or 'Earnings' not in fundamentals:
            LOGGER.debug("%s: No earnings data in EODHD fundamentals for %s", BLOCK_NAME, symbol)
            return None
        
        earnings_history = fundamentals['Earnings'].get('History', {})
        if not earnings_history:
            LOGGER.debug("%s: No earnings history in EODHD for %s", BLOCK_NAME, symbol)
            return None
        
        # Extract dates from earnings history
        # Format: {'2025-01-30': {'epsActual': 1.85, ...}, '2024-10-31': {...}, ...}
        dates_list = []
        for date_str in earnings_history.keys():
            try:
                dt = pd.to_datetime(date_str)
                dates_list.append(dt)
            except Exception:
                continue
        
        if not dates_list:
            return None
        
        LOGGER.info("%s: Fetched %d earnings dates from EODHD for %s", BLOCK_NAME, len(dates_list), symbol)
        return sorted(dates_list)
        
    except Exception as e:
        LOGGER.warning("%s: Failed to fetch earnings from EODHD for %s: %s", BLOCK_NAME, symbol, e)
        return None


def _fetch_macro_events_from_eodhd(
    start_date: str,
    end_date: str,
) -> Optional[pd.DataFrame]:
    """Fetch macro events directly from EODHD economic-events API (raw data source).
    
    Returns DataFrame with columns: date, event_type (fomc/cpi/nfp/gdp/pce)
    """
    try:
        from src.data_sources.eodhd_provider import get_eodhd_provider
        
        eodhd = get_eodhd_provider()
        
        # Fetch US economic events using correct method name
        events = eodhd.get_economic_events(
            start_date=start_date,
            end_date=end_date,
            country="US",
            limit=1000,
        )
        
        if not events:
            LOGGER.debug("%s: No economic events from EODHD API", BLOCK_NAME)
            return None
        
        # Parse events and classify by type
        parsed_events = []
        for ev in events:
            if not isinstance(ev, dict):
                continue
            
            # Get event date
            date_str = ev.get("date") or ev.get("datetime") or ev.get("Date")
            if not date_str:
                continue
            
            try:
                dt = pd.to_datetime(date_str).normalize()
            except Exception:
                continue
            
            # Get event name/type
            event_name = (
                ev.get("event") or ev.get("type") or 
                ev.get("name") or ev.get("title") or ""
            ).lower()
            
            # Classify the event
            for event_type, patterns in _MACRO_EVENT_PATTERNS.items():
                if any(p in event_name for p in patterns):
                    parsed_events.append({"date": dt, "event_type": event_type})
                    break
        
        if not parsed_events:
            LOGGER.debug("%s: No major macro events found in EODHD response", BLOCK_NAME)
            return None
        
        df = pd.DataFrame(parsed_events)
        LOGGER.info("%s: Fetched %d macro events from EODHD", BLOCK_NAME, len(df))
        return df
        
    except Exception as e:
        LOGGER.warning("%s: Failed to fetch macro events from EODHD: %s", BLOCK_NAME, e)
        return None


def _compute_days_to_next_event(
    panel_dates: pd.Series,
    event_dates: List[pd.Timestamp],
) -> pd.Series:
    """For each panel date, compute days until next event in list."""
    event_dates_sorted = sorted(event_dates)
    
    result = []
    for dt in panel_dates:
        dt_ts = pd.to_datetime(dt).normalize()
        # Find next event >= dt
        future_events = [e for e in event_dates_sorted if e >= dt_ts]
        if future_events:
            days_to = (future_events[0] - dt_ts).days
            result.append(float(days_to))
        else:
            result.append(999.0)  # No future event found
    
    return pd.Series(result, index=panel_dates.index)


def _compute_event_risk_features(
    panel: pd.DataFrame,
    earnings_dates: Optional[List[pd.Timestamp]],
    macro_df: Optional[pd.DataFrame],
) -> pd.DataFrame:
    """Compute event risk features from earnings and macro calendars.
    
    Args:
        panel: DataFrame with 'date' column
        earnings_dates: List of earnings announcement dates from EODHD
        macro_df: DataFrame with 'date' and 'event_type' columns from EODHD
    """
    result = pd.DataFrame(index=panel.index)
    result["date"] = panel["date"] if "date" in panel.columns else panel.index
    
    n_rows = len(panel)
    
    # Initialize all features with safe defaults
    result[f"{BLOCK_NAME}_earnings_next_1d"] = 0.0
    result[f"{BLOCK_NAME}_earnings_next_3d"] = 0.0
    result[f"{BLOCK_NAME}_earnings_imminent"] = 0.0
    result[f"{BLOCK_NAME}_macro_next_1d"] = 0.0
    result[f"{BLOCK_NAME}_macro_imminent"] = 0.0
    result[f"{BLOCK_NAME}_score"] = 0.0
    
    # Process earnings calendar
    if earnings_dates is not None and len(earnings_dates) > 0:
        try:
            # Compute days to next earnings for each panel date
            panel_dates = pd.to_datetime(result["date"])
            days_to_earnings = _compute_days_to_next_event(panel_dates, earnings_dates)
            
            # Binary flags
            result[f"{BLOCK_NAME}_earnings_next_1d"] = (days_to_earnings <= 1).astype(float)
            result[f"{BLOCK_NAME}_earnings_next_3d"] = (days_to_earnings <= 3).astype(float)
            
            # Continuous imminent score: exp(-days/7) so ~0.5 at 5 days, ~0.1 at 16 days
            result[f"{BLOCK_NAME}_earnings_imminent"] = np.exp(-days_to_earnings.clip(lower=0) / 7.0)
            
            LOGGER.info("%s: Computed earnings features from %d earnings dates", BLOCK_NAME, len(earnings_dates))
            
        except Exception as e:
            LOGGER.warning("%s: Failed to process earnings calendar: %s", BLOCK_NAME, e)
    
    # Process macro calendar
    if macro_df is not None and not macro_df.empty:
        try:
            panel_dates = pd.to_datetime(result["date"])
            
            # Compute days to next event for each major macro event type
            days_to_each_event = {}
            for event_type in MAJOR_MACRO_EVENTS:
                event_dates = macro_df[macro_df["event_type"] == event_type]["date"].tolist()
                if event_dates:
                    days_to_each_event[event_type] = _compute_days_to_next_event(panel_dates, event_dates)
            
            if days_to_each_event:
                # Find minimum days to any major macro event
                all_days = pd.DataFrame(days_to_each_event)
                min_days_to_macro = all_days.min(axis=1)
            else:
                min_days_to_macro = pd.Series([999.0] * n_rows)
            
            # Binary flag: any major macro event within 1 day
            result[f"{BLOCK_NAME}_macro_next_1d"] = (min_days_to_macro <= 1).astype(float)
            
            # Continuous imminent score
            result[f"{BLOCK_NAME}_macro_imminent"] = np.exp(-min_days_to_macro.clip(lower=0) / 5.0)
            
            LOGGER.info("%s: Computed macro features from %d events", BLOCK_NAME, len(macro_df))
            
        except Exception as e:
            LOGGER.warning("%s: Failed to process macro calendar: %s", BLOCK_NAME, e)
    
    # Composite event risk score: max of earnings and macro imminent scores
    # Weighted: earnings are more impactful for individual names
    earnings_weight = 0.7
    macro_weight = 0.3
    
    result[f"{BLOCK_NAME}_score"] = (
        earnings_weight * result[f"{BLOCK_NAME}_earnings_imminent"] +
        macro_weight * result[f"{BLOCK_NAME}_macro_imminent"]
    ).clip(0.0, 1.0)
    
    # Governance columns
    has_earnings = 1.0 if (earnings_dates is not None and len(earnings_dates) > 0) else 0.0
    has_macro = 1.0 if (macro_df is not None and not macro_df.empty) else 0.0
    result[f"{BLOCK_NAME}_has_data"] = max(has_earnings, has_macro)
    result[f"{BLOCK_NAME}_activity"] = (has_earnings + has_macro) / 2.0
    result[f"{BLOCK_NAME}_days_since_update"] = 0.0  # Always current (calendar-based)
    result[f"{BLOCK_NAME}_confidence"] = (has_earnings * 0.6 + has_macro * 0.4)
    
    return result


def build(
    symbol: str,
    horizon: int,
    start: str,
    end: str,
    out_path: str,
    raw_source_cfg: Optional[Dict[str, Any]] = None,
    compute_cfg: Optional[Dict[str, Any]] = None,
) -> None:
    """Materialize event_risk_hf block outputs.
    
    Args:
        symbol: Stock symbol (e.g., "AAPL")
        horizon: Forecast horizon in days (e.g., 63)
        start: Start date string (YYYY-MM-DD)
        end: End date string (YYYY-MM-DD)
        out_path: Output parquet path
        raw_source_cfg: Optional raw data source config
        compute_cfg: Optional compute configuration
    """
    LOGGER.info(
        "%s: building block signal for %s h%s [%s → %s]",
        BLOCK_NAME,
        symbol,
        horizon,
        start,
        end,
    )
    
    # Prepare base payload and cache directory
    payload, cache_dir = prepare_block_payload(
        block_name=BLOCK_NAME,
        symbol=symbol,
        horizon=horizon,
        start=start,
        end=end,
        out_path=out_path,
        raw_source_cfg=raw_source_cfg,
        compute_cfg=compute_cfg,
    )
    
    # Fetch earnings calendar directly from EODHD raw data
    earnings_dates = _fetch_earnings_dates_from_eodhd(symbol, start, end)
    if earnings_dates:
        LOGGER.info("%s: fetched %d earnings dates from EODHD for %s", BLOCK_NAME, len(earnings_dates), symbol)
    else:
        LOGGER.warning("%s: no earnings dates available from EODHD for %s", BLOCK_NAME, symbol)
    
    # Fetch macro calendar directly from EODHD raw data
    macro_df = _fetch_macro_events_from_eodhd(start, end)
    if macro_df is not None and not macro_df.empty:
        LOGGER.info("%s: fetched %d macro events from EODHD", BLOCK_NAME, len(macro_df))
    else:
        LOGGER.warning("%s: no macro events available from EODHD", BLOCK_NAME)
    
    # Compute event risk features
    event_risk_features = _compute_event_risk_features(payload, earnings_dates, macro_df)
    
    # Merge with payload
    if "date" in payload.columns and "date" in event_risk_features.columns:
        payload = payload.merge(
            event_risk_features.drop(columns=["date"], errors="ignore"),
            left_index=True,
            right_index=True,
            how="left",
        )
    else:
        for col in event_risk_features.columns:
            if col != "date":
                payload[col] = event_risk_features[col].values
    
    # Write output
    out_file = Path(out_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    payload.to_parquet(out_file, index=False)
    
    # Update block cache
    if should_update_block_cache(out_file, raw_source_cfg, compute_cfg):
        try:
            update_block_cache(payload, cache_dir, symbol, horizon, BLOCK_NAME)
        except Exception as exc:
            LOGGER.warning("%s: block cache update skipped (%s)", BLOCK_NAME, exc)
    
    LOGGER.info("%s: wrote %d rows to %s", BLOCK_NAME, len(payload), out_file)


def get_feature_columns() -> List[str]:
    """Return list of feature columns produced by this block."""
    return [
        f"{BLOCK_NAME}_earnings_next_1d",
        f"{BLOCK_NAME}_earnings_next_3d",
        f"{BLOCK_NAME}_earnings_imminent",
        f"{BLOCK_NAME}_macro_next_1d",
        f"{BLOCK_NAME}_macro_imminent",
        f"{BLOCK_NAME}_score",
        f"{BLOCK_NAME}_has_data",
        f"{BLOCK_NAME}_activity",
        f"{BLOCK_NAME}_days_since_update",
        f"{BLOCK_NAME}_confidence",
    ]


def get_feature_roles() -> Dict[str, str]:
    """Return role mapping for all features (all RISK for portfolio routing)."""
    return {
        f"{BLOCK_NAME}_earnings_next_1d": "risk",
        f"{BLOCK_NAME}_earnings_next_3d": "risk",
        f"{BLOCK_NAME}_earnings_imminent": "risk",
        f"{BLOCK_NAME}_macro_next_1d": "risk",
        f"{BLOCK_NAME}_macro_imminent": "risk",
        f"{BLOCK_NAME}_score": "risk",
        f"{BLOCK_NAME}_has_data": "hygiene",
        f"{BLOCK_NAME}_activity": "hygiene",
        f"{BLOCK_NAME}_days_since_update": "hygiene",
        f"{BLOCK_NAME}_confidence": "hygiene",
    }
