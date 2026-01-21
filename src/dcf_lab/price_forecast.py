from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.statespace.sarimax import SARIMAX

from .news import score_sentiment
from .providers.dev_yf import YFinanceProvider

# Constants
ADJ_CLOSE_LITERAL = "adj close"

# Check for FinBERT sentiment analysis availability
try:
    from .finbert_sentiment import (
        _lazy_import_transformers,
        build_finbert_sentiment_features,
    )

    # Trigger lazy import to check if transformers is actually available
    FINBERT_AVAILABLE = _lazy_import_transformers()
    if not FINBERT_AVAILABLE:
        import logging
        logging.getLogger(__name__).debug("FinBERT not available, using basic sentiment analysis (transformers not installed)")
except ImportError:
    FINBERT_AVAILABLE = False
    import logging
    logging.getLogger(__name__).debug("FinBERT not available, using basic sentiment analysis (import error)")


# Define constants
COL_DATE = "date"
COL_ADJ_CLOSE = "adj_close"
COL_CLOSE = "close"
DEFAULT_COLUMNS = [COL_DATE, COL_ADJ_CLOSE]


def _process_multi_index_columns(df):
    """Helper function to process DataFrames with MultiIndex columns"""
    data = {COL_DATE: df.index}

    # Extract data from each column using first level as the new column name
    for col in df.columns:
        col_name = col[0].lower()  # Use first level (e.g., 'Close', 'Open')
        data[col_name] = df[col].values

    # Create new DataFrame with flattened column names
    df_processed = pd.DataFrame(data)

    # Ensure we have adj_close column if needed
    if COL_CLOSE in df_processed.columns and COL_ADJ_CLOSE not in df_processed.columns:
        df_processed[COL_ADJ_CLOSE] = df_processed[COL_CLOSE]

    return df_processed


def _process_standard_columns(df):
    """Helper function to process DataFrames with standard columns"""
    # Standardize columns to lowercase
    df_processed = df.rename(columns=str.lower)

    # Handle date in index
    df_processed = df_processed.reset_index()

    # Ensure we have an adj_close column
    if ADJ_CLOSE_LITERAL in df_processed.columns:
        df_processed = df_processed.rename(
            columns={ADJ_CLOSE_LITERAL: COL_ADJ_CLOSE})

    if COL_ADJ_CLOSE not in df_processed.columns and COL_CLOSE in df_processed.columns:
        df_processed[COL_ADJ_CLOSE] = df_processed[COL_CLOSE]

    # Ensure date column is correctly named
    if COL_DATE not in df_processed.columns and "Date" in df_processed.columns:
        df_processed = df_processed.rename(columns={"Date": COL_DATE})
    elif COL_DATE not in df_processed.columns and df_processed.columns[0] != COL_DATE:
        df_processed = df_processed.rename(
            columns={df_processed.columns[0]: COL_DATE})

    return df_processed


def fetch_prices(
        ticker: str,
        start: str = "2012-01-01",
        end: str | None = None,
        interval: str = "1d") -> pd.DataFrame:
    """
    Fetch historical price data for a ticker from Yahoo Finance.
    Updated to handle different yfinance versions and column formats.
    """
    try:
        from .utils import sanitize_dt_index
        
        print(
            f"Fetching data for {ticker} from {start} to {end if end else 'now'}")
        df = yf.download(
            ticker,
            start=start,
            end=end,
            interval=interval,
            auto_adjust=True,
            progress=False)

        if df is None or df.empty:
            print(f"No data retrieved for {ticker}")
            return pd.DataFrame(columns=DEFAULT_COLUMNS)

        print(f"Data shape: {df.shape}, columns: {df.columns.tolist()}")

        # Process DataFrame based on column type
        if isinstance(df.columns, pd.MultiIndex):
            df = _process_multi_index_columns(df)
        else:
            df = _process_standard_columns(df)

        # Global datetime sanitation: make timestamps tz-naive UTC
        df = sanitize_dt_index(df)
        
        # Convert date column to datetime and clean data
        print(f"Processed columns: {df.columns.tolist()}")
        df[COL_DATE] = pd.to_datetime(df[COL_DATE], errors="coerce")
        result = df.dropna(
            subset=[
                COL_DATE,
                COL_ADJ_CLOSE]).sort_values(COL_DATE)
        print(f"Final data shape: {result.shape}")
        return result

    except Exception as e:
        print(f"Error fetching price data for {ticker}: {e}")
        return pd.DataFrame(columns=DEFAULT_COLUMNS)


def add_tech_indicators(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["ret_1d"] = out["adj_close"].pct_change()
    out["ret_5d"] = out["adj_close"].pct_change(5)
    out["ma20"] = out["adj_close"].rolling(20).mean()
    out["ma50"] = out["adj_close"].rolling(50).mean()
    out["mom_10"] = out["adj_close"].pct_change(10)
    out["vol_20"] = out["adj_close"].pct_change().rolling(20).std()
    # RSI
    delta = out["adj_close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean().abs()
    rs = gain / loss
    out["rsi14"] = 100 - (100 / (1 + rs))
    # MACD
    ema12 = out["adj_close"].ewm(span=12, adjust=False).mean()
    ema26 = out["adj_close"].ewm(span=26, adjust=False).mean()
    out["macd"] = ema12 - ema26
    out["macd_sig"] = out["macd"].ewm(span=9, adjust=False).mean()
    return out.dropna()


def build_sentiment_features(ticker: str, days: int = 60) -> pd.DataFrame:
    prov = YFinanceProvider()
    # Use timezone-aware datetime objects
    end = datetime.now(tz=timezone.utc)
    start = end - timedelta(days=days)
    items = prov.get_news(ticker, days=days)

    if not items:
        # Return empty DataFrame with all possible columns
        return pd.DataFrame(
            columns=[
                "date",
                "sent_7d",
                "sent_30d",
                "finbert_sent_7d",
                "finbert_sent_30d"]).set_index("date")

    # Try to use FinBERT if available
    if FINBERT_AVAILABLE:
        try:
            print(
                f"Using FinBERT for sentiment analysis on {len(items)} news items")
            finbert_sent = build_finbert_sentiment_features(
                news_items=items,
                start_date=pd.Timestamp(start),
                end_date=pd.Timestamp(end)
            )
            # If FinBERT analysis was successful, return that data
            if not finbert_sent.empty:
                # Also add traditional sentiment for backward compatibility
                basic_sent = _build_basic_sentiment_features(items, start, end)
                # Combine both sentiment analyses
                combined = pd.concat([basic_sent, finbert_sent], axis=1)
                return combined.fillna(method="ffill")
        except Exception as e:
            print(
                f"Error using FinBERT sentiment: {str(e)}, falling back to basic sentiment")

    # Fallback to basic sentiment
    return _build_basic_sentiment_features(items, start, end)


def _build_basic_sentiment_features(items, start, end) -> pd.DataFrame:
    """Legacy basic sentiment analysis function"""
    rows = []
    for it in items:
        ts = it.get("date")
        try:
            # Use timezone-aware datetime object
            dt = datetime.fromtimestamp(int(ts), tz=timezone.utc)
        except Exception:
            dt = None
        if dt and start <= dt <= end:
            s = score_sentiment(it.get("headline", "")) or 0.0
            rows.append({"date": dt.date(), "score": s})
    if not rows:
        return pd.DataFrame(
            columns=[
                "date",
                "sent_7d",
                "sent_30d"]).set_index("date")
    df = pd.DataFrame(rows)
    daily = df.groupby("date")["score"].mean().to_frame("score")
    daily.index = pd.to_datetime(daily.index)
    daily = daily.asfreq("D").fillna(method="ffill")
    daily["sent_7d"] = daily["score"].rolling(7, min_periods=1).mean()
    daily["sent_30d"] = daily["score"].rolling(30, min_periods=1).mean()
    return daily[["sent_7d", "sent_30d"]]


def make_supervised(df: pd.DataFrame,
                    lookback: int = 20) -> tuple[np.ndarray,
                                                 np.ndarray,
                                                 list[str]]:
    # Base features
    base_cols = [
        "ret_1d",
        "ret_5d",
        "ma20",
        "ma50",
        "mom_10",
        "vol_20",
        "rsi14",
        "macd",
        "macd_sig"]

    # Add sentiment features - include both basic and FinBERT if available
    sentiment_cols = ["sent_7d", "sent_30d"]
    finbert_cols = ["finbert_sent_7d", "finbert_sent_30d"]

    # Check which sentiment columns are available in the dataframe
    available_sent_cols = [col for col in sentiment_cols if col in df.columns]
    available_finbert_cols = [col for col in finbert_cols if col in df.columns]

    # Combine all available features
    cols = base_cols + available_sent_cols + available_finbert_cols

    # Filter to only include columns that exist in the dataframe
    available_cols = [c for c in cols if c in df.columns]

    if len(available_cols) < 3:  # Require at least a few features
        print(
            f"Warning: Not enough feature columns available. Found: {available_cols}")
        return np.array([]), np.array([]), []

    # Use only available columns
    use = df[["adj_close"] + available_cols].dropna()
    ret_next = use["adj_close"].pct_change().shift(-1)

    X, y = [], []
    for i in range(lookback, len(use)-1):
        X.append(use[available_cols].iloc[i-lookback:i].values.flatten())
        y.append(ret_next.iloc[i])

    names = [f"{c}_t-{k}" for k in range(lookback, 0, -1)
             for c in available_cols]

    if not X:
        return np.array([]), np.array([]), []

    return np.asarray(X), np.asarray(y), names


def _handle_multiindex_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Handle MultiIndex columns by flattening them"""
    # Create a mapping that prioritizes the first level (Close, Open, etc.)
    column_mapping = {col: col[0].lower() for col in df.columns}

    # Rename columns using the mapping
    return df.rename(columns=column_mapping)


def _handle_date_column(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure the DataFrame has a proper date column"""
    result = df.copy()

    # Handle Date in index
    if result.index.name == 'Date' or (
        hasattr(
            result.index,
            'name') and result.index.name == 'Date'):
        result = result.reset_index()
    elif isinstance(result.index, pd.DatetimeIndex):
        result = result.reset_index()
        if 'index' in result.columns:
            result = result.rename(columns={'index': 'date'})

    # Ensure date column exists
    if "date" not in result.columns:
        # Try to find a datetime column
        date_col = next(
            (col for col in result.columns
             if pd.api.types.is_datetime64_any_dtype(result[col])),
            None)
        if date_col:
            result = result.rename(columns={date_col: "date"})
        # Try the Date column
        elif 'Date' in result.columns:
            result = result.rename(columns={'Date': 'date'})
        # Default to first column
        elif result.columns[0] != "date":
            result = result.rename(columns={result.columns[0]: "date"})

    return result


def _handle_price_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure the DataFrame has adj_close column"""
    result = df.copy()

    # Ensure adj_close column exists
    if COL_ADJ_CLOSE not in result.columns:
        # Try common column names for adjusted close prices
        for col in [COL_CLOSE, ADJ_CLOSE_LITERAL]:
            if col in result.columns:
                result[COL_ADJ_CLOSE] = result[col]
                break

    return result


def _ensure_cols(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure DataFrame has the required columns (date, adj_close) for forecasting.
    Enhanced to handle different column formats from yfinance.
    """
    if df is None or df.empty:
        # Return empty DataFrame with required columns
        return pd.DataFrame(columns=[COL_DATE, COL_ADJ_CLOSE])

    print(f"Input columns: {df.columns.tolist()}")
    result = df.copy()

    # Handle MultiIndex columns - flatten them first
    if isinstance(result.columns, pd.MultiIndex):
        result = _handle_multiindex_columns(result)

    # Standardize column names to lowercase
    if not isinstance(result.columns, pd.MultiIndex):
        result.columns = [
            c.lower() if isinstance(
                c, str) else c for c in result.columns]

    # Handle date columns
    result = _handle_date_column(result)

    # Ensure price columns
    result = _handle_price_columns(result)

    # Ensure date is datetime type
    try:
        result[COL_DATE] = pd.to_datetime(result[COL_DATE], errors="coerce")

        # Remove rows with missing critical data
        result = result.dropna(
            subset=[
                COL_DATE,
                COL_ADJ_CLOSE]).sort_values(COL_DATE)
        print(f"Output columns after ensuring: {result.columns.tolist()}")
        return result
    except KeyError as e:
        print(f"Error in _ensure_cols: {str(e)}")
        # If we failed to find or create the date column, return empty
        # DataFrame
        return pd.DataFrame(columns=["date", "adj_close"])


def rf_short_term_forecast(df: pd.DataFrame,
                           horizon: int = 5) -> Tuple[pd.Series,
                                                      Dict[str,
                                                           float],
                                                      Dict[str,
                                                           float]]:
    # Features + sentiment
    df = _ensure_cols(df)
    print(
        f"Debug - rf_short_term_forecast - df shape after _ensure_cols: {df.shape}")

    # Get sentiment features
    try:
        ticker = df.attrs.get("ticker", "")
        print(f"Debug - rf_short_term_forecast - using ticker: {ticker}")
        sent = build_sentiment_features(ticker, days=90)
        sent_shape = sent.shape if not sent.empty else "Empty"
        print(
            f"Debug - rf_short_term_forecast - sentiment data shape: {sent_shape}")
    except Exception as e:
        print(
            f"Debug - rf_short_term_forecast - error getting sentiment: {str(e)}")
        # Create empty sentiment DataFrame to continue
        sent = pd.DataFrame(
            columns=[
                "sent_7d",
                "sent_30d"]).set_index(
            pd.DatetimeIndex(
                []))

    # Add technical indicators
    try:
        print("Debug - rf_short_term_forecast - adding technical indicators")
        x = add_tech_indicators(df)
        print(
            f"Debug - rf_short_term_forecast - tech indicators added, shape: {x.shape}")

        # Join with sentiment data
        x = x.set_index("date").join(
            sent, how="left").fillna(
            method="ffill").reset_index()
        print(
            f"Debug - rf_short_term_forecast - after joining sentiment, shape: {x.shape}")

        # Create supervised learning dataset
        X, y, names = make_supervised(x, lookback=20)
        x_shape = X.shape if isinstance(X, np.ndarray) else "Not array"
        y_shape = y.shape if isinstance(y, np.ndarray) else "Not array"
        print(
            f"Debug - rf_short_term_forecast - supervised data shape: X:{x_shape}, y:{y_shape}")

        if len(X) < 30:
            print(
                f"Debug - rf_short_term_forecast - insufficient data points: {len(X)}, need at least 30")
            return pd.Series(dtype=float), {}, {}

        # Train model
        scaler = StandardScaler()
        x_scaled = scaler.fit_transform(X)
        rf = RandomForestRegressor(n_estimators=300, random_state=42)
        rf.fit(x_scaled, y)
        print("Debug - rf_short_term_forecast - model trained successfully")

    except Exception as e:
        print(
            f"Debug - rf_short_term_forecast - error in processing: {str(e)}")
        return pd.Series(dtype=float), {}, {}

    # iterative forecast on returns
    work = x.copy()
    preds = []
    for _ in range(horizon):
        tmp = add_tech_indicators(
            _ensure_cols(work)).set_index("date").join(
            sent,
            how="left").fillna(
            method="ffill").reset_index()
        x_input, _, _ = make_supervised(tmp, lookback=20)
        x_input_last = x_input[-1:]
        yhat = float(rf.predict(scaler.transform(x_input_last))[0])
        preds.append(yhat)
        last = work.iloc[-1]["adj_close"]
        work = pd.concat([work, pd.DataFrame({"date": [pd.to_datetime(
            work.iloc[-1]["date"]) + pd.Timedelta(days=1)], "adj_close": [last*(1+yhat)]})], ignore_index=True)
    idx = pd.date_range(start=df["date"].iloc[-1] +
                        pd.Timedelta(days=1), periods=horizon, freq="D")
    # SHAP-like feature importance proxy from RF (mean decrease in impurity)
    importances = {
        names[i]: float(v) for i,
        v in enumerate(
            rf.feature_importances_)}
    top_imp = dict(
        sorted(
            importances.items(),
            key=lambda kv: kv[1],
            reverse=True)[
            :10])
    return pd.Series(
        preds, index=idx), top_imp, {
        "mae": float(
            np.mean(
                np.abs(
                    y - rf.predict(x_scaled))))}


def arima_forecast_close(df: pd.DataFrame, horizon: int = 5) -> pd.Series:
    y = df.set_index("date")["adj_close"].astype(float)
    if len(y) < 10:
        return pd.Series(dtype=float)
    res = SARIMAX(y, order=(1, 1, 1)).fit(disp=False)
    fc = res.get_forecast(steps=horizon)
    idx = pd.date_range(
        start=y.index[-1] + pd.Timedelta(days=1), periods=horizon, freq="D")
    return pd.Series(fc.predicted_mean, index=idx)


def long_term_monthly(df: pd.DataFrame, months: int = 12) -> pd.Series:
    m = df.set_index("date")["adj_close"].resample("M").last().dropna()
    if len(m) < 12:
        return pd.Series(dtype=float)
    res = SARIMAX(m, order=(1, 1, 1), seasonal_order=(
        1, 0, 1, 12)).fit(disp=False)
    fc = res.get_forecast(steps=months)
    return pd.Series(fc.predicted_mean)
