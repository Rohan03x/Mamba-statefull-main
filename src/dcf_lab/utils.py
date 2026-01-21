
import pandas as pd


def clamp(x, lo, hi): return max(lo, min(hi, x))


def try_float(x, default=None):
    try:
        return float(x)
    except Exception:
        return default


def sanitize_dt_index(df: pd.DataFrame, col: str = None) -> pd.DataFrame:
    """
    Force tz-naive UTC for DateTimeIndex or a datetime column.
    Safe to call many times - converts tz-aware to UTC then strips timezone.
    
    This prevents "Cannot compare tz-naive and tz-aware timestamps" errors
    in regime detection and other time-series operations.
    
    Args:
        df: DataFrame to sanitize
        col: Optional column name. If None, sanitizes the index.
    
    Returns:
        DataFrame with tz-naive UTC timestamps
        
    Examples:
        # Sanitize index after loading
        df = load_prices(...)
        df = sanitize_dt_index(df)
        
        # Sanitize a specific column
        df = sanitize_dt_index(df, col='date')
        
        # Make a timestamp tz-naive
        t0 = pd.to_datetime(t0)
        if t0.tz is not None:
            t0 = t0.tz_convert("UTC").tz_localize(None)
    """
    if col is None:
        # Sanitize index
        idx = pd.to_datetime(df.index, utc=True, errors="coerce")
        # Convert to UTC then make tz-naive
        if idx.tz is not None:
            idx = idx.tz_convert("UTC").tz_localize(None)
        df.index = idx
        return df
    else:
        # Sanitize a specific column
        s = pd.to_datetime(df[col], utc=True, errors="coerce")
        if hasattr(s.dt, 'tz') and s.dt.tz is not None:
            s = s.dt.tz_convert("UTC").dt.tz_localize(None)
        df[col] = s
        return df


def wacc(coe, cod, equity, debt, tax):
    """
    Calculate Weighted Average Cost of Capital

    Args:
        coe: Cost of equity
        cod: Cost of debt
        equity: Equity value
        debt: Debt value
        tax: Tax rate

    Returns:
        Weighted Average Cost of Capital
    """
    equity_val = float(equity or 0)
    debt_val = float(debt or 0)
    total_val = equity_val + debt_val

    if total_val <= 0:
        return None

    cod_after_tax = float(cod or 0) * (1 - float(tax or 0))
    return (equity_val/total_val) * float(coe or 0) + \
        (debt_val/total_val) * cod_after_tax


def gordon_terminal_value(cf1, r, g):
    if r is None or g is None or r <= g:
        return None
    return cf1 / (r - g)
