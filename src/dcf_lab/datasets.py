from __future__ import annotations

import pandas as pd


def _to_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def align_financials(
        inc: pd.DataFrame,
        bal: pd.DataFrame,
        cf: pd.DataFrame) -> pd.DataFrame:
    """Return a single DataFrame indexed by date with key engineered columns.

    Expects columns close to the canonical naming used in mapping.py.
    """
    inc = inc.copy()
    bal = bal.copy()
    cf = cf.copy()
    for df in (inc, bal, cf):
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            df.dropna(subset=["date"], inplace=True)
            df.sort_values("date", inplace=True)
    out = pd.DataFrame()
    out["date"] = inc.get("date")
    out.set_index("date", inplace=True)

    def put(df: pd.DataFrame, col: str, name: str):
        if col in df.columns:
            out[name] = df.set_index("date")[col]

    put(inc, "totalRevenue", "revenue")
    put(inc, "grossProfit", "gross_profit")
    put(inc, "operatingIncome", "ebit")
    put(inc, "depreciationAndAmortization", "da")
    put(bal, "totalCurrentAssets", "tca")
    put(bal, "totalCurrentLiabilities", "tcl")
    put(cf, "capitalExpenditures", "capex")

    out = out.dropna(how="all")
    # Engineer ratios
    rev = _to_num(
        out["revenue"]) if "revenue" in out.columns else pd.Series(
        dtype=float)
    if len(rev):
        out["gross_margin"] = _to_num(out.get("gross_profit")) / rev
        out["ebit_margin"] = _to_num(out.get("ebit")) / rev
        out["da_to_rev"] = _to_num(out.get("da")) / rev
        out["capex_to_rev"] = _to_num(out.get("capex")).abs() / rev
    if "tca" in out.columns and "tcl" in out.columns and "revenue" in out.columns:
        out["nwc_to_rev"] = (_to_num(out["tca"]) -
                             _to_num(out["tcl"])) / _to_num(out["revenue"])
    out["rev_growth"] = _to_num(out["revenue"]).pct_change()
    return out


def make_supervised(df: pd.DataFrame, target: str, horizon: int = 1):
    """Create X,y for next-period prediction of target.

    Uses lag-1 features and contemporaneous ratios; drops NaNs.
    """
    d = df.copy()
    y = d[target].shift(-horizon)
    X = d[[c for c in d.columns if c not in (target,)]]
    # simple selection
    X = X[[c for c in X.columns if X[c].dtype != "O"]]
    Z = pd.concat([X, y.rename("y")], axis=1).dropna()
    return Z.drop(columns=["y"]).values, Z["y"].values


class DatasetManager:
    """Minimal dataset manager for integration."""
    def __init__(self):
        pass

    def load_financials(self, income: pd.DataFrame, balance: pd.DataFrame, cashflow: pd.DataFrame) -> pd.DataFrame:
        return align_financials(income, balance, cashflow)

    def make_supervised(self, df: pd.DataFrame, target: str, horizon: int = 1):
        return make_supervised(df, target, horizon)
