from __future__ import annotations

import os
from typing import Dict, List

import pandas as pd

from ..settings import CIQ_EXCEL_PATH


class CIQExcelProvider:
    """Read standardized financials from an Excel export.

    Expected sheet names (case-insensitive best effort):
      - Income or IS
      - Balance or BS
      - CashFlow or CF

    Each sheet should contain a 'date' column or an equivalent and common
    standardized columns as per mapping.
    """

    def __init__(self) -> None:
        if not CIQ_EXCEL_PATH or not os.path.exists(CIQ_EXCEL_PATH):
            # Keep lazy failure so the app can render controls first
            pass

    def _load(self) -> Dict[str, pd.DataFrame]:
        if not CIQ_EXCEL_PATH or not os.path.exists(CIQ_EXCEL_PATH):
            raise FileNotFoundError("CIQ_EXCEL_PATH not set or file missing")
        xls = pd.ExcelFile(CIQ_EXCEL_PATH)
        sheets = {s.lower(): s for s in xls.sheet_names}

        def find_sheet(cands: List[str]) -> str | None:
            for c in cands:
                for k, v in sheets.items():
                    if c.lower() in k:
                        return v
            return None
        is_name = find_sheet(["income", "is"]) or xls.sheet_names[0]
        bs_name = find_sheet(["balance", "bs"]) or xls.sheet_names[1]
        cf_name = find_sheet(["cash", "cf"]) or xls.sheet_names[2]
        return {
            "income": pd.read_excel(xls, is_name),
            "balance": pd.read_excel(xls, bs_name),
            "cashflow": pd.read_excel(xls, cf_name),
        }

    def get_profile(self, ticker: str) -> Dict:
        return {"ticker": ticker}

    def get_market(self, ticker: str) -> Dict:
        return {"ticker": ticker}

    def get_financials(self, ticker: str,
                       years: int = 5) -> Dict[str, List[Dict]]:
        dfs = self._load()
        out: Dict[str, List[Dict]] = {}
        for k, df in dfs.items():
            df = df.copy()
            if "date" in df.columns:
                df["date"] = pd.to_datetime(df["date"], errors="coerce")
                df = df.sort_values("date").dropna(subset=["date"]).tail(years)
            out[k] = df.to_dict(orient="records")
        return out

    def get_estimates(self, ticker: str) -> Dict:
        # Optional: parse another sheet named Estimates if present
        return {}

    def get_news(self, ticker: str, days: int = 7) -> List[Dict]:
        return []
