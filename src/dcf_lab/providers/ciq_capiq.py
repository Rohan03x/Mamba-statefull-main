from __future__ import annotations

import os
from datetime import datetime
from typing import Dict, List

import pandas as pd

from ..mnemonics_excel import load_mnemonics_from_excel


def _looks_like_date(x) -> bool:
    try:
        if isinstance(x, str) and ("-" in x or "/" in x):
            datetime.fromisoformat(x.replace("Z", "").replace("/", "-"))
            return True
    except Exception:
        pass
    return False


class CIQCapIQProvider:
    """Adapter using the third‑party capiq-python wrapper (SPQL over BasicAuth).

    Requirements:
      - Install: `pip install git+https://github.com/faaez/capiq-python.git`
      - Env: `CAPIQ_USERNAME`, `CAPIQ_PASSWORD`
    Notes:
      - This wrapper returns row arrays; we heuristically map [value, PeriodDate]
        pairs to a tidy time series.
    """

    def __init__(self) -> None:
        try:
            from capiq.capiq_client import CapIQClient  # type: ignore
        except Exception as e:  # pragma: no cover
            raise RuntimeError(
                "capiq-python not installed. Run: pip install git+https://github.com/faaez/capiq-python.git"
            ) from e
        user = os.environ.get("CAPIQ_USERNAME")
        pwd = os.environ.get("CAPIQ_PASSWORD")
        if not user or not pwd:
            raise RuntimeError(
                "Set CAPIQ_USERNAME and CAPIQ_PASSWORD in .env to use ciq_capiq provider")
        verify = os.environ.get("CAPIQ_VERIFY", "true").lower() != "false"
        self.client = CapIQClient(user, pwd, verify=verify, debug=False)
        self.default_exchange = os.environ.get(
            "CIQ_DEFAULT_EXCHANGE", "NASDAQ")
        self.ident_prefix = os.environ.get("CIQ_GDS_IDENT_PREFIX", "ticker:")
        # Mnemonic maps (align with ciq_gds)
        self.inc_mnems: Dict[str, str] = {
            "IQ_TOTAL_REV": "totalRevenue",
            "IQ_EBITDA": "ebitda",
            "IQ_EBITA": "ebitda",
            "IQ_EBIT": "operatingIncome",
            "IQ_NI": "netIncome",
            "IQ_GP": "grossProfit",
            "IQ_INT_EXPENSE": "interestExpense",
            "IQ_TAX_PROVISION": "incomeTaxExpense",
        }
        self.bs_mnems: Dict[str, str] = {
            "IQ_CASH_NEAR_CASH": "cashAndCashEquivalentsAtCarryingValue",
            "IQ_ST_DEBT": "shortTermDebt",
            "IQ_LT_DEBT": "longTermDebt",
            "IQ_TOTAL_DEBT": "totalDebt",
            "IQ_TOTAL_ASSETS": "totalAssets",
            "IQ_TOTAL_LIAB": "totalLiabilities",
            "IQ_TOTAL_EQUITY": "totalShareholderEquity",
            "IQ_TOT_CUR_ASSET": "totalCurrentAssets",
            "IQ_TOT_CUR_LIAB": "totalCurrentLiabilities",
            "IQ_NET_PPE": "propertyPlantEquipmentNet",
        }
        self.cf_mnems: Dict[str, str] = {
            "IQ_CAPEX": "capitalExpenditures",
            "IQ_DEP_AMORT": "depreciationAndAmortization",
        }
        # Optional Excel override
        xlsx = os.environ.get("CIQ_MNEMONICS_XLSX")
        try:
            if xlsx and os.path.exists(xlsx):
                over = load_mnemonics_from_excel(xlsx)
                if over.get("inc"):
                    self.inc_mnems = {v: k for k, v in over["inc"].items()}
                if over.get("bs"):
                    self.bs_mnems = {v: k for k, v in over["bs"].items()}
                if over.get("cf"):
                    self.cf_mnems = {v: k for k, v in over["c"].items()}
        except Exception:
            pass

    def _ident(self, ticker: str) -> str:
        if ":" in ticker:
            return ticker
        return f"{ticker}:{self.default_exchange}"

    @staticmethod
    def _series_to_df(rows: List[List], alias: str) -> pd.DataFrame:
        # Heuristic: prefer [value, PeriodDate] pairs; else try reverse
        records = []
        for r in rows or []:
            if not isinstance(r, (list, tuple)) or len(r) == 0:
                continue
            if len(r) >= 2 and _looks_like_date(r[1]):
                records.append({"date": r[1], alias: r[0]})
            elif _looks_like_date(r[0]):
                records.append(
                    {"date": r[0], alias: r[1] if len(r) > 1 else None})
        df = pd.DataFrame(records)
        if not df.empty and "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            df = df.dropna(subset=["date"]).sort_values("date")
        return df

    def _gdshe_df(self,
                  ident: str,
                  mapping: Dict[str,
                                str],
                  period: str) -> pd.DataFrame:
        mnems = list(mapping.keys())
        keys = mnems[:]  # return keys per wrapper API
        props = [{"PERIODTYPE": period, "METADATATAG": "PeriodDate"}
                 for _ in mnems]
        out = self.client.gdshe([ident], mnems, keys, properties=props)
        data = out.get(ident, {}) if isinstance(out, dict) else {}
        frames = []
        for mnem, alias in mapping.items():
            rows = data.get(mnem) or []
            frames.append(self._series_to_df(rows, alias))
        if not frames:
            return pd.DataFrame()
        df = None
        for f in frames:
            if f is None or f.empty:
                continue
            df = f if df is None else pd.merge(df, f, on="date", how="outer")
        return df if df is not None else pd.DataFrame()

    # ------------------------------ API --------------------------------
    def get_profile(self, ticker: str) -> Dict:
        # Minimal profile; name via GDSP group could be added if desired
        return {"ticker": ticker}

    def get_market(self, ticker: str) -> Dict:
        ident = self._ident(ticker)
        mnems = [
            "IQ_CLOSEPRICE",
            "IQ_SHARES_DILUTED",
            "IQ_MARKET_CAP",
            "IQ_BETA"]
        keys = ["price", "shares", "marketcap", "beta"]
        props = [{} for _ in mnems]
        out = self.client.gdsp([ident], mnems, keys, properties=props)
        d = out.get(ident, {}) if isinstance(out, dict) else {}
        # Convert numerics when possible
        res = {"ticker": ticker}
        for k in keys:
            try:
                res[k] = float(d.get(k)) if d.get(k) is not None else None
            except Exception:
                res[k] = d.get(k)
        return res

    def get_financials(self, ticker: str,
                       years: int = 5) -> Dict[str, List[Dict]]:
        ident = self._ident(ticker)
        inc_period = os.environ.get("CIQ_GDS_INC_PERIOD") or f"IQ_FY-{years}"
        bs_period = os.environ.get("CIQ_GDS_BS_PERIOD") or f"IQ_FY-{years}"
        cf_period = os.environ.get("CIQ_GDS_CF_PERIOD") or f"IQ_FY-{years}"
        inc_df = self._gdshe_df(ident, self.inc_mnems, inc_period)
        bs_df = self._gdshe_df(ident, self.bs_mnems, bs_period)
        cf_df = self._gdshe_df(ident, self.cf_mnems, cf_period)
        return {
            "income": inc_df.to_dict(orient="records"),
            "balance": bs_df.to_dict(orient="records"),
            "cashflow": cf_df.to_dict(orient="records"),
        }

    def get_estimates(self, ticker: str) -> Dict:
        return {}

    def get_news(self, ticker: str, days: int = 7) -> List[Dict]:
        return []
