from __future__ import annotations

import hashlib
import json
import os
import time
from typing import Dict, List, Optional, Tuple

import pandas as pd
import requests

from ..mnemonics_excel import load_mnemonics_from_excel
from ..settings import CACHE_DIR, CACHE_TTL_FINANCIALS_SEC, CACHE_TTL_MARKET_SEC


def _cache_path(prefix: str, obj: object) -> str:
    key = hashlib.md5(
        (prefix +
         json.dumps(
             obj,
             sort_keys=True)).encode()).hexdigest()
    return os.path.join(CACHE_DIR, f"{key}.json")


class CIQGDSProvider:
    """S&P CIQ GDS Client Service provider using SPQL via POST.

    Requires an access token (CIQ_ACCESS_TOKEN) or a token URL with username/password
    to fetch it. Uses the /v3/clientservice.json endpoint and supports GDSP/GDSHE.
    """

    def __init__(self) -> None:
        self.base_url = os.environ.get(
            "CIQ_BASE_URL",
            "https://api-ciq.marketintelligence.spglobal.com/gdsapi/rest").rstrip("/")
        self.token_url = os.environ.get("CIQ_TOKEN_URL")
        self.username = os.environ.get("CIQ_USERNAME")
        self.password = os.environ.get("CIQ_PASSWORD")
        self._token: Optional[str] = os.environ.get("CIQ_ACCESS_TOKEN")
        self._token_expiry = time.time() + int(os.environ.get("CIQ_ACCESS_TOKEN_TTL", "0") or 0)
        self.default_exchange = os.environ.get(
            "CIQ_DEFAULT_EXCHANGE", "NASDAQ")
        self.ident_prefix = os.environ.get("CIQ_GDS_IDENT_PREFIX", "ticker:")
        os.makedirs(CACHE_DIR, exist_ok=True)

        # Default mnemonic maps (override via env if needed)
        self.inc_mnems: Dict[str, str] = json.loads(
            os.environ.get(
                "CIQ_GDS_INC_MNEMS",
                json.dumps(
                    {
                        "IQ_TOTAL_REV": "totalRevenue",
                        "IQ_EBITDA": "ebitda",
                        "IQ_EBITA": "ebitda",
                        "IQ_EBIT": "operatingIncome",
                        "IQ_NI": "netIncome",
                        "IQ_GP": "grossProfit",
                        # Optional extras used in drivers when available
                        "IQ_INT_EXPENSE": "interestExpense",
                        "IQ_TAX_PROVISION": "incomeTaxExpense",
                    }
                ),
            )
        )
        self.bs_mnems: Dict[str, str] = json.loads(
            os.environ.get(
                "CIQ_GDS_BS_MNEMS",
                json.dumps(
                    {
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
                ),
            )
        )
        self.cf_mnems: Dict[str, str] = json.loads(
            os.environ.get(
                "CIQ_GDS_CF_MNEMS",
                json.dumps(
                    {
                        "IQ_CAPEX": "capitalExpenditures",
                        "IQ_DEP_AMORT": "depreciationAndAmortization",
                        # Optional: operating cash flow (not required for FCFF
                        # but useful)
                        "IQ_CASH_OPER": "cashFromOperations",
                    }
                ),
            )
        )
        self.market_mnems: Dict[str, str] = json.loads(
            os.environ.get(
                "CIQ_GDS_MKT_MNEMS",
                json.dumps(
                    {
                        # mnemonic -> output field name
                        "IQ_LAST_PRICE": "price",
                        "IQ_SHARES_DILUTED": "shares",
                        "IQ_BETA": "beta",
                        "IQ_MARKET_CAP": "marketcap",
                    }
                ),
            )
        )
        # Optional: override mnemonics from an Excel mapping file
        xlsx = os.environ.get("CIQ_MNEMONICS_XLSX")
        try:
            if xlsx and os.path.exists(xlsx):
                over = load_mnemonics_from_excel(xlsx)
                if over.get("inc"):
                    # invert mapping: canonical -> mnemonic to mnemonic ->
                    # canonical
                    self.inc_mnems = {v: k for k, v in over["inc"].items()}
                if over.get("bs"):
                    self.bs_mnems = {v: k for k, v in over["bs"].items()}
                if over.get("cf"):
                    self.cf_mnems = {v: k for k, v in over["c"].items()}
                if over.get("market"):
                    self.market_mnems = {
                        v: k for k, v in over["market"].items()}
        except Exception:
            pass

    # --------------------------- token mgmt ----------------------------
    def _get_token(self) -> Optional[str]:
        if self._token and time.time() < self._token_expiry - 60:
            return self._token
        if not self.token_url or not (self.username and self.password):
            return self._token
        try:
            r = requests.post(
                self.token_url,
                data={"username": self.username, "password": self.password},
                headers={"Content-Type": "application/x-www-form-urlencoded"},
                timeout=30,
            )
            r.raise_for_status()
            p = r.json()
            self._token = p.get("access_token")
            ttl = int(p.get("expires_in") or p.get(
                "expires_in_seconds") or 3600)
            self._token_expiry = time.time() + ttl
        except Exception:
            pass
        return self._token

    def _headers(self) -> Dict[str, str]:
        tok = self._get_token()
        return {"Authorization": f"Bearer {tok}",
                "Accept": "application/json"} if tok else {"Accept": "application/json"}

    # ----------------------------- SPQL --------------------------------
    def _clientservice_url(self) -> str:
        return f"{self.base_url}/v3/clientservice.json"

    def _post_spql(
            self,
            input_requests: List[Dict],
            ttl: int,
            cache_key: str) -> Dict:
        fp = _cache_path(cache_key, input_requests)
        if os.path.exists(fp) and time.time() - os.path.getmtime(fp) < ttl:
            with open(fp, "r", encoding="utf-8") as f:
                return json.load(f)
        url = self._clientservice_url()
        headers = self._headers()
        payload = {"inputRequests": input_requests}
        r = requests.post(url, headers=headers, json=payload, timeout=60)
        # On 401, force refresh once if we can
        if r.status_code == 401 and self.token_url and (
                self.username and self.password):
            self._token = None
            self._token_expiry = 0
            headers = self._headers()
            r = requests.post(url, headers=headers, json=payload, timeout=60)
        try:
            r.raise_for_status()
            data = r.json()
        except Exception as e:
            data = {
                "_error": str(e),
                "_status": getattr(
                    r,
                    "status_code",
                    None)}
        with open(fp, "w", encoding="utf-8") as f:
            json.dump(data, f)
        # Write a stable debug file to help diagnose issues without exposing
        # secrets
        try:
            dbg = {
                "_ts": int(
                    time.time()), "url": url, "status": getattr(
                    r, "status_code", None), "has_error": "_error" in data, "responses": len(
                    ((data or {}).get("outputResponses") or (
                        data or {}).get("OutputResponses") or []) or []), "keys": list(
                        (data or {}).keys()) if isinstance(
                            data, dict) else None, }
            with open(os.path.join(CACHE_DIR, "last_gds.json"), "w", encoding="utf-8") as df:
                json.dump(dbg, df)
        except Exception:
            pass
        return data

    @staticmethod
    def _collect_series(out: Dict) -> Dict[str, List[Tuple[str, float]]]:
        """Return mapping mnemonic -> list of (date, value)."""
        series_map: Dict[str, List[Tuple[str, float]]] = {}
        responses = out.get("outputResponses") or out.get(
            "OutputResponses") or []
        for resp in responses or []:
            mnem = (resp.get("inputRequest", {}) or {}).get("mnemonic")
            data = (resp.get("data", {}) or {})
            # Accept a variety of shapes from SPQL
            ser = (
                data.get("series")
                or data.get("Series")
                or data.get("dataPoints")
                or data.get("values")
                or data.get("SeriesData")
                or []
            )
            items: List[Tuple[str, float]] = []
            for dp in ser:
                md = (dp.get("metadata", {}) or {})
                date = (
                    md.get("PeriodDate")
                    or md.get("PERIODDATE")
                    or md.get("AsOfDate")
                    or md.get("ASOFDATE")
                    or dp.get("asOfDate")
                    or dp.get("ASOFDATE")
                    or dp.get("date")
                )
                val = dp.get("value") if "value" in dp else dp.get("VALUE")
                if date is None:
                    continue
                items.append(
                    (str(date), float(val) if val is not None else float("nan")))
            if mnem:
                series_map[mnem] = items
        return series_map

    # --------------------------- helpers -------------------------------
    def _ident(self, ticker: str) -> str:
        if ":" in ticker:
            return ticker
        # Prefer explicit ticker:exchange, otherwise use default exchange
        return f"{ticker}:{self.default_exchange}"

    def _ident_variants(self, ident: str) -> List[str]:
        variants = [ident]
        p = (self.ident_prefix or "").lower()
        if p and not ident.lower().startswith(p):
            variants.append(f"{self.ident_prefix}{ident}")
        return variants

    def _build_gdshe(
            self,
            ident: str,
            mnemonics: List[str],
            years: int,
            period: Optional[str] = None) -> List[Dict]:
        # Match your working examples: use periodType strings like IQ_FY-5 or
        # IQ_FQ-12
        if not period:
            period = f"IQ_FY-{years}"
        props = {"periodType": period, "metadataTag": "PeriodDate"}
        return [{"function": "GDSHE", "identifier": ident,
                 "mnemonic": m, "properties": props} for m in mnemonics]

    # ---------------------------- API ----------------------------------
    def get_profile(self, ticker: str) -> Dict:
        # Basic stub: SPQL can fetch more, but not required for modeling
        return {"ticker": ticker}

    def get_market(self, ticker: str) -> Dict:
        ident = self._ident(ticker)
        reqs = [
            {"function": "GDSP", "identifier": ident, "mnemonic": m}
            for m in self.market_mnems.keys()
        ]
        out = self._post_spql(
            reqs,
            CACHE_TTL_MARKET_SEC,
            f"ciq_gds_market_{ident}")
        result: Dict[str, float] = {}
        for resp in out.get("outputResponses", []) or []:
            mnem = (resp.get("inputRequest", {}) or {}).get("mnemonic")
            field = self.market_mnems.get(mnem)
            val = (resp.get("data", {}) or {}).get("value")
            if field and val is not None:
                try:
                    result[field] = float(val)
                except Exception:
                    pass
        # Compute marketcap if missing and price/shares present
        if "marketcap" not in result and all(
                k in result for k in ("price", "shares")):
            result["marketcap"] = float(
                result["price"]) * float(result["shares"])
        result["ticker"] = ticker
        return result

    def get_financials(self, ticker: str,
                       years: int = 5) -> Dict[str, List[Dict]]:
        ident = self._ident(ticker)
        # Period overrides via env to match org preferences
        inc_period = os.environ.get("CIQ_GDS_INC_PERIOD") or f"IQ_FY-{years}"
        bs_period = os.environ.get("CIQ_GDS_BS_PERIOD") or f"IQ_FY-{years}"
        cf_period = os.environ.get("CIQ_GDS_CF_PERIOD") or f"IQ_FY-{years}"

        # Income statement (try identifier variants e.g., with ticker: prefix)
        inc_out = {}
        for ident_try in self._ident_variants(ident):
            inc_reqs = self._build_gdshe(ident_try, list(
                self.inc_mnems.keys()), years, inc_period)
            inc_out = self._post_spql(
                inc_reqs,
                CACHE_TTL_FINANCIALS_SEC,
                f"ciq_gds_inc_{ident_try}_{years}")
            if len((inc_out or {}).get("outputResponses", []) or []) > 0:
                ident = ident_try
                break
        inc_series = self._collect_series(inc_out)
        inc_rows: Dict[str, Dict] = {}
        for mnem, alias in self.inc_mnems.items():
            for date, val in inc_series.get(mnem, []):
                inc_rows.setdefault(date, {"date": date})[alias] = val
        inc_df = pd.DataFrame(list(inc_rows.values())
                              ) if inc_rows else pd.DataFrame()

        # Balance sheet
        bs_reqs = self._build_gdshe(ident, list(
            self.bs_mnems.keys()), years, bs_period)
        bs_out = self._post_spql(
            bs_reqs,
            CACHE_TTL_FINANCIALS_SEC,
            f"ciq_gds_bs_{ident}_{years}")
        bs_series = self._collect_series(bs_out)
        bs_rows: Dict[str, Dict] = {}
        for mnem, alias in self.bs_mnems.items():
            for date, val in bs_series.get(mnem, []):
                bs_rows.setdefault(date, {"date": date})[alias] = val
        bs_df = pd.DataFrame(list(bs_rows.values())
                             ) if bs_rows else pd.DataFrame()

        # Cash flow
        cf_reqs = self._build_gdshe(ident, list(
            self.cf_mnems.keys()), years, cf_period)
        cf_out = self._post_spql(
            cf_reqs,
            CACHE_TTL_FINANCIALS_SEC,
            f"ciq_gds_cf_{ident}_{years}")
        cf_series = self._collect_series(cf_out)
        cf_rows: Dict[str, Dict] = {}
        for mnem, alias in self.cf_mnems.items():
            for date, val in cf_series.get(mnem, []):
                cf_rows.setdefault(date, {"date": date})[alias] = val
        cf_df = pd.DataFrame(list(cf_rows.values())
                             ) if cf_rows else pd.DataFrame()

        # Sort and limit just in case
        for df in (inc_df, bs_df, cf_df):
            if not df.empty and "date" in df.columns:
                df["date"] = pd.to_datetime(df["date"], errors="coerce")
                df.sort_values("date", inplace=True)

        return {
            "income": inc_df.to_dict(orient="records"),
            "balance": bs_df.to_dict(orient="records"),
            "cashflow": cf_df.to_dict(orient="records"),
        }

    def get_estimates(self, ticker: str) -> Dict:
        # Could be implemented via SPQL mnemonics for estimates if available
        return {}

    def get_news(self, ticker: str, days: int = 7) -> List[Dict]:
        # SPQL focus is fundamentals/time series; news may require a different
        # API
        return []
