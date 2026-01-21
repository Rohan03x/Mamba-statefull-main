from __future__ import annotations

import json
import os
import time
from typing import Dict, List, Optional

import requests

from ..settings import (
    CACHE_DIR,
    CACHE_TTL_FINANCIALS_SEC,
    CACHE_TTL_MARKET_SEC,
    CACHE_TTL_NEWS_SEC,
)


class CIQAPIProvider:
    """S&P Capital IQ API adapter.

    Notes
    -----
    - This is a skeleton adapter. Endpoints and field mapping may differ
      depending on entitlements and API version. Configure:
        CIQ_BASE_URL, CIQ_CLIENT_ID, CIQ_CLIENT_SECRET, CIQ_TENANT (optional)
    - Implements simple on-disk caching and basic token management.
    - Normalization to canonical fields is handled by mapping utilities
      elsewhere. Here we aim to return raw standardized rows as close as
      possible, using common field names where unambiguous.
    """

    def __init__(self) -> None:
        self.base_url = os.environ.get("CIQ_BASE_URL", "").rstrip("/")
        self.client_id = os.environ.get("CIQ_CLIENT_ID")
        self.client_secret = os.environ.get("CIQ_CLIENT_SECRET")
        self.tenant = os.environ.get("CIQ_TENANT")
        self._token: Optional[str] = None
        self._token_expiry: float = 0.0
        # Accept pre-supplied tokens via env
        env_tok = os.environ.get("CIQ_ACCESS_TOKEN")
        if env_tok:
            self._token = env_tok
            ttl = int(os.environ.get("CIQ_ACCESS_TOKEN_TTL", "3600") or 3600)
            self._token_expiry = time.time() + ttl
        if not self.client_id or not self.client_secret:
            # Defer hard failure until first request so the rest of the app can
            # still load.
            pass
        # Endpoint path overrides (so you can match your deployment without
        # code changes)
        self.paths = {
            "profile": os.environ.get(
                "CIQ_PROFILE_PATH",
                "/v1/companies/profile"),
            "market": os.environ.get(
                "CIQ_MARKET_PATH",
                "/v1/market/quote"),
            "financials": os.environ.get(
                "CIQ_FINANCIALS_PATH",
                "/v1/financials/standardized"),
            "estimates": os.environ.get(
                "CIQ_ESTIMATES_PATH",
                "/v1/estimates/consensus"),
            "news": os.environ.get(
                "CIQ_NEWS_PATH",
                "/v1/news"),
        }

    # --------------------------- HTTP helpers ---------------------------
    def _auth_headers(self) -> Dict[str, str]:
        tok = self._get_token()
        return {"Authorization": f"Bearer {tok}"} if tok else {}

    def _get_token(self) -> Optional[str]:
        # Token strategy is API-dependent; keep generic
        if self._token and time.time() < self._token_expiry - 60:
            return self._token
        token_url = os.environ.get("CIQ_TOKEN_URL") or (
            f"{self.base_url}/oauth/token" if self.base_url else None
        )
        if not token_url:
            # Maybe an env token was provided — return whatever we have
            return self._token
        # Try refresh_token first if present
        refresh = os.environ.get("CIQ_REFRESH_TOKEN")
        if refresh and self.client_id and self.client_secret:
            try:
                r = requests.post(
                    token_url,
                    data={
                        "grant_type": "refresh_token",
                        "refresh_token": refresh,
                        "client_id": self.client_id,
                        "client_secret": self.client_secret,
                    },
                    timeout=30,
                )
                r.raise_for_status()
                payload = r.json()
                self._token = payload.get("access_token") or self._token
                ttl = int(payload.get("expires_in", 3600))
                self._token_expiry = time.time() + ttl
                return self._token
            except Exception:
                pass
        # Fallback to resource owner password (username/password) if available
        username = os.environ.get("CIQ_USERNAME")
        password = os.environ.get("CIQ_PASSWORD")
        if username and password:
            try:
                r = requests.post(
                    token_url, data={
                        "username": username, "password": password, }, headers={
                        "Content-Type": "application/x-www-form-urlencoded"}, timeout=30, )
                r.raise_for_status()
                payload = r.json()
                self._token = payload.get("access_token") or self._token
                ttl = int(payload.get("expires_in") or payload.get(
                    "expires_in_seconds") or 3600)
                self._token_expiry = time.time() + ttl
                # Capture refresh token if provided
                rt = payload.get("refresh_token")
                if rt:
                    os.environ["CIQ_REFRESH_TOKEN"] = rt
                return self._token
            except Exception:
                pass

        # Fallback to client_credentials
        if self.client_id and self.client_secret:
            data = {
                "grant_type": "client_credentials",
                "client_id": self.client_id,
                "client_secret": self.client_secret,
            }
            if self.tenant:
                data["tenant"] = self.tenant
            try:
                r = requests.post(token_url, data=data, timeout=30)
                r.raise_for_status()
                payload = r.json()
                self._token = payload.get("access_token")
                ttl = int(payload.get("expires_in", 3600))
                self._token_expiry = time.time() + ttl
            except Exception:
                self._token = None
                self._token_expiry = 0.0
        return self._token

    def _get_json(
            self,
            url: str,
            params: Dict,
            ttl: int,
            cache_key: str) -> Dict:
        os.makedirs(CACHE_DIR, exist_ok=True)
        fp = os.path.join(CACHE_DIR, cache_key + ".json")
        if os.path.exists(fp) and (time.time() - os.path.getmtime(fp)) < ttl:
            with open(fp, "r", encoding="utf-8") as f:
                return json.load(f)
        headers = {"Accept": "application/json", **self._auth_headers()}
        r = requests.get(url, params=params, headers=headers, timeout=40)
        try:
            r.raise_for_status()
            data = r.json()
        except Exception as e:
            # Write a minimal error stub for easier debugging in UI without
            # leaking secrets
            data = {
                "_error": str(e),
                "_status": getattr(
                    r,
                    "status_code",
                    None),
                "_url": url}
        with open(fp, "w", encoding="utf-8") as f:
            json.dump(data, f)
        return data

    # ---------------------------- Endpoints -----------------------------
    def get_profile(self, ticker: str) -> Dict:
        # Placeholder: endpoint path varies by deployment
        if not self.base_url:
            return {"ticker": ticker}
        path = self.paths["profile"]
        url = path if path.startswith("http") else f"{self.base_url}{path}"
        data = self._get_json(url,
                              {"ticker": ticker},
                              CACHE_TTL_FINANCIALS_SEC,
                              f"ciq_profile_{ticker}")
        return data or {"ticker": ticker}

    def get_market(self, ticker: str) -> Dict:
        if not self.base_url:
            return {"ticker": ticker}
        path = self.paths["market"]
        url = path if path.startswith("http") else f"{self.base_url}{path}"
        data = self._get_json(url,
                              {"ticker": ticker},
                              CACHE_TTL_MARKET_SEC,
                              f"ciq_market_{ticker}")
        return data or {"ticker": ticker}

    def get_financials(self, ticker: str,
                       years: int = 5) -> Dict[str, List[Dict]]:
        if not self.base_url:
            return {"income": [], "balance": [], "cashflow": []}
        path = self.paths["financials"]
        url = path if path.startswith("http") else f"{self.base_url}{path}"
        data = self._get_json(
            url,
            {"ticker": ticker, "years": years},
            CACHE_TTL_FINANCIALS_SEC,
            f"ciq_fin_{ticker}_{years}",
        )
        # Expect {income: [...], balance: [...], cashflow: [...]} or similar
        return {
            "income": list(data.get("income", [])),
            "balance": list(data.get("balance", [])),
            "cashflow": list(data.get("cashflow", [])),
        }

    def get_estimates(self, ticker: str) -> Dict:
        if not self.base_url:
            return {}
        path = self.paths["estimates"]
        url = path if path.startswith("http") else f"{self.base_url}{path}"
        return self._get_json(url,
                              {"ticker": ticker},
                              CACHE_TTL_FINANCIALS_SEC,
                              f"ciq_est_{ticker}")

    def get_news(self, ticker: str, days: int = 7) -> List[Dict]:
        if not self.base_url:
            return []
        path = self.paths["news"]
        url = path if path.startswith("http") else f"{self.base_url}{path}"
        data = self._get_json(url,
                              {"ticker": ticker,
                               "days": days},
                              CACHE_TTL_NEWS_SEC,
                              f"ciq_news_{ticker}_{days}")
        items = data.get("items") if isinstance(data, dict) else data
        return list(items) if items else []
