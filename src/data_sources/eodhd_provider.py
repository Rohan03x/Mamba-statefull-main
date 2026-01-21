"""
EODHD (End of Day Historical Data) API Provider
================================================

Comprehensive data provider for:
- Historical stock prices (EOD and intraday)
- Fundamentals (financial statements)
- Dividends, splits
- Options data
- Economic indicators
- News (limited)

API Documentation: https://eodhistoricaldata.com/financial-apis/
Pricing: $79.99/month for comprehensive data access
"""

import os
import logging
import time
import atexit
import hashlib
import json
import re
import tempfile
from collections import Counter
from typing import Dict, List, Optional, Any, Iterable
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import numpy as np
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logger = logging.getLogger(__name__)


class EODHDProvider:
    """
    EODHD API Provider - Professional financial data
    
    Features:
    - Historical EOD prices (50+ years for US stocks)
    - Intraday data (1m, 5m, 1h intervals)
    - Fundamental data (financials, ratios, earnings)
    - Dividends and splits
    - Options chains (historical)
    - Economic data
    - News articles
    
    Rate Limits:
    - Standard plan: 100,000 API calls/day
    - Professional plan: Unlimited calls
    """
    
    BASE_URL = "https://eodhistoricaldata.com/api"
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize EODHD provider
        
        Args:
            api_key: EODHD API key (or set EODHD_API_KEY env var)
        """
        self.api_key = api_key or os.getenv('EODHD_API_KEY')
        if not self.api_key:
            logger.warning("No EODHD API key provided. Set EODHD_API_KEY environment variable.")
        
        # Setup session with retries
        self.session = requests.Session()
        retries = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504]
        )
        self.session.mount('https://', HTTPAdapter(max_retries=retries))
        
        self._last_request_time = 0
        # Default to 10 requests/second per process. Note this does not protect against
        # multi-process fanout; tune via env var when running in parallel.
        try:
            self._min_request_interval = float(os.getenv("EODHD_MIN_REQUEST_INTERVAL", "0.1"))
        except Exception:
            self._min_request_interval = 0.1
        self._request_count = 0

        # Caching (default ON): reduces redundant calls across windows/families.
        # - EODHD_CACHE=0 disables
        # - EODHD_CACHE_TTL_SECONDS sets TTL (default 30d)
        # - EODHD_CACHE_DIR sets location (default: data/cache/eodhd)
        cache_flag = os.getenv("EODHD_CACHE", "1").strip().lower()
        self._cache_enabled = cache_flag not in {"0", "false", "no", "off"}
        try:
            # EODHD data (historical eod/div/options snapshots) is effectively immutable.
            # A longer default TTL prevents re-downloading the same payloads every day
            # and helps keep total daily API calls below quota when running many windows.
            self._cache_ttl_seconds = int(os.getenv("EODHD_CACHE_TTL_SECONDS", str(30 * 24 * 3600)))
        except Exception:
            self._cache_ttl_seconds = 30 * 24 * 3600
        cache_root = os.getenv("EODHD_CACHE_DIR", "")
        if cache_root:
            self._cache_dir = Path(cache_root)
        else:
            # Use centralized cache paths (with legacy fallback)
            try:
                from src.cache_paths import resolve_eodhd_cache_dir
                self._cache_dir = resolve_eodhd_cache_dir()
            except ImportError:
                # Fallback to legacy path
                self._cache_dir = Path(__file__).resolve().parents[2] / "data" / "cache" / "eodhd"
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._memory_cache: Dict[str, Any] = {}
        self._cache_hits = 0
        self._cache_misses = 0

        # Range-aware in-process caches (default ON) to avoid re-downloading overlapping
        # time ranges across walk-forward windows.
        range_cache_flag = os.getenv("EODHD_RANGE_CACHE", "1").strip().lower()
        self._range_cache_enabled = range_cache_flag not in {"0", "false", "no", "off"}
        self._eod_range_cache: Dict[tuple, pd.DataFrame] = {}
        self._eod_range_coverage: Dict[tuple, tuple] = {}  # (symbol, period) -> (start_ts, end_ts)
        self._options_cache: Dict[tuple, pd.DataFrame] = {}  # (symbol, date) -> df
        self._range_cache_hits = 0
        self._range_cache_misses = 0

        # Optional request tracing (disabled by default).
        # Enable with: EODHD_TRACE=1
        self._trace_enabled = os.getenv("EODHD_TRACE", "").strip().lower() in {"1", "true", "yes", "on"}
        self._endpoint_counts: Counter[str] = Counter()
        self._category_counts: Counter[str] = Counter()
        if self._trace_enabled:
            atexit.register(self._dump_trace_summary)

        # Circuit breaker: when EODHD returns 401/402 we stop hammering the API.
        # - 401 usually means invalid key
        # - 402 is often returned for quota exhaustion or plan restrictions
        try:
            self._circuit_break_seconds = int(os.getenv("EODHD_CIRCUIT_BREAK_SECONDS", "600"))
        except Exception:
            self._circuit_break_seconds = 600
        self._disabled_until_ts: float = 0.0
        self._disabled_reason: str = ""

        # Persist circuit-break state across processes so high-parallel runs don't
        # create retry storms when quota/plan restrictions are hit.
        self._circuit_state_path = self._cache_dir / "_circuit_break.json"

    def _read_circuit_state(self) -> tuple[float, str]:
        """Return (disabled_until_ts, reason) from disk, or (0.0, '')."""
        try:
            if not self._circuit_state_path.exists():
                return 0.0, ""
            raw = self._circuit_state_path.read_text(encoding="utf-8")
            payload = json.loads(raw)
            until_ts = float(payload.get("disabled_until_ts", 0.0) or 0.0)
            reason = str(payload.get("reason", "") or "")
            return until_ts, reason
        except Exception:
            return 0.0, ""

    def _write_circuit_state(self, disabled_until_ts: float, reason: str) -> None:
        payload = {
            "disabled_until_ts": float(disabled_until_ts or 0.0),
            "reason": str(reason or ""),
            "written_at": float(time.time()),
        }
        try:
            self._circuit_state_path.parent.mkdir(parents=True, exist_ok=True)
            tmp_dir = self._circuit_state_path.parent
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=str(tmp_dir),
                delete=False,
                prefix="._circuit_",
                suffix=".json",
            ) as tmp:
                tmp.write(json.dumps(payload, sort_keys=True))
                tmp_path = Path(tmp.name)
            tmp_path.replace(self._circuit_state_path)
        except Exception:
            # Best effort; do not fail requests due to circuit file write issues.
            return
        
    def _rate_limit(self):
        """Rate limiting to avoid overwhelming API"""
        current_time = time.time()
        elapsed = current_time - self._last_request_time
        if elapsed < self._min_request_interval:
            time.sleep(self._min_request_interval - elapsed)
        self._last_request_time = time.time()
        self._request_count += 1
        
    def _make_request(
        self,
        endpoint: str,
        params: Dict = None,
        *,
        timeout_seconds: int = 30,
        cache: Optional[bool] = None,
    ) -> Any:
        """
        Make API request with error handling
        
        Args:
            endpoint: API endpoint path
            params: Query parameters
            
        Returns:
            JSON response data or None on error
        """
        if not self.api_key:
            logger.error("EODHD API key not configured")
            return None

        # Short-circuit when we know EODHD is temporarily unusable.
        now = time.time()
        if self._disabled_until_ts and now < self._disabled_until_ts:
            if self._trace_enabled:
                logger.debug(
                    "⛔ EODHD circuit-break active (%.0fs remaining): %s",
                    max(0.0, self._disabled_until_ts - now),
                    self._disabled_reason or "disabled",
                )
            return None

        # Cross-process circuit-break: consult disk state.
        disk_until, disk_reason = self._read_circuit_state()
        if disk_until and now < disk_until:
            self._disabled_until_ts = disk_until
            self._disabled_reason = disk_reason or self._disabled_reason
            if self._trace_enabled:
                logger.debug(
                    "⛔ EODHD circuit-break active (disk) (%.0fs remaining): %s",
                    max(0.0, disk_until - now),
                    disk_reason or "disabled",
                )
            return None
        
        # Prepare params (do not include api_token in cache key)
        params = params or {}
        cache_key = self._cache_key(endpoint, params)

        use_cache = self._cache_enabled if cache is None else bool(cache)

        # Cache lookup (memory -> disk)
        if use_cache:
            if cache_key in self._memory_cache:
                self._cache_hits += 1
                if self._trace_enabled:
                    logger.debug("📦 EODHD cache hit (memory) endpoint=%s", endpoint)
                return self._memory_cache[cache_key]

            cached = self._load_disk_cache(cache_key)
            if cached is not None:
                self._cache_hits += 1
                self._memory_cache[cache_key] = cached
                if self._trace_enabled:
                    logger.debug("📦 EODHD cache hit (disk) endpoint=%s", endpoint)
                return cached

        self._cache_misses += 1

        self._rate_limit()

        if self._trace_enabled:
            self._endpoint_counts[endpoint] += 1
            category = endpoint.split("/", 1)[0] if endpoint else ""
            self._category_counts[category] += 1

        params = dict(params)
        params['api_token'] = self.api_key
        params['fmt'] = 'json'
        
        url = f"{self.BASE_URL}/{endpoint}"

        def _scrub_secrets(text: object) -> str:
            """Best-effort removal of api_token from error strings."""
            try:
                return re.sub(r"(api_token=)[^&\s]+", r"\1***", str(text))
            except Exception:
                return "<scrubbed>"

        # Track last HTTP status per endpoint (best-effort observability).
        if not hasattr(self, "_last_http_status"):
            self._last_http_status: Dict[str, Optional[int]] = {}
        
        try:
            response = self.session.get(url, params=params, timeout=int(timeout_seconds))
            response.raise_for_status()

            try:
                self._last_http_status[endpoint] = int(getattr(response, "status_code", 200) or 200)
            except Exception:
                self._last_http_status[endpoint] = 200
            
            # Handle different response types
            content_type = response.headers.get('content-type', '')
            if 'application/json' in content_type:
                payload = response.json()
            else:
                payload = response.text

            if use_cache:
                self._memory_cache[cache_key] = payload
                self._save_disk_cache(cache_key, payload)

            return payload
                
        except requests.exceptions.HTTPError as e:
            status = getattr(e.response, "status_code", None)
            try:
                self._last_http_status[endpoint] = int(status) if status is not None else None
            except Exception:
                self._last_http_status[endpoint] = None
            if not hasattr(self, "_warned_404_endpoints"):
                self._warned_404_endpoints: set[str] = set()
            if status == 401:
                logger.error("EODHD API: Invalid API key")
                self._disabled_reason = "401 invalid API key"
                self._disabled_until_ts = time.time() + float(self._circuit_break_seconds)
                self._write_circuit_state(self._disabled_until_ts, self._disabled_reason)
            elif status == 404:
                # Many EODHD endpoints are plan-gated and can return 404; don't spam logs.
                if endpoint not in self._warned_404_endpoints:
                    self._warned_404_endpoints.add(endpoint)
                    logger.warning("EODHD API endpoint not found (404) endpoint=%s", endpoint)
            elif status == 402:
                # EODHD sometimes uses 402 for quota exhaustion as well.
                body = ""
                try:
                    body = (e.response.text or "")[:500]
                except Exception:
                    body = ""
                logger.error(
                    "EODHD API: 402 Payment Required for endpoint=%s. "
                    "This can mean plan restriction or daily call quota exhausted. response=%s",
                    endpoint,
                    _scrub_secrets(body).replace("\n", " ").strip(),
                )
                self._disabled_reason = "402 payment/quota"
                self._disabled_until_ts = time.time() + float(self._circuit_break_seconds)
                self._write_circuit_state(self._disabled_until_ts, self._disabled_reason)
            elif status == 429:
                logger.error("EODHD API: Rate limit exceeded")
            else:
                body = ""
                try:
                    body = (e.response.text or "")[:500]
                except Exception:
                    body = ""
                logger.error(
                    "EODHD API HTTP error (status=%s) endpoint=%s: %s",
                    status,
                    endpoint,
                    _scrub_secrets(body or e),
                )
            return None

        except requests.exceptions.Timeout as e:
            logger.error(
                "EODHD API timeout endpoint=%s timeout_seconds=%s: %s",
                endpoint,
                timeout_seconds,
                _scrub_secrets(e),
            )
            return None

        except requests.exceptions.RequestException as e:
            logger.error("EODHD API request failed endpoint=%s: %s", endpoint, _scrub_secrets(e))
            return None

        except Exception as e:
            logger.error("EODHD API unexpected error endpoint=%s: %s", endpoint, _scrub_secrets(e))
            return None

    def get_us_quote_delayed(self, symbols: str | Iterable[str]) -> pd.DataFrame:
        """Fetch delayed top-of-book quote snapshot for US equities.

        Uses EODHD Live v2 "extended quotes" endpoint (`us-quote-delayed`). This is a
        *snapshot* (not a historical time series). We deliberately bypass the provider's
        long-lived cache for this endpoint to avoid serving stale quotes.

        Returns:
            DataFrame indexed by symbol with columns:
              - bid_price, bid_size, ask_price, ask_size, mid_price, spread_abs, spread_bps
              - last_trade_price, last_trade_time, bid_time, ask_time, timestamp
              - has_data (0/1)
        """

        if isinstance(symbols, str):
            raw_symbols = [symbols]
        else:
            raw_symbols = list(symbols)

        normalized: list[str] = []
        for sym in raw_symbols:
            s = str(sym).strip()
            if not s:
                continue
            if "." not in s:
                s = f"{s}.US"
            normalized.append(s)

        if not normalized:
            return pd.DataFrame()

        payload = self._make_request(
            "us-quote-delayed",
            {"s": ",".join(normalized)},
            timeout_seconds=10,
            cache=False,
        )
        if not payload or not isinstance(payload, dict):
            return pd.DataFrame()

        data_obj = payload.get("data")
        if not isinstance(data_obj, dict):
            return pd.DataFrame()

        rows: list[dict[str, object]] = []
        for sym in normalized:
            entry = data_obj.get(sym) or data_obj.get(sym.upper())
            if not isinstance(entry, dict):
                rows.append({"symbol": sym, "has_data": 0.0})
                continue

            bid_price = entry.get("bidPrice")
            ask_price = entry.get("askPrice")
            bid_size = entry.get("bidSize")
            ask_size = entry.get("askSize")
            last_trade_price = entry.get("lastTradePrice")

            def _f(x: object) -> float:
                try:
                    if x is None:
                        return float("nan")
                    return float(x)
                except Exception:
                    return float("nan")

            bid_p = _f(bid_price)
            ask_p = _f(ask_price)
            bid_s = _f(bid_size)
            ask_s = _f(ask_size)
            mid = (bid_p + ask_p) / 2.0 if np.isfinite(bid_p) and np.isfinite(ask_p) else float("nan")
            spread = (ask_p - bid_p) if np.isfinite(bid_p) and np.isfinite(ask_p) else float("nan")
            spread_bps = (1e4 * spread / mid) if np.isfinite(spread) and np.isfinite(mid) and mid != 0 else float("nan")

            rows.append(
                {
                    "symbol": sym,
                    "bid_price": bid_p,
                    "ask_price": ask_p,
                    "bid_size": bid_s,
                    "ask_size": ask_s,
                    "mid_price": float(mid) if mid == mid else float("nan"),
                    "spread_abs": float(spread) if spread == spread else float("nan"),
                    "spread_bps": float(spread_bps) if spread_bps == spread_bps else float("nan"),
                    "last_trade_price": _f(last_trade_price),
                    "last_trade_time": entry.get("lastTradeTime"),
                    "bid_time": entry.get("bidTime"),
                    "ask_time": entry.get("askTime"),
                    "timestamp": entry.get("timestamp"),
                    "has_data": 1.0,
                }
            )

        df = pd.DataFrame(rows)
        if df.empty:
            return df
        df = df.set_index("symbol").sort_index()
        return df

    def get_delisted_companies(
        self,
        exchange_code: str = "US",
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        *,
        timeout_seconds: int = 120,
        chunk_days: int = 7,
        max_chunks: int = 500,
    ) -> Optional[pd.DataFrame]:
        """Fetch delisted company metadata (survivorship control).

        EODHD docs: "Delisted Stock Companies Data".

        Notes:
        - The EODHD endpoint can be slow/large; we fetch in small date chunks.
        - Returns an empty DataFrame (not None) when the request succeeds but no rows exist.
        """

        # Candidate endpoints (EODHD docs have varied across time).
        ex = str(exchange_code or "").strip().upper() or "US"
        endpoints = [
            # Most common variants observed in the wild.
            f"delisted-companies/{ex}",
            "delisted-companies",
            f"delisted/{ex}",
            # Doc / legacy variants some accounts see.
            f"delisted-companies-data/{ex}",
            "delisted-companies-data",
            f"delisted-companies-list/{ex}",
            "delisted-companies-list",
        ]

        start_ts = pd.to_datetime(start_date, errors="coerce") if start_date else pd.NaT
        end_ts = pd.to_datetime(end_date, errors="coerce") if end_date else pd.NaT
        if pd.isna(start_ts):
            start_ts = pd.Timestamp("2000-01-01")
        if pd.isna(end_ts):
            end_ts = pd.Timestamp.utcnow().normalize()
        start_ts = pd.Timestamp(start_ts).normalize()
        end_ts = pd.Timestamp(end_ts).normalize()
        if end_ts < start_ts:
            return pd.DataFrame(columns=[])

        step = pd.Timedelta(days=max(1, int(chunk_days)))
        cur = start_ts
        out_frames: List[pd.DataFrame] = []
        chunks = 0

        while cur <= end_ts and chunks < int(max_chunks):
            chunk_end = min(end_ts, cur + step - pd.Timedelta(days=1))
            params = {
                "from": cur.strftime("%Y-%m-%d"),
                "to": chunk_end.strftime("%Y-%m-%d"),
            }

            payload = None
            all_endpoints_404 = True
            for ep in endpoints:
                payload = self._make_request(ep, params=params, timeout_seconds=int(timeout_seconds))
                status = None
                try:
                    status = getattr(self, "_last_http_status", {}).get(ep)
                except Exception:
                    status = None
                # If this endpoint looks genuinely missing, don't keep trying it on every chunk.
                if status == 404:
                    continue
                all_endpoints_404 = False
                if payload is not None:
                    break

            # If every endpoint variant returned 404, the API is likely not available
            # for this account (or docs changed). Avoid spamming 3*chunks errors.
            if all_endpoints_404:
                return pd.DataFrame(columns=[])

            if payload is None:
                # Best-effort: skip this chunk rather than failing the whole run.
                cur = chunk_end + pd.Timedelta(days=1)
                chunks += 1
                continue

            # Normalize payload shapes.
            rows = self._normalize_eodhd_indexed_dict(payload)
            if not rows:
                cur = chunk_end + pd.Timedelta(days=1)
                chunks += 1
                continue

            df = pd.DataFrame(rows)
            if df.empty:
                cur = chunk_end + pd.Timedelta(days=1)
                chunks += 1
                continue

            # Best-effort normalize date columns.
            date_col = None
            for c in ("DelistedDate", "delistedDate", "delisted_date", "Date", "date"):
                if c in df.columns:
                    date_col = c
                    break
            if date_col is not None:
                df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
            out_frames.append(df)

            cur = chunk_end + pd.Timedelta(days=1)
            chunks += 1

        if not out_frames:
            return pd.DataFrame(columns=[])

        out = pd.concat(out_frames, axis=0, ignore_index=True)
        return out

    def _cache_key(self, endpoint: str, params: Dict) -> str:
        """Stable cache key for endpoint + params (excluding auth + fmt)."""
        safe_params = {k: v for k, v in (params or {}).items() if k not in {"api_token", "fmt"}}
        try:
            payload = json.dumps({"endpoint": endpoint, "params": safe_params}, sort_keys=True, default=str)
        except Exception:
            payload = str((endpoint, tuple(sorted(safe_params.items()))))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _cache_path(self, cache_key: str) -> Path:
        return self._cache_dir / f"{cache_key}.json"

    def _load_disk_cache(self, cache_key: str) -> Any:
        try:
            path = self._cache_path(cache_key)
            if not path.exists():
                return None
            # TTL check
            age = time.time() - path.stat().st_mtime
            if self._cache_ttl_seconds > 0 and age > self._cache_ttl_seconds:
                return None
            with path.open("r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None

    def _save_disk_cache(self, cache_key: str, payload: Any) -> None:
        try:
            path = self._cache_path(cache_key)
            tmp = path.with_suffix(path.suffix + ".tmp")
            with tmp.open("w", encoding="utf-8") as f:
                json.dump(payload, f)
            tmp.replace(path)
        except Exception:
            # Cache must never break data fetch
            return

    def _dump_trace_summary(self, top_n: int = 20) -> None:
        """Emit a summary of EODHD requests made during this process."""
        try:
            if not self._trace_enabled:
                return
            total = int(self._request_count)
            logger.warning("📡 EODHD_TRACE summary: total_requests=%d", total)
            if self._cache_enabled:
                logger.warning(
                    "📡 EODHD_TRACE cache: hits=%d misses=%d (ttl_seconds=%d dir=%s)",
                    int(self._cache_hits),
                    int(self._cache_misses),
                    int(self._cache_ttl_seconds),
                    str(self._cache_dir),
                )
            if getattr(self, "_range_cache_enabled", False):
                logger.warning(
                    "📡 EODHD_TRACE range-cache: hits=%d misses=%d",
                    int(getattr(self, "_range_cache_hits", 0)),
                    int(getattr(self, "_range_cache_misses", 0)),
                )
            if self._category_counts:
                top_categories = ", ".join(
                    f"{k}:{v}" for k, v in self._category_counts.most_common(10) if k
                )
                logger.warning("📡 EODHD_TRACE by category: %s", top_categories)
            if self._endpoint_counts:
                top_endpoints = ", ".join(
                    f"{k}:{v}" for k, v in self._endpoint_counts.most_common(top_n)
                )
                logger.warning("📡 EODHD_TRACE top endpoints: %s", top_endpoints)
        except Exception as exc:
            logger.debug("Failed to dump EODHD_TRACE summary: %s", exc)
    
    def get_eod_prices(
        self,
        symbol: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        period: str = "d"
    ) -> Optional[pd.DataFrame]:
        """
        Get end-of-day historical prices
        
        Args:
            symbol: Stock symbol (e.g., 'AAPL' or 'AAPL.US')
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            period: Data period - 'd' (daily), 'w' (weekly), 'm' (monthly)
            
        Returns:
            DataFrame with OHLCV data
        """
        # Ensure symbol has exchange suffix
        if '.' not in symbol:
            symbol = f"{symbol}.US"

        cache_key = (symbol, period)
        req_start = pd.to_datetime(start_date) if start_date else None
        req_end = pd.to_datetime(end_date) if end_date else None

        # Fast path: serve from in-memory range cache when covered.
        if self._range_cache_enabled and cache_key in self._eod_range_cache and req_start is not None and req_end is not None:
            cov = self._eod_range_coverage.get(cache_key)
            if cov is not None:
                cov_start, cov_end = cov
                if cov_start <= req_start and cov_end >= req_end:
                    self._range_cache_hits += 1
                    cached_df = self._eod_range_cache[cache_key]
                    return cached_df.loc[(cached_df.index >= req_start) & (cached_df.index <= req_end)].copy()
        
        # When partially cached, expand the fetch to cover the union once.
        fetch_start = start_date
        fetch_end = end_date
        if self._range_cache_enabled and cache_key in self._eod_range_cache and req_start is not None and req_end is not None:
            cov = self._eod_range_coverage.get(cache_key)
            if cov is not None:
                cov_start, cov_end = cov
                fetch_start = min(cov_start, req_start).strftime('%Y-%m-%d')
                fetch_end = max(cov_end, req_end).strftime('%Y-%m-%d')

        params = {'period': period}
        if fetch_start:
            params['from'] = fetch_start
        if fetch_end:
            params['to'] = fetch_end
        
        self._range_cache_misses += 1
        data = self._make_request(f"eod/{symbol}", params)
        
        if not data:
            return None
        
        try:
            df = pd.DataFrame(data)
            if df.empty:
                return None
            
            # Parse date and set as index
            df['date'] = pd.to_datetime(df['date'])
            df = df.set_index('date')
            
            # Rename columns to match yfinance convention
            column_map = {
                'open': 'Open',
                'high': 'High',
                'low': 'Low',
                'close': 'Close',
                'adjusted_close': 'Adj Close',
                'volume': 'Volume'
            }
            df = df.rename(columns=column_map)
            
            # Convert to numeric
            for col in ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            df = df.sort_index()
            if self._range_cache_enabled:
                self._eod_range_cache[cache_key] = df
                self._eod_range_coverage[cache_key] = (df.index.min(), df.index.max())

            if req_start is not None and req_end is not None:
                return df.loc[(df.index >= req_start) & (df.index <= req_end)].copy()
            return df
            
        except Exception as e:
            logger.error(f"Failed to parse EODHD price data: {e}")
            return None
    
    def get_intraday_prices(
        self,
        symbol: str,
        interval: str = "5m",
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> Optional[pd.DataFrame]:
        """
        Get intraday historical prices
        
        Args:
            symbol: Stock symbol (e.g., 'AAPL.US')
            interval: Time interval - '1m', '5m', '1h'
            start_date: Start datetime (YYYY-MM-DD HH:MM:SS)
            end_date: End datetime (YYYY-MM-DD HH:MM:SS)
            
        Returns:
            DataFrame with intraday OHLCV data
        """
        if '.' not in symbol:
            symbol = f"{symbol}.US"
        
        params = {'interval': interval}
        if start_date:
            params['from'] = int(pd.Timestamp(start_date).timestamp())
        if end_date:
            params['to'] = int(pd.Timestamp(end_date).timestamp())
        
        data = self._make_request(f"intraday/{symbol}", params)
        
        if not data:
            return None
        
        try:
            df = pd.DataFrame(data)
            if df.empty:
                return None
            
            # Parse timestamp
            df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
            df = df.set_index('datetime')
            
            # Rename columns
            column_map = {
                'open': 'Open',
                'high': 'High',
                'low': 'Low',
                'close': 'Close',
                'volume': 'Volume'
            }
            df = df.rename(columns=column_map)
            
            # Convert to numeric
            for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            return df
            
        except Exception as e:
            logger.error(f"Failed to parse EODHD intraday data: {e}")
            return None
    
    def get_fundamentals(self, symbol: str, *, timeout_seconds: int = 30) -> Optional[Dict]:
        """
        Get fundamental data for a symbol
        
        Args:
            symbol: Stock symbol
            
        Returns:
            Dict with comprehensive fundamental data including:
            - General company info
            - Financial statements (annual/quarterly)
            - Earnings history
            - Financial ratios
        """
        if '.' not in symbol:
            symbol = f"{symbol}.US"
        
        data = self._make_request(f"fundamentals/{symbol}", timeout_seconds=int(timeout_seconds))
        
        return data if data else None

    def get_stock_screener(
        self,
        *,
        filters: Optional[list[list[object]]] = None,
        signals: Optional[object] = None,
        sort: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
        timeout_seconds: int = 60,
    ) -> Optional[pd.DataFrame]:
        """Call the EODHD Stock Market Screener API.

        EODHD docs show this endpoint as:
        - https://eodhd.com/api/screener

        Our provider uses BASE_URL=https://eodhistoricaldata.com/api, so this call
        hits https://eodhistoricaldata.com/api/screener.

        Args:
            filters: Nested list of triplets: [[field, op, value], ...]
                     Example: [["exchange", "=", "US"], ["market_capitalization", ">", 1e9]]
            signals: Either a comma-separated string ("bookvalue_neg,200d_new_lo") or
                     an iterable of signal names.
            sort: Sort spec like "market_capitalization.desc".
            limit: Max results (docs: 1..100).
            offset: Offset for paging (docs: 0..999).

        Returns:
            DataFrame of results, or None if the request failed.
        """

        params: Dict[str, object] = {
            "limit": int(limit),
            "offset": int(offset),
        }
        if sort:
            params["sort"] = str(sort)
        if filters is not None:
            # Screener expects a string parameter shaped like JSON.
            params["filters"] = json.dumps(filters)
        if signals:
            if isinstance(signals, str):
                params["signals"] = signals
            elif isinstance(signals, Iterable):
                params["signals"] = ",".join(str(s) for s in signals)
            else:
                params["signals"] = str(signals)

        payload = self._make_request("screener", params=params, timeout_seconds=int(timeout_seconds))
        if payload is None or payload == "":
            return None

        rows: Optional[list] = None
        if isinstance(payload, list):
            rows = payload
        elif isinstance(payload, dict):
            for key in ("data", "results", "items"):
                v = payload.get(key)
                if isinstance(v, list):
                    rows = v
                    break
            if rows is None:
                # Best-effort: sometimes APIs return a dict-per-row or error details.
                rows = [payload]
        else:
            return None

        try:
            df = pd.DataFrame(rows)
            return df
        except Exception as e:
            logger.error("Failed to parse EODHD screener payload: %s", e)
            return None

    def get_technical_indicator(
        self,
        symbol: str,
        *,
        function: str,
        start_date: str,
        end_date: str,
        order: str = "a",
        splitadjusted_only: int = 1,
        **params: Any,
    ) -> Optional[pd.DataFrame]:
        """Fetch a technical-indicator time series from EODHD.

        Uses the EODHD Technical Analysis Indicators API:
        https://eodhd.com/financial-apis/technical-indicators-api/

        Returns a DataFrame indexed by date with one or more indicator columns.
        """
        if '.' not in symbol:
            symbol = f"{symbol}.US"

        request_params: Dict[str, Any] = {
            "from": start_date,
            "to": end_date,
            "order": order,
            "fmt": "json",
            "function": function,
        }
        # splitadjusted_only is supported for many functions (sma/ema/rsi/macd/etc)
        if splitadjusted_only is not None:
            request_params["splitadjusted_only"] = int(splitadjusted_only)
        request_params.update(params)

        data = self._make_request(f"technical/{symbol}", request_params)
        if not data:
            return None
        if not isinstance(data, list):
            return None

        try:
            df = pd.DataFrame(data)
            if df.empty or "date" not in df.columns:
                return None
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            df = df.dropna(subset=["date"]).set_index("date").sort_index()
            # Convert all remaining columns to numeric where possible.
            for col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
            return df
        except Exception as e:
            logger.error("Failed to parse EODHD technical indicator %s for %s: %s", function, symbol, e)
            return None

    def get_ml_framework_features(
        self,
        symbol: str,
        *,
        start_date: str,
        end_date: str,
        lookback_days: int = 365,
    ) -> Optional[pd.DataFrame]:
        """Return the ml_framework feature bundle sourced entirely from EODHD.

        This intentionally avoids locally-deriving indicators; the values come from
        EODHD's Technical Indicators API.
        """
        start_ts = pd.to_datetime(start_date)
        lookback_start = (start_ts - pd.Timedelta(days=int(lookback_days))).strftime("%Y-%m-%d")

        pieces: List[pd.DataFrame] = []

        def _add(df: Optional[pd.DataFrame], rename: Dict[str, str]) -> None:
            if df is None or df.empty:
                return
            keep = df.rename(columns=rename)
            pieces.append(keep)

        # Moving averages / bands
        _add(
            self.get_technical_indicator(symbol, function="sma", start_date=lookback_start, end_date=end_date, period=20),
            {"sma": "ml_sma_20"},
        )
        _add(
            self.get_technical_indicator(symbol, function="sma", start_date=lookback_start, end_date=end_date, period=50),
            {"sma": "ml_sma_50"},
        )
        _add(
            self.get_technical_indicator(symbol, function="ema", start_date=lookback_start, end_date=end_date, period=12),
            {"ema": "ml_ema_12"},
        )
        _add(
            self.get_technical_indicator(symbol, function="ema", start_date=lookback_start, end_date=end_date, period=26),
            {"ema": "ml_ema_26"},
        )
        _add(
            self.get_technical_indicator(symbol, function="bbands", start_date=lookback_start, end_date=end_date, period=20),
            {"uband": "ml_bbands_upper", "mband": "ml_bbands_middle", "lband": "ml_bbands_lower"},
        )

        # Momentum / volatility
        _add(
            self.get_technical_indicator(symbol, function="rsi", start_date=lookback_start, end_date=end_date, period=14),
            {"rsi": "ml_rsi_14"},
        )
        _add(
            self.get_technical_indicator(symbol, function="atr", start_date=lookback_start, end_date=end_date, period=14),
            {"atr": "ml_atr_14"},
        )

        # MACD (multi-output)
        _add(
            self.get_technical_indicator(
                symbol,
                function="macd",
                start_date=lookback_start,
                end_date=end_date,
                fast_period=12,
                slow_period=26,
                signal_period=9,
            ),
            {"macd": "ml_macd", "signal": "ml_macd_signal", "divergence": "ml_macd_divergence"},
        )

        if not pieces:
            return None

        out = pd.concat(pieces, axis=1).sort_index()
        out = out[~out.index.duplicated(keep="last")]

        # Restrict to requested range AFTER pulling long lookback.
        out = out[(out.index >= pd.to_datetime(start_date)) & (out.index <= pd.to_datetime(end_date))]
        if out.empty:
            return None

        return out
    
    def get_dividends(
        self,
        symbol: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> Optional[pd.DataFrame]:
        """
        Get dividend history
        
        Args:
            symbol: Stock symbol
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            
        Returns:
            DataFrame with dividend dates and amounts
        """
        if '.' not in symbol:
            symbol = f"{symbol}.US"
        
        params = {}
        if start_date:
            params['from'] = start_date
        if end_date:
            params['to'] = end_date
        
        data = self._make_request(f"div/{symbol}", params)
        
        if not data:
            return None
        
        try:
            df = pd.DataFrame(data)
            if df.empty:
                return None
            
            df['date'] = pd.to_datetime(df['date'])
            df = df.set_index('date')
            df['value'] = pd.to_numeric(df['value'], errors='coerce')
            
            return df
            
        except Exception as e:
            logger.error(f"Failed to parse EODHD dividend data: {e}")
            return None

    def get_splits(
        self,
        symbol: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> Optional[pd.DataFrame]:
        """Get historical stock splits from EODHD.

        Docs: https://eodhd.com/financial-apis/api-splits-dividends/

        Endpoint:
          /splits/{symbol}

        Returns a DataFrame indexed by split effective date.
        """

        if '.' not in symbol:
            symbol = f"{symbol}.US"

        params: Dict[str, Any] = {}
        if start_date:
            params['from'] = start_date
        if end_date:
            params['to'] = end_date

        data = self._make_request(f"splits/{symbol}", params)
        if data is None:
            return None

        # Empty list is valid: no splits in range.
        if isinstance(data, list) and len(data) == 0:
            return pd.DataFrame(columns=[])

        try:
            df = pd.DataFrame(data)
        except Exception as exc:
            logger.error("Failed to parse EODHD splits data: %s", exc)
            return None

        if df is None or df.empty:
            return pd.DataFrame(columns=[])

        date_col = None
        for c in ("date", "Date", "split_date", "SplitDate"):
            if c in df.columns:
                date_col = c
                break
        if date_col is None:
            return None

        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
        df = df.dropna(subset=[date_col]).set_index(date_col).sort_index()
        return df
    
    def get_options(
        self,
        symbol: str,
        date: Optional[str] = None
    ) -> Optional[pd.DataFrame]:
        """
        Get options chain data
        
        Args:
            symbol: Stock symbol
            date: Expiration date or trading date (YYYY-MM-DD)
            
        Returns:
            DataFrame with options data (calls and puts)
        """
        if '.' not in symbol:
            symbol = f"{symbol}.US"

        cache_key = (symbol, date or "")
        if self._range_cache_enabled and cache_key in self._options_cache:
            self._range_cache_hits += 1
            return self._options_cache[cache_key].copy()
        
        params = {}
        if date:
            params['date'] = date
        
        self._range_cache_misses += 1
        data = self._make_request(f"options/{symbol}", params)
        
        if not data or 'data' not in data:
            return None
        
        try:
            options_list = []
            for item in data['data']:
                if 'options' in item:
                    expiration_date = item.get('expirationDate')
                    
                    # EODHD uses uppercase 'CALL' and 'PUT', not 'calls' and 'puts'
                    for option_type_raw, option_type_normalized in [('CALL', 'calls'), ('PUT', 'puts')]:
                        if option_type_raw in item['options']:
                            for opt in item['options'][option_type_raw]:
                                opt['type'] = option_type_normalized  # Normalize to lowercase
                                opt['expirationDate'] = expiration_date
                                options_list.append(opt)
            
            if not options_list:
                return None
            
            df = pd.DataFrame(options_list)
            
            # Normalize column names to match expected format
            column_mapping = {
                'lastPrice': 'last',  # EODHD uses 'lastPrice', normalize to 'last'
            }
            df = df.rename(columns=column_mapping)
            
            # Convert numeric columns
            numeric_cols = ['strike', 'bid', 'ask', 'last', 'volume', 
                          'openInterest', 'impliedVolatility']
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            if self._range_cache_enabled:
                self._options_cache[cache_key] = df
            return df
            
        except Exception as e:
            logger.error(f"Failed to parse EODHD options data: {e}")
            return None
    
    def get_news(
        self,
        symbol: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        limit: Optional[int] = None
    ) -> Optional[List[Dict]]:
        """
        Get news articles for a symbol with automatic chunking for historical data
        
        EODHD returns only the most recent N articles within a date range, not all articles.
        To get full historical coverage, we chunk the date range into monthly segments.

        Args:
            symbol: Stock symbol
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            limit: Maximum number of articles (None for all available via chunking)
            
        Returns:
            List of news article dicts
        """
        if '.' not in symbol:
            symbol = f"{symbol}.US"
        
        # If no limit specified, chunk by month to get complete historical coverage
        if limit is None and start_date and end_date:
            logger.info(f"🔄 Chunking news fetch for {symbol}: {start_date} → {end_date}")
            all_articles = []
            seen_ids = set()  # Deduplicate articles that might appear in multiple chunks
            
            # Parse date range
            start_dt = pd.to_datetime(start_date)
            end_dt = pd.to_datetime(end_date)
            
            # Chunk into monthly segments
            current = start_dt
            chunk_size = 31  # ~1 month
            
            while current <= end_dt:
                chunk_end = min(current + pd.Timedelta(days=chunk_size), end_dt)
                
                params = {
                    's': symbol,
                    'limit': 1000,  # Get up to 1000 per chunk
                    'from': current.strftime('%Y-%m-%d'),
                    'to': chunk_end.strftime('%Y-%m-%d')
                }
                
                batch = self._make_request(f"news", params)
                if batch:
                    for article in batch:
                        # Deduplicate by URL or date+title
                        article_id = article.get('link') or f"{article.get('date')}_{article.get('title')}"
                        if article_id not in seen_ids:
                            seen_ids.add(article_id)
                            all_articles.append(article)
                
                current = chunk_end + pd.Timedelta(days=1)
            
            logger.info(f"✅ Chunking complete for {symbol}: {len(all_articles)} total articles from {len(seen_ids)} unique sources")
            data = all_articles if all_articles else None
        else:
            # Single request with explicit limit or no date range
            params = {'s': symbol}
            if limit is not None:
                params['limit'] = limit
            if start_date:
                params['from'] = start_date
            if end_date:
                params['to'] = end_date
            
            data = self._make_request(f"news", params)
        
        if not data:
            return None
        
        # EODHD API with 's' parameter returns pre-filtered results
        # But verify symbols field exists and contains our ticker
        filtered_news = []
        symbol_ticker = symbol.split('.')[0]  # Extract AAPL from AAPL.US
        for article in data:
            article_symbols = article.get('symbols', [])
            if isinstance(article_symbols, list):
                # Check if any symbol in the list matches our ticker
                if any(symbol_ticker in s for s in article_symbols):
                    filtered_news.append(article)
            elif isinstance(article_symbols, str):
                # Handle legacy string format
                if symbol_ticker in article_symbols:
                    filtered_news.append(article)
            else:
                # If no symbols field, trust the API's filtering
                filtered_news.append(article)
        
        return filtered_news if filtered_news else None

    def get_insider_transactions(
        self,
        ticker: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        limit: int = 1000,
    ) -> Optional[List[Dict[str, Any]]]:
        """Get SEC Form-4 insider transactions from EODHD.

        Docs: https://eodhd.com/financial-apis/insider-transactions-api/

        Args:
            ticker: Stock symbol (e.g., 'AAPL' or 'AAPL.US')
            start_date/end_date: YYYY-MM-DD bounds
            limit: max rows (1..1000)

        Returns:
            List of transaction dicts (raw payload) or None on error.
        """

        # Ensure symbol has exchange suffix
        if '.' not in ticker:
            ticker = f"{ticker}.US"

        params: Dict[str, Any] = {
            'code': ticker,
            'limit': int(limit) if limit is not None else 1000,
        }
        if start_date:
            params['from'] = start_date
        if end_date:
            params['to'] = end_date

        data = self._make_request("insider-transactions", params)
        # Important: an empty list is a valid response (no filings in range).
        # Only treat None as an error.
        if data is None:
            return None

        if isinstance(data, list):
            return [x for x in data if isinstance(x, dict)]

        # Defensive: some APIs wrap the list.
        if isinstance(data, dict):
            for key in ("data", "results", "items"):
                payload = data.get(key)
                if isinstance(payload, list):
                    return [x for x in payload if isinstance(x, dict)]
        return None

    def get_economic_events(
        self,
        *,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        country: Optional[str] = None,
        event_type: Optional[str] = None,
        comparison: Optional[str] = None,
        limit: int = 1000,
        max_pages: int = 25,
    ) -> Optional[List[Dict[str, Any]]]:
        """Fetch economic events calendar data from EODHD.

        Docs: https://eodhd.com/financial-apis/economic-events-data-api/

        Endpoint:
          /economic-events

        Args:
            start_date/end_date: YYYY-MM-DD bounds (optional)
            country: ISO 3166-1 alpha-2 (e.g. "US") (optional)
            event_type: event type string (optional, exact match on EODHD side)
            comparison: one of {mom, qoq, yoy} (optional)
            limit: page size (1..1000)
            max_pages: safety cap for pagination

        Returns:
            List of raw event dicts or None on error.
        """

        params_base: Dict[str, Any] = {}
        if start_date:
            params_base['from'] = start_date
        if end_date:
            params_base['to'] = end_date
        if country:
            params_base['country'] = str(country).strip().upper()
        if event_type:
            params_base['type'] = str(event_type)
        if comparison:
            params_base['comparison'] = str(comparison)

        page_limit = int(limit) if limit is not None else 1000
        page_limit = max(1, min(1000, page_limit))

        out: List[Dict[str, Any]] = []

        # Per EODHD docs, offset is limited (often 0..1000). Requesting beyond that
        # can yield HTTP 422. We cap pagination accordingly.
        max_offset = 1000

        offset = 0
        for _ in range(int(max_pages) if max_pages is not None else 1):
            params = dict(params_base)
            params['limit'] = page_limit
            params['offset'] = int(offset)

            data = self._make_request('economic-events', params)
            if data is None:
                return None

            # EODHD usually returns a list. Be defensive about wrappers.
            items: Optional[List[Any]] = None
            if isinstance(data, list):
                items = data
            elif isinstance(data, dict):
                for key in ('data', 'results', 'items'):
                    payload = data.get(key)
                    if isinstance(payload, list):
                        items = payload
                        break

            if items is None:
                return None

            batch = [x for x in items if isinstance(x, dict)]
            out.extend(batch)

            # Stop when API returns no items.
            if len(items) == 0:
                break

            # Stop when fewer than requested rows returned.
            if len(items) < page_limit:
                break

            next_offset = offset + page_limit
            if next_offset > max_offset:
                break
            offset = next_offset

        return out

    def get_historical_market_cap(
        self,
        ticker: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> Optional[pd.DataFrame]:
        """Get historical market cap series from EODHD.

        Note: EODHD provides weekly market cap points (not daily).

        Docs: https://eodhd.com/financial-apis/historical-market-capitalization-api/

        Returns:
            DataFrame indexed by date with required column: 'market_cap'.
            May also include optional columns if the provider emits them:
            - 'float_market_cap'
            - 'shares_float'
        """

        if '.' not in ticker:
            ticker = f"{ticker}.US"

        params: Dict[str, Any] = {}
        if start_date:
            params['from'] = start_date
        if end_date:
            params['to'] = end_date

        data = self._make_request(f"historical-market-cap/{ticker}", params)
        if data is None:
            return None

        # EODHD can return:
        # - list[dict] rows
        # - dict keyed by numeric strings: {"0": {...}, "1": {...}}
        rows = self._normalize_eodhd_indexed_dict(data)
        if not rows:
            return None

        try:
            df = pd.DataFrame(rows)
        except Exception:
            return None

        if df.empty:
            return None

        date_col = None
        for candidate in ("date", "Date", "datetime"):
            if candidate in df.columns:
                date_col = candidate
                break
        if date_col is None:
            return None

        cap_col = None
        for candidate in (
            "market_cap",
            "marketCap",
            "MarketCap",
            "marketCapitalization",
            "MarketCapitalization",
            "value",
        ):
            if candidate in df.columns:
                cap_col = candidate
                break
        if cap_col is None:
            return None

        float_cap_col = None
        for candidate in (
            "float_market_cap",
            "floatMarketCap",
            "FloatMarketCap",
            "floatMarketCapitalization",
            "shares_float_market_cap",
        ):
            if candidate in df.columns:
                float_cap_col = candidate
                break

        shares_float_col = None
        for candidate in (
            "shares_float",
            "sharesFloat",
            "SharesFloat",
            "float_shares",
            "floatShares",
        ):
            if candidate in df.columns:
                shares_float_col = candidate
                break

        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        df = df.dropna(subset=[date_col]).set_index(date_col).sort_index()
        try:
            df.index = pd.DatetimeIndex(df.index).tz_localize(None).normalize()
        except Exception:
            pass

        rename_map: Dict[str, str] = {cap_col: "market_cap"}
        if float_cap_col is not None:
            rename_map[float_cap_col] = "float_market_cap"
        if shares_float_col is not None:
            rename_map[shares_float_col] = "shares_float"
        df = df.rename(columns=rename_map)

        df["market_cap"] = pd.to_numeric(df["market_cap"], errors="coerce")
        if "float_market_cap" in df.columns:
            df["float_market_cap"] = pd.to_numeric(df["float_market_cap"], errors="coerce")
        if "shares_float" in df.columns:
            df["shares_float"] = pd.to_numeric(df["shares_float"], errors="coerce")

        keep_cols = ["market_cap"]
        for c in ("float_market_cap", "shares_float"):
            if c in df.columns:
                keep_cols.append(c)
        df = df[keep_cols]
        return df

    @staticmethod
    def _normalize_eodhd_indexed_dict(obj: Any) -> List[Dict[str, Any]]:
        """EODHD sometimes returns dicts keyed by numeric strings ("0","1",...)."""
        if obj is None:
            return []
        if isinstance(obj, list):
            return [x for x in obj if isinstance(x, dict)]
        if isinstance(obj, dict):
            # If already a payload dict (not an indexed dict), wrap it.
            if all(isinstance(k, str) and k.isdigit() for k in obj.keys()):
                items: List[Dict[str, Any]] = []
                for k in sorted(obj.keys(), key=lambda s: int(s)):
                    v = obj.get(k)
                    if isinstance(v, dict):
                        items.append(v)
                return items
            return [obj]
        return []

    def get_exchange_details(
        self,
        exchange_code: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """Get exchange details (trading hours, holidays, early closes).

        Docs: https://eodhd.com/financial-apis/exchanges-api-trading-hours-and-stock-market-holidays/

        Endpoint:
          /exchange-details/{EXCHANGE_CODE}

        Notes:
        - ExchangeHolidays and ExchangeEarlyCloseDays are often dicts keyed by "0","1",...
        """

        if not exchange_code:
            return None

        params: Dict[str, Any] = {}
        if start_date:
            params["from"] = start_date
        if end_date:
            params["to"] = end_date

        data = self._make_request(f"exchange-details/{exchange_code}", params=params)
        if not isinstance(data, dict) or not data:
            return None
        return data

    def get_exchange_holidays(
        self,
        exchange_code: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> Optional[pd.DataFrame]:
        details = self.get_exchange_details(exchange_code, start_date=start_date, end_date=end_date)
        if not details:
            return None

        rows = self._normalize_eodhd_indexed_dict(details.get("ExchangeHolidays"))
        if not rows:
            return pd.DataFrame(columns=["date", "holiday", "type"])  # empty but valid

        df = pd.DataFrame(rows)
        # Observed keys: Holiday, Date, Type
        date_col = "Date" if "Date" in df.columns else ("date" if "date" in df.columns else None)
        if date_col is None:
            return pd.DataFrame(columns=["date", "holiday", "type"])

        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        df = df.dropna(subset=[date_col]).copy()
        df = df.rename(columns={date_col: "date", "Holiday": "holiday", "Type": "type"})
        df["date"] = df["date"].dt.tz_localize(None).dt.normalize()
        keep = [c for c in ("date", "holiday", "type") if c in df.columns]
        df = df[keep].sort_values("date").drop_duplicates(subset=["date"])
        return df

    def get_exchange_early_close_days(
        self,
        exchange_code: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> Optional[pd.DataFrame]:
        details = self.get_exchange_details(exchange_code, start_date=start_date, end_date=end_date)
        if not details:
            return None

        rows = self._normalize_eodhd_indexed_dict(details.get("ExchangeEarlyCloseDays"))
        if not rows:
            return pd.DataFrame(columns=["date", "holiday", "type"])  # empty but valid

        df = pd.DataFrame(rows)
        date_col = "Date" if "Date" in df.columns else ("date" if "date" in df.columns else None)
        if date_col is None:
            return pd.DataFrame(columns=["date", "holiday", "type"])

        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        df = df.dropna(subset=[date_col]).copy()
        df = df.rename(columns={date_col: "date", "Holiday": "holiday", "Type": "type"})
        df["date"] = df["date"].dt.tz_localize(None).dt.normalize()
        keep = [c for c in ("date", "holiday", "type") if c in df.columns]
        df = df[keep].sort_values("date").drop_duplicates(subset=["date"])
        return df

    def get_exchanges_list(self) -> Optional[pd.DataFrame]:
        """List supported exchanges.

        Endpoint:
          /exchanges-list/
        """

        data = self._make_request("exchanges-list")
        if data is None:
            return None
        try:
            if isinstance(data, list):
                return pd.DataFrame(data)
            if isinstance(data, dict):
                # Some clients wrap; accept single dict.
                return pd.DataFrame([data])
        except Exception:
            return None
        return None

    def get_exchange_symbol_list(self, exchange_code: str, *, delisted: bool = False) -> Optional[pd.DataFrame]:
        """List tickers on an exchange.

        Endpoint:
          /exchange-symbol-list/{EXCHANGE_CODE}
        """

        if not exchange_code:
            return None
        params: Dict[str, Any] = {}
        if delisted:
            params["delisted"] = 1
        data = self._make_request(f"exchange-symbol-list/{exchange_code}", params=params)
        if data is None:
            return None
        try:
            if isinstance(data, list):
                return pd.DataFrame(data)
            if isinstance(data, dict):
                return pd.DataFrame([data])
        except Exception:
            return None
        return None

    def get_symbol_change_history(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> Optional[pd.DataFrame]:
        """US symbol change history (renames, ticker changes).

        Endpoint:
          /symbol-change-history
        """

        params: Dict[str, Any] = {}
        if start_date:
            params["from"] = start_date
        if end_date:
            params["to"] = end_date
        data = self._make_request("symbol-change-history", params=params)
        if data is None:
            return None
        try:
            if isinstance(data, list):
                df = pd.DataFrame(data)
            elif isinstance(data, dict):
                df = pd.DataFrame([data])
            else:
                return None
            return df
        except Exception:
            return None
    
    def get_profile(self, ticker: str) -> Dict:
        """
        Get company profile (compatible with AutoProvider interface)
        
        Args:
            ticker: Stock symbol
            
        Returns:
            Dict with company info
        """
        fundamentals = self.get_fundamentals(ticker)
        
        if not fundamentals or 'General' not in fundamentals:
            return {}
        
        general = fundamentals['General']

        def _first_key(d: Dict[str, Any], keys: List[str]) -> Any:
            for k in keys:
                if k in d and d.get(k) is not None:
                    return d.get(k)
            return None

        type_raw = _first_key(
            general,
            [
                "Type",
                "type",
                "InstrumentType",
                "instrumentType",
                "AssetType",
                "assetType",
                "Category",
                "category",
            ],
        )
        type_norm = str(type_raw or "").strip().upper()

        is_etf_raw = _first_key(general, ["IsETF", "IsEtf", "is_etf", "ETF", "Etf"])
        is_adr_raw = _first_key(general, ["IsADR", "IsAdr", "is_adr", "ADR", "Adr"])
        is_fund_raw = _first_key(
            general,
            [
                "IsFund",
                "IsFUND",
                "is_fund",
                "IsMutualFund",
                "is_mutual_fund",
                "MutualFund",
            ],
        )

        # Best-effort derivation from Type when explicit flags are absent.
        if is_etf_raw is None and "ETF" in type_norm:
            is_etf_raw = True
        if is_adr_raw is None and "ADR" in type_norm:
            is_adr_raw = True
        if is_fund_raw is None and ("FUND" in type_norm or "MUTUAL" in type_norm):
            is_fund_raw = True
        
        return {
            'name': general.get('Name', ''),
            'sector': general.get('Sector', ''),
            'industry': general.get('Industry', ''),
            'description': general.get('Description', ''),
            'country': general.get('CountryName', ''),
            'exchange': general.get('Exchange', ''),
            'currency': general.get('CurrencyCode', 'USD'),
            'market_cap': general.get('MarketCapitalization'),
            'employees': general.get('FullTimeEmployees'),
            'founded': general.get('IPODate'),
            # Instrument classification (Tier-1 uses these as hard structural exclusions).
            'type': type_raw,
            'is_etf': is_etf_raw,
            'is_adr': is_adr_raw,
            'is_fund': is_fund_raw,
        }
    
    def get_market(self, ticker: str) -> Dict:
        """
        Get current market data (compatible with AutoProvider interface)
        
        Args:
            ticker: Stock symbol
            
        Returns:
            Dict with market metrics
        """
        # Get latest EOD price
        df = self.get_eod_prices(ticker, period='d')
        
        if df is None or df.empty:
            return {}
        
        latest = df.iloc[-1]
        
        return {
            'price': float(latest.get('Close', 0)),
            'volume': int(latest.get('Volume', 0)),
            'market_cap': None,  # Need fundamentals for this
            'pe_ratio': None,
            'dividend_yield': None,
        }
    
    def get_financials(self, ticker: str, years: int = 5) -> Dict[str, List[Dict]]:
        """
        Get financial statements (compatible with AutoProvider interface)
        
        Args:
            ticker: Stock symbol
            years: Number of years of data
            
        Returns:
            Dict with 'income', 'balance', 'cashflow' lists
        """
        fundamentals = self.get_fundamentals(ticker)
        
        if not fundamentals or 'Financials' not in fundamentals:
            return {"income": [], "balance": [], "cashflow": []}
        
        financials = fundamentals['Financials']
        
        result = {
            "income": [],
            "balance": [],
            "cashflow": []
        }
        
        # Process annual financials
        if 'Income_Statement' in financials and 'yearly' in financials['Income_Statement']:
            for year, data in list(financials['Income_Statement']['yearly'].items())[:years]:
                result["income"].append({
                    'date': year,
                    'revenue': data.get('totalRevenue'),
                    'gross_profit': data.get('grossProfit'),
                    'operating_income': data.get('operatingIncome'),
                    'net_income': data.get('netIncome'),
                    'eps': data.get('eps'),
                })
        
        if 'Balance_Sheet' in financials and 'yearly' in financials['Balance_Sheet']:
            for year, data in list(financials['Balance_Sheet']['yearly'].items())[:years]:
                result["balance"].append({
                    'date': year,
                    'total_assets': data.get('totalAssets'),
                    'total_liabilities': data.get('totalLiab'),
                    'total_equity': data.get('totalStockholderEquity'),
                    'cash': data.get('cash'),
                })
        
        if 'Cash_Flow' in financials and 'yearly' in financials['Cash_Flow']:
            for year, data in list(financials['Cash_Flow']['yearly'].items())[:years]:
                result["cashflow"].append({
                    'date': year,
                    'operating_cashflow': data.get('totalCashFromOperatingActivities'),
                    'investing_cashflow': data.get('totalCashflowsFromInvestingActivities'),
                    'financing_cashflow': data.get('totalCashFromFinancingActivities'),
                    'free_cashflow': data.get('freeCashFlow'),
                })
        
        return result
    
    def get_estimates(self, ticker: str) -> Dict:
        """
        Get analyst estimates (compatible with AutoProvider interface)
        
        Args:
            ticker: Stock symbol
            
        Returns:
            Dict with earnings estimates
        """
        fundamentals = self.get_fundamentals(ticker)
        
        if not fundamentals or 'Earnings' not in fundamentals:
            return {}
        
        earnings = fundamentals['Earnings']
        
        return {
            'earnings_history': earnings.get('History', []),
            'earnings_trend': earnings.get('Trend', []),
            'revenue_estimates': earnings.get('Annual', {}),
        }


# Singleton instance
_eodhd_provider = None


def get_eodhd_provider() -> EODHDProvider:
    """Get singleton EODHD provider instance"""
    global _eodhd_provider
    if _eodhd_provider is None:
        _eodhd_provider = EODHDProvider()
    return _eodhd_provider
