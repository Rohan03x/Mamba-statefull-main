"""Configuration settings for valuation models and providers."""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# Load env early so apps pick up PROVIDER and creds
# 1) project root .env, 2) package .env next to this file
try:
    load_dotenv(override=False)  # current working dir / parents
    pkg_env = Path(__file__).resolve().parents[1] / ".env"
    if pkg_env.exists():
        load_dotenv(dotenv_path=pkg_env, override=False)
except Exception:
    pass

DEFAULT_RISK_FREE_RATE: float = 0.04
DEFAULT_MARKET_RISK_PREMIUM: float = 0.05
DEFAULT_BETA_LOOKBACK_DAYS: int = 365

# Cache configuration
CACHE_TTL_SECONDS: int = 6 * 3600
CACHE_TTL_MARKET_SEC: int = int(os.environ.get("TTL_MARKET_SEC", 300))
CACHE_TTL_FINANCIALS_SEC: int = int(
    os.environ.get(
        "TTL_FINANCIALS_SEC",
        24 * 3600))
CACHE_TTL_NEWS_SEC: int = int(os.environ.get("TTL_NEWS_SEC", 30 * 60))

# Use centralized cache path
try:
    from src.cache_paths import SHARED_CACHE_ROOT
    CACHE_DIR: str = os.environ.get("DCF_LAB_CACHE_DIR", str(SHARED_CACHE_ROOT))
except ImportError:
    CACHE_DIR: str = os.environ.get("DCF_LAB_CACHE_DIR", "cache/shared")

# Provider-specific environment
CIQ_EXCEL_PATH: str | None = os.environ.get("CIQ_EXCEL_PATH")
