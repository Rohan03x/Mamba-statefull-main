"""
Configuration module for DCF Suite API keys and settings.
"""

import os
from typing import Optional


class APIKeys:
    """API key management for various data providers."""
    
    def __init__(self):
        self._keys = {}
        self._load_from_env()
    
    def _load_from_env(self):
        """Load API keys from environment variables."""
        env_mappings = {
            'ALPHA_VANTAGE_API_KEY': 'alpha_vantage',
            'FRED_API_KEY': 'fred',
            'FINNHUB_API_KEY': 'finnhub',
            'IEX_API_KEY': 'iex',
            'POLYGON_API_KEY': 'polygon',
            'QUANDL_API_KEY': 'quandl'
        }
        
        for env_var, key_name in env_mappings.items():
            value = os.environ.get(env_var)
            if value:
                self._keys[key_name] = value
    
    def get(self, provider: str) -> Optional[str]:
        """Get API key for a specific provider."""
        return self._keys.get(provider.lower())
    
    def set(self, provider: str, key: str):
        """Set API key for a specific provider."""
        self._keys[provider.lower()] = key
    
    def has_key(self, provider: str) -> bool:
        """Check if API key exists for provider."""
        return provider.lower() in self._keys
    
    def list_configured(self) -> list:
        """List all configured providers."""
        return list(self._keys.keys())


# Global API keys instance
api_keys = APIKeys()
