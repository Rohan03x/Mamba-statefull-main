from __future__ import annotations

from typing import Dict, List


class CIQSnowflakeProvider:
    """Placeholder Snowflake provider.

    Implement using your organization’s Snowflake share/Xpressfeed schema.
    This stub keeps the interface consistent.
    """

    def get_profile(self, ticker: str) -> Dict:
        raise NotImplementedError("Snowflake adapter not implemented yet")

    def get_market(self, ticker: str) -> Dict:
        raise NotImplementedError("Snowflake adapter not implemented yet")

    def get_financials(self, ticker: str,
                       years: int = 5) -> Dict[str, List[Dict]]:
        raise NotImplementedError("Snowflake adapter not implemented yet")

    def get_estimates(self, ticker: str) -> Dict:
        raise NotImplementedError("Snowflake adapter not implemented yet")

    def get_news(self, ticker: str, days: int = 7) -> List[Dict]:
        raise NotImplementedError("Snowflake adapter not implemented yet")
