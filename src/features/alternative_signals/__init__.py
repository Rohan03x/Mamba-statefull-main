"""
Alternative Signals Feature Family
===================================

Hedge-fund-grade signals using only free data sources (refactored Jan 2026).

TOTAL: 38 features across 8 categories (down from 42, higher quality)

Categories:
1. Price Anomalies (7) - Normalized microstructure signals
2. Volume Anomalies (6) - Z-scored order flow indicators
3. Earnings/Calendar (6) - High-alpha event signals (removed weak calendar features)
4. Free Sentiment (4) - News volume changes and z-scores
5. Realized Volatility (6) - Multi-horizon vol estimators
6. Cross-Asset (5) - Market factor exposures
7. Intraday (4) - Best intraday patterns only (removed 4 noisy features)
8. Macro Interactions (4) - NEW: Gap×VIX, overnight_z, range_z, beta×VIX

Hedge-fund discipline applied:
- All features normalized/z-scored (no raw counts)
- Removed calendar overfitting (week_of_year, day_of_week)
- Removed noisy intraday features (midday_drift, session_momentum, etc.)
- Added interaction features for regime awareness
- Continuous signals only (no categorical labels)
"""

from .price_anomalies import extract_price_anomalies
from .volume_anomalies import extract_volume_anomalies
from .earnings_seasonality import extract_earnings_seasonality
from .free_sentiment import extract_free_sentiment
from .realized_volatility import extract_realized_volatility
from .cross_asset_relations import extract_cross_asset_relations
from .intraday_anomalies import extract_intraday_anomalies
from .macro_interactions import extract_macro_interactions

__all__ = [
    'extract_price_anomalies',
    'extract_volume_anomalies',
    'extract_earnings_seasonality',
    'extract_free_sentiment',
    'extract_realized_volatility',
    'extract_cross_asset_relations',
    'extract_intraday_anomalies',
    'extract_macro_interactions',
]
