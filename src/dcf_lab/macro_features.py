"""
Macro Economic Features Integration for AI Forecasting

This module connects FRED economic data to the AI forecasting pipeline,
providing regime-aware macro features that enhance price prediction accuracy.

Key Features:
- Real-time macro indicator ingestion from FRED
- Regime-specific feature engineering
- Economic cycle detection and scoring
- Yield curve feature extraction
- Inflation/employment/growth composite indicators

Integration Points:
- Feeds into ai_price_forecast.py feature pipeline
- Enhances regime_detection.py with macro context
- Provides risk-free rate inputs for options anchoring
"""

import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

from .providers.enhanced_fred import EnhancedFREDProvider, EnhancedFREDConfig
from .providers.fred_provider import FREDProvider

logger = logging.getLogger(__name__)


@dataclass
class MacroFeatureConfig:
    """Configuration for macro feature engineering"""
    lookback_months: int = 24
    regime_window: int = 12
    yield_curve_maturities: Optional[List[str]] = None
    key_indicators: Optional[List[str]] = None
    enable_vintage_data: bool = False
    
    def __post_init__(self):
        if self.yield_curve_maturities is None:
            self.yield_curve_maturities = ['3MO', '2Y', '5Y', '10Y', '30Y']
        
        if self.key_indicators is None:
            self.key_indicators = [
                'FEDFUNDS', 'DGS10', 'DGS2', 'CPIAUCSL', 'UNRATE',
                'GDP', 'INDPRO', 'VIXCLS', 'T5YIFR'
            ]


class MacroFeatureEngineer:
    """
    Macro Economic Feature Engineering for AI Forecasting
    
    Integrates FRED economic data into AI-ready features for enhanced
    price forecasting with regime awareness.
    """
    
    def __init__(self, config: Optional[MacroFeatureConfig] = None):
        self.config = config or MacroFeatureConfig()
        
        # Initialize FRED providers
        try:
            fred_config = EnhancedFREDConfig()
            self.enhanced_fred = EnhancedFREDProvider(fred_config)
            self.fred = FREDProvider()
            self.providers_available = True
            logger.info("✅ FRED providers initialized successfully")
        except Exception as e:
            logger.warning(f"⚠️ FRED providers initialization failed: {e}")
            self.providers_available = False
    
    def get_macro_features(self, ticker: str,
                           end_date: Optional[str] = None) -> Dict[str, Any]:
        """
        Get comprehensive macro features for AI forecasting
        
        Args:
            ticker: Stock ticker for context
            end_date: End date for feature extraction
            
        Returns:
            Dictionary with macro features and regime indicators
        """
        if not self.providers_available:
            logger.warning("FRED providers not available, returning empty")
            return self._get_fallback_features()
        
        try:
            # Calculate date range
            if end_date is None:
                end_date = datetime.now().strftime('%Y-%m-%d')
            
            start_date = (datetime.strptime(end_date, '%Y-%m-%d') -
                          timedelta(days=self.config.lookback_months * 30)
                          ).strftime('%Y-%m-%d')
            
            # Get core macro indicators
            macro_data = self._fetch_core_indicators(start_date, end_date)
            
            # Engineer features
            features = {}
            features.update(self._engineer_yield_curve_features(macro_data))
            features.update(self._engineer_inflation_features(macro_data))
            features.update(self._engineer_employment_features(macro_data))
            features.update(self._engineer_growth_features(macro_data))
            features.update(
                self._engineer_monetary_policy_features(macro_data))
            features.update(self._engineer_regime_features(macro_data))
            features.update(self._engineer_cycle_features(macro_data))
            
            # Add meta information
            features['macro_data_quality'] = self._assess_data_quality(
                macro_data)
            features['macro_last_updated'] = end_date
            features['macro_regime_confidence'] = (
                self._calculate_regime_confidence(features))
            
            logger.info(
                f"✅ Generated {len(features)} macro features for {ticker}")
            return features
            
        except Exception as e:
            logger.error(f"❌ Error generating macro features: {e}")
            return self._get_fallback_features()
    
    def _fetch_core_indicators(self, start_date: str,
                               end_date: str) -> pd.DataFrame:
        """Fetch core macro indicators from FRED"""
        try:
            # Use basic FRED provider to get time series with proper column names
            macro_df = self.fred.get_multiple_series(
                series_ids=self.config.key_indicators,
                start_date=start_date,
                end_date=end_date
            )
            
            if not macro_df.empty:
                return macro_df
            
            # Fallback to enhanced FRED if basic fails
            dashboard = self.enhanced_fred.get_economic_dashboard(
                lookback_months=self.config.lookback_months)
            
            if 'indicators' in dashboard:
                indicators_df = pd.DataFrame([dashboard['indicators']])
                indicators_df.index = [pd.to_datetime(end_date)]
                #  Map friendly names to FRED series IDs for compatibility
                column_mapping = {
                    'gdp_real': 'GDP',
                    'unemployment_rate': 'UNRATE',
                    'cpi_all': 'CPIAUCSL',
                    'fed_funds_rate': 'FEDFUNDS',
                    'industrial_production': 'INDPRO',
                    'vix': 'VIXCLS',
                }
                indicators_df = indicators_df.rename(columns=column_mapping)
                return indicators_df
            
            return pd.DataFrame()
            
        except Exception as e:
            logger.warning(f"Error fetching FRED data: {e}")
            return pd.DataFrame()
    
    def _engineer_yield_curve_features(
            self, macro_data: pd.DataFrame) -> Dict[str, float]:
        """Engineer yield curve-based features"""
        features = {}
        
        try:
            # Yield curve slope (10Y - 2Y)
            if 'DGS10' in macro_data.columns and 'DGS2' in macro_data.columns:
                slope = (macro_data['DGS10'].iloc[-1] -
                         macro_data['DGS2'].iloc[-1])
                features['yield_curve_slope'] = float(slope)
                
                # Inversion indicator
                features['yield_curve_inverted'] = 1.0 if slope < 0 else 0.0
                
                # Steepening/flattening trend
                if len(macro_data) >= 20:
                    slope_20d_ago = (macro_data['DGS10'].iloc[-20] -
                                     macro_data['DGS2'].iloc[-20])
                    features['yield_curve_steepening'] = float(
                        slope - slope_20d_ago)
            
            # Fed funds vs 10Y spread (monetary policy stance)
            if ('FEDFUNDS' in macro_data.columns and
                    'DGS10' in macro_data.columns):
                policy_spread = (macro_data['DGS10'].iloc[-1] -
                                 macro_data['FEDFUNDS'].iloc[-1])
                features['monetary_policy_spread'] = float(policy_spread)
            
            # Real rates (using 5Y TIPS breakeven if available)
            if 'T5YIFR' in macro_data.columns and 'DGS5' in macro_data.columns:
                real_rate = (macro_data['DGS5'].iloc[-1] -
                             macro_data['T5YIFR'].iloc[-1])
                features['real_5y_rate'] = float(real_rate)
        
        except Exception as e:
            logger.warning(f"Error engineering yield curve features: {e}")
        
        return features
    
    def _engineer_inflation_features(
            self, macro_data: pd.DataFrame) -> Dict[str, float]:
        """Engineer inflation-based features"""
        features = {}
        
        try:
            if 'CPIAUCSL' in macro_data.columns:
                cpi_series = macro_data['CPIAUCSL'].dropna()
                
                if len(cpi_series) >= 12:
                    # YoY inflation rate
                    current_cpi = cpi_series.iloc[-1]
                    year_ago_cpi = cpi_series.iloc[-12]
                    inflation_rate = ((current_cpi / year_ago_cpi) - 1) * 100
                    features['inflation_rate_yoy'] = float(inflation_rate)
                    
                    # Inflation acceleration (3M vs 6M annualized)
                    if len(cpi_series) >= 6:
                        recent_3m = ((cpi_series.iloc[-1] /
                                      cpi_series.iloc[-3]) - 1) * 400
                        recent_6m = ((cpi_series.iloc[-3] /
                                      cpi_series.iloc[-6]) - 1) * 200
                        features['inflation_acceleration'] = float(
                            recent_3m - recent_6m)
                    
                    # Inflation volatility (12M rolling std of MoM changes)
                    mom_changes = cpi_series.pct_change() * 100
                    inflation_vol = mom_changes.rolling(12).std().iloc[-1]
                    features['inflation_volatility'] = float(inflation_vol)
            
            # Breakeven inflation expectations if available
            if 'T5YIFR' in macro_data.columns:
                expectations_5y = macro_data['T5YIFR'].iloc[-1]
                features['inflation_expectations_5y'] = float(expectations_5y)
        
        except Exception as e:
            logger.warning(f"Error engineering inflation features: {e}")
        
        return features
    
    def _engineer_employment_features(
            self, macro_data: pd.DataFrame) -> Dict[str, float]:
        """Engineer employment-based features"""
        features = {}
        
        try:
            if 'UNRATE' in macro_data.columns:
                unemployment = macro_data['UNRATE'].dropna()
                
                if len(unemployment) >= 12:
                    current_unemp = unemployment.iloc[-1]
                    features['unemployment_rate'] = float(current_unemp)
                    
                    # Change from year ago
                    year_ago_unemp = unemployment.iloc[-12]
                    unemp_change = current_unemp - year_ago_unemp
                    features['unemployment_change_yoy'] = float(unemp_change)
                    
                    # Trend (3M moving average slope)
                    if len(unemployment) >= 3:
                        recent_avg = unemployment.rolling(3).mean()
                        if len(recent_avg) >= 3:
                            trend = recent_avg.iloc[-1] - recent_avg.iloc[-3]
                        else:
                            trend = 0
                        features['unemployment_trend'] = float(trend)
                    
                    # Distance from historical median (recession indicator)
                    if len(unemployment) >= 60:
                        historical_median = unemployment.rolling(60).median()
                        median_val = historical_median.iloc[-1]
                        vs_trend = current_unemp - median_val
                        features['unemployment_vs_trend'] = float(vs_trend)
        
        except Exception as e:
            logger.warning(f"Error engineering employment features: {e}")
        
        return features
    
    def _engineer_growth_features(
            self, macro_data: pd.DataFrame) -> Dict[str, float]:
        """Engineer growth-based features"""
        features = {}
        
        try:
            # Industrial production as high-frequency growth proxy
            if 'INDPRO' in macro_data.columns:
                ip_series = macro_data['INDPRO'].dropna()
                
                if len(ip_series) >= 12:
                    # YoY industrial production growth
                    current_ip = ip_series.iloc[-1]
                    year_ago_ip = ip_series.iloc[-12]
                    ip_growth = ((current_ip / year_ago_ip) - 1) * 100
                    features['industrial_production_growth'] = float(ip_growth)
                    
                    # 3M momentum
                    if len(ip_series) >= 3:
                        ratio = ip_series.iloc[-1] / ip_series.iloc[-3]
                        momentum = ((ratio) - 1) * 400
                        feature_name = 'industrial_production_momentum'
                        features[feature_name] = float(momentum)
        
        except Exception as e:
            logger.warning(f"Error engineering growth features: {e}")
        
        return features
    
    def _engineer_monetary_policy_features(
            self, macro_data: pd.DataFrame) -> Dict[str, float]:
        """Engineer monetary policy features"""
        features = {}
        
        try:
            if 'FEDFUNDS' in macro_data.columns:
                fed_funds = macro_data['FEDFUNDS'].dropna()
                
                if len(fed_funds) >= 12:
                    current_rate = fed_funds.iloc[-1]
                    features['fed_funds_rate'] = float(current_rate)
                    
                    # Change from year ago (tightening/easing cycle)
                    year_ago_rate = fed_funds.iloc[-12]
                    rate_change = current_rate - year_ago_rate
                    features['fed_funds_change_yoy'] = float(rate_change)
                    
                    # Policy stance classification
                    if rate_change > 0.5:
                        features['monetary_policy_stance'] = 1.0  # Tightening
                    elif rate_change < -0.5:
                        features['monetary_policy_stance'] = -1.0  # Easing
                    else:
                        features['monetary_policy_stance'] = 0.0  # Neutral
                    
                    # Rate level context (low/normal/high)
                    if len(fed_funds) >= 120:  # 10 years of history
                        ranked = fed_funds.rolling(120).rank().iloc[-1]
                        rate_percentile = (ranked / 120) * 100
                        feature_key = 'fed_funds_percentile'
                        features[feature_key] = float(rate_percentile)
        
        except Exception as e:
            logger.warning(f"Error engineering monetary policy features: {e}")
        
        return features
    
    def _engineer_regime_features(
            self, macro_data: pd.DataFrame) -> Dict[str, float]:
        """Engineer regime classification features"""
        features = {}
        
        try:
            # Market stress indicator using VIX if available
            features.update(self._calculate_market_stress(macro_data))
            
            # Economic conditions composite
            features.update(self._calculate_economic_conditions())
        
        except Exception as e:
            logger.warning(f"Error engineering regime features: {e}")
        
        return features
    
    def _calculate_market_stress(self,
                                 macro_data: pd.DataFrame) -> Dict[str, float]:
        """Calculate market stress indicators"""
        features = {}
        
        if 'VIXCLS' in macro_data.columns:
            vix = macro_data['VIXCLS'].dropna()
            if not vix.empty:
                current_vix = vix.iloc[-1]
                features['market_stress_vix'] = float(current_vix)
                
                # Stress classification
                if current_vix > 30:
                    features['market_regime_stress'] = 1.0  # High stress
                elif current_vix > 20:
                    features['market_regime_stress'] = 0.5  # Moderate
                else:
                    features['market_regime_stress'] = 0.0  # Low stress
        
        return features
    
    def _calculate_economic_conditions(self) -> Dict[str, float]:
        """Calculate economic conditions composite"""
        # Placeholder for future composite calculation
        return {}
    
    def _engineer_cycle_features(
            self, features: Dict[str, float]) -> Dict[str, float]:
        """Engineer economic cycle features"""
        cycle_features = {}
        
        try:
            # Business cycle indicators
            cycle_indicators = []
            
            # Employment cycle (unemployment gap from trend)
            if 'unemployment_vs_trend' in features:
                unemployment_gap = features['unemployment_vs_trend']
                # Invert so positive = expansion
                cycle_indicators.append(-unemployment_gap)
            
            # Growth cycle
            if 'industrial_production_growth' in features:
                ip_growth = features['industrial_production_growth']
                # Normalize to roughly [-1, 1]
                cycle_indicators.append(ip_growth / 5)
            
            # Financial conditions cycle
            if 'monetary_policy_spread' in features:
                policy_spread = features['monetary_policy_spread']
                cycle_indicators.append(policy_spread / 3)  # Normalize
            
            if cycle_indicators:
                # Composite cycle score
                cycle_score = np.mean(cycle_indicators)
                cycle_features['business_cycle_score'] = float(cycle_score)
                
                # Cycle phase classification
                if cycle_score > 0.5:
                    cycle_features['cycle_phase'] = 3.0  # Expansion
                elif cycle_score > 0:
                    cycle_features['cycle_phase'] = 2.0  # Recovery
                elif cycle_score > -0.5:
                    cycle_features['cycle_phase'] = 1.0  # Slowdown
                else:
                    cycle_features['cycle_phase'] = 0.0  # Contraction
        
        except Exception as e:
            logger.warning(f"Error engineering cycle features: {e}")
        
        return cycle_features
    
    def _assess_data_quality(self, macro_data: pd.DataFrame) -> float:
        """Assess quality of macro data for reliability scoring"""
        if macro_data.empty:
            return 0.0
        
        # Calculate completeness
        total_cells = macro_data.size
        non_null_cells = macro_data.count().sum()
        completeness = non_null_cells / total_cells if total_cells > 0 else 0
        
        # Recency penalty (data older than 1 month gets penalty)
        latest_date = macro_data.index[-1] if not macro_data.empty else datetime.now()
        days_old = (datetime.now() - pd.to_datetime(latest_date)).days
        recency_score = max(0, 1 - days_old / 60)  # Linear penalty over 60 days
        
        return float(completeness * recency_score)
    
    def _calculate_regime_confidence(self, features: Dict[str, float]) -> float:
        """Calculate confidence in regime classification"""
        confidence_factors = []
        
        # Data quality
        if 'macro_data_quality' in features:
            confidence_factors.append(features['macro_data_quality'])
        
        # Feature coverage (more features = higher confidence)
        key_feature_groups = [
            'yield_curve_slope', 'inflation_rate_yoy', 'unemployment_rate',
            'fed_funds_rate', 'industrial_production_growth'
        ]
        coverage = sum(1 for f in key_feature_groups if f in features) / len(key_feature_groups)
        confidence_factors.append(coverage)
        
        # Economic conditions clarity (extreme values are more confident)
        if 'economic_conditions_score' in features:
            conditions = features['economic_conditions_score']
            clarity = abs(conditions - 0.5) * 2  # Distance from neutral
            confidence_factors.append(clarity)
        
        return float(np.mean(confidence_factors)) if confidence_factors else 0.5
    
    def _get_fallback_features(self) -> Dict[str, Any]:
        """Return minimal fallback features when FRED data unavailable"""
        return {
            'macro_data_quality': 0.0,
            'macro_regime_confidence': 0.0,
            'economic_conditions_score': 0.5,  # Neutral
            'cycle_phase': 1.0,  # Assume normal conditions
            'fallback_mode': True
        }
    
    def get_regime_indicators(self, features: Dict[str, float]) -> Dict[str, Any]:
        """
        Extract regime indicators from macro features for integration
        with regime detection system
        """
        regime_indicators = {}
        
        # Map features to regime detector expected format
        if 'fed_funds_change_yoy' in features:
            regime_indicators['fed_funds_change'] = features['fed_funds_change_yoy']
        
        if 'unemployment_rate' in features:
            regime_indicators['unemployment_rate'] = features['unemployment_rate']
        
        if 'inflation_rate_yoy' in features:
            regime_indicators['inflation_rate'] = features['inflation_rate_yoy']
        
        if 'industrial_production_growth' in features:
            regime_indicators['gdp_growth'] = features['industrial_production_growth']  # Proxy
        
        if 'fed_funds_rate' in features:
            regime_indicators['fed_funds_rate'] = features['fed_funds_rate']
        
        # Add composite indicators
        if 'economic_conditions_score' in features:
            regime_indicators['economic_conditions'] = features['economic_conditions_score']
        
        if 'business_cycle_score' in features:
            regime_indicators['business_cycle'] = features['business_cycle_score']
        
        return regime_indicators