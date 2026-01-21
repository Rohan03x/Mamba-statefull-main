"""
Regime Detection & Switching for Adaptive Market Forecasting

This module implements sophisticated regime detection methods to identify
different market states (bull, bear, volatile, stable) and adapt forecasting
models accordingly.

Key Features:
- Hidden Markov Models (HMM) for regime identification
- Gaussian Mixture Models (GMM) for volatility clustering
- Volatility-based regime detection
- Regime-adaptive forecasting with different parameters per regime
- Smooth regime transitions and probability estimation
- Real-time regime monitoring and alerts

Supported Regimes:
- Bull Market: High returns, low volatility, positive momentum
- Bear Market: Low/negative returns, high volatility, negative momentum
- High Volatility: Unstable markets with large price swings
- Low Volatility: Stable markets with small price movements
- Transition: Mixed signals, regime uncertainty
"""

import logging
import warnings
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from scipy import stats

# Optional dependencies with fallbacks
try:
    from sklearn.mixture import GaussianMixture
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    warnings.warn(
        "sklearn not available - some regime detection features disabled")

try:
    import hmmlearn  # type: ignore  # noqa: F401
    from hmmlearn import hmm  # type: ignore  # noqa: F401
    HAS_HMMLEARN = True
except ImportError:
    HAS_HMMLEARN = False
    # Don't warn - we'll implement simple HMM alternative

logger = logging.getLogger(__name__)


@dataclass
class RegimeState:
    """Current regime information"""
    regime_id: int
    regime_name: str
    probability: float
    duration: int  # Days in current regime
    confidence: float  # Confidence in regime identification
    characteristics: Dict[str, float]
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None


@dataclass
class RegimeTransition:
    """Regime transition information"""
    from_regime: int
    to_regime: int
    transition_date: datetime
    transition_probability: float
    duration_previous: int


class SimpleHMM:
    """Simple Hidden Markov Model implementation as fallback"""

    def __init__(self, n_components=3, n_iter=100, tol=1e-4):
        self.n_components = n_components
        self.n_iter = n_iter
        self.tol = tol

        # Model parameters
        self.startprob_ = None
        self.transmat_ = None
        self.means_ = None
        self.covars_ = None
        self.fitted = False

    def fit(self, X):
        """Fit HMM using EM algorithm (simplified)"""
        X = np.array(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        _, _ = X.shape  # Get shape info but don't need to store

        # Initialize parameters randomly
        rng = np.random.default_rng(42)  # Use modern random generator
        self.startprob_ = np.ones(self.n_components) / self.n_components
        self.transmat_ = rng.random((self.n_components, self.n_components))
        self.transmat_ = self.transmat_ / \
            self.transmat_.sum(axis=1, keepdims=True)

        # Initialize means with k-means-like approach
        percentiles = np.linspace(10, 90, self.n_components)
        self.means_ = np.percentile(X, percentiles, axis=0).T
        self.covars_ = np.array([np.cov(X.T)
                                for _ in range(self.n_components)])

        # Simple EM iterations (simplified)
        for _ in range(self.n_iter):
            # E-step: compute posterior probabilities (simplified)
            posteriors = self._compute_posteriors(X)

            # M-step: update parameters
            old_means = self.means_.copy()

            # Update means
            for k in range(self.n_components):
                weights = posteriors[:, k]
                if weights.sum() > 0:
                    self.means_[k] = np.average(X, axis=0, weights=weights)

            # Check convergence
            if np.allclose(old_means, self.means_, atol=self.tol):
                break

        self.fitted = True
        return self

    def _compute_posteriors(self, X):
        """Compute posterior probabilities (simplified)"""
        log_probs = np.zeros((len(X), self.n_components))

        for k in range(self.n_components):
            # Simplified likelihood computation
            diff = X - self.means_[k]
            log_probs[:, k] = -0.5 * np.sum(diff**2, axis=1)

        # Convert to probabilities
        log_probs -= log_probs.max(axis=1,
                                   keepdims=True)  # Numerical stability
        probs = np.exp(log_probs)
        probs /= probs.sum(axis=1, keepdims=True)

        return probs

    def predict(self, X):
        """Predict most likely regime sequence"""
        if not self.fitted:
            raise ValueError("Model must be fitted first")

        posteriors = self._compute_posteriors(X)
        return np.argmax(posteriors, axis=1)

    def predict_proba(self, X):
        """Predict regime probabilities"""
        if not self.fitted:
            raise ValueError("Model must be fitted first")

        return self._compute_posteriors(X)


class VolatilityRegimeDetector:
    """Volatility-based regime detection using rolling statistics"""

    def __init__(
        self,
        window_short: int = 20,
        window_long: int = 60,
        volatility_threshold_high: float = 0.75,
        volatility_threshold_low: float = 0.25
    ):
        self.window_short = window_short
        self.window_long = window_long
        self.volatility_threshold_high = volatility_threshold_high
        self.volatility_threshold_low = volatility_threshold_low

    def detect_regimes(self, returns: pd.Series) -> pd.Series:
        """Detect regimes based on volatility and returns"""
        # Calculate rolling statistics
        vol_short = returns.rolling(self.window_short).std() * np.sqrt(252)
        vol_long = returns.rolling(self.window_long).std() * np.sqrt(252)
        returns_short = returns.rolling(self.window_short).mean() * 252

        # Normalize volatility relative to long-term average
        vol_ratio = vol_short / vol_long

        # Define regime rules
        regimes = pd.Series(index=returns.index, dtype=int)

        # High volatility regimes
        high_vol = vol_ratio > (1 + self.volatility_threshold_high)

        # Low volatility regimes
        low_vol = vol_ratio < (1 - self.volatility_threshold_low)

        # Bull/Bear based on returns
        # Positive returns, not high vol
        bull_market = (returns_short > 0.05) & ~high_vol
        # Negative returns or high vol
        bear_market = (returns_short < -0.05) | high_vol

        # Assign regime labels
        regimes.loc[bull_market] = 0  # Bull
        regimes.loc[bear_market] = 1  # Bear
        regimes.loc[high_vol & ~bear_market] = 2  # High Volatility
        regimes.loc[low_vol & ~bull_market] = 3  # Low Volatility

        # Forward fill any missing values
        regimes = regimes.ffill().fillna(0)

        return regimes.astype(int)


class MarketRegimeDetector:
    """Main regime detection system with multiple methods"""

    def __init__(
        self,
        detection_method: str = 'hmm',
        n_regimes: int = 4,
        lookback_window: int = 252,
        min_regime_duration: int = 5
    ):
        self.detection_method = detection_method
        self.n_regimes = n_regimes
        self.lookback_window = lookback_window
        self.min_regime_duration = min_regime_duration

        # Model components
        self.hmm_model = None
        self.gmm_model = None
        self.volatility_detector = None

        # Fitted data and results
        self.regime_history = None
        self.regime_probabilities = None
        self.regime_characteristics = {}
        self.transition_matrix = None
        self.current_regime = None

        # Regime names mapping
        self.regime_names = {
            0: "Bull Market",
            1: "Bear Market",
            2: "High Volatility",
            3: "Low Volatility"
        }

    def fit(self, returns_data: pd.Series) -> 'MarketRegimeDetector':
        """Fit regime detection model to historical data"""
        logger.info(f"Fitting regime detector with {len(returns_data)} observations")

        # Prepare features for regime detection
        features = self._prepare_features(returns_data)

        # Fit the appropriate model
        if self.detection_method == 'hmm':
            self._fit_hmm(features)
        elif self.detection_method == 'gmm':
            self._fit_gmm(features)
        elif self.detection_method == 'volatility':
            self._fit_volatility_detector(returns_data)
        else:
            raise ValueError(f"Unknown detection method: {self.detection_method}")

        # Detect regimes for the full history
        regimes = self.detect_regimes(returns_data)
        self.regime_history = regimes

        # Calculate regime characteristics
        self._calculate_regime_characteristics(returns_data, regimes)

        # Estimate transition matrix
        self._estimate_transition_matrix(regimes)

        # Set current regime
        if len(regimes) > 0:
            self.current_regime = self._get_current_regime_state(
                returns_data, regimes)

        logger.info(f"Regime detection fitted successfully using {self.detection_method}")
        return self

    def _prepare_features(self, returns: pd.Series) -> np.ndarray:
        """Prepare features for regime detection"""
        # Calculate various market indicators
        vol_5 = returns.rolling(5).std() * np.sqrt(252)
        vol_20 = returns.rolling(20).std() * np.sqrt(252)
        vol_60 = returns.rolling(60).std() * np.sqrt(252)

        ret_5 = returns.rolling(5).mean() * 252
        ret_20 = returns.rolling(20).mean() * 252

        # Volume proxy (use absolute returns as proxy)
        volume_proxy = returns.abs().rolling(20).mean()

        # Momentum indicators
        momentum = returns.rolling(10).sum()

        # Calculate ratios carefully to avoid division issues
        # Add small epsilon to avoid division by zero
        vol_ratio = vol_20 / (vol_60 + 1e-8)
        # Add small epsilon to avoid division by zero
        return_vol_ratio = ret_20 / (vol_20 + 1e-8)

        # Combine features - ensure all are 1D Series
        features = pd.DataFrame({
            'returns': returns,
            'volatility_short': vol_5,
            'volatility_medium': vol_20,
            'volatility_long': vol_60,
            'returns_short': ret_5,
            'returns_medium': ret_20,
            'volume_proxy': volume_proxy,
            'momentum': momentum,
            'vol_ratio': vol_ratio,
            'return_vol_ratio': return_vol_ratio
        }, index=returns.index)  # Explicitly pass the index

        # Remove NaN values and normalize
        features = features.dropna()

        # Use subset of most informative features
        selected_features = [
            'returns',
            'volatility_medium',
            'returns_medium',
            'vol_ratio']
        return features[selected_features].values

    def _fit_hmm(self, features: np.ndarray):
        """Fit Hidden Markov Model"""
        if HAS_HMMLEARN:
            try:
                from hmmlearn.hmm import GaussianHMM  # type: ignore
                self.hmm_model = GaussianHMM(
                    n_components=self.n_regimes,
                    covariance_type="full",
                    n_iter=100
                )
                self.hmm_model.fit(features)
                logger.info("Using hmmlearn package for HMM")
            except Exception as e:
                logger.warning(f"hmmlearn failed: {e}, using simple HMM")
                self._fit_simple_hmm(features)
        else:
            self._fit_simple_hmm(features)

    def _fit_simple_hmm(self, features: np.ndarray):
        """Fit simple HMM implementation"""
        self.hmm_model = SimpleHMM(n_components=self.n_regimes)
        self.hmm_model.fit(features)
        logger.info("Using simple HMM implementation")

    def _fit_gmm(self, features: np.ndarray):
        """Fit Gaussian Mixture Model"""
        if not HAS_SKLEARN:
            raise ImportError("sklearn required for GMM")

        self.gmm_model = GaussianMixture(
            n_components=self.n_regimes,
            covariance_type='full',
            max_iter=100,
            random_state=42
        )
        self.gmm_model.fit(features)
        logger.info("GMM fitted successfully")

    def _fit_volatility_detector(self, _returns: pd.Series):
        """Fit volatility-based detector"""
        self.volatility_detector = VolatilityRegimeDetector()
        logger.info("Volatility-based detector initialized")

    def detect_regimes(self, returns_data: pd.Series) -> pd.Series:
        """Detect regimes for given return data"""
        if self.detection_method == 'volatility':
            return self.volatility_detector.detect_regimes(returns_data)

        # Prepare features
        features = self._prepare_features(returns_data)

        if self.detection_method == 'hmm':
            regimes = self.hmm_model.predict(features)
            # Get probabilities if available
            if hasattr(self.hmm_model, 'predict_proba'):
                self.regime_probabilities = self.hmm_model.predict_proba(
                    features)
        elif self.detection_method == 'gmm':
            regimes = self.gmm_model.predict(features)
            self.regime_probabilities = self.gmm_model.predict_proba(features)

        # Create series with proper index - align with features index
        features_index = self._get_features_index(returns_data)
        regime_series = pd.Series(index=features_index, data=regimes)

        # Apply minimum duration filter
        regime_series = self._smooth_regimes(regime_series)

        return regime_series

    def _get_features_index(self, returns_data: pd.Series) -> pd.Index:
        """Get the proper index for features after NaN removal"""
        # Calculate various market indicators (same as _prepare_features)
        vol_5 = returns_data.rolling(5).std() * np.sqrt(252)
        vol_20 = returns_data.rolling(20).std() * np.sqrt(252)
        vol_60 = returns_data.rolling(60).std() * np.sqrt(252)

        ret_5 = returns_data.rolling(5).mean() * 252
        ret_20 = returns_data.rolling(20).mean() * 252

        # Volume proxy (use absolute returns as proxy)
        volume_proxy = returns_data.abs().rolling(20).mean()

        # Momentum indicators
        momentum = returns_data.rolling(10).sum()

        # Calculate ratios carefully to avoid division issues
        # Add small epsilon to avoid division by zero
        vol_ratio = vol_20 / (vol_60 + 1e-8)
        # Add small epsilon to avoid division by zero
        return_vol_ratio = ret_20 / (vol_20 + 1e-8)

        # Combine features
        features = pd.DataFrame({
            'returns': returns_data,
            'volatility_short': vol_5,
            'volatility_medium': vol_20,
            'volatility_long': vol_60,
            'returns_short': ret_5,
            'returns_medium': ret_20,
            'volume_proxy': volume_proxy,
            'momentum': momentum,
            'vol_ratio': vol_ratio,
            'return_vol_ratio': return_vol_ratio
        }, index=returns_data.index)  # Explicitly pass the index

        # Remove NaN values and return index
        features_clean = features.dropna()
        return features_clean.index

    def _smooth_regimes(self, regimes: pd.Series) -> pd.Series:
        """Apply minimum duration filter to reduce regime switching noise"""
        smoothed = regimes.copy()

        i = 0
        while i < len(smoothed):
            current_regime = smoothed.iloc[i]

            # Find end of current regime
            j = i
            while j < len(smoothed) and smoothed.iloc[j] == current_regime:
                j += 1

            duration = j - i

            # If duration too short, assign to previous regime
            if duration < self.min_regime_duration and i > 0:
                smoothed.iloc[i:j] = smoothed.iloc[i-1]

            i = j

        return smoothed

    def _calculate_regime_characteristics(
            self, returns: pd.Series, regimes: pd.Series):
        """Calculate characteristics for each regime"""
        for regime_id in range(self.n_regimes):
            # Align indices properly
            aligned_returns = returns.reindex(
                regimes.index, method='nearest').fillna(0)
            mask = regimes == regime_id
            regime_returns = aligned_returns[mask]

            if len(regime_returns) > 0:
                characteristics = {
                    'mean_return': regime_returns.mean() * 252,  # Annualized
                    'volatility': regime_returns.std() * np.sqrt(252),  # Annualized
                    'sharpe_ratio': (regime_returns.mean() * 252) / (regime_returns.std() * np.sqrt(252) + 1e-8),
                    'skewness': stats.skew(regime_returns),
                    'kurtosis': stats.kurtosis(regime_returns),
                    'max_drawdown': self._calculate_max_drawdown(regime_returns),
                    'frequency': mask.sum() / len(mask),
                    'avg_duration': self._calculate_avg_duration(regimes, regime_id)
                }
            else:
                characteristics = dict.fromkeys([
                    'mean_return', 'volatility', 'sharpe_ratio', 'skewness',
                    'kurtosis', 'max_drawdown', 'frequency', 'avg_duration'
                ], 0.0)

            self.regime_characteristics[regime_id] = characteristics

            logger.info(f"Regime {regime_id} ({self.regime_names.get(regime_id, 'Unknown')}): Return={characteristics['mean_return']:.2%}, Vol={characteristics['volatility']:.2%}, Freq={characteristics['frequency']:.1%}")

    def _calculate_max_drawdown(self, returns: pd.Series) -> float:
        """Calculate maximum drawdown"""
        if len(returns) == 0:
            return 0.0

        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        return abs(drawdown.min())

    def _calculate_avg_duration(
            self,
            regimes: pd.Series,
            regime_id: int) -> float:
        """Calculate average duration of regime"""
        durations = []
        current_duration = 0

        for regime in regimes:
            if regime == regime_id:
                current_duration += 1
            else:
                if current_duration > 0:
                    durations.append(current_duration)
                current_duration = 0

        # Don't forget the last regime if it ends the series
        if current_duration > 0:
            durations.append(current_duration)

        return np.mean(durations) if durations else 0.0

    def _estimate_transition_matrix(self, regimes: pd.Series):
        """Estimate regime transition probabilities"""
        transitions = np.zeros((self.n_regimes, self.n_regimes))

        for i in range(len(regimes) - 1):
            from_regime = regimes.iloc[i]
            to_regime = regimes.iloc[i + 1]
            transitions[from_regime, to_regime] += 1

        # Normalize to probabilities
        row_sums = transitions.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1  # Avoid division by zero
        self.transition_matrix = transitions / row_sums

        logger.info("Transition matrix estimated")

    def _get_current_regime_state(
            self,
            returns: pd.Series,
            regimes: pd.Series) -> Optional[RegimeState]:
        """Get current regime state information"""
        if len(regimes) == 0:
            return None

        current_regime_id = regimes.iloc[-1]
        regime_name = self.regime_names.get(
            current_regime_id, f"Regime {current_regime_id}")

        # Calculate duration in current regime
        duration = 1
        for i in range(len(regimes) - 2, -1, -1):
            if regimes.iloc[i] == current_regime_id:
                duration += 1
            else:
                break

        # Get probability if available
        probability = 1.0
        if self.regime_probabilities is not None and len(
                self.regime_probabilities) > 0:
            probability = self.regime_probabilities[-1, current_regime_id]

        # Get characteristics
        characteristics = self.regime_characteristics.get(
            current_regime_id, {})

        # Calculate confidence based on duration and probability
        confidence = min(0.95, 0.5 + 0.1 * duration + 0.4 * probability)

        return RegimeState(
            regime_id=current_regime_id,
            regime_name=regime_name,
            probability=probability,
            duration=duration,
            confidence=confidence,
            characteristics=characteristics,
            start_date=returns.index[-duration] if duration <= len(
                returns) else None
        )

    def forecast_regime_probabilities(self, horizon: int = 5) -> np.ndarray:
        """Forecast regime probabilities for future periods"""
        if self.transition_matrix is None or self.current_regime is None:
            # Equal probabilities if no information
            return np.ones((horizon, self.n_regimes)) / self.n_regimes

        # Current regime probabilities (one-hot encoding for simplicity)
        current_probs = np.zeros(self.n_regimes)
        current_probs[self.current_regime.regime_id] = 1.0

        # Forecast using transition matrix
        forecasts = np.zeros((horizon, self.n_regimes))
        prob_vec = current_probs.copy()

        for h in range(horizon):
            prob_vec = prob_vec @ self.transition_matrix
            forecasts[h] = prob_vec

        return forecasts

    def get_regime_adjusted_parameters(
            self, base_params: Dict[str, float]) -> Dict[str, float]:
        """Adjust model parameters based on current regime"""
        if self.current_regime is None:
            return base_params

        adjusted_params = base_params.copy()
        characteristics = self.current_regime.characteristics

        # Adjust parameters based on regime characteristics
        if 'volatility' in characteristics:
            # Normalize to typical vol
            vol_factor = characteristics['volatility'] / 0.15
            vol_factor = np.clip(vol_factor, 0.5, 3.0)  # Reasonable bounds

            # Increase ensemble diversity in high vol regimes
            if 'ensemble_diversity' in adjusted_params:
                adjusted_params['ensemble_diversity'] *= vol_factor

            # Adjust learning rates
            if 'learning_rate' in adjusted_params:
                # Lower LR in high vol
                adjusted_params['learning_rate'] *= (1.0 / vol_factor)

        if 'mean_return' in characteristics:
            # Adjust momentum parameters based on trend strength
            trend_strength = abs(characteristics['mean_return'])
            if 'momentum_factor' in adjusted_params:
                adjusted_params['momentum_factor'] *= (
                    1.0 + trend_strength * 2)

        return adjusted_params


class RegimeAdaptiveForecaster:
    """Forecasting system that adapts to market regimes"""

    def __init__(
        self,
        regime_detector: MarketRegimeDetector,
        base_model_params: Optional[Dict[str, Any]] = None
    ):
        self.regime_detector = regime_detector
        self.base_model_params = base_model_params or {}
        self.regime_models = {}  # Store different models per regime

    def fit(self, returns_data: pd.Series) -> None:
        """Fit regime-adaptive forecasting models"""
        # First fit the regime detector
        self.regime_detector.fit(returns_data)

        # Train separate models for each regime if needed
        regimes = self.regime_detector.regime_history

        for regime_id in range(self.regime_detector.n_regimes):
            # Align indices properly for regime data extraction
            aligned_returns = returns_data.reindex(
                regimes.index, method='nearest').fillna(0)
            regime_mask = regimes == regime_id
            regime_data = aligned_returns[regime_mask]

            if len(regime_data) > 20:  # Minimum data requirement
                # Get regime-adjusted parameters
                adjusted_params = self.regime_detector.get_regime_adjusted_parameters(
                    self.base_model_params)

                # Store adjusted parameters for this regime
                self.regime_models[regime_id] = {
                    'params': adjusted_params,
                    'data_size': len(regime_data),
                    'regime_name': self.regime_detector.regime_names.get(
                        regime_id,
                        f"Regime {regime_id}")}

        logger.info(
            f"Regime-adaptive forecaster fitted with {len(self.regime_models)} regime-specific models")

    def forecast(self, horizon: int = 5) -> Dict[str, Any]:
        """Generate regime-adaptive forecast"""
        current_regime = self.regime_detector.current_regime

        if current_regime is None:
            return {"error": "No current regime detected"}

        # Get regime-specific parameters
        regime_params = self.regime_models.get(
            current_regime.regime_id,
            {'params': self.base_model_params}
        )

        # Forecast regime probabilities
        regime_forecast = self.regime_detector.forecast_regime_probabilities(
            horizon)

        # Build comprehensive forecast result
        forecast_result = {
            'current_regime': {
                'id': current_regime.regime_id,
                'name': current_regime.regime_name,
                'probability': current_regime.probability,
                'duration': current_regime.duration,
                'confidence': current_regime.confidence,
                'characteristics': current_regime.characteristics
            },
            'regime_forecast': regime_forecast,
            'adjusted_parameters': regime_params['params'],
            'horizon': horizon,
            'regime_names': self.regime_detector.regime_names,
            'transition_matrix': self.regime_detector.transition_matrix
        }

        return forecast_result


def test_regime_detection_system():
    """Test the regime detection and switching system"""
    print("=== TESTING REGIME DETECTION & SWITCHING SYSTEM ===")

    # Generate synthetic market data with different regimes
    rng = np.random.default_rng(42)  # Use modern random generator
    n_days = 1000

    # Create regime switching data
    regime_lengths = [200, 300, 200, 300]  # Length of each regime
    regime_params = [
        {'mean': 0.001, 'vol': 0.015},  # Bull market
        {'mean': -0.002, 'vol': 0.025},  # Bear market
        {'mean': 0.0005, 'vol': 0.035},  # High volatility
        {'mean': 0.0003, 'vol': 0.008}  # Low volatility
    ]

    returns_data = []
    true_regimes = []

    for i, length in enumerate(regime_lengths):
        params = regime_params[i]
        regime_returns = rng.normal(params['mean'], params['vol'], length)
        returns_data.extend(regime_returns)
        true_regimes.extend([i] * length)

    # Convert to pandas series
    dates = pd.date_range('2020-01-01', periods=len(returns_data), freq='D')
    returns_series = pd.Series(returns_data, index=dates)
    true_regimes_series = pd.Series(true_regimes, index=dates)

    print(f"Generated {len(returns_data)} days of synthetic market data")
    print(f"True regime distribution: {np.bincount(true_regimes)}")

    # Test different detection methods
    methods = ['volatility', 'hmm']
    if HAS_SKLEARN:
        methods.append('gmm')

    for method in methods:
        print(f"\n--- Testing {method.upper()} Method ---")

        try:
            # Initialize detector
            detector = MarketRegimeDetector(
                detection_method=method,
                n_regimes=4,
                lookback_window=500
            )

            # Fit detector
            detector.fit(returns_series)
            print(f"✓ {method.upper()} detector fitted successfully")

            # Get detected regimes
            detected_regimes = detector.regime_history
            print(f"  Detected regime distribution: {np.bincount(detected_regimes)}")

            # Show current regime
            current = detector.current_regime
            if current:
                print(f"  Current regime: {current.regime_name} (ID: {current.regime_id})")
                print(f"  Duration: {current.duration} days, Confidence: {current.confidence:.2f}")
                vol_pct = current.characteristics.get('volatility', 0) if hasattr(current, 'characteristics') else 0
                print(f"  Characteristics: Vol={vol_pct:.2%}")

            # Test regime forecasting
            if hasattr(detector, 'forecast_regime_probabilities'):
                regime_forecast = detector.forecast_regime_probabilities(
                    horizon=10)
                print(
                    f"  10-day regime forecast shape: {regime_forecast.shape}")

            # Test adaptive forecaster
            print("  Testing adaptive forecaster...")
            base_params = {'learning_rate': 0.01, 'ensemble_diversity': 1.0}
            adaptive_forecaster = RegimeAdaptiveForecaster(
                detector, base_params)
            adaptive_forecaster.fit(returns_series)

            forecast_result = adaptive_forecaster.forecast(horizon=5)
            print(
                f"  ✓ Adaptive forecast generated for {forecast_result['horizon']} days")
            print(
                f"  Adjusted learning rate: {forecast_result['adjusted_parameters'].get('learning_rate', 'N/A')}")

        except Exception as e:
            print(f"  ✗ Error with {method} method: {str(e)}")
            import traceback
            traceback.print_exc()

    # Test regime characteristics
    print("\n--- Regime Characteristics Analysis ---")
    if 'detector' in locals():
        for regime_id, characteristics in detector.regime_characteristics.items():
            regime_name = detector.regime_names.get(
                regime_id, f"Regime {regime_id}")
            print(f"  {regime_name}:")
            print(
                f"    Annual Return: {characteristics.get('mean_return', 0):.2%}")
            print(
                f"    Annual Volatility: {characteristics.get('volatility', 0):.2%}")
            print(
                f"    Sharpe Ratio: {characteristics.get('sharpe_ratio', 0):.2f}")
            print(f"    Frequency: {characteristics.get('frequency', 0):.1%}")
            print(f"    Avg Duration: {characteristics.get('avg_duration', 0):.1f} days")

    print("\n=== REGIME DETECTION TESTING COMPLETE ===")


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)

    # Run tests
    test_regime_detection_system()
