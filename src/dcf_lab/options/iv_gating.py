"""
IV Extremeness Detection and Gating

This module implements mechanisms to detect when implied volatility is extreme
and adaptively gate the options expert to up-weight it during these periods.

Key components:
- IV extremeness detection using percentiles and statistical measures
- Adaptive gating networks that learn when to trust options signals
- Integration with ensemble framework for end-to-end learning
- Options expert weight adjustment based on market conditions
"""

import numpy as np
from typing import Dict, List, Any
from dataclasses import dataclass
from datetime import datetime
from scipy import stats
import logging

logger = logging.getLogger(__name__)

@dataclass
class ExtremenessMetrics:
    """Metrics for IV extremeness detection"""
    iv_percentile: float
    iv_zscore: float
    term_structure_slope: float
    skew_level: float
    extremeness_score: float
    regime_indicator: str  # 'low', 'normal', 'high', 'extreme'
    confidence: float
    metadata: Dict[str, Any]

@dataclass
class GatingDecision:
    """Gating decision for options expert"""
    options_weight: float
    feature_weight: float
    extremeness_factor: float
    regime_adjustment: float
    final_weight: float
    reasoning: str
    confidence: float
    metadata: Dict[str, Any]

class IVExtremenessDetector:
    """Detect when implied volatility is in extreme regimes"""
    
    def __init__(self, lookback_window: int = 252, 
                 percentile_thresholds: Dict[str, float] = None):
        """
        Args:
            lookback_window: Days to look back for percentile calculations
            percentile_thresholds: Thresholds for regime classification
        """
        self.lookback_window = lookback_window
        self.percentile_thresholds = percentile_thresholds or {
            'low': 20.0,
            'normal_low': 40.0,
            'normal_high': 60.0,
            'high': 80.0,
            'extreme': 90.0
        }
        
        self.iv_history = []
        self.term_structure_history = []
        self.skew_history = []
    
    def update_history(self, iv_data: Dict[str, float], timestamp: datetime = None):
        """
        Update IV history with new data
        
        Args:
            iv_data: Dict with 'atm_iv', 'term_structure_slope', 'skew', etc.
            timestamp: Timestamp for the data
        """
        
        if timestamp is None:
            timestamp = datetime.now()
        
        # Store data with timestamp
        data_point = {
            'timestamp': timestamp,
            'atm_iv': iv_data.get('atm_iv', 0.0),
            'term_structure_slope': iv_data.get('term_structure_slope', 0.0),
            'skew': iv_data.get('skew', 0.0),
            **iv_data
        }
        
        self.iv_history.append(data_point)
        
        # Maintain window size
        if len(self.iv_history) > self.lookback_window:
            self.iv_history = self.iv_history[-self.lookback_window:]
    
    def detect_extremeness(self, current_iv: float,
                          term_structure_slope: float = 0.0,
                          skew: float = 0.0) -> ExtremenessMetrics:
        """
        Detect IV extremeness for current conditions
        
        Args:
            current_iv: Current ATM implied volatility
            term_structure_slope: Term structure slope
            skew: Current skew level
            
        Returns:
            ExtremenessMetrics with extremeness assessment
        """
        
        if len(self.iv_history) < 30:  # Need minimum history
            return ExtremenessMetrics(
                iv_percentile=50.0,
                iv_zscore=0.0,
                term_structure_slope=term_structure_slope,
                skew_level=skew,
                extremeness_score=0.5,
                regime_indicator='normal',
                confidence=0.3,  # Low confidence with limited history
                metadata={'insufficient_history': True, 'history_length': len(self.iv_history)}
            )
        
        # Calculate percentile
        historical_ivs = [point['atm_iv'] for point in self.iv_history]
        iv_percentile = stats.percentileofscore(historical_ivs, current_iv)
        
        # Calculate z-score
        iv_mean = np.mean(historical_ivs)
        iv_std = np.std(historical_ivs)
        iv_zscore = (current_iv - iv_mean) / iv_std if iv_std > 0 else 0.0
        
        # Determine regime
        regime = self._classify_regime(iv_percentile)
        
        # Calculate composite extremeness score
        extremeness_score = self._calculate_extremeness_score(
            iv_percentile, abs(iv_zscore), term_structure_slope, skew
        )
        
        # Confidence based on history length and consistency
        confidence = self._calculate_confidence(iv_percentile, iv_zscore)
        
        return ExtremenessMetrics(
            iv_percentile=iv_percentile,
            iv_zscore=iv_zscore,
            term_structure_slope=term_structure_slope,
            skew_level=skew,
            extremeness_score=extremeness_score,
            regime_indicator=regime,
            confidence=confidence,
            metadata={
                'history_length': len(self.iv_history),
                'iv_mean': iv_mean,
                'iv_std': iv_std,
                'current_iv': current_iv
            }
        )
    
    def _classify_regime(self, percentile: float) -> str:
        """Classify IV regime based on percentile"""
        
        if percentile >= self.percentile_thresholds['extreme']:
            return 'extreme'
        elif percentile >= self.percentile_thresholds['high']:
            return 'high'
        elif percentile >= self.percentile_thresholds['normal_high']:
            return 'normal'
        elif percentile >= self.percentile_thresholds['normal_low']:
            return 'normal'
        elif percentile >= self.percentile_thresholds['low']:
            return 'low'
        else:
            return 'very_low'
    
    def _calculate_extremeness_score(self, percentile: float, abs_zscore: float,
                                   term_structure_slope: float, skew: float) -> float:
        """Calculate composite extremeness score (0-1)"""
        
        # Percentile component (0-1)
        percentile_score = min(max(percentile / 100.0, 0.0), 1.0)
        
        # Z-score component (capped at 3 sigma)
        zscore_score = min(abs_zscore / 3.0, 1.0)
        
        # Term structure component (steep contango/backwardation)
        ts_score = min(abs(term_structure_slope) / 0.1, 1.0)  # Normalize by 10% slope
        
        # Skew component
        skew_score = min(abs(skew) / 0.2, 1.0)  # Normalize by 20% skew
        
        # Weighted combination
        weights = [0.4, 0.3, 0.2, 0.1]  # Percentile gets highest weight
        scores = [percentile_score, zscore_score, ts_score, skew_score]
        
        extremeness_score = np.average(scores, weights=weights)
        return extremeness_score
    
    def _calculate_confidence(self, percentile: float, zscore: float) -> float:
        """Calculate confidence in extremeness assessment"""
        
        # More confident when we have sufficient history
        history_confidence = min(len(self.iv_history) / self.lookback_window, 1.0)
        
        # More confident when signals are consistent
        if abs(zscore) > 2.0 and (percentile > 80 or percentile < 20):
            signal_confidence = 0.9
        elif abs(zscore) > 1.5 and (percentile > 70 or percentile < 30):
            signal_confidence = 0.7
        elif abs(zscore) > 1.0 and (percentile > 60 or percentile < 40):
            signal_confidence = 0.5
        else:
            signal_confidence = 0.3
        
        # Combined confidence
        return (history_confidence * 0.6 + signal_confidence * 0.4)

class AdaptiveGatingNetwork:
    """Neural network that learns when to gate options expert"""
    
    def __init__(self, input_features: List[str], hidden_size: int = 64):
        """
        Args:
            input_features: List of feature names for gating decisions
            hidden_size: Hidden layer size
        """
        self.input_features = input_features
        self.hidden_size = hidden_size
        self.model = None
        self.is_trained = False
        self.training_history = []
    
    def _build_model(self, input_size: int):
        """Build the gating network model"""
        import torch.nn as nn
        
        self.model = nn.Sequential(
            nn.Linear(input_size, self.hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(self.hidden_size, self.hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_size // 2, 1),
            nn.Sigmoid()  # Output between 0 and 1
        )
        
        return self.model
    
    def prepare_gating_features(self, extremeness_metrics: ExtremenessMetrics,
                              market_features: Dict[str, float]) -> np.ndarray:
        """
        Prepare features for gating decision
        
        Args:
            extremeness_metrics: IV extremeness metrics
            market_features: Additional market features
            
        Returns:
            Feature vector for gating network
        """
        
        features = [
            extremeness_metrics.iv_percentile / 100.0,
            extremeness_metrics.iv_zscore / 3.0,  # Normalize to [-1, 1] roughly
            extremeness_metrics.extremeness_score,
            extremeness_metrics.confidence,
            extremeness_metrics.term_structure_slope,
            extremeness_metrics.skew_level,
        ]
        
        # Add market features
        for feature_name in self.input_features:
            if feature_name in market_features:
                features.append(market_features[feature_name])
            else:
                features.append(0.0)  # Default if missing
        
        return np.array(features)
    
    def train(self, training_data: List[Dict], validation_split: float = 0.2) -> Dict[str, Any]:
        """
        Train the gating network
        
        Args:
            training_data: List of dicts with 'features', 'target_weight', 'outcome'
            validation_split: Fraction for validation
            
        Returns:
            Training results
        """
        import torch
        import torch.nn as nn
        import torch.optim as optim
        
        if len(training_data) < 10:
            raise ValueError("Need at least 10 training samples")
        
        # Prepare data
        features_list = []
        targets_list = []
        
        for sample in training_data:
            features_list.append(sample['features'])
            targets_list.append(sample['target_weight'])
        
        features_array = np.array(features_list)
        targets_array = np.array(targets_list)
        
        # Build model
        input_size = features_array.shape[1]
        if self.model is None:
            self.model = self._build_model(input_size)
        
        # Split data
        n_samples = len(features_array)
        n_val = int(n_samples * validation_split)
        n_train = n_samples - n_val
        
        train_features = torch.FloatTensor(features_array[:n_train])
        train_targets = torch.FloatTensor(targets_array[:n_train]).unsqueeze(-1)
        
        if n_val > 0:
            val_features = torch.FloatTensor(features_array[n_train:])
            val_targets = torch.FloatTensor(targets_array[n_train:]).unsqueeze(-1)
        else:
            val_features = val_targets = None
        
        # Training setup
        optimizer = optim.Adam(self.model.parameters(), lr=0.001, weight_decay=0.01)
        criterion = nn.MSELoss()
        
        # Training loop
        epochs = 100
        train_losses = []
        val_losses = []
        
        for epoch in range(epochs):
            # Training
            self.model.train()
            optimizer.zero_grad()
            
            outputs = self.model(train_features)
            loss = criterion(outputs, train_targets)
            loss.backward()
            optimizer.step()
            
            train_losses.append(loss.item())
            
            # Validation
            if val_features is not None:
                self.model.eval()
                with torch.no_grad():
                    val_outputs = self.model(val_features)
                    val_loss = criterion(val_outputs, val_targets)
                    val_losses.append(val_loss.item())
        
        self.is_trained = True
        
        results = {
            'train_loss': train_losses[-1],
            'val_loss': val_losses[-1] if val_losses else None,
            'epochs': epochs,
            'input_size': input_size
        }
        
        self.training_history.append(results)
        return results
    
    def predict_gate_weight(self, extremeness_metrics: ExtremenessMetrics,
                           market_features: Dict[str, float]) -> float:
        """
        Predict optimal gate weight for options expert
        
        Args:
            extremeness_metrics: IV extremeness metrics
            market_features: Market features
            
        Returns:
            Gate weight (0-1) for options expert
        """
        
        if not self.is_trained:
            # Default rule-based gating if not trained
            return self._rule_based_gating(extremeness_metrics)
        
        import torch
        
        features = self.prepare_gating_features(extremeness_metrics, market_features)
        features_tensor = torch.FloatTensor(features).unsqueeze(0)
        
        self.model.eval()
        with torch.no_grad():
            weight = self.model(features_tensor).item()
        
        return weight
    
    def _rule_based_gating(self, extremeness_metrics: ExtremenessMetrics) -> float:
        """Rule-based gating as fallback"""
        
        # Simple rule: higher weight when IV is extreme
        if extremeness_metrics.regime_indicator == 'extreme':
            return 0.8
        elif extremeness_metrics.regime_indicator == 'high':
            return 0.6
        elif extremeness_metrics.regime_indicator == 'low':
            return 0.3
        else:
            return 0.4  # Normal regime

class OptionsExpertGate:
    """Main gating system for options expert in ensemble"""
    
    def __init__(self, lookback_window: int = 252):
        self.extremeness_detector = IVExtremenessDetector(lookback_window)
        self.gating_network = None
        self.gating_history = []
        
        # Regime-specific weight adjustments
        self.regime_weights = {
            'very_low': 0.2,
            'low': 0.3,
            'normal': 0.4,
            'high': 0.6,
            'extreme': 0.8
        }
    
    def initialize_gating_network(self, input_features: List[str]):
        """Initialize the adaptive gating network"""
        self.gating_network = AdaptiveGatingNetwork(input_features)
    
    def make_gating_decision(self, current_iv: float,
                           market_features: Dict[str, float],
                           base_options_weight: float = 0.3) -> GatingDecision:
        """
        Make gating decision for options expert
        
        Args:
            current_iv: Current implied volatility
            market_features: Market features for gating
            base_options_weight: Base weight for options expert
            
        Returns:
            GatingDecision with weight adjustments
        """
        
        # Extract IV components from market features
        term_structure_slope = market_features.get('term_structure_slope', 0.0)
        skew = market_features.get('skew', 0.0)
        
        # Detect extremeness
        extremeness_metrics = self.extremeness_detector.detect_extremeness(
            current_iv, term_structure_slope, skew
        )
        
        # Get adaptive weight if network is available
        if self.gating_network and self.gating_network.is_trained:
            adaptive_weight = self.gating_network.predict_gate_weight(
                extremeness_metrics, market_features
            )
            reasoning = "adaptive_network"
        else:
            adaptive_weight = self.regime_weights.get(
                extremeness_metrics.regime_indicator, 0.4
            )
            reasoning = "rule_based"
        
        # Apply extremeness factor
        extremeness_factor = extremeness_metrics.extremeness_score
        
        # Calculate final weight
        regime_adjustment = adaptive_weight
        final_weight = base_options_weight * (1 + extremeness_factor * regime_adjustment)
        final_weight = np.clip(final_weight, 0.0, 1.0)
        
        # Complementary feature weight
        feature_weight = 1.0 - final_weight
        
        decision = GatingDecision(
            options_weight=final_weight,
            feature_weight=feature_weight,
            extremeness_factor=extremeness_factor,
            regime_adjustment=regime_adjustment,
            final_weight=final_weight,
            reasoning=reasoning,
            confidence=extremeness_metrics.confidence,
            metadata={
                'extremeness_metrics': extremeness_metrics,
                'base_weight': base_options_weight,
                'adaptive_weight': adaptive_weight,
                'regime': extremeness_metrics.regime_indicator
            }
        )
        
        # Store decision history
        self.gating_history.append({
            'timestamp': datetime.now(),
            'decision': decision,
            'current_iv': current_iv,
            'market_features': market_features
        })
        
        return decision
    
    def update_iv_history(self, iv_data: Dict[str, float], timestamp: datetime = None):
        """Update IV history for extremeness detection"""
        self.extremeness_detector.update_history(iv_data, timestamp)
    
    def train_gating_network(self, training_data: List[Dict]) -> Dict[str, Any]:
        """
        Train the adaptive gating network
        
        Args:
            training_data: Training samples with features and target weights
            
        Returns:
            Training results
        """
        if self.gating_network is None:
            raise ValueError("Gating network not initialized")
        
        return self.gating_network.train(training_data)
    
    def get_gating_statistics(self) -> Dict[str, Any]:
        """Get statistics on gating decisions"""
        
        if not self.gating_history:
            return {'message': 'No gating history available'}
        
        decisions = [entry['decision'] for entry in self.gating_history]
        
        stats = {
            'total_decisions': len(decisions),
            'avg_options_weight': np.mean([d.options_weight for d in decisions]),
            'avg_extremeness_factor': np.mean([d.extremeness_factor for d in decisions]),
            'avg_confidence': np.mean([d.confidence for d in decisions]),
            'regime_distribution': {},
            'reasoning_distribution': {}
        }
        
        # Regime distribution
        regimes = [d.metadata['regime'] for d in decisions]
        unique_regimes = set(regimes)
        for regime in unique_regimes:
            stats['regime_distribution'][regime] = regimes.count(regime) / len(regimes)
        
        # Reasoning distribution
        reasoning_types = [d.reasoning for d in decisions]
        unique_reasoning = set(reasoning_types)
        for reasoning in unique_reasoning:
            stats['reasoning_distribution'][reasoning] = reasoning_types.count(reasoning) / len(reasoning_types)
        
        return stats