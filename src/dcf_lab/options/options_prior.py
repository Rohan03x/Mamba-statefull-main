"""
Options-based Neutral Prior Builder

This module creates neutral priors from options market data, serving as baseline
expectations that can be blended with other features in the learning framework.

Key components:
- Neutral prior construction from implied moves
- Distribution modeling from options skew
- Prior calibration and validation
- Integration with blending framework
"""

import numpy as np
from typing import Dict, List, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
from scipy.stats import norm, skewnorm, t as t_dist
import logging

from .implied_moves import ImpliedMove

logger = logging.getLogger(__name__)

@dataclass
class ImpliedDistribution:
    """Implied probability distribution from options"""
    distribution_type: str  # 'normal', 'skewed_normal', 't_distribution'
    parameters: Dict[str, float]
    confidence_levels: Dict[float, Tuple[float, float]]  # confidence -> (lower, upper)
    expected_move: float
    expected_move_percent: float
    metadata: Dict

@dataclass
class PriorCalibration:
    """Prior calibration results"""
    calibration_score: float
    coverage_errors: Dict[str, float]
    bias_metrics: Dict[str, float]
    calibration_plot_data: Dict
    validation_period: Tuple[datetime, datetime]
    sample_size: int

class NeutralPriorBuilder:
    """Build neutral priors from options-implied moves"""
    
    def __init__(self, distribution_type: str = 'skewed_normal'):
        """
        Args:
            distribution_type: Type of distribution to fit ('normal', 'skewed_normal', 't_distribution')
        """
        self.distribution_type = distribution_type
        self.supported_distributions = ['normal', 'skewed_normal', 't_distribution']
        
        if distribution_type not in self.supported_distributions:
            raise ValueError(f"Distribution type must be one of {self.supported_distributions}")
    
    def build_prior(self, implied_move: ImpliedMove, 
                   spot_price: float,
                   target_horizons: List[int] = [1, 7, 30]) -> Dict[int, ImpliedDistribution]:
        """
        Build neutral priors for different time horizons
        
        Args:
            implied_move: Implied move from options
            spot_price: Current underlying price
            target_horizons: List of target horizons in days
            
        Returns:
            Dict mapping horizon -> ImpliedDistribution
        """
        priors = {}
        
        for horizon in target_horizons:
            # Scale implied move to target horizon
            scaled_move = self._scale_move_to_horizon(implied_move, horizon)
            
            # Build distribution
            distribution = self._build_distribution(scaled_move, spot_price, horizon)
            priors[horizon] = distribution
        
        return priors
    
    def _scale_move_to_horizon(self, implied_move: ImpliedMove, target_days: int) -> ImpliedMove:
        """Scale implied move to target horizon using sqrt(time) scaling"""
        
        # Get original horizon from metadata
        original_days = implied_move.metadata.get('days_to_expiry', 30)
        if original_days <= 0:
            original_days = 30  # Default assumption
        
        # Time scaling factor
        time_scale = np.sqrt(target_days / original_days)
        
        # Scale all move metrics
        scaled_move = ImpliedMove(
            one_sigma_move=implied_move.one_sigma_move * time_scale,
            one_sigma_percent=implied_move.one_sigma_percent * time_scale,
            two_sigma_move=implied_move.two_sigma_move * time_scale,
            two_sigma_percent=implied_move.two_sigma_percent * time_scale,
            straddle_move=implied_move.straddle_move * time_scale,
            straddle_percent=implied_move.straddle_percent * time_scale,
            confidence_interval=(
                implied_move.confidence_interval[0] * time_scale,
                implied_move.confidence_interval[1] * time_scale
            ),
            calculation_method=implied_move.calculation_method,
            metadata={
                **implied_move.metadata,
                'scaled_to_days': target_days,
                'time_scale_factor': time_scale,
                'original_days': original_days
            }
        )
        
        return scaled_move
    
    def _build_distribution(self, implied_move: ImpliedMove, 
                          spot_price: float, horizon: int) -> ImpliedDistribution:
        """Build probability distribution from implied move"""
        
        # Convert to return space (log returns)
        expected_return = 0.0  # Neutral prior assumption
        volatility = implied_move.one_sigma_percent / 100  # Convert to decimal
        
        if self.distribution_type == 'normal':
            return self._build_normal_distribution(expected_return, volatility, spot_price, horizon, implied_move)
        elif self.distribution_type == 'skewed_normal':
            return self._build_skewed_normal_distribution(expected_return, volatility, spot_price, horizon, implied_move)
        elif self.distribution_type == 't_distribution':
            return self._build_t_distribution(expected_return, volatility, spot_price, horizon, implied_move)
    
    def _build_normal_distribution(self, mu: float, sigma: float, 
                                  spot_price: float, horizon: int,
                                  implied_move: ImpliedMove) -> ImpliedDistribution:
        """Build normal distribution prior"""
        
        # Calculate confidence intervals
        confidence_levels = {}
        for conf in [0.68, 0.80, 0.95, 0.99]:
            z_score = norm.ppf(0.5 + conf/2)
            lower_return = mu - z_score * sigma
            upper_return = mu + z_score * sigma
            
            # Convert back to price levels
            lower_price = spot_price * np.exp(lower_return)
            upper_price = spot_price * np.exp(upper_return)
            
            confidence_levels[conf] = (lower_price, upper_price)
        
        return ImpliedDistribution(
            distribution_type='normal',
            parameters={'mu': mu, 'sigma': sigma},
            confidence_levels=confidence_levels,
            expected_move=implied_move.one_sigma_move,
            expected_move_percent=implied_move.one_sigma_percent,
            metadata={
                'spot_price': spot_price,
                'horizon_days': horizon,
                'volatility_annual': sigma * np.sqrt(252),  # Annualized vol
                'implied_move_data': implied_move.metadata
            }
        )
    
    def _build_skewed_normal_distribution(self, mu: float, sigma: float,
                                        spot_price: float, horizon: int,
                                        implied_move: ImpliedMove) -> ImpliedDistribution:
        """Build skewed normal distribution prior"""
        
        # Default skewness (slightly negative for equity markets)
        default_skew = -0.2
        
        # Calculate confidence intervals using skewed normal
        confidence_levels = {}
        for conf in [0.68, 0.80, 0.95, 0.99]:
            lower_quantile = (1 - conf) / 2
            upper_quantile = 1 - lower_quantile
            
            lower_return = skewnorm.ppf(lower_quantile, default_skew, loc=mu, scale=sigma)
            upper_return = skewnorm.ppf(upper_quantile, default_skew, loc=mu, scale=sigma)
            
            # Convert to price levels
            lower_price = spot_price * np.exp(lower_return)
            upper_price = spot_price * np.exp(upper_return)
            
            confidence_levels[conf] = (lower_price, upper_price)
        
        return ImpliedDistribution(
            distribution_type='skewed_normal',
            parameters={'mu': mu, 'sigma': sigma, 'skew': default_skew},
            confidence_levels=confidence_levels,
            expected_move=implied_move.one_sigma_move,
            expected_move_percent=implied_move.one_sigma_percent,
            metadata={
                'spot_price': spot_price,
                'horizon_days': horizon,
                'volatility_annual': sigma * np.sqrt(252),
                'skewness': default_skew,
                'implied_move_data': implied_move.metadata
            }
        )
    
    def _build_t_distribution(self, mu: float, sigma: float,
                            spot_price: float, horizon: int,
                            implied_move: ImpliedMove) -> ImpliedDistribution:
        """Build t-distribution prior (fat tails)"""
        
        # Default degrees of freedom (lower = fatter tails)
        default_df = 5.0
        
        # Calculate confidence intervals using t-distribution
        confidence_levels = {}
        for conf in [0.68, 0.80, 0.95, 0.99]:
            t_value = t_dist.ppf(0.5 + conf/2, default_df)
            
            lower_return = mu - t_value * sigma
            upper_return = mu + t_value * sigma
            
            # Convert to price levels
            lower_price = spot_price * np.exp(lower_return)
            upper_price = spot_price * np.exp(upper_return)
            
            confidence_levels[conf] = (lower_price, upper_price)
        
        return ImpliedDistribution(
            distribution_type='t_distribution',
            parameters={'mu': mu, 'sigma': sigma, 'df': default_df},
            confidence_levels=confidence_levels,
            expected_move=implied_move.one_sigma_move,
            expected_move_percent=implied_move.one_sigma_percent,
            metadata={
                'spot_price': spot_price,
                'horizon_days': horizon,
                'volatility_annual': sigma * np.sqrt(252),
                'degrees_of_freedom': default_df,
                'implied_move_data': implied_move.metadata
            }
        )

class OptionsPrior:
    """Main interface for options-based priors"""
    
    def __init__(self, distribution_type: str = 'skewed_normal'):
        self.prior_builder = NeutralPriorBuilder(distribution_type)
        self.calibrator = PriorCalibrator()
    
    def create_prior(self, implied_move: ImpliedMove,
                    spot_price: float,
                    horizon: int = 1) -> ImpliedDistribution:
        """
        Create options-based prior for given horizon
        
        Args:
            implied_move: Computed implied move from options
            spot_price: Current underlying price
            horizon: Target horizon in days
            
        Returns:
            ImpliedDistribution representing the prior
        """
        priors = self.prior_builder.build_prior(implied_move, spot_price, [horizon])
        return priors[horizon]
    
    def create_multi_horizon_priors(self, implied_move: ImpliedMove,
                                   spot_price: float,
                                   horizons: List[int] = [1, 7, 30]) -> Dict[int, ImpliedDistribution]:
        """Create priors for multiple horizons"""
        return self.prior_builder.build_prior(implied_move, spot_price, horizons)
    
    def get_prior_prediction(self, prior: ImpliedDistribution,
                           confidence_level: float = 0.68) -> Tuple[float, float]:
        """
        Get prediction interval from prior
        
        Args:
            prior: ImpliedDistribution
            confidence_level: Desired confidence level
            
        Returns:
            Tuple of (lower_bound, upper_bound)
        """
        if confidence_level not in prior.confidence_levels:
            # Find closest available confidence level
            available_levels = list(prior.confidence_levels.keys())
            closest_level = min(available_levels, key=lambda x: abs(x - confidence_level))
            logger.warning(f"Confidence level {confidence_level} not available, using {closest_level}")
            confidence_level = closest_level
        
        return prior.confidence_levels[confidence_level]
    
    def get_prior_point_estimate(self, prior: ImpliedDistribution,
                               current_price: float) -> float:
        """
        Get point estimate from prior (expected return)
        
        Args:
            prior: ImpliedDistribution  
            current_price: Current price for scaling
            
        Returns:
            Expected return
        """
        # For neutral prior, expected return is typically 0
        # This could be enhanced with risk-free rate adjustments
        return 0.0

class PriorCalibrator:
    """Calibrate and validate options priors against realized outcomes"""
    
    def __init__(self):
        self.calibration_history = []
    
    def calibrate_prior(self, priors: List[ImpliedDistribution],
                       realized_outcomes: List[float],
                       confidence_levels: List[float] = [0.68, 0.80, 0.95]) -> PriorCalibration:
        """
        Calibrate priors against realized outcomes
        
        Args:
            priors: List of ImpliedDistribution objects
            realized_outcomes: List of realized returns/moves
            confidence_levels: Confidence levels to evaluate
            
        Returns:
            PriorCalibration results
        """
        
        if len(priors) != len(realized_outcomes):
            raise ValueError("Priors and outcomes must have same length")
        
        # Calculate coverage errors
        coverage_errors = {}
        for conf_level in confidence_levels:
            predicted_coverage = self._calculate_coverage(priors, realized_outcomes, conf_level)
            coverage_error = abs(predicted_coverage - conf_level)
            coverage_errors[f"coverage_{conf_level}"] = coverage_error
        
        # Calculate bias metrics
        bias_metrics = self._calculate_bias_metrics(priors, realized_outcomes)
        
        # Overall calibration score (lower is better)
        calibration_score = np.mean(list(coverage_errors.values())) + bias_metrics.get('mean_bias', 0)
        
        # Create calibration plot data
        calibration_plot_data = self._create_calibration_plot_data(priors, realized_outcomes, confidence_levels)
        
        calibration_result = PriorCalibration(
            calibration_score=calibration_score,
            coverage_errors=coverage_errors,
            bias_metrics=bias_metrics,
            calibration_plot_data=calibration_plot_data,
            validation_period=(datetime.now() - timedelta(days=len(priors)), datetime.now()),
            sample_size=len(priors)
        )
        
        self.calibration_history.append(calibration_result)
        return calibration_result
    
    def _calculate_coverage(self, priors: List[ImpliedDistribution],
                          outcomes: List[float], confidence_level: float) -> float:
        """Calculate actual coverage rate for given confidence level"""
        
        covered_count = 0
        total_count = 0
        
        for prior, outcome in zip(priors, outcomes):
            if confidence_level in prior.confidence_levels:
                lower, upper = prior.confidence_levels[confidence_level]
                if lower <= outcome <= upper:
                    covered_count += 1
                total_count += 1
        
        return covered_count / total_count if total_count > 0 else 0.0
    
    def _calculate_bias_metrics(self, priors: List[ImpliedDistribution],
                              outcomes: List[float]) -> Dict[str, float]:
        """Calculate bias metrics for prior predictions"""
        
        # For neutral priors, we mainly look at symmetry and accuracy of intervals
        biases = []
        interval_widths = []
        
        for prior, outcome in zip(priors, outcomes):
            # Get 68% interval as baseline
            if 0.68 in prior.confidence_levels:
                lower, upper = prior.confidence_levels[0.68]
                interval_center = (lower + upper) / 2
                spot_price = prior.metadata.get('spot_price', (lower + upper) / 2)
                
                # Bias relative to center vs spot price
                bias = (interval_center - spot_price) / spot_price
                biases.append(bias)
                
                # Interval width analysis
                width = (upper - lower) / spot_price
                interval_widths.append(width)
        
        return {
            'mean_bias': np.mean(biases) if biases else 0.0,
            'bias_std': np.std(biases) if biases else 0.0,
            'mean_interval_width': np.mean(interval_widths) if interval_widths else 0.0,
            'interval_width_std': np.std(interval_widths) if interval_widths else 0.0
        }
    
    def _create_calibration_plot_data(self, priors: List[ImpliedDistribution],
                                    outcomes: List[float],
                                    confidence_levels: List[float]) -> Dict:
        """Create data for calibration plots"""
        
        plot_data = {
            'predicted_coverage': confidence_levels,
            'actual_coverage': [],
            'sample_sizes': []
        }
        
        for conf_level in confidence_levels:
            actual_coverage = self._calculate_coverage(priors, outcomes, conf_level)
            plot_data['actual_coverage'].append(actual_coverage)
            
            # Count available samples for this confidence level
            sample_count = sum(1 for prior in priors if conf_level in prior.confidence_levels)
            plot_data['sample_sizes'].append(sample_count)
        
        return plot_data