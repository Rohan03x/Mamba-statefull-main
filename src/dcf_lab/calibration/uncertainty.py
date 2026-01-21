"""
Comprehensive Uncertainty Quantification Framework

This module provides sophisticated uncertainty quantification tools for
financial ML models, combining both aleatoric (data) and epistemic (model)
uncertainty. Includes calibration diagnostics, coverage analysis, and
reliability assessment.

Key Features:
- Multi-source uncertainty quantification
- Calibration quality assessment
- Coverage validation and monitoring
- Reliability diagrams and calibration curves
- Time-series aware uncertainty analysis
- Integration with ensemble and conformal methods

Uncertainty quantification is crucial for financial ML as it enables:
- Risk-aware trading decisions
- Portfolio optimization under uncertainty
- Model confidence assessment
- Outlier and regime change detection
- Regulatory compliance and audit trails

References:
- Gal, Y. & Ghahramani, Z. (2016). Dropout as a Bayesian approximation
- Lakshminarayanan, B. et al. (2017). Simple and scalable predictive uncertainty
- Kuleshov, V. et al. (2018). Accurate uncertainties for deep learning
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Any
import logging

import matplotlib.pyplot as plt
from sklearn.metrics import brier_score_loss
from scipy import stats

logger = logging.getLogger(__name__)


@dataclass
class UncertaintyMetrics:
    """Comprehensive uncertainty quantification metrics"""
    
    # Calibration metrics
    calibration_error: float  # Expected Calibration Error (ECE)
    max_calibration_error: float  # Maximum Calibration Error (MCE)
    brier_score: float  # Overall prediction quality
    reliability_score: float  # Calibration component of Brier score
    resolution_score: float  # Discrimination ability
    uncertainty_score: float  # Base rate uncertainty
    
    # Coverage metrics (for intervals)
    coverage_probability: Optional[float] = None
    average_width: Optional[float] = None
    conditional_coverage: Optional[Dict[str, float]] = None
    
    # Sharpness and informativeness
    entropy: float = 0.0  # Prediction entropy
    mutual_information: float = 0.0  # MI between predictions and outcomes
    prediction_variance: float = 0.0  # Variance of predictions
    
    # Temporal metrics
    coverage_by_time: Optional[Dict[str, float]] = None
    calibration_drift: Optional[float] = None
    
    # Confidence metrics
    confidence_correlation: float = 0.0  # Correlation between confidence and accuracy
    overconfidence_ratio: float = 0.0  # Fraction of overconfident predictions
    underconfidence_ratio: float = 0.0  # Fraction of underconfident predictions
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary"""
        result = {
            'calibration_error': self.calibration_error,
            'max_calibration_error': self.max_calibration_error,
            'brier_score': self.brier_score,
            'reliability_score': self.reliability_score,
            'resolution_score': self.resolution_score,
            'uncertainty_score': self.uncertainty_score,
            'entropy': self.entropy,
            'mutual_information': self.mutual_information,
            'prediction_variance': self.prediction_variance,
            'confidence_correlation': self.confidence_correlation,
            'overconfidence_ratio': self.overconfidence_ratio,
            'underconfidence_ratio': self.underconfidence_ratio
        }
        
        if self.coverage_probability is not None:
            result['coverage_probability'] = self.coverage_probability
        if self.average_width is not None:
            result['average_width'] = self.average_width
        if self.conditional_coverage is not None:
            result['conditional_coverage'] = self.conditional_coverage
        if self.coverage_by_time is not None:
            result['coverage_by_time'] = self.coverage_by_time
        if self.calibration_drift is not None:
            result['calibration_drift'] = self.calibration_drift
            
        return result


@dataclass
class ReliabilityDiagram:
    """Data for plotting reliability diagrams"""
    
    bin_centers: np.ndarray
    bin_accuracies: np.ndarray
    bin_confidences: np.ndarray
    bin_counts: np.ndarray
    bin_boundaries: np.ndarray
    
    # Calibration metrics
    ece: float
    mce: float
    perfectly_calibrated_line: np.ndarray
    
    def plot(self, save_path: Optional[str] = None, 
            show_histogram: bool = True) -> plt.Figure:
        """Plot reliability diagram"""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8)) if show_histogram else plt.subplots(1, 1, figsize=(10, 6))
        
        # Main reliability plot
        main_ax = ax1 if show_histogram else fig.gca()
        
        # Plot reliability curve
        main_ax.plot(self.bin_confidences, self.bin_accuracies, 'o-', 
                    label=f'Model (ECE={self.ece:.3f})', linewidth=2, markersize=8)
        
        # Plot perfect calibration line
        main_ax.plot([0, 1], [0, 1], '--', color='gray', 
                    label='Perfect Calibration', alpha=0.8)
        
        # Formatting
        main_ax.set_xlabel('Mean Predicted Probability')
        main_ax.set_ylabel('Fraction of Positives')
        main_ax.set_title('Reliability Diagram (Calibration Curve)')
        main_ax.legend()
        main_ax.grid(True, alpha=0.3)
        main_ax.set_xlim([0, 1])
        main_ax.set_ylim([0, 1])
        
        # Add ECE and MCE annotations
        main_ax.text(0.02, 0.98, f'ECE: {self.ece:.4f}\nMCE: {self.mce:.4f}', 
                    transform=main_ax.transAxes, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        if show_histogram:
            # Histogram of prediction confidences
            ax2.bar(self.bin_centers, self.bin_counts, width=0.08, alpha=0.7, 
                   color='skyblue', edgecolor='black')
            ax2.set_xlabel('Mean Predicted Probability')
            ax2.set_ylabel('Count')
            ax2.set_title('Distribution of Predictions')
            ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Reliability diagram saved to {save_path}")
        
        return fig


class UncertaintyQuantifier:
    """Comprehensive uncertainty quantification system"""
    
    def __init__(self, n_bins: int = 10):
        self.n_bins = n_bins
        
    def analyze_calibration(self, y_true: np.ndarray, 
                          y_prob: np.ndarray,
                          sample_weight: Optional[np.ndarray] = None) -> UncertaintyMetrics:
        """Comprehensive calibration analysis"""
        
        # Basic calibration metrics
        ece = self._expected_calibration_error(y_true, y_prob)
        mce = self._maximum_calibration_error(y_true, y_prob)
        brier = brier_score_loss(y_true, y_prob, sample_weight=sample_weight)
        
        # Decompose Brier score
        reliability, resolution, uncertainty = self._brier_decomposition(y_true, y_prob)
        
        # Information-theoretic metrics
        entropy = self._prediction_entropy(y_prob)
        mutual_info = self._mutual_information(y_true, y_prob)
        pred_variance = np.var(y_prob)
        
        # Confidence analysis
        conf_corr = self._confidence_accuracy_correlation(y_true, y_prob)
        over_conf, under_conf = self._confidence_ratios(y_true, y_prob)
        
        return UncertaintyMetrics(
            calibration_error=ece,
            max_calibration_error=mce,
            brier_score=brier,
            reliability_score=reliability,
            resolution_score=resolution,
            uncertainty_score=uncertainty,
            entropy=entropy,
            mutual_information=mutual_info,
            prediction_variance=pred_variance,
            confidence_correlation=conf_corr,
            overconfidence_ratio=over_conf,
            underconfidence_ratio=under_conf
        )
    
    def create_reliability_diagram(self, y_true: np.ndarray, 
                                 y_prob: np.ndarray) -> ReliabilityDiagram:
        """Create reliability diagram data"""
        
        # Compute calibration curve
        bin_boundaries = np.linspace(0, 1, self.n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        bin_centers = []
        bin_accuracies = []
        bin_confidences = []
        bin_counts = []
        
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (y_prob > bin_lower) & (y_prob <= bin_upper)
            prop_in_bin = in_bin.mean()
            
            if prop_in_bin > 0:
                accuracy_in_bin = y_true[in_bin].mean()
                avg_confidence_in_bin = y_prob[in_bin].mean()
                count_in_bin = in_bin.sum()
            else:
                accuracy_in_bin = 0.0
                avg_confidence_in_bin = (bin_lower + bin_upper) / 2
                count_in_bin = 0
            
            bin_centers.append((bin_lower + bin_upper) / 2)
            bin_accuracies.append(accuracy_in_bin)
            bin_confidences.append(avg_confidence_in_bin)
            bin_counts.append(count_in_bin)
        
        # Calculate calibration metrics
        ece = self._expected_calibration_error(y_true, y_prob)
        mce = self._maximum_calibration_error(y_true, y_prob)
        
        return ReliabilityDiagram(
            bin_centers=np.array(bin_centers),
            bin_accuracies=np.array(bin_accuracies),
            bin_confidences=np.array(bin_confidences),
            bin_counts=np.array(bin_counts),
            bin_boundaries=bin_boundaries,
            ece=ece,
            mce=mce,
            perfectly_calibrated_line=np.linspace(0, 1, 100)
        )
    
    def analyze_coverage(self, y_true: np.ndarray,
                        lower_bounds: np.ndarray,
                        upper_bounds: np.ndarray,
                        target_coverage: float = 0.9) -> Dict[str, float]:
        """Analyze prediction interval coverage"""
        
        # Basic coverage metrics
        in_interval = (y_true >= lower_bounds) & (y_true <= upper_bounds)
        empirical_coverage = np.mean(in_interval)
        coverage_gap = abs(empirical_coverage - target_coverage)
        
        # Interval width analysis
        widths = upper_bounds - lower_bounds
        average_width = np.mean(widths)
        width_std = np.std(widths)
        
        # Violation analysis
        lower_violations = np.mean(y_true < lower_bounds)
        upper_violations = np.mean(y_true > upper_bounds)
        
        # Efficiency score (coverage / width)
        efficiency = empirical_coverage / average_width if average_width > 0 else 0
        
        return {
            'empirical_coverage': empirical_coverage,
            'target_coverage': target_coverage,
            'coverage_gap': coverage_gap,
            'average_width': average_width,
            'width_std': width_std,
            'lower_violations': lower_violations,
            'upper_violations': upper_violations,
            'efficiency_score': efficiency
        }
    
    def temporal_coverage_analysis(self, y_true: np.ndarray,
                                 lower_bounds: np.ndarray,
                                 upper_bounds: np.ndarray,
                                 timestamps: np.ndarray,
                                 window_size: int = 50) -> Dict[str, np.ndarray]:
        """Analyze coverage over time using rolling windows"""
        
        n_samples = len(y_true)
        n_windows = n_samples - window_size + 1
        
        coverage_over_time = np.zeros(n_windows)
        width_over_time = np.zeros(n_windows)
        timestamps_windows = np.zeros(n_windows, dtype='datetime64[ns]')
        
        for i in range(n_windows):
            start_idx = i
            end_idx = i + window_size
            
            window_true = y_true[start_idx:end_idx]
            window_lower = lower_bounds[start_idx:end_idx]
            window_upper = upper_bounds[start_idx:end_idx]
            
            # Coverage in this window
            in_interval = (window_true >= window_lower) & (window_true <= window_upper)
            coverage_over_time[i] = np.mean(in_interval)
            
            # Average width in this window
            width_over_time[i] = np.mean(window_upper - window_lower)
            
            # Window timestamp (center)
            timestamps_windows[i] = timestamps[start_idx + window_size // 2]
        
        return {
            'timestamps': timestamps_windows,
            'coverage': coverage_over_time,
            'width': width_over_time
        }
    
    def _expected_calibration_error(self, y_true: np.ndarray, 
                                  y_prob: np.ndarray) -> float:
        """Calculate Expected Calibration Error (ECE)"""
        bin_boundaries = np.linspace(0, 1, self.n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        ece = 0.0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (y_prob > bin_lower) & (y_prob <= bin_upper)
            prop_in_bin = in_bin.mean()
            
            if prop_in_bin > 0:
                accuracy_in_bin = y_true[in_bin].mean()
                avg_confidence_in_bin = y_prob[in_bin].mean()
                ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
        
        return ece
    
    def _maximum_calibration_error(self, y_true: np.ndarray, 
                                 y_prob: np.ndarray) -> float:
        """Calculate Maximum Calibration Error (MCE)"""
        bin_boundaries = np.linspace(0, 1, self.n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        mce = 0.0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (y_prob > bin_lower) & (y_prob <= bin_upper)
            
            if in_bin.sum() > 0:
                accuracy_in_bin = y_true[in_bin].mean()
                avg_confidence_in_bin = y_prob[in_bin].mean()
                mce = max(mce, np.abs(avg_confidence_in_bin - accuracy_in_bin))
        
        return mce
    
    def _brier_decomposition(self, y_true: np.ndarray, 
                           y_prob: np.ndarray) -> Tuple[float, float, float]:
        """Decompose Brier score into reliability, resolution, and uncertainty"""
        
        # Overall base rate
        base_rate = np.mean(y_true)
        
        # Bin predictions
        bin_boundaries = np.linspace(0, 1, self.n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        reliability = 0.0
        resolution = 0.0
        
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (y_prob > bin_lower) & (y_prob <= bin_upper)
            n_in_bin = in_bin.sum()
            
            if n_in_bin > 0:
                prop_in_bin = n_in_bin / len(y_true)
                accuracy_in_bin = y_true[in_bin].mean()
                avg_confidence_in_bin = y_prob[in_bin].mean()
                
                # Reliability component
                reliability += prop_in_bin * (avg_confidence_in_bin - accuracy_in_bin) ** 2
                
                # Resolution component  
                resolution += prop_in_bin * (accuracy_in_bin - base_rate) ** 2
        
        # Uncertainty (base rate uncertainty)
        uncertainty = base_rate * (1 - base_rate)
        
        return reliability, resolution, uncertainty
    
    def _prediction_entropy(self, y_prob: np.ndarray) -> float:
        """Calculate prediction entropy"""
        # Clip probabilities to avoid log(0)
        y_prob_clipped = np.clip(y_prob, 1e-15, 1 - 1e-15)
        
        # Binary entropy
        entropy = -(y_prob_clipped * np.log2(y_prob_clipped) + 
                   (1 - y_prob_clipped) * np.log2(1 - y_prob_clipped))
        
        return np.mean(entropy)
    
    def _mutual_information(self, y_true: np.ndarray, 
                          y_prob: np.ndarray) -> float:
        """Calculate mutual information between predictions and outcomes"""
        # Discretize predictions into bins
        bin_edges = np.linspace(0, 1, self.n_bins + 1)
        pred_bins = np.digitize(y_prob, bin_edges) - 1
        pred_bins = np.clip(pred_bins, 0, self.n_bins - 1)
        
        # Calculate mutual information
        mi = 0.0
        n_total = len(y_true)
        
        for pred_bin in range(self.n_bins):
            for outcome in [0, 1]:
                # Joint probability
                joint_mask = (pred_bins == pred_bin) & (y_true == outcome)
                p_joint = np.sum(joint_mask) / n_total
                
                if p_joint > 0:
                    # Marginal probabilities
                    p_pred = np.sum(pred_bins == pred_bin) / n_total
                    p_outcome = np.sum(y_true == outcome) / n_total
                    
                    if p_pred > 0 and p_outcome > 0:
                        mi += p_joint * np.log2(p_joint / (p_pred * p_outcome))
        
        return mi
    
    def _confidence_accuracy_correlation(self, y_true: np.ndarray, 
                                       y_prob: np.ndarray) -> float:
        """Calculate correlation between confidence and accuracy"""
        # Accuracy indicator (correct predictions)
        accuracy = (y_prob >= 0.5) == y_true.astype(bool)
        
        # Confidence (distance from 0.5)
        confidence = np.abs(y_prob - 0.5)
        
        # Calculate correlation
        if len(np.unique(confidence)) > 1 and len(np.unique(accuracy)) > 1:
            correlation = np.corrcoef(confidence, accuracy.astype(float))[0, 1]
            return correlation if not np.isnan(correlation) else 0.0
        else:
            return 0.0
    
    def _confidence_ratios(self, y_true: np.ndarray, 
                         y_prob: np.ndarray) -> Tuple[float, float]:
        """Calculate overconfidence and underconfidence ratios"""
        
        # Predicted class and confidence
        predicted_class = (y_prob >= 0.5).astype(int)
        confidence = np.maximum(y_prob, 1 - y_prob)  # Max probability
        
        # Correct predictions
        correct = (predicted_class == y_true)
        
        # Overconfident: high confidence but wrong
        overconfident = (confidence > 0.8) & (~correct)
        overconfidence_ratio = np.mean(overconfident)
        
        # Underconfident: low confidence but correct
        underconfident = (confidence < 0.6) & correct
        underconfidence_ratio = np.mean(underconfident)
        
        return overconfidence_ratio, underconfidence_ratio


class CalibrationDiagnostics:
    """Advanced calibration diagnostics and monitoring"""
    
    def __init__(self):
        self.uncertainty_quantifier = UncertaintyQuantifier()
    
    def comprehensive_analysis(self, y_true: np.ndarray,
                             y_prob: np.ndarray,
                             timestamps: Optional[np.ndarray] = None,
                             feature_matrix: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Run comprehensive calibration analysis"""
        
        results = {}
        
        # Basic uncertainty metrics
        metrics = self.uncertainty_quantifier.analyze_calibration(y_true, y_prob)
        results['uncertainty_metrics'] = metrics.to_dict()
        
        # Reliability diagram
        reliability_diagram = self.uncertainty_quantifier.create_reliability_diagram(y_true, y_prob)
        results['reliability_diagram'] = {
            'bin_centers': reliability_diagram.bin_centers.tolist(),
            'bin_accuracies': reliability_diagram.bin_accuracies.tolist(),
            'bin_confidences': reliability_diagram.bin_confidences.tolist(),
            'bin_counts': reliability_diagram.bin_counts.tolist(),
            'ece': reliability_diagram.ece,
            'mce': reliability_diagram.mce
        }
        
        # Temporal analysis if timestamps provided
        if timestamps is not None:
            temporal_analysis = self._temporal_calibration_analysis(
                y_true, y_prob, timestamps
            )
            results['temporal_analysis'] = temporal_analysis
        
        # Feature-conditional analysis if features provided
        if feature_matrix is not None:
            conditional_analysis = self._conditional_calibration_analysis(
                y_true, y_prob, feature_matrix
            )
            results['conditional_analysis'] = conditional_analysis
        
        return results
    
    def _temporal_calibration_analysis(self, y_true: np.ndarray,
                                     y_prob: np.ndarray,
                                     timestamps: np.ndarray,
                                     window_size: int = 50) -> Dict[str, Any]:
        """Analyze calibration over time"""
        
        n_samples = len(y_true)
        n_windows = max(1, n_samples - window_size + 1)
        
        ece_over_time = []
        brier_over_time = []
        window_timestamps = []
        
        for i in range(n_windows):
            start_idx = i
            end_idx = min(i + window_size, n_samples)
            
            window_true = y_true[start_idx:end_idx]
            window_prob = y_prob[start_idx:end_idx]
            
            # Calculate metrics for this window
            window_ece = self.uncertainty_quantifier._expected_calibration_error(
                window_true, window_prob
            )
            window_brier = brier_score_loss(window_true, window_prob)
            
            ece_over_time.append(window_ece)
            brier_over_time.append(window_brier)
            window_timestamps.append(timestamps[start_idx + (end_idx - start_idx) // 2])
        
        # Calculate calibration drift (trend in ECE)
        if len(ece_over_time) > 1:
            time_indices = np.arange(len(ece_over_time))
            slope, _, _, _, _ = stats.linregress(time_indices, ece_over_time)
            calibration_drift = slope
        else:
            calibration_drift = 0.0
        
        return {
            'timestamps': [ts.isoformat() if hasattr(ts, 'isoformat') else str(ts) 
                         for ts in window_timestamps],
            'ece_over_time': ece_over_time,
            'brier_over_time': brier_over_time,
            'calibration_drift': calibration_drift
        }
    
    def _conditional_calibration_analysis(self, y_true: np.ndarray,
                                        y_prob: np.ndarray,
                                        feature_matrix: np.ndarray,
                                        n_bins: int = 3) -> Dict[str, Any]:
        """Analyze calibration conditional on features"""
        
        conditional_results = {}
        
        for feature_idx in range(feature_matrix.shape[1]):
            feature = feature_matrix[:, feature_idx]
            
            # Create feature bins
            quantiles = np.quantile(feature, np.linspace(0, 1, n_bins + 1))
            
            bin_results = []
            for i in range(len(quantiles) - 1):
                in_bin = (feature >= quantiles[i]) & (feature < quantiles[i + 1])
                
                if np.sum(in_bin) > 10:  # Need sufficient samples
                    bin_true = y_true[in_bin]
                    bin_prob = y_prob[in_bin]
                    
                    bin_ece = self.uncertainty_quantifier._expected_calibration_error(
                        bin_true, bin_prob
                    )
                    bin_brier = brier_score_loss(bin_true, bin_prob)
                    
                    bin_results.append({
                        'feature_range': [quantiles[i], quantiles[i + 1]],
                        'n_samples': np.sum(in_bin),
                        'ece': bin_ece,
                        'brier_score': bin_brier
                    })
            
            conditional_results[f'feature_{feature_idx}'] = bin_results
        
        return conditional_results