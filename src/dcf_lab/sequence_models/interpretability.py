"""
Interpretability Analysis for Sequence Models

This module provides comprehensive interpretability analysis for TFT and 
Transformer models with focus on:

1. Attention weight analysis and visualization
2. Variable importance scoring across feature types
3. Temporal pattern detection in attention
4. Sanity checks for model behavior validation
5. Feature attribution and contribution analysis

Key components:
- Attention pattern analysis
- Variable importance computation
- Temporal attention visualization
- Model behavior validation
- Feature contribution tracking
"""

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)

@dataclass
class InterpretabilityConfig:
    """Configuration for interpretability analysis"""
    
    # Attention analysis
    analyze_attention_weights: bool = True
    attention_aggregation_method: str = 'mean'  # 'mean', 'max', 'last_layer'
    attention_threshold: float = 0.1
    
    # Variable importance
    compute_variable_importance: bool = True
    importance_method: str = 'gradient'  # 'gradient', 'permutation', 'integrated_gradients'
    importance_baseline: str = 'zero'  # 'zero', 'mean', 'median'
    
    # Temporal analysis
    analyze_temporal_patterns: bool = True
    temporal_window_size: int = 10
    
    # Sanity checks
    perform_sanity_checks: bool = True
    sanity_check_features: List[str] = field(default_factory=list)
    
    # Visualization
    create_visualizations: bool = True
    save_plots: bool = False
    plot_directory: str = 'interpretability_plots'

class AttentionAnalyzer:
    """Analyze attention patterns in sequence models"""
    
    def __init__(self, config: InterpretabilityConfig = None):
        self.config = config or InterpretabilityConfig()
        self.attention_history = []
    
    def analyze_attention_weights(self, attention_weights: torch.Tensor,
                                input_timestamps: Optional[List] = None,
                                feature_names: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Analyze attention weight patterns
        
        Args:
            attention_weights: [batch, heads, seq_len, seq_len] or [batch, seq_len, seq_len]
            input_timestamps: Optional timestamps for temporal analysis
            feature_names: Optional feature names
            
        Returns:
            Attention analysis results
        """
        
        if attention_weights is None:
            return {'error': 'No attention weights provided'}
        
        # Convert to numpy for analysis
        if isinstance(attention_weights, torch.Tensor):
            attention_weights = attention_weights.detach().cpu().numpy()
        
        # Handle different attention weight dimensions
        if len(attention_weights.shape) == 4:
            # Multi-head attention: aggregate across heads
            if self.config.attention_aggregation_method == 'mean':
                attention_matrix = attention_weights.mean(axis=1)  # Average across heads
            elif self.config.attention_aggregation_method == 'max':
                attention_matrix = attention_weights.max(axis=1)
            else:
                attention_matrix = attention_weights[:, -1]  # Use last head
        else:
            attention_matrix = attention_weights
        
        # Average across batch dimension
        if len(attention_matrix.shape) == 3:
            attention_matrix = attention_matrix.mean(axis=0)
        
        analysis_results = {}
        
        # Basic attention statistics
        analysis_results['attention_stats'] = {
            'mean_attention': float(attention_matrix.mean()),
            'std_attention': float(attention_matrix.std()),
            'max_attention': float(attention_matrix.max()),
            'min_attention': float(attention_matrix.min()),
            'sparsity': float((attention_matrix < self.config.attention_threshold).mean())
        }
        
        # Temporal attention patterns
        if self.config.analyze_temporal_patterns:
            temporal_analysis = self._analyze_temporal_attention(attention_matrix, input_timestamps)
            analysis_results['temporal_patterns'] = temporal_analysis
        
        # Attention concentration
        attention_entropy = self._compute_attention_entropy(attention_matrix)
        analysis_results['attention_entropy'] = attention_entropy
        
        # Position-wise attention
        position_attention = self._analyze_position_attention(attention_matrix)
        analysis_results['position_attention'] = position_attention
        
        # Store for history
        self.attention_history.append(analysis_results)
        
        return analysis_results
    
    def _analyze_temporal_attention(self, attention_matrix: np.ndarray,
                                   timestamps: Optional[List] = None) -> Dict[str, Any]:
        """Analyze temporal patterns in attention"""
        
        seq_len = attention_matrix.shape[0]
        
        # Recency bias analysis
        recency_weights = []
        for i in range(seq_len):
            # Attention from position i to more recent positions
            recent_attention = attention_matrix[i, max(0, i-self.config.temporal_window_size):i+1]
            recency_weights.append(recent_attention.sum() if len(recent_attention) > 0 else 0)
        
        # Attention decay analysis
        attention_decay = []
        for i in range(seq_len):
            if i > 0:
                # Attention to positions relative to current
                relative_attention = attention_matrix[i, :i]
                if len(relative_attention) > 1:
                    # Compute decay pattern
                    distances = np.arange(1, len(relative_attention) + 1)
                    correlation = np.corrcoef(distances, relative_attention[::-1])[0, 1]
                    attention_decay.append(correlation)
        
        # Long-range dependencies
        long_range_threshold = max(5, seq_len // 4)
        long_range_attention = []
        
        for i in range(seq_len):
            if i >= long_range_threshold:
                distant_attention = attention_matrix[i, :i-long_range_threshold]
                long_range_attention.append(distant_attention.sum())
        
        return {
            'recency_bias': {
                'mean': float(np.mean(recency_weights)),
                'std': float(np.std(recency_weights)),
                'pattern': recency_weights
            },
            'attention_decay': {
                'mean_correlation': float(np.mean(attention_decay)) if attention_decay else 0.0,
                'decay_pattern': attention_decay
            },
            'long_range_dependencies': {
                'mean_long_range': float(np.mean(long_range_attention)) if long_range_attention else 0.0,
                'long_range_pattern': long_range_attention
            }
        }
    
    def _compute_attention_entropy(self, attention_matrix: np.ndarray) -> Dict[str, float]:
        """Compute attention entropy measures"""
        
        # Row-wise entropy (how dispersed each query's attention is)
        row_entropies = []
        for i in range(attention_matrix.shape[0]):
            attention_row = attention_matrix[i]
            # Add small epsilon to avoid log(0)
            attention_row = attention_row + 1e-8
            entropy = -np.sum(attention_row * np.log(attention_row))
            row_entropies.append(entropy)
        
        # Column-wise entropy (how dispersed attention to each key is)
        col_entropies = []
        for j in range(attention_matrix.shape[1]):
            attention_col = attention_matrix[:, j]
            attention_col = attention_col + 1e-8
            entropy = -np.sum(attention_col * np.log(attention_col))
            col_entropies.append(entropy)
        
        return {
            'mean_row_entropy': float(np.mean(row_entropies)),
            'mean_col_entropy': float(np.mean(col_entropies)),
            'total_entropy': float(np.mean(row_entropies) + np.mean(col_entropies)),
            'row_entropy_std': float(np.std(row_entropies)),
            'col_entropy_std': float(np.std(col_entropies))
        }
    
    def _analyze_position_attention(self, attention_matrix: np.ndarray) -> Dict[str, Any]:
        """Analyze position-based attention patterns"""
        
        attention_matrix.shape[0]
        
        # Diagonal attention (self-attention strength)
        diagonal_attention = np.diag(attention_matrix)
        
        # Off-diagonal patterns
        upper_triangular = np.triu(attention_matrix, k=1)
        lower_triangular = np.tril(attention_matrix, k=-1)
        
        # Position preference
        position_weights = attention_matrix.sum(axis=0)  # Total attention to each position
        
        return {
            'diagonal_attention': {
                'mean': float(diagonal_attention.mean()),
                'std': float(diagonal_attention.std()),
                'pattern': diagonal_attention.tolist()
            },
            'upper_triangular_sum': float(upper_triangular.sum()),
            'lower_triangular_sum': float(lower_triangular.sum()),
            'position_preferences': {
                'weights': position_weights.tolist(),
                'most_attended_position': int(position_weights.argmax()),
                'least_attended_position': int(position_weights.argmin())
            }
        }

class VariableImportanceScorer:
    """Compute variable importance for interpretability"""
    
    def __init__(self, model: torch.nn.Module, config: InterpretabilityConfig = None):
        self.model = model
        self.config = config or InterpretabilityConfig()
        self.importance_history = []
    
    def compute_importance(self, inputs: Dict[str, torch.Tensor],
                          targets: torch.Tensor,
                          feature_names: Optional[Dict[str, List[str]]] = None) -> Dict[str, Any]:
        """
        Compute variable importance using specified method
        
        Args:
            inputs: Model inputs by feature type
            targets: Target values
            feature_names: Feature names by input type
            
        Returns:
            Importance scores by feature type and individual features
        """
        
        self.model.eval()
        
        if self.config.importance_method == 'gradient':
            importance_scores = self._compute_gradient_importance(inputs, targets)
        elif self.config.importance_method == 'permutation':
            importance_scores = self._compute_permutation_importance(inputs, targets)
        elif self.config.importance_method == 'integrated_gradients':
            importance_scores = self._compute_integrated_gradients(inputs, targets)
        else:
            raise ValueError(f"Unknown importance method: {self.config.importance_method}")
        
        # Add feature names if provided
        if feature_names:
            importance_scores = self._add_feature_names(importance_scores, feature_names)
        
        self.importance_history.append(importance_scores)
        return importance_scores
    
    def _compute_gradient_importance(self, inputs: Dict[str, torch.Tensor],
                                   targets: torch.Tensor) -> Dict[str, Any]:
        """Compute importance using gradient-based attribution"""
        
        importance_scores = {}
        
        # Enable gradients for inputs
        for key, tensor in inputs.items():
            tensor.requires_grad_(True)
        
        # Forward pass
        outputs = self.model(inputs)
        
        # Compute loss (use median quantile prediction)
        loss = 0.0
        horizon_count = 0
        
        for horizon_key, horizon_outputs in outputs.items():
            for target_key, target_outputs in horizon_outputs.items():
                if 'q50' in target_outputs:  # Use median prediction
                    prediction = target_outputs['q50']
                    
                    # Get corresponding target
                    if len(targets.shape) == 3:  # [batch, horizons, targets]
                        horizon_idx = int(horizon_key.split('_')[1]) - 1
                        target_idx = int(target_key.split('_')[1])
                        target_values = targets[:, horizon_idx, target_idx]
                    else:
                        target_values = targets.squeeze()
                    
                    # MSE loss
                    horizon_loss = F.mse_loss(prediction, target_values)
                    loss += horizon_loss
                    horizon_count += 1
        
        if horizon_count > 0:
            loss = loss / horizon_count
        
        # Backward pass
        loss.backward()
        
        # Extract gradients
        for key, tensor in inputs.items():
            if tensor.grad is not None:
                # Compute importance as gradient magnitude
                grad_magnitude = torch.abs(tensor.grad).mean(dim=0)
                
                if len(grad_magnitude.shape) > 1:
                    # Average across time dimension for sequences
                    grad_magnitude = grad_magnitude.mean(dim=0)
                
                importance_scores[key] = {
                    'feature_importance': grad_magnitude.detach().cpu().numpy().tolist(),
                    'total_importance': float(grad_magnitude.sum()),
                    'mean_importance': float(grad_magnitude.mean()),
                    'std_importance': float(grad_magnitude.std())
                }
        
        return importance_scores
    
    def _compute_permutation_importance(self, inputs: Dict[str, torch.Tensor],
                                      targets: torch.Tensor) -> Dict[str, Any]:
        """Compute importance using permutation-based method"""
        
        importance_scores = {}
        
        with torch.no_grad():
            # Baseline prediction
            baseline_outputs = self.model(inputs)
            baseline_loss = self._compute_prediction_loss(baseline_outputs, targets)
            
            # Permute each feature type
            for key, tensor in inputs.items():
                feature_losses = []
                
                # Permute each feature dimension
                for feature_idx in range(tensor.shape[-1]):
                    # Create permuted input
                    permuted_inputs = inputs.copy()
                    permuted_tensor = tensor.clone()
                    
                    # Permute specific feature across batch
                    perm_indices = torch.randperm(tensor.shape[0])
                    if len(tensor.shape) == 2:  # Static features
                        permuted_tensor[:, feature_idx] = permuted_tensor[perm_indices, feature_idx]
                    else:  # Sequence features
                        permuted_tensor[:, :, feature_idx] = permuted_tensor[perm_indices, :, feature_idx]
                    
                    permuted_inputs[key] = permuted_tensor
                    
                    # Compute loss with permuted feature
                    permuted_outputs = self.model(permuted_inputs)
                    permuted_loss = self._compute_prediction_loss(permuted_outputs, targets)
                    
                    # Importance as loss increase
                    importance = permuted_loss - baseline_loss
                    feature_losses.append(float(importance))
                
                importance_scores[key] = {
                    'feature_importance': feature_losses,
                    'total_importance': sum(feature_losses),
                    'mean_importance': np.mean(feature_losses),
                    'std_importance': np.std(feature_losses)
                }
        
        return importance_scores
    
    def _compute_integrated_gradients(self, inputs: Dict[str, torch.Tensor],
                                    targets: torch.Tensor,
                                    steps: int = 50) -> Dict[str, Any]:
        """Compute importance using integrated gradients"""
        
        importance_scores = {}
        
        # Create baseline inputs
        baseline_inputs = {}
        for key, tensor in inputs.items():
            if self.config.importance_baseline == 'zero':
                baseline_inputs[key] = torch.zeros_like(tensor)
            elif self.config.importance_baseline == 'mean':
                baseline_inputs[key] = torch.full_like(tensor, tensor.mean())
            else:  # median
                baseline_inputs[key] = torch.full_like(tensor, tensor.median())
        
        # Interpolate between baseline and actual inputs
        for key, tensor in inputs.items():
            baseline_tensor = baseline_inputs[key]
            
            # Accumulate gradients along interpolation path
            integrated_gradients = torch.zeros_like(tensor)
            
            for step in range(steps):
                # Interpolation factor
                alpha = step / float(steps)
                
                # Interpolated input
                interpolated_inputs = inputs.copy()
                interpolated_tensor = baseline_tensor + alpha * (tensor - baseline_tensor)
                interpolated_tensor.requires_grad_(True)
                interpolated_inputs[key] = interpolated_tensor
                
                # Forward and backward pass
                outputs = self.model(interpolated_inputs)
                loss = self._compute_prediction_loss(outputs, targets)
                loss.backward()
                
                # Accumulate gradients
                integrated_gradients += interpolated_tensor.grad
            
            # Average gradients and multiply by input difference
            integrated_gradients = integrated_gradients / steps
            integrated_gradients = integrated_gradients * (tensor - baseline_tensor)
            
            # Compute importance magnitude
            importance_magnitude = torch.abs(integrated_gradients).mean(dim=0)
            
            if len(importance_magnitude.shape) > 1:
                importance_magnitude = importance_magnitude.mean(dim=0)
            
            importance_scores[key] = {
                'feature_importance': importance_magnitude.detach().cpu().numpy().tolist(),
                'total_importance': float(importance_magnitude.sum()),
                'mean_importance': float(importance_magnitude.mean()),
                'std_importance': float(importance_magnitude.std())
            }
        
        return importance_scores
    
    def _compute_prediction_loss(self, outputs: Dict[str, Dict], targets: torch.Tensor) -> torch.Tensor:
        """Compute prediction loss for importance calculation"""
        
        total_loss = 0.0
        count = 0
        
        for horizon_key, horizon_outputs in outputs.items():
            for target_key, target_outputs in horizon_outputs.items():
                if 'q50' in target_outputs:
                    prediction = target_outputs['q50']
                    
                    # Get corresponding target
                    if len(targets.shape) == 3:
                        horizon_idx = int(horizon_key.split('_')[1]) - 1
                        target_idx = int(target_key.split('_')[1])
                        target_values = targets[:, horizon_idx, target_idx]
                    else:
                        target_values = targets.squeeze()
                    
                    loss = F.mse_loss(prediction, target_values)
                    total_loss += loss
                    count += 1
        
        return total_loss / max(count, 1)
    
    def _add_feature_names(self, importance_scores: Dict[str, Any],
                          feature_names: Dict[str, List[str]]) -> Dict[str, Any]:
        """Add feature names to importance scores"""
        
        enhanced_scores = {}
        
        for key, scores in importance_scores.items():
            enhanced_scores[key] = scores.copy()
            
            if key in feature_names:
                names = feature_names[key]
                feature_importance = scores['feature_importance']
                
                # Create named importance mapping
                named_importance = {}
                for i, (name, importance) in enumerate(zip(names, feature_importance)):
                    named_importance[name] = importance
                
                enhanced_scores[key]['named_importance'] = named_importance
                enhanced_scores[key]['feature_names'] = names
        
        return enhanced_scores

class InterpretabilityChecker:
    """Perform sanity checks and validation of model interpretability"""
    
    def __init__(self, model: torch.nn.Module, config: InterpretabilityConfig = None):
        self.model = model
        self.config = config or InterpretabilityConfig()
    
    def perform_sanity_checks(self, inputs: Dict[str, torch.Tensor],
                            targets: torch.Tensor,
                            feature_names: Optional[Dict[str, List[str]]] = None) -> Dict[str, Any]:
        """
        Perform comprehensive sanity checks
        
        Args:
            inputs: Model inputs
            targets: Target values
            feature_names: Feature names for analysis
            
        Returns:
            Sanity check results
        """
        
        sanity_results = {}
        
        # 1. Model sensitivity check
        sanity_results['sensitivity_check'] = self._check_model_sensitivity(inputs, targets)
        
        # 2. Feature randomization check
        sanity_results['randomization_check'] = self._check_feature_randomization(inputs, targets)
        
        # 3. Input scaling robustness
        sanity_results['scaling_robustness'] = self._check_scaling_robustness(inputs, targets)
        
        # 4. Prediction consistency
        sanity_results['consistency_check'] = self._check_prediction_consistency(inputs)
        
        # 5. Attention sanity
        if hasattr(self.model, 'get_attention_weights'):
            sanity_results['attention_sanity'] = self._check_attention_sanity(inputs)
        
        return sanity_results
    
    def _check_model_sensitivity(self, inputs: Dict[str, torch.Tensor],
                               targets: torch.Tensor) -> Dict[str, Any]:
        """Check if model is sensitive to input changes"""
        
        self.model.eval()
        
        with torch.no_grad():
            # Baseline prediction
            baseline_outputs = self.model(inputs)
            baseline_predictions = self._extract_median_predictions(baseline_outputs)
            
            # Add noise to inputs
            noise_levels = [0.01, 0.05, 0.1]
            sensitivity_results = {}
            
            for noise_level in noise_levels:
                noisy_inputs = {}
                
                for key, tensor in inputs.items():
                    noise = torch.randn_like(tensor) * noise_level * tensor.std()
                    noisy_inputs[key] = tensor + noise
                
                # Noisy prediction
                noisy_outputs = self.model(noisy_inputs)
                noisy_predictions = self._extract_median_predictions(noisy_outputs)
                
                # Compute sensitivity
                prediction_change = torch.abs(noisy_predictions - baseline_predictions).mean()
                sensitivity_results[f'noise_{noise_level}'] = float(prediction_change)
        
        return {
            'noise_sensitivity': sensitivity_results,
            'is_sensitive': any(change > 0.001 for change in sensitivity_results.values())
        }
    
    def _check_feature_randomization(self, inputs: Dict[str, torch.Tensor],
                                   targets: torch.Tensor) -> Dict[str, Any]:
        """Check model performance with randomized features"""
        
        self.model.eval()
        
        with torch.no_grad():
            # Baseline loss
            baseline_outputs = self.model(inputs)
            baseline_loss = self._compute_loss(baseline_outputs, targets)
            
            randomization_results = {}
            
            # Randomize each feature type
            for key, tensor in inputs.items():
                randomized_inputs = inputs.copy()
                
                # Randomize features while preserving distribution
                randomized_tensor = tensor[torch.randperm(tensor.shape[0])]
                randomized_inputs[key] = randomized_tensor
                
                # Compute loss with randomized features
                randomized_outputs = self.model(randomized_inputs)
                randomized_loss = self._compute_loss(randomized_outputs, targets)
                
                # Performance degradation
                degradation = (randomized_loss - baseline_loss) / (baseline_loss + 1e-8)
                randomization_results[key] = {
                    'loss_increase': float(randomized_loss - baseline_loss),
                    'relative_degradation': float(degradation)
                }
        
        return {
            'randomization_effects': randomization_results,
            'meaningful_features': [k for k, v in randomization_results.items() 
                                  if v['relative_degradation'] > 0.05]
        }
    
    def _check_scaling_robustness(self, inputs: Dict[str, torch.Tensor],
                                targets: torch.Tensor) -> Dict[str, Any]:
        """Check robustness to input scaling"""
        
        self.model.eval()
        
        with torch.no_grad():
            # Baseline prediction
            baseline_outputs = self.model(inputs)
            baseline_predictions = self._extract_median_predictions(baseline_outputs)
            
            scaling_factors = [0.5, 2.0, 10.0]
            scaling_results = {}
            
            for scale in scaling_factors:
                scaled_inputs = {}
                
                for key, tensor in inputs.items():
                    scaled_inputs[key] = tensor * scale
                
                # Scaled prediction
                scaled_outputs = self.model(scaled_inputs)
                scaled_predictions = self._extract_median_predictions(scaled_outputs)
                
                # Prediction change
                prediction_change = torch.abs(scaled_predictions - baseline_predictions).mean()
                scaling_results[f'scale_{scale}'] = float(prediction_change)
        
        return {
            'scaling_sensitivity': scaling_results,
            'is_scale_robust': all(change < 1.0 for change in scaling_results.values())
        }
    
    def _check_prediction_consistency(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """Check prediction consistency across multiple forward passes"""
        
        self.model.eval()
        
        predictions_list = []
        
        with torch.no_grad():
            for _ in range(5):
                outputs = self.model(inputs)
                predictions = self._extract_median_predictions(outputs)
                predictions_list.append(predictions)
        
        # Compute consistency metrics
        all_predictions = torch.stack(predictions_list)
        prediction_std = all_predictions.std(dim=0).mean()
        prediction_range = (all_predictions.max(dim=0)[0] - all_predictions.min(dim=0)[0]).mean()
        
        return {
            'prediction_std': float(prediction_std),
            'prediction_range': float(prediction_range),
            'is_consistent': float(prediction_std) < 0.01
        }
    
    def _check_attention_sanity(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """Check attention mechanism sanity"""
        
        self.model.eval()
        
        with torch.no_grad():
            # Get attention weights
            self.model(inputs)
            attention_weights = self.model.get_attention_weights()
            
            if attention_weights is None:
                return {'error': 'No attention weights available'}
            
            # Basic attention sanity checks
            attention_np = attention_weights.detach().cpu().numpy()
            
            # Check if attention sums to 1 (approximately)
            attention_sums = attention_np.sum(axis=-1)
            sum_check = np.abs(attention_sums - 1.0).mean()
            
            # Check for degenerate attention (all weight on one position)
            max_attention = attention_np.max(axis=-1)
            degenerate_ratio = (max_attention > 0.9).mean()
            
            # Check for uniform attention (no focus)
            seq_len = attention_np.shape[-1]
            uniform_threshold = 1.0 / seq_len + 0.1
            uniform_ratio = (max_attention < uniform_threshold).mean()
        
        return {
            'attention_sum_error': float(sum_check),
            'degenerate_attention_ratio': float(degenerate_ratio),
            'uniform_attention_ratio': float(uniform_ratio),
            'passes_sum_check': float(sum_check) < 0.1,
            'has_focused_attention': float(degenerate_ratio) < 0.8,
            'has_selective_attention': float(uniform_ratio) < 0.8
        }
    
    def _extract_median_predictions(self, outputs: Dict[str, Dict]) -> torch.Tensor:
        """Extract median predictions for consistency checking"""
        
        predictions = []
        
        for horizon_key, horizon_outputs in outputs.items():
            for target_key, target_outputs in horizon_outputs.items():
                if 'q50' in target_outputs:
                    predictions.append(target_outputs['q50'])
        
        if predictions:
            return torch.cat(predictions, dim=-1)
        else:
            return torch.tensor([0.0])
    
    def _compute_loss(self, outputs: Dict[str, Dict], targets: torch.Tensor) -> torch.Tensor:
        """Compute loss for sanity checking"""
        
        total_loss = 0.0
        count = 0
        
        for horizon_key, horizon_outputs in outputs.items():
            for target_key, target_outputs in horizon_outputs.items():
                if 'q50' in target_outputs:
                    prediction = target_outputs['q50']
                    
                    # Use first target for simplicity
                    if len(targets.shape) > 1:
                        target_values = targets[:, 0, 0] if len(targets.shape) == 3 else targets[:, 0]
                    else:
                        target_values = targets
                    
                    loss = F.mse_loss(prediction, target_values)
                    total_loss += loss
                    count += 1
        
        return total_loss / max(count, 1)

class ModelExplainer:
    """Main interpretability interface combining all analysis components"""
    
    def __init__(self, model: torch.nn.Module, config: InterpretabilityConfig = None):
        self.model = model
        self.config = config or InterpretabilityConfig()
        
        # Initialize components
        self.attention_analyzer = AttentionAnalyzer(config)
        self.importance_scorer = VariableImportanceScorer(model, config)
        self.sanity_checker = InterpretabilityChecker(model, config)
        
        # Results storage
        self.explanation_history = []
    
    def explain_prediction(self, inputs: Dict[str, torch.Tensor],
                         targets: torch.Tensor,
                         feature_names: Optional[Dict[str, List[str]]] = None) -> Dict[str, Any]:
        """
        Complete interpretability analysis
        
        Args:
            inputs: Model inputs
            targets: Target values
            feature_names: Feature names for analysis
            
        Returns:
            Comprehensive explanation results
        """
        
        explanation_results = {
            'timestamp': pd.Timestamp.now(),
            'model_type': type(self.model).__name__
        }
        
        # 1. Attention analysis
        if self.config.analyze_attention_weights and hasattr(self.model, 'get_attention_weights'):
            logger.info("Analyzing attention weights...")
            self.model(inputs)
            attention_weights = self.model.get_attention_weights()
            
            attention_analysis = self.attention_analyzer.analyze_attention_weights(
                attention_weights, feature_names=feature_names.get('observed_past') if feature_names else None
            )
            explanation_results['attention_analysis'] = attention_analysis
        
        # 2. Variable importance
        if self.config.compute_variable_importance:
            logger.info("Computing variable importance...")
            importance_analysis = self.importance_scorer.compute_importance(
                inputs, targets, feature_names
            )
            explanation_results['variable_importance'] = importance_analysis
        
        # 3. Sanity checks
        if self.config.perform_sanity_checks:
            logger.info("Performing sanity checks...")
            sanity_analysis = self.sanity_checker.perform_sanity_checks(
                inputs, targets, feature_names
            )
            explanation_results['sanity_checks'] = sanity_analysis
        
        # 4. Overall interpretability summary
        explanation_results['summary'] = self._create_explanation_summary(explanation_results)
        
        # Store results
        self.explanation_history.append(explanation_results)
        
        return explanation_results
    
    def _create_explanation_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Create summary of interpretability analysis"""
        
        summary = {
            'model_interpretable': True,
            'interpretation_confidence': 'high',
            'key_findings': [],
            'recommendations': []
        }
        
        # Analyze attention results
        if 'attention_analysis' in results:
            attention = results['attention_analysis']
            
            if 'attention_stats' in attention:
                sparsity = attention['attention_stats']['sparsity']
                if sparsity > 0.8:
                    summary['key_findings'].append('High attention sparsity - model focuses on few inputs')
                elif sparsity < 0.2:
                    summary['key_findings'].append('Low attention sparsity - model considers many inputs')
            
            if 'temporal_patterns' in attention:
                recency_bias = attention['temporal_patterns']['recency_bias']['mean']
                if recency_bias > 0.7:
                    summary['key_findings'].append('Strong recency bias - model favors recent information')
        
        # Analyze importance results
        if 'variable_importance' in results:
            importance = results['variable_importance']
            
            # Find most important feature types
            feature_importance = {}
            for feature_type, scores in importance.items():
                feature_importance[feature_type] = scores['total_importance']
            
            if feature_importance:
                most_important = max(feature_importance, key=feature_importance.get)
                summary['key_findings'].append(f'Most important feature type: {most_important}')
        
        # Analyze sanity check results
        if 'sanity_checks' in results:
            sanity = results['sanity_checks']
            
            if 'randomization_check' in sanity:
                meaningful_features = sanity['randomization_check']['meaningful_features']
                if len(meaningful_features) == 0:
                    summary['interpretation_confidence'] = 'low'
                    summary['recommendations'].append('Model may not be using features meaningfully')
            
            if 'attention_sanity' in sanity:
                attention_sanity = sanity['attention_sanity']
                if not attention_sanity.get('passes_sum_check', True):
                    summary['recommendations'].append('Attention mechanism may have numerical issues')
        
        return summary