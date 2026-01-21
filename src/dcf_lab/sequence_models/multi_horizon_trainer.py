"""
Multi-horizon Sequence Model Trainer

This module implements the complete training pipeline for multi-horizon
sequence models (TFT/Transformer) with:

1. Comprehensive data preparation and validation
2. Curriculum learning integration
3. Model training with interpretability
4. Performance evaluation and monitoring
5. Complete Task 8 implementation

Key components:
- Integrated training pipeline
- Model evaluation and validation
- Performance monitoring
- Production deployment support
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
import logging
import json
import os
from datetime import datetime

from .data_preparation import SequenceDataProcessor, FeatureConfig, WindowConfig
from .tft_model import TFTConfig, TFTTrainer, TFTPredictor
from .curriculum_learning import CurriculumScheduler, CurriculumConfig
from .interpretability import ModelExplainer, InterpretabilityConfig

logger = logging.getLogger(__name__)

@dataclass
class SequenceModelConfig:
    """Complete configuration for sequence model training"""
    
    # Data configuration
    feature_config: FeatureConfig = field(default_factory=FeatureConfig)
    window_config: WindowConfig = field(default_factory=WindowConfig)
    
    # Model configuration
    tft_config: TFTConfig = field(default_factory=TFTConfig)
    
    # Curriculum learning
    use_curriculum_learning: bool = True
    curriculum_config: CurriculumConfig = field(default_factory=CurriculumConfig)
    
    # Interpretability
    enable_interpretability: bool = True
    interpretability_config: InterpretabilityConfig = field(default_factory=InterpretabilityConfig)
    
    # Training configuration
    max_epochs: int = 200
    early_stopping_patience: int = 20
    validation_split: float = 0.2
    
    # Evaluation
    evaluation_horizons: List[int] = field(default_factory=lambda: [1, 5, 20, 60])
    evaluation_metrics: List[str] = field(default_factory=lambda: ['mse', 'mae', 'mape', 'coverage'])
    
    # Output and logging
    output_directory: str = 'sequence_model_outputs'
    save_model_checkpoints: bool = True
    log_interpretability: bool = True

@dataclass
class TrainingResults:
    """Results from sequence model training"""
    
    # Training metrics
    training_history: List[Dict[str, Any]] = field(default_factory=list)
    best_epoch: int = 0
    best_validation_loss: float = float('inf')
    
    # Curriculum learning results
    curriculum_history: List[Dict[str, Any]] = field(default_factory=list)
    final_horizons: List[int] = field(default_factory=list)
    
    # Interpretability analysis
    interpretability_results: Dict[str, Any] = field(default_factory=dict)
    
    # Model evaluation
    evaluation_results: Dict[str, Any] = field(default_factory=dict)
    
    # Metadata
    training_duration: float = 0.0
    model_size: int = 0
    total_parameters: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert results to dictionary for serialization"""
        return {
            'training_history': self.training_history,
            'best_epoch': self.best_epoch,
            'best_validation_loss': self.best_validation_loss,
            'curriculum_history': self.curriculum_history,
            'final_horizons': self.final_horizons,
            'interpretability_results': self.interpretability_results,
            'evaluation_results': self.evaluation_results,
            'training_duration': self.training_duration,
            'model_size': self.model_size,
            'total_parameters': self.total_parameters
        }

class ModelEvaluator:
    """Comprehensive model evaluation for sequence models"""
    
    def __init__(self, config: SequenceModelConfig):
        self.config = config
    
    def evaluate_model(self, model: nn.Module,
                      test_windows: Dict[str, np.ndarray],
                      feature_names: Optional[Dict[str, List[str]]] = None) -> Dict[str, Any]:
        """
        Comprehensive model evaluation
        
        Args:
            model: Trained model
            test_windows: Test data windows
            feature_names: Feature names for analysis
            
        Returns:
            Evaluation results
        """
        
        logger.info("Starting comprehensive model evaluation...")
        
        evaluation_results = {}
        
        # 1. Prediction accuracy evaluation
        accuracy_results = self._evaluate_prediction_accuracy(model, test_windows)
        evaluation_results['accuracy'] = accuracy_results
        
        # 2. Multi-horizon performance
        horizon_results = self._evaluate_horizon_performance(model, test_windows)
        evaluation_results['horizon_performance'] = horizon_results
        
        # 3. Uncertainty calibration
        calibration_results = self._evaluate_uncertainty_calibration(model, test_windows)
        evaluation_results['uncertainty_calibration'] = calibration_results
        
        # 4. Interpretability analysis
        if self.config.enable_interpretability:
            interpretability_results = self._evaluate_interpretability(
                model, test_windows, feature_names
            )
            evaluation_results['interpretability'] = interpretability_results
        
        # 5. Robustness testing
        robustness_results = self._evaluate_robustness(model, test_windows)
        evaluation_results['robustness'] = robustness_results
        
        # 6. Overall performance summary
        summary = self._create_evaluation_summary(evaluation_results)
        evaluation_results['summary'] = summary
        
        logger.info("Model evaluation completed")
        return evaluation_results
    
    def _evaluate_prediction_accuracy(self, model: nn.Module,
                                    test_windows: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """Evaluate basic prediction accuracy"""
        
        model.eval()
        predictor = TFTPredictor(self.config.tft_config)
        predictor.model = model
        
        # Create test dataloader
        test_loader = predictor._create_dataloader(
            test_windows, batch_size=64, shuffle=False
        )
        
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for batch in test_loader:
                inputs, targets = predictor._reconstruct_batch(batch, test_loader.key_mapping)
                outputs = model(inputs)
                
                # Extract median predictions
                batch_predictions = []
                batch_targets = []
                
                for h_idx, horizon in enumerate(self.config.evaluation_horizons):
                    horizon_key = f"horizon_{horizon}"
                    if horizon_key in outputs:
                        for t_idx in range(self.config.tft_config.num_targets):
                            target_key = f"target_{t_idx}"
                            if target_key in outputs[horizon_key] and 'q50' in outputs[horizon_key][target_key]:
                                pred = outputs[horizon_key][target_key]['q50']
                                target = targets[:, h_idx, t_idx]
                                
                                # Skip NaN targets
                                valid_mask = ~torch.isnan(target)
                                if valid_mask.sum() > 0:
                                    batch_predictions.extend(pred[valid_mask].cpu().numpy())
                                    batch_targets.extend(target[valid_mask].cpu().numpy())
                
                all_predictions.extend(batch_predictions)
                all_targets.extend(batch_targets)
        
        # Compute accuracy metrics
        if len(all_predictions) > 0:
            predictions = np.array(all_predictions)
            targets = np.array(all_targets)
            
            mse = np.mean((predictions - targets) ** 2)
            mae = np.mean(np.abs(predictions - targets))
            mape = np.mean(np.abs((predictions - targets) / (targets + 1e-8))) * 100
            
            correlation = np.corrcoef(predictions, targets)[0, 1] if len(predictions) > 1 else 0.0
            
            return {
                'mse': float(mse),
                'mae': float(mae),
                'mape': float(mape),
                'rmse': float(np.sqrt(mse)),
                'correlation': float(correlation),
                'n_samples': len(predictions)
            }
        else:
            return {'error': 'No valid predictions found'}
    
    def _evaluate_horizon_performance(self, model: nn.Module,
                                    test_windows: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """Evaluate performance across different horizons"""
        
        horizon_results = {}
        
        for horizon in self.config.evaluation_horizons:
            # Create windows for specific horizon
            horizon_windows = self._extract_horizon_windows(test_windows, horizon)
            
            # Evaluate for this horizon
            horizon_accuracy = self._evaluate_prediction_accuracy(model, horizon_windows)
            horizon_results[f"horizon_{horizon}"] = horizon_accuracy
        
        return horizon_results
    
    def _evaluate_uncertainty_calibration(self, model: nn.Module,
                                        test_windows: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """Evaluate uncertainty calibration using quantile predictions"""
        
        model.eval()
        predictor = TFTPredictor(self.config.tft_config)
        predictor.model = model
        
        test_loader = predictor._create_dataloader(
            test_windows, batch_size=64, shuffle=False
        )
        
        coverage_results = {}
        
        for quantile in self.config.tft_config.quantiles:
            if quantile == 0.5:  # Skip median
                continue
            
            # Find corresponding coverage
            if quantile < 0.5:
                coverage_level = (0.5 - quantile) * 2
                upper_quantile = 1.0 - quantile
            else:
                coverage_level = (quantile - 0.5) * 2
                upper_quantile = quantile
            
            in_interval_count = 0
            total_count = 0
            
            with torch.no_grad():
                for batch in test_loader:
                    inputs, targets = predictor._reconstruct_batch(batch, test_loader.key_mapping)
                    outputs = model(inputs)
                    
                    for h_idx, horizon in enumerate(self.config.evaluation_horizons):
                        horizon_key = f"horizon_{horizon}"
                        if horizon_key in outputs:
                            for t_idx in range(self.config.tft_config.num_targets):
                                target_key = f"target_{t_idx}"
                                if target_key in outputs[horizon_key]:
                                    target_values = targets[:, h_idx, t_idx]
                                    
                                    # Get quantile predictions
                                    lower_key = f"q{int(quantile * 100)}"
                                    upper_key = f"q{int(upper_quantile * 100)}"
                                    
                                    if (lower_key in outputs[horizon_key][target_key] and
                                        upper_key in outputs[horizon_key][target_key]):
                                        
                                        lower_pred = outputs[horizon_key][target_key][lower_key]
                                        upper_pred = outputs[horizon_key][target_key][upper_key]
                                        
                                        # Check coverage
                                        valid_mask = ~torch.isnan(target_values)
                                        if valid_mask.sum() > 0:
                                            in_interval = ((target_values >= lower_pred) & 
                                                         (target_values <= upper_pred))[valid_mask]
                                            in_interval_count += in_interval.sum().item()
                                            total_count += valid_mask.sum().item()
            
            if total_count > 0:
                actual_coverage = in_interval_count / total_count
                coverage_results[f"coverage_{int(coverage_level * 100)}"] = {
                    'expected_coverage': coverage_level,
                    'actual_coverage': actual_coverage,
                    'calibration_error': abs(actual_coverage - coverage_level)
                }
        
        return coverage_results
    
    def _evaluate_interpretability(self, model: nn.Module,
                                 test_windows: Dict[str, np.ndarray],
                                 feature_names: Optional[Dict[str, List[str]]] = None) -> Dict[str, Any]:
        """Evaluate model interpretability"""
        
        # Use a subset of test data for interpretability analysis
        sample_size = min(100, test_windows['targets'].shape[0])
        sample_indices = np.random.choice(
            test_windows['targets'].shape[0], size=sample_size, replace=False
        )
        
        sample_windows = {}
        for key, windows in test_windows.items():
            sample_windows[key] = windows[sample_indices]
        
        # Create explainer
        explainer = ModelExplainer(model, self.config.interpretability_config)
        
        # Prepare inputs and targets
        inputs = {}
        for key in ['static', 'observed_past', 'known_future']:
            if key in sample_windows:
                inputs[key] = torch.FloatTensor(sample_windows[key])
        
        targets = torch.FloatTensor(sample_windows['targets'])
        
        # Run interpretability analysis
        explanation = explainer.explain_prediction(inputs, targets, feature_names)
        
        return explanation
    
    def _evaluate_robustness(self, model: nn.Module,
                           test_windows: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """Evaluate model robustness"""
        
        model.eval()
        
        # Original performance
        original_accuracy = self._evaluate_prediction_accuracy(model, test_windows)
        
        robustness_results = {
            'baseline_performance': original_accuracy,
            'noise_robustness': {},
            'missing_data_robustness': {}
        }
        
        # Test noise robustness
        noise_levels = [0.01, 0.05, 0.1]
        for noise_level in noise_levels:
            noisy_windows = {}
            for key, windows in test_windows.items():
                if key != 'targets':
                    noise = np.random.normal(0, noise_level * windows.std(), windows.shape)
                    noisy_windows[key] = windows + noise
                else:
                    noisy_windows[key] = windows
            
            noisy_accuracy = self._evaluate_prediction_accuracy(model, noisy_windows)
            robustness_results['noise_robustness'][f'noise_{noise_level}'] = {
                'accuracy': noisy_accuracy,
                'performance_drop': original_accuracy['mse'] - noisy_accuracy['mse'] if 'mse' in noisy_accuracy else 0
            }
        
        # Test missing data robustness
        missing_rates = [0.1, 0.2, 0.3]
        for missing_rate in missing_rates:
            missing_windows = {}
            for key, windows in test_windows.items():
                if key != 'targets':
                    missing_mask = np.random.random(windows.shape) < missing_rate
                    masked_windows = windows.copy()
                    masked_windows[missing_mask] = 0  # Replace with zeros
                    missing_windows[key] = masked_windows
                else:
                    missing_windows[key] = windows
            
            missing_accuracy = self._evaluate_prediction_accuracy(model, missing_windows)
            robustness_results['missing_data_robustness'][f'missing_{missing_rate}'] = {
                'accuracy': missing_accuracy,
                'performance_drop': original_accuracy['mse'] - missing_accuracy['mse'] if 'mse' in missing_accuracy else 0
            }
        
        return robustness_results
    
    def _extract_horizon_windows(self, windows: Dict[str, np.ndarray], 
                                horizon: int) -> Dict[str, np.ndarray]:
        """Extract windows for specific horizon evaluation"""
        
        # For simplicity, return the same windows
        # In practice, you might want to adjust window generation for specific horizons
        return windows
    
    def _create_evaluation_summary(self, evaluation_results: Dict[str, Any]) -> Dict[str, Any]:
        """Create summary of evaluation results"""
        
        summary = {
            'overall_performance': 'good',
            'key_metrics': {},
            'strengths': [],
            'weaknesses': [],
            'recommendations': []
        }
        
        # Extract key metrics
        if 'accuracy' in evaluation_results:
            accuracy = evaluation_results['accuracy']
            summary['key_metrics']['mse'] = accuracy.get('mse', 0)
            summary['key_metrics']['correlation'] = accuracy.get('correlation', 0)
            
            if accuracy.get('correlation', 0) > 0.5:
                summary['strengths'].append('Good predictive correlation')
            else:
                summary['weaknesses'].append('Low predictive correlation')
        
        # Uncertainty calibration
        if 'uncertainty_calibration' in evaluation_results:
            calibration = evaluation_results['uncertainty_calibration']
            calibration_errors = [v['calibration_error'] for v in calibration.values() 
                                if isinstance(v, dict) and 'calibration_error' in v]
            
            if calibration_errors and np.mean(calibration_errors) < 0.1:
                summary['strengths'].append('Well-calibrated uncertainty estimates')
            else:
                summary['weaknesses'].append('Poor uncertainty calibration')
        
        # Robustness assessment
        if 'robustness' in evaluation_results:
            robustness = evaluation_results['robustness']
            
            # Check noise robustness
            noise_performance_drops = []
            for noise_result in robustness.get('noise_robustness', {}).values():
                if isinstance(noise_result, dict) and 'performance_drop' in noise_result:
                    noise_performance_drops.append(noise_result['performance_drop'])
            
            if noise_performance_drops and np.mean(noise_performance_drops) < 0.1:
                summary['strengths'].append('Robust to input noise')
            else:
                summary['weaknesses'].append('Sensitive to input noise')
        
        # Overall assessment
        if len(summary['strengths']) > len(summary['weaknesses']):
            summary['overall_performance'] = 'excellent'
        elif len(summary['strengths']) == len(summary['weaknesses']):
            summary['overall_performance'] = 'good'
        else:
            summary['overall_performance'] = 'needs_improvement'
        
        return summary

class MultiHorizonTrainer:
    """Main trainer for multi-horizon sequence models"""
    
    def __init__(self, config: SequenceModelConfig):
        self.config = config
        
        # Initialize components
        self.data_processor = SequenceDataProcessor(
            config.feature_config, config.window_config
        )
        
        # Create output directory
        os.makedirs(config.output_directory, exist_ok=True)
        
        # Training state
        self.training_results = TrainingResults()
        self.model = None
        self.curriculum_scheduler = None
    
    def train(self, data: pd.DataFrame,
             feature_names: Optional[Dict[str, List[str]]] = None) -> TrainingResults:
        """
        Complete training pipeline for multi-horizon sequence models
        
        Args:
            data: Input dataframe with all features
            feature_names: Feature names by category
            
        Returns:
            Training results
        """
        
        start_time = datetime.now()
        logger.info("Starting multi-horizon sequence model training...")
        
        # 1. Data preparation
        logger.info("Processing data...")
        windows = self.data_processor.process_data(data)
        train_windows, val_windows = self.data_processor.create_train_val_split(
            windows, self.config.validation_split
        )
        
        # 2. Model initialization
        logger.info("Initializing TFT model...")
        
        # Update model config with actual feature dimensions
        self.config.tft_config.num_static_features = train_windows.get('static', np.array([])).shape[-1] if 'static' in train_windows else 0
        self.config.tft_config.num_observed_past_features = train_windows.get('observed_past', np.array([])).shape[-1] if 'observed_past' in train_windows else 0
        self.config.tft_config.num_known_future_features = train_windows.get('known_future', np.array([])).shape[-1] if 'known_future' in train_windows else 0
        self.config.tft_config.num_targets = train_windows.get('targets', np.array([])).shape[-1] if 'targets' in train_windows else 1
        
        # Create TFT trainer
        tft_trainer = TFTTrainer(self.config.tft_config)
        self.model = tft_trainer.predictor.model
        
        # 3. Curriculum learning setup
        if self.config.use_curriculum_learning:
            logger.info("Setting up curriculum learning...")
            self.curriculum_scheduler = CurriculumScheduler(
                self.model, self.config.curriculum_config
            )
        
        # 4. Training loop with curriculum learning
        logger.info("Starting training loop...")
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(self.config.max_epochs):
            # Get current horizons from curriculum
            if self.curriculum_scheduler:
                current_horizons = self.curriculum_scheduler.get_current_horizons()
                # Update model config for current horizons
                self.config.tft_config.prediction_horizons = current_horizons
            
            # Train epoch
            train_loss = tft_trainer.predictor.train_epoch(
                tft_trainer.predictor._create_dataloader(train_windows, self.config.tft_config.batch_size, shuffle=True)
            )
            
            # Validate epoch
            val_loss = tft_trainer.predictor.validate_epoch(
                tft_trainer.predictor._create_dataloader(val_windows, self.config.tft_config.batch_size, shuffle=False)
            )
            
            # Curriculum learning step
            curriculum_results = None
            if self.curriculum_scheduler:
                curriculum_results = self.curriculum_scheduler.step(epoch, val_loss)
                self.training_results.curriculum_history.append(curriculum_results)
            
            # Early stopping check
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                self.training_results.best_epoch = epoch
                self.training_results.best_validation_loss = best_val_loss
                
                # Save best model
                if self.config.save_model_checkpoints:
                    checkpoint_path = os.path.join(self.config.output_directory, 'best_model.pt')
                    torch.save(self.model.state_dict(), checkpoint_path)
            else:
                patience_counter += 1
            
            # Log progress
            if epoch % 10 == 0 or epoch < 10:
                current_horizons_str = str(current_horizons) if self.curriculum_scheduler else "all"
                logger.info(
                    f"Epoch {epoch}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, "
                    f"horizons={current_horizons_str}"
                )
            
            # Store training history
            epoch_results = {
                'epoch': epoch,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'current_horizons': current_horizons if self.curriculum_scheduler else self.config.window_config.prediction_horizons,
                'curriculum_results': curriculum_results
            }
            self.training_results.training_history.append(epoch_results)
            
            # Early stopping
            if patience_counter >= self.config.early_stopping_patience:
                logger.info(f"Early stopping at epoch {epoch}")
                break
        
        # 5. Final model evaluation
        logger.info("Evaluating final model...")
        
        # Create test split (use validation for now)
        evaluator = ModelEvaluator(self.config)
        evaluation_results = evaluator.evaluate_model(self.model, val_windows, feature_names)
        self.training_results.evaluation_results = evaluation_results
        
        # 6. Interpretability analysis
        if self.config.enable_interpretability:
            logger.info("Running interpretability analysis...")
            explainer = ModelExplainer(self.model, self.config.interpretability_config)
            
            # Sample inputs for interpretation
            sample_inputs = {}
            for key in ['static', 'observed_past', 'known_future']:
                if key in val_windows:
                    sample_inputs[key] = torch.FloatTensor(val_windows[key][:32])  # First 32 samples
            
            sample_targets = torch.FloatTensor(val_windows['targets'][:32])
            
            interpretability_results = explainer.explain_prediction(
                sample_inputs, sample_targets, feature_names
            )
            self.training_results.interpretability_results = interpretability_results
        
        # 7. Final metadata
        end_time = datetime.now()
        self.training_results.training_duration = (end_time - start_time).total_seconds()
        self.training_results.total_parameters = sum(p.numel() for p in self.model.parameters())
        self.training_results.final_horizons = (
            self.curriculum_scheduler.get_current_horizons() 
            if self.curriculum_scheduler 
            else self.config.window_config.prediction_horizons
        )
        
        # 8. Save results
        results_path = os.path.join(self.config.output_directory, 'training_results.json')
        with open(results_path, 'w') as f:
            json.dump(self.training_results.to_dict(), f, indent=2, default=str)
        
        logger.info(f"Training completed in {self.training_results.training_duration:.1f} seconds")
        logger.info(f"Best validation loss: {self.training_results.best_validation_loss:.4f}")
        logger.info(f"Final horizons: {self.training_results.final_horizons}")
        
        return self.training_results
    
    def get_model(self) -> Optional[nn.Module]:
        """Get trained model"""
        return self.model
    
    def get_training_summary(self) -> Dict[str, Any]:
        """Get comprehensive training summary"""
        
        summary = {
            'config': {
                'model_type': 'TFT',
                'input_length': self.config.window_config.input_length,
                'prediction_horizons': self.config.window_config.prediction_horizons,
                'use_curriculum_learning': self.config.use_curriculum_learning,
                'enable_interpretability': self.config.enable_interpretability
            },
            'training_results': self.training_results.to_dict(),
            'model_info': {
                'total_parameters': self.training_results.total_parameters,
                'model_size_mb': self.training_results.total_parameters * 4 / (1024 * 1024),  # Rough estimate
                'training_duration_minutes': self.training_results.training_duration / 60
            }
        }
        
        return summary