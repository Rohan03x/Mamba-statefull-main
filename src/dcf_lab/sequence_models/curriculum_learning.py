"""
Curriculum Learning for Multi-horizon Sequence Models

This module implements curriculum learning strategies for progressive
multi-horizon training:

1. Start with shorter horizons → gradually extend to longer horizons
2. Freeze embeddings when stable to prevent catastrophic forgetting
3. Monitor stability and progression metrics
4. Adaptive scheduling based on performance

Key components:
- Progressive horizon scheduling
- Embedding stabilization and freezing
- Performance monitoring and adaptation
- Curriculum progression strategies
"""

import numpy as np
import torch.nn as nn
from typing import Dict, List, Any
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)

@dataclass
class CurriculumConfig:
    """Configuration for curriculum learning"""
    
    # Horizon progression strategy
    progression_strategy: str = 'linear'  # 'linear', 'exponential', 'adaptive'
    
    # Initial and final horizons
    initial_horizons: List[int] = field(default_factory=lambda: [1])
    final_horizons: List[int] = field(default_factory=lambda: [1, 5, 20, 60])
    
    # Progression schedule
    horizon_progression_epochs: List[int] = field(default_factory=lambda: [20, 40, 60])
    
    # Stability monitoring
    stability_window: int = 10
    stability_threshold: float = 0.05  # Max relative change in loss
    min_stable_epochs: int = 5
    
    # Embedding freezing
    freeze_embeddings_after_stable: bool = True
    freeze_threshold_epochs: int = 15
    unfreeze_on_new_horizon: bool = True
    
    # Adaptive scheduling
    use_adaptive_scheduling: bool = True
    performance_patience: int = 5
    performance_threshold: float = 0.02
    
    # Validation criteria
    min_epochs_per_stage: int = 10
    max_epochs_per_stage: int = 50

class StabilityMonitor:
    """Monitor training stability for curriculum decisions"""
    
    def __init__(self, window_size: int = 10, threshold: float = 0.05):
        self.window_size = window_size
        self.threshold = threshold
        
        self.loss_history = []
        self.stability_history = []
    
    def update(self, loss: float) -> Dict[str, Any]:
        """
        Update with new loss value and compute stability metrics
        
        Args:
            loss: Current loss value
            
        Returns:
            Stability metrics
        """
        
        self.loss_history.append(loss)
        
        # Keep only recent history
        if len(self.loss_history) > self.window_size * 2:
            self.loss_history = self.loss_history[-self.window_size * 2:]
        
        # Compute stability metrics
        stability_metrics = self._compute_stability()
        self.stability_history.append(stability_metrics)
        
        return stability_metrics
    
    def _compute_stability(self) -> Dict[str, Any]:
        """Compute various stability metrics"""
        
        if len(self.loss_history) < self.window_size:
            return {
                'is_stable': False,
                'relative_change': 1.0,
                'trend': 'unknown',
                'volatility': 1.0,
                'sample_size': len(self.loss_history)
            }
        
        recent_losses = self.loss_history[-self.window_size:]
        
        # Relative change over window
        relative_change = abs(recent_losses[-1] - recent_losses[0]) / (recent_losses[0] + 1e-8)
        
        # Trend analysis
        if len(recent_losses) >= 3:
            early_avg = np.mean(recent_losses[:len(recent_losses)//2])
            late_avg = np.mean(recent_losses[len(recent_losses)//2:])
            
            if late_avg < early_avg * 0.95:
                trend = 'improving'
            elif late_avg > early_avg * 1.05:
                trend = 'degrading'
            else:
                trend = 'stable'
        else:
            trend = 'unknown'
        
        # Volatility (coefficient of variation)
        mean_loss = np.mean(recent_losses)
        std_loss = np.std(recent_losses)
        volatility = std_loss / (mean_loss + 1e-8)
        
        # Overall stability decision
        is_stable = (
            relative_change < self.threshold and
            volatility < self.threshold and
            trend in ['improving', 'stable']
        )
        
        return {
            'is_stable': is_stable,
            'relative_change': relative_change,
            'trend': trend,
            'volatility': volatility,
            'mean_loss': mean_loss,
            'std_loss': std_loss
        }
    
    def get_stability_score(self) -> float:
        """Get overall stability score (0-1, higher is more stable)"""
        
        if not self.stability_history:
            return 0.0
        
        recent_stability = self.stability_history[-min(5, len(self.stability_history)):]
        stable_count = sum(1 for s in recent_stability if s['is_stable'])
        
        return stable_count / len(recent_stability)

class EmbeddingFreezer:
    """Manage embedding freezing and unfreezing"""
    
    def __init__(self, model: nn.Module):
        self.model = model
        self.frozen_parameters = set()
        self.freeze_history = []
    
    def freeze_embeddings(self, layer_names: List[str] = None) -> Dict[str, Any]:
        """
        Freeze embedding layers
        
        Args:
            layer_names: Specific layer names to freeze, None for default embeddings
            
        Returns:
            Freeze operation results
        """
        
        if layer_names is None:
            # Default embedding layers to freeze
            layer_names = [
                'static_embedding',
                'observed_embedding', 
                'known_future_embedding'
            ]
        
        frozen_count = 0
        frozen_layers = []
        
        for name, param in self.model.named_parameters():
            for layer_name in layer_names:
                if layer_name in name:
                    param.requires_grad = False
                    self.frozen_parameters.add(name)
                    frozen_count += 1
                    frozen_layers.append(name)
                    break
        
        freeze_info = {
            'frozen_count': frozen_count,
            'frozen_layers': frozen_layers,
            'total_frozen': len(self.frozen_parameters)
        }
        
        self.freeze_history.append({
            'action': 'freeze',
            'info': freeze_info
        })
        
        logger.info(f"Froze {frozen_count} embedding parameters")
        return freeze_info
    
    def unfreeze_embeddings(self, layer_names: List[str] = None) -> Dict[str, Any]:
        """
        Unfreeze embedding layers
        
        Args:
            layer_names: Specific layer names to unfreeze, None for all frozen
            
        Returns:
            Unfreeze operation results  
        """
        
        unfrozen_count = 0
        unfrozen_layers = []
        
        if layer_names is None:
            # Unfreeze all frozen parameters
            for name, param in self.model.named_parameters():
                if name in self.frozen_parameters:
                    param.requires_grad = True
                    unfrozen_count += 1
                    unfrozen_layers.append(name)
            
            self.frozen_parameters.clear()
        else:
            # Unfreeze specific layers
            for name, param in self.model.named_parameters():
                for layer_name in layer_names:
                    if layer_name in name and name in self.frozen_parameters:
                        param.requires_grad = True
                        self.frozen_parameters.remove(name)
                        unfrozen_count += 1
                        unfrozen_layers.append(name)
                        break
        
        unfreeze_info = {
            'unfrozen_count': unfrozen_count,
            'unfrozen_layers': unfrozen_layers,
            'remaining_frozen': len(self.frozen_parameters)
        }
        
        self.freeze_history.append({
            'action': 'unfreeze',
            'info': unfreeze_info
        })
        
        logger.info(f"Unfroze {unfrozen_count} embedding parameters")
        return unfreeze_info
    
    def get_freeze_status(self) -> Dict[str, Any]:
        """Get current freeze status"""
        
        total_params = sum(1 for _ in self.model.parameters())
        frozen_params = len(self.frozen_parameters)
        
        return {
            'total_parameters': total_params,
            'frozen_parameters': frozen_params,
            'frozen_ratio': frozen_params / max(total_params, 1),
            'frozen_layer_names': list(self.frozen_parameters)
        }

class HorizonProgression:
    """Manage progression through prediction horizons"""
    
    def __init__(self, config: CurriculumConfig):
        self.config = config
        self.current_stage = 0
        self.progression_history = []
        
        # Generate progression schedule
        self.progression_schedule = self._generate_schedule()
    
    def _generate_schedule(self) -> List[Dict[str, Any]]:
        """Generate horizon progression schedule"""
        
        schedule = []
        
        if self.config.progression_strategy == 'linear':
            # Linear progression through horizons
            all_horizons = sorted(set(self.config.final_horizons))
            
            for i, horizon in enumerate(all_horizons):
                stage_horizons = all_horizons[:i+1]
                
                schedule.append({
                    'stage': i,
                    'horizons': stage_horizons,
                    'target_epochs': self.config.horizon_progression_epochs[min(i, len(self.config.horizon_progression_epochs)-1)] if self.config.horizon_progression_epochs else 20,
                    'description': f"Horizons {stage_horizons}"
                })
        
        elif self.config.progression_strategy == 'exponential':
            # Exponential growth in horizons
            sorted_horizons = sorted(self.config.final_horizons)
            
            stage = 0
            current_horizons = [sorted_horizons[0]]
            
            while len(current_horizons) < len(sorted_horizons):
                schedule.append({
                    'stage': stage,
                    'horizons': current_horizons.copy(),
                    'target_epochs': self.config.horizon_progression_epochs[min(stage, len(self.config.horizon_progression_epochs)-1)] if self.config.horizon_progression_epochs else 20,
                    'description': f"Horizons {current_horizons}"
                })
                
                # Add next horizon(s)
                next_count = min(len(current_horizons), len(sorted_horizons) - len(current_horizons))
                for _ in range(max(1, next_count)):
                    if len(current_horizons) < len(sorted_horizons):
                        current_horizons.append(sorted_horizons[len(current_horizons)])
                
                stage += 1
            
            # Final stage with all horizons
            schedule.append({
                'stage': stage,
                'horizons': sorted_horizons,
                'target_epochs': self.config.horizon_progression_epochs[-1] if self.config.horizon_progression_epochs else 30,
                'description': f"All horizons {sorted_horizons}"
            })
        
        else:  # adaptive or default
            # Simple progression from initial to final
            schedule.append({
                'stage': 0,
                'horizons': self.config.initial_horizons,
                'target_epochs': self.config.horizon_progression_epochs[0] if self.config.horizon_progression_epochs else 20,
                'description': f"Initial horizons {self.config.initial_horizons}"
            })
            
            schedule.append({
                'stage': 1,
                'horizons': self.config.final_horizons,
                'target_epochs': self.config.horizon_progression_epochs[-1] if self.config.horizon_progression_epochs else 40,
                'description': f"Final horizons {self.config.final_horizons}"
            })
        
        return schedule
    
    def get_current_horizons(self) -> List[int]:
        """Get current active horizons"""
        
        if self.current_stage < len(self.progression_schedule):
            return self.progression_schedule[self.current_stage]['horizons']
        else:
            return self.config.final_horizons
    
    def should_progress(self, epoch: int, stability_score: float, 
                       performance_improving: bool = True) -> bool:
        """
        Determine if should progress to next stage
        
        Args:
            epoch: Current epoch within stage
            stability_score: Stability score from monitor
            performance_improving: Whether performance is improving
            
        Returns:
            True if should progress to next stage
        """
        
        if self.current_stage >= len(self.progression_schedule) - 1:
            return False  # Already at final stage
        
        current_stage_info = self.progression_schedule[self.current_stage]
        target_epochs = current_stage_info['target_epochs']
        
        # Minimum epochs requirement
        if epoch < self.config.min_epochs_per_stage:
            return False
        
        # Maximum epochs limit
        if epoch >= self.config.max_epochs_per_stage:
            return True
        
        # Adaptive progression criteria
        if self.config.use_adaptive_scheduling:
            # Need both stability and target epochs
            stability_ready = stability_score > 0.7
            epoch_ready = epoch >= target_epochs * 0.5
            
            if stability_ready and epoch_ready and performance_improving:
                return True
        else:
            # Simple epoch-based progression
            if epoch >= target_epochs:
                return True
        
        return False
    
    def progress_to_next_stage(self) -> Dict[str, Any]:
        """Progress to next horizon stage"""
        
        if self.current_stage >= len(self.progression_schedule) - 1:
            logger.warning("Already at final stage, cannot progress further")
            return {'progressed': False, 'reason': 'already_at_final_stage'}
        
        old_stage = self.current_stage
        old_horizons = self.get_current_horizons()
        
        self.current_stage += 1
        new_horizons = self.get_current_horizons()
        
        progression_info = {
            'progressed': True,
            'old_stage': old_stage,
            'new_stage': self.current_stage,
            'old_horizons': old_horizons,
            'new_horizons': new_horizons,
            'added_horizons': [h for h in new_horizons if h not in old_horizons]
        }
        
        self.progression_history.append(progression_info)
        
        logger.info(f"Progressed to stage {self.current_stage}: {old_horizons} → {new_horizons}")
        return progression_info
    
    def get_progression_status(self) -> Dict[str, Any]:
        """Get current progression status"""
        
        current_horizons = self.get_current_horizons()
        total_stages = len(self.progression_schedule)
        
        return {
            'current_stage': self.current_stage,
            'total_stages': total_stages,
            'current_horizons': current_horizons,
            'progression_ratio': (self.current_stage + 1) / total_stages,
            'is_final_stage': self.current_stage >= total_stages - 1,
            'progression_history': self.progression_history
        }

class CurriculumScheduler:
    """Main curriculum learning scheduler"""
    
    def __init__(self, model: nn.Module, config: CurriculumConfig):
        self.config = config
        self.model = model
        
        # Components
        self.stability_monitor = StabilityMonitor(
            window_size=config.stability_window,
            threshold=config.stability_threshold
        )
        self.embedding_freezer = EmbeddingFreezer(model)
        self.horizon_progression = HorizonProgression(config)
        
        # State tracking
        self.current_epoch_in_stage = 0
        self.stable_epochs_count = 0
        self.embeddings_frozen = False
        self.curriculum_history = []
    
    def step(self, epoch: int, loss: float, performance_metrics: Dict[str, float] = None) -> Dict[str, Any]:
        """
        Execute one curriculum step
        
        Args:
            epoch: Global epoch number
            loss: Current validation loss
            performance_metrics: Additional performance metrics
            
        Returns:
            Curriculum step results
        """
        
        step_results = {
            'epoch': epoch,
            'loss': loss,
            'actions_taken': []
        }
        
        # Update stability monitoring
        stability_metrics = self.stability_monitor.update(loss)
        step_results['stability'] = stability_metrics
        
        # Track stability
        if stability_metrics['is_stable']:
            self.stable_epochs_count += 1
        else:
            self.stable_epochs_count = 0
        
        # Embedding freezing logic
        if (self.config.freeze_embeddings_after_stable and 
            not self.embeddings_frozen and
            self.stable_epochs_count >= self.config.freeze_threshold_epochs):
            
            freeze_info = self.embedding_freezer.freeze_embeddings()
            self.embeddings_frozen = True
            step_results['actions_taken'].append(('freeze_embeddings', freeze_info))
        
        # Horizon progression logic
        stability_score = self.stability_monitor.get_stability_score()
        performance_improving = stability_metrics['trend'] in ['improving', 'stable']
        
        should_progress = self.horizon_progression.should_progress(
            self.current_epoch_in_stage, stability_score, performance_improving
        )
        
        if should_progress:
            # Unfreeze embeddings if configured
            if self.config.unfreeze_on_new_horizon and self.embeddings_frozen:
                unfreeze_info = self.embedding_freezer.unfreeze_embeddings()
                self.embeddings_frozen = False
                step_results['actions_taken'].append(('unfreeze_embeddings', unfreeze_info))
            
            # Progress to next stage
            progression_info = self.horizon_progression.progress_to_next_stage()
            if progression_info['progressed']:
                self.current_epoch_in_stage = 0
                self.stable_epochs_count = 0
                step_results['actions_taken'].append(('progress_horizon', progression_info))
        
        # Update counters
        self.current_epoch_in_stage += 1
        
        # Status updates
        step_results['horizon_status'] = self.horizon_progression.get_progression_status()
        step_results['freeze_status'] = self.embedding_freezer.get_freeze_status()
        step_results['stability_score'] = stability_score
        step_results['stable_epochs'] = self.stable_epochs_count
        step_results['epoch_in_stage'] = self.current_epoch_in_stage
        
        # Store in history
        self.curriculum_history.append(step_results)
        
        return step_results
    
    def get_current_horizons(self) -> List[int]:
        """Get currently active prediction horizons"""
        return self.horizon_progression.get_current_horizons()
    
    def get_curriculum_summary(self) -> Dict[str, Any]:
        """Get comprehensive curriculum status summary"""
        
        return {
            'config': self.config,
            'current_horizons': self.get_current_horizons(),
            'progression_status': self.horizon_progression.get_progression_status(),
            'freeze_status': self.embedding_freezer.get_freeze_status(),
            'stability_score': self.stability_monitor.get_stability_score(),
            'stable_epochs': self.stable_epochs_count,
            'epoch_in_stage': self.current_epoch_in_stage,
            'total_curriculum_steps': len(self.curriculum_history)
        }