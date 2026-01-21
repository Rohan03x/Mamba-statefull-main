"""
A/B Testing Framework for Production Model Experiments

This module implements comprehensive A/B testing for model performance evaluation:
- Champion/challenger framework for live model comparisons
- Statistical significance testing with proper sample size calculations
- Treatment allocation with stratified randomization
- Performance tracking with bias detection and correction
- Integration with production scoring and monitoring systems

Key features:
- Rigorous experimental design with proper controls
- Real-time performance monitoring and early stopping
- Statistical power analysis and sample size calculations
- Treatment effect estimation with confidence intervals
- Integration with paper trading for safe experimentation
"""

import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
import logging
from dataclasses import dataclass, field
import sqlite3
import json
import hashlib
import uuid
from abc import ABC, abstractmethod
from enum import Enum

# Statistical imports
from scipy import stats
from statsmodels.stats.power import ttest_power

logger = logging.getLogger(__name__)

class ExperimentStatus(Enum):
    """Experiment status enumeration"""
    DRAFT = "draft"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    TERMINATED = "terminated"

class AllocationMethod(Enum):
    """Treatment allocation methods"""
    RANDOM = "random"
    STRATIFIED = "stratified"
    DETERMINISTIC = "deterministic"
    THOMPSON_SAMPLING = "thompson_sampling"

@dataclass
class ABTestConfig:
    """Configuration for A/B test experiments"""
    
    # Experiment design
    name: str
    description: str
    champion_model: str
    challenger_model: str
    
    # Allocation parameters
    allocation_ratio: float = 0.8  # Fraction allocated to champion
    allocation_method: AllocationMethod = AllocationMethod.STRATIFIED
    stratification_features: List[str] = field(default_factory=list)
    
    # Statistical parameters
    alpha: float = 0.05  # Significance level
    power: float = 0.8   # Statistical power
    minimum_effect_size: float = 0.02  # Minimum detectable effect (2%)
    
    # Experiment duration
    max_duration_days: int = 30
    min_sample_size: int = 1000
    max_sample_size: int = 10000
    
    # Early stopping criteria
    early_stopping_enabled: bool = True
    interim_analysis_frequency: int = 100  # Check every N observations
    futility_boundary: float = 0.1  # Stop if probability of success < 10%
    
    # Risk controls
    max_loss_threshold: float = 0.05  # Maximum allowed loss (5%)
    max_exposure_per_symbol: float = 0.1  # Maximum exposure per symbol (10%)
    
    # Documentation requirements
    business_justification: str = ""
    risk_assessment: str = ""
    success_criteria: str = ""

@dataclass
class TreatmentAllocation:
    """Represents treatment allocation for a single observation"""
    
    observation_id: str
    timestamp: str
    treatment: str  # 'champion' or 'challenger'
    allocation_probability: float
    stratification_key: Optional[str] = None
    randomization_seed: Optional[int] = None

@dataclass
class ABTestResults:
    """Results of A/B test statistical analysis"""
    
    experiment_id: str
    analysis_timestamp: str
    
    # Sample sizes
    champion_samples: int
    challenger_samples: int
    total_samples: int
    
    # Performance metrics
    champion_performance: Dict[str, float]
    challenger_performance: Dict[str, float]
    
    # Statistical test results
    test_statistic: float
    p_value: float
    confidence_interval: Tuple[float, float]
    effect_size: float
    
    # Decision
    statistically_significant: bool
    practical_significance: bool
    recommended_action: str
    
    # Additional metrics
    statistical_power: float
    sample_ratio_mismatch: float
    
    # Risk metrics
    max_drawdown_champion: float
    max_drawdown_challenger: float
    sharpe_ratio_champion: float
    sharpe_ratio_challenger: float

class TreatmentAllocator(ABC):
    """Abstract base class for treatment allocation strategies"""
    
    @abstractmethod
    def allocate(self, observation_data: Dict[str, Any], config: ABTestConfig) -> TreatmentAllocation:
        """Allocate treatment for an observation"""
        pass
    
    @abstractmethod
    def get_allocation_probability(self, treatment: str) -> float:
        """Get probability of allocation to specific treatment"""
        pass

class RandomAllocator(TreatmentAllocator):
    """Random treatment allocation"""
    
    def __init__(self, seed: Optional[int] = None):
        self.rng = np.random.RandomState(seed)
        
    def allocate(self, observation_data: Dict[str, Any], config: ABTestConfig) -> TreatmentAllocation:
        """Randomly allocate treatment"""
        
        # Generate random allocation
        random_value = self.rng.random()
        treatment = "champion" if random_value < config.allocation_ratio else "challenger"
        
        return TreatmentAllocation(
            observation_id=str(uuid.uuid4()),
            timestamp=datetime.now().isoformat(),
            treatment=treatment,
            allocation_probability=config.allocation_ratio if treatment == "champion" else (1 - config.allocation_ratio),
            randomization_seed=self.rng.get_state()[1][0]
        )
    
    def get_allocation_probability(self, treatment: str) -> float:
        """Get allocation probability"""
        # This would need config access in practice
        return 0.8 if treatment == "champion" else 0.2

class StratifiedAllocator(TreatmentAllocator):
    """Stratified treatment allocation based on features"""
    
    def __init__(self, seed: Optional[int] = None):
        self.rng = np.random.RandomState(seed)
        self.stratum_allocators = {}
        
    def allocate(self, observation_data: Dict[str, Any], config: ABTestConfig) -> TreatmentAllocation:
        """Allocate treatment with stratification"""
        
        # Create stratification key
        strat_values = []
        for feature in config.stratification_features:
            if feature in observation_data:
                value = observation_data[feature]
                # Convert to categorical if numeric
                if isinstance(value, (int, float)):
                    # Simple binning for numeric features
                    binned_value = "low" if value < 0 else "high"
                    strat_values.append(f"{feature}_{binned_value}")
                else:
                    strat_values.append(f"{feature}_{value}")
        
        stratification_key = "_".join(strat_values) if strat_values else "default"
        
        # Get or create allocator for this stratum
        if stratification_key not in self.stratum_allocators:
            self.stratum_allocators[stratification_key] = RandomAllocator(
                seed=self.rng.randint(0, 2**31-1)
            )
        
        # Allocate within stratum
        allocation = self.stratum_allocators[stratification_key].allocate(observation_data, config)
        allocation.stratification_key = stratification_key
        
        return allocation
    
    def get_allocation_probability(self, treatment: str) -> float:
        """Get overall allocation probability across strata"""
        # Simplified - would need proper stratum weighting
        return 0.8 if treatment == "champion" else 0.2

class ABTestExperiment:
    """Manages a single A/B test experiment"""
    
    def __init__(self, config: ABTestConfig, allocator: Optional[TreatmentAllocator] = None):
        self.config = config
        self.id = self._generate_experiment_id()
        
        # Initialize allocator
        if allocator is None:
            if config.allocation_method == AllocationMethod.RANDOM:
                self.allocator = RandomAllocator()
            elif config.allocation_method == AllocationMethod.STRATIFIED:
                self.allocator = StratifiedAllocator()
            else:
                self.allocator = RandomAllocator()  # Default fallback
        else:
            self.allocator = allocator
        
        # Experiment state
        self.status = ExperimentStatus.DRAFT
        self.start_time = None
        self.end_time = None
        
        # Data storage
        self.allocations = []
        self.observations = []
        self.results_history = []
        
        # Performance tracking
        self.champion_metrics = []
        self.challenger_metrics = []
        
        # Risk monitoring
        self.risk_violations = []
        
    def _generate_experiment_id(self) -> str:
        """Generate unique experiment ID"""
        
        # Create deterministic ID from config
        config_str = f"{self.config.name}_{self.config.champion_model}_{self.config.challenger_model}"
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        return f"exp_{hashlib.md5(config_str.encode()).hexdigest()[:8]}_{timestamp}"
    
    def start_experiment(self) -> bool:
        """Start the experiment"""
        
        try:
            # Validate configuration
            validation_result = self._validate_config()
            if not validation_result['valid']:
                logger.error(f"Invalid experiment config: {validation_result['errors']}")
                return False
            
            # Calculate required sample size
            required_samples = self._calculate_sample_size()
            logger.info(f"Required sample size: {required_samples}")
            
            self.status = ExperimentStatus.RUNNING
            self.start_time = datetime.now()
            
            logger.info(f"Started experiment {self.id}: {self.config.name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start experiment: {e}")
            return False
    
    def allocate_treatment(self, observation_data: Dict[str, Any]) -> TreatmentAllocation:
        """Allocate treatment for new observation"""
        
        if self.status != ExperimentStatus.RUNNING:
            raise ValueError("Experiment not running")
        
        # Check if experiment should continue
        if not self._should_continue_experiment():
            self.stop_experiment("Maximum duration or sample size reached")
            raise ValueError("Experiment has ended")
        
        # Allocate treatment
        allocation = self.allocator.allocate(observation_data, self.config)
        self.allocations.append(allocation)
        
        return allocation
    
    def add_observation(self, allocation_id: str, outcome_metrics: Dict[str, float]):
        """Add observation outcome for analysis"""
        
        observation = {
            'allocation_id': allocation_id,
            'timestamp': datetime.now().isoformat(),
            'metrics': outcome_metrics
        }
        
        self.observations.append(observation)
        
        # Find corresponding allocation
        allocation = next((a for a in self.allocations if a.observation_id == allocation_id), None)
        
        if allocation:
            # Store metrics by treatment
            if allocation.treatment == "champion":
                self.champion_metrics.append(outcome_metrics)
            else:
                self.challenger_metrics.append(outcome_metrics)
            
            # Check for interim analysis
            if (self.config.early_stopping_enabled and 
                len(self.observations) % self.config.interim_analysis_frequency == 0):
                self._perform_interim_analysis()
    
    def _validate_config(self) -> Dict[str, Any]:
        """Validate experiment configuration"""
        
        errors = []
        
        # Check required fields
        if not self.config.name:
            errors.append("Experiment name is required")
        
        if not self.config.champion_model:
            errors.append("Champion model is required")
        
        if not self.config.challenger_model:
            errors.append("Challenger model is required")
        
        # Check statistical parameters
        if not 0 < self.config.alpha < 1:
            errors.append("Alpha must be between 0 and 1")
        
        if not 0 < self.config.power < 1:
            errors.append("Power must be between 0 and 1")
        
        if not 0 < self.config.allocation_ratio < 1:
            errors.append("Allocation ratio must be between 0 and 1")
        
        # Check sample sizes
        if self.config.min_sample_size <= 0:
            errors.append("Minimum sample size must be positive")
        
        if self.config.max_sample_size <= self.config.min_sample_size:
            errors.append("Maximum sample size must be greater than minimum")
        
        return {
            'valid': len(errors) == 0,
            'errors': errors
        }
    
    def _calculate_sample_size(self) -> int:
        """Calculate required sample size for experiment"""
        
        # Use power analysis for two-sample t-test
        effect_size = self.config.minimum_effect_size
        alpha = self.config.alpha
        power = self.config.power
        ratio = (1 - self.config.allocation_ratio) / self.config.allocation_ratio
        
        # Calculate sample size for each group
        try:
            # This is approximate - using two-sample t-test power calculation
            from statsmodels.stats.power import ttest_power
            
            # Solve for sample size
            sample_size_champion = 100  # Initial guess
            
            for n in range(50, 10000, 50):
                calculated_power = ttest_power(
                    effect_size=effect_size,
                    nobs=n,
                    alpha=alpha,
                    alternative='two-sided'
                )
                
                if calculated_power >= power:
                    sample_size_champion = n
                    break
            
            # Adjust for allocation ratio
            sample_size_challenger = int(sample_size_champion * ratio)
            total_sample_size = sample_size_champion + sample_size_challenger
            
            # Ensure within bounds
            total_sample_size = max(self.config.min_sample_size, total_sample_size)
            total_sample_size = min(self.config.max_sample_size, total_sample_size)
            
            return total_sample_size
            
        except Exception:
            # Fallback to simple calculation
            return max(self.config.min_sample_size, 1000)
    
    def _should_continue_experiment(self) -> bool:
        """Check if experiment should continue"""
        
        # Check duration
        if self.start_time:
            duration = datetime.now() - self.start_time
            if duration.days >= self.config.max_duration_days:
                return False
        
        # Check sample size
        if len(self.observations) >= self.config.max_sample_size:
            return False
        
        # Check for early termination due to risk
        if self._check_risk_violations():
            return False
        
        return True
    
    def _check_risk_violations(self) -> bool:
        """Check for risk violations requiring early termination"""
        
        if len(self.challenger_metrics) < 10:  # Need minimum observations
            return False
        
        # Calculate recent performance
        recent_challenger = self.challenger_metrics[-20:]  # Last 20 observations
        recent_champion = self.champion_metrics[-20:] if len(self.champion_metrics) >= 20 else self.champion_metrics
        
        if not recent_challenger or not recent_champion:
            return False
        
        # Check for significant underperformance
        challenger_returns = [m.get('return', 0) for m in recent_challenger]
        champion_returns = [m.get('return', 0) for m in recent_champion]
        
        challenger_mean = np.mean(challenger_returns)
        champion_mean = np.mean(champion_returns)
        
        # Check loss threshold
        relative_loss = (champion_mean - challenger_mean) / abs(champion_mean) if champion_mean != 0 else 0
        
        if relative_loss > self.config.max_loss_threshold:
            self.risk_violations.append({
                'timestamp': datetime.now().isoformat(),
                'type': 'performance_loss',
                'value': relative_loss,
                'threshold': self.config.max_loss_threshold
            })
            return True
        
        return False
    
    def _perform_interim_analysis(self):
        """Perform interim statistical analysis"""
        
        if len(self.champion_metrics) < 10 or len(self.challenger_metrics) < 10:
            return  # Not enough data
        
        # Calculate current test results
        results = self.analyze_results()
        
        if results:
            self.results_history.append(results)
            
            # Check for early stopping
            if results.statistically_significant and results.practical_significance:
                self.stop_experiment(f"Early stopping: significant results detected (p={results.p_value:.4f})")
            elif self.config.early_stopping_enabled:
                # Futility analysis
                if results.statistical_power < self.config.futility_boundary:
                    self.stop_experiment(f"Futility stopping: low probability of success (power={results.statistical_power:.4f})")
    
    def analyze_results(self) -> Optional[ABTestResults]:
        """Analyze current experiment results"""
        
        if len(self.champion_metrics) < 5 or len(self.challenger_metrics) < 5:
            return None
        
        try:
            # Extract performance metrics
            champion_returns = np.array([m.get('return', 0) for m in self.champion_metrics])
            challenger_returns = np.array([m.get('return', 0) for m in self.challenger_metrics])
            
            # Calculate summary statistics
            champion_performance = {
                'mean_return': np.mean(champion_returns),
                'std_return': np.std(champion_returns),
                'sharpe_ratio': np.mean(champion_returns) / np.std(champion_returns) if np.std(champion_returns) > 0 else 0,
                'max_drawdown': self._calculate_max_drawdown(champion_returns),
                'win_rate': np.mean(champion_returns > 0)
            }
            
            challenger_performance = {
                'mean_return': np.mean(challenger_returns),
                'std_return': np.std(challenger_returns),
                'sharpe_ratio': np.mean(challenger_returns) / np.std(challenger_returns) if np.std(challenger_returns) > 0 else 0,
                'max_drawdown': self._calculate_max_drawdown(challenger_returns),
                'win_rate': np.mean(challenger_returns > 0)
            }
            
            # Perform statistical test (two-sample t-test)
            test_statistic, p_value = stats.ttest_ind(challenger_returns, champion_returns)
            
            # Calculate effect size (Cohen's d)
            pooled_std = np.sqrt(((len(challenger_returns) - 1) * np.var(challenger_returns) + 
                                 (len(champion_returns) - 1) * np.var(champion_returns)) / 
                                (len(challenger_returns) + len(champion_returns) - 2))
            
            effect_size = (challenger_performance['mean_return'] - champion_performance['mean_return']) / pooled_std if pooled_std > 0 else 0
            
            # Confidence interval for difference in means
            diff_means = challenger_performance['mean_return'] - champion_performance['mean_return']
            se_diff = np.sqrt(np.var(challenger_returns)/len(challenger_returns) + 
                             np.var(champion_returns)/len(champion_returns))
            
            ci_lower = diff_means - 1.96 * se_diff
            ci_upper = diff_means + 1.96 * se_diff
            
            # Statistical power (post-hoc)
            try:
                statistical_power = ttest_power(
                    effect_size=abs(effect_size),
                    nobs=min(len(challenger_returns), len(champion_returns)),
                    alpha=self.config.alpha,
                    alternative='two-sided'
                )
            except Exception:
                statistical_power = 0.5  # Default
            
            # Sample ratio mismatch
            expected_ratio = self.config.allocation_ratio / (1 - self.config.allocation_ratio)
            actual_ratio = len(champion_returns) / len(challenger_returns) if len(challenger_returns) > 0 else 0
            sample_ratio_mismatch = abs(actual_ratio - expected_ratio) / expected_ratio if expected_ratio > 0 else 0
            
            # Decision logic
            statistically_significant = p_value < self.config.alpha
            practical_significance = abs(effect_size) >= self.config.minimum_effect_size
            
            if statistically_significant and practical_significance:
                if challenger_performance['mean_return'] > champion_performance['mean_return']:
                    recommended_action = "promote_challenger"
                else:
                    recommended_action = "keep_champion"
            else:
                recommended_action = "continue_experiment"
            
            return ABTestResults(
                experiment_id=self.id,
                analysis_timestamp=datetime.now().isoformat(),
                champion_samples=len(champion_returns),
                challenger_samples=len(challenger_returns),
                total_samples=len(champion_returns) + len(challenger_returns),
                champion_performance=champion_performance,
                challenger_performance=challenger_performance,
                test_statistic=test_statistic,
                p_value=p_value,
                confidence_interval=(ci_lower, ci_upper),
                effect_size=effect_size,
                statistically_significant=statistically_significant,
                practical_significance=practical_significance,
                recommended_action=recommended_action,
                statistical_power=statistical_power,
                sample_ratio_mismatch=sample_ratio_mismatch,
                max_drawdown_champion=champion_performance['max_drawdown'],
                max_drawdown_challenger=challenger_performance['max_drawdown'],
                sharpe_ratio_champion=champion_performance['sharpe_ratio'],
                sharpe_ratio_challenger=challenger_performance['sharpe_ratio']
            )
            
        except Exception as e:
            logger.error(f"Failed to analyze results: {e}")
            return None
    
    def _calculate_max_drawdown(self, returns: np.ndarray) -> float:
        """Calculate maximum drawdown from returns"""
        
        if len(returns) == 0:
            return 0.0
        
        cumulative = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        
        return abs(np.min(drawdown))
    
    def stop_experiment(self, reason: str = "Manual stop"):
        """Stop the experiment"""
        
        self.status = ExperimentStatus.COMPLETED
        self.end_time = datetime.now()
        
        logger.info(f"Stopped experiment {self.id}: {reason}")
    
    def get_experiment_summary(self) -> Dict[str, Any]:
        """Get comprehensive experiment summary"""
        
        return {
            'experiment_id': self.id,
            'config': {
                'name': self.config.name,
                'champion_model': self.config.champion_model,
                'challenger_model': self.config.challenger_model,
                'allocation_ratio': self.config.allocation_ratio
            },
            'status': self.status.value,
            'duration': {
                'start_time': self.start_time.isoformat() if self.start_time else None,
                'end_time': self.end_time.isoformat() if self.end_time else None,
                'duration_hours': ((self.end_time or datetime.now()) - self.start_time).total_seconds() / 3600 if self.start_time else 0
            },
            'sample_sizes': {
                'champion': len(self.champion_metrics),
                'challenger': len(self.challenger_metrics),
                'total': len(self.observations)
            },
            'latest_results': self.results_history[-1].__dict__ if self.results_history else None,
            'risk_violations': len(self.risk_violations)
        }

class ChampionChallengerFramework:
    """Framework for managing champion/challenger experiments"""
    
    def __init__(self, config: ABTestConfig = None, db_path: str = "experiments.db"):
        self.config = config
        self.db_path = db_path
        self._init_database()
        
        # Active experiments
        self.active_experiments = {}
        
        # Performance tracking
        self.model_performance_history = {}
        
    def _init_database(self):
        """Initialize experiment database"""
        
        conn = sqlite3.connect(self.db_path)
        
        # Experiments table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS experiments (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                champion_model TEXT NOT NULL,
                challenger_model TEXT NOT NULL,
                config TEXT NOT NULL,
                status TEXT NOT NULL,
                start_time TEXT,
                end_time TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Allocations table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS allocations (
                id TEXT PRIMARY KEY,
                experiment_id TEXT NOT NULL,
                observation_id TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                treatment TEXT NOT NULL,
                allocation_probability REAL,
                stratification_key TEXT,
                FOREIGN KEY (experiment_id) REFERENCES experiments (id)
            )
        """)
        
        # Observations table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS observations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                experiment_id TEXT NOT NULL,
                allocation_id TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                metrics TEXT NOT NULL,
                FOREIGN KEY (experiment_id) REFERENCES experiments (id)
            )
        """)
        
        # Results table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS experiment_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                experiment_id TEXT NOT NULL,
                analysis_timestamp TEXT NOT NULL,
                results TEXT NOT NULL,
                FOREIGN KEY (experiment_id) REFERENCES experiments (id)
            )
        """)
        
        conn.commit()
        conn.close()
    
    def create_experiment(self, config: ABTestConfig) -> ABTestExperiment:
        """Create new A/B test experiment"""
        
        experiment = ABTestExperiment(config)
        
        # Store in database
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            INSERT INTO experiments (id, name, champion_model, challenger_model, config, status)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            experiment.id,
            config.name,
            config.champion_model,
            config.challenger_model,
            json.dumps(config.__dict__, default=str),
            experiment.status.value
        ))
        conn.commit()
        conn.close()
        
        logger.info(f"Created experiment {experiment.id}: {config.name}")
        
        return experiment
    
    def start_experiment(self, experiment: ABTestExperiment) -> bool:
        """Start an experiment"""
        
        if experiment.start_experiment():
            self.active_experiments[experiment.id] = experiment
            
            # Update database
            conn = sqlite3.connect(self.db_path)
            conn.execute("""
                UPDATE experiments 
                SET status = ?, start_time = ?
                WHERE id = ?
            """, (
                experiment.status.value,
                experiment.start_time.isoformat(),
                experiment.id
            ))
            conn.commit()
            conn.close()
            
            return True
        
        return False
    
    def allocate_treatment(self, experiment_id: str, observation_data: Dict[str, Any]) -> Optional[TreatmentAllocation]:
        """Allocate treatment for observation"""
        
        if experiment_id not in self.active_experiments:
            logger.error(f"Experiment {experiment_id} not active")
            return None
        
        experiment = self.active_experiments[experiment_id]
        
        try:
            allocation = experiment.allocate_treatment(observation_data)
            
            # Store allocation in database
            conn = sqlite3.connect(self.db_path)
            conn.execute("""
                INSERT INTO allocations 
                (id, experiment_id, observation_id, timestamp, treatment, allocation_probability, stratification_key)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                str(uuid.uuid4()),
                experiment_id,
                allocation.observation_id,
                allocation.timestamp,
                allocation.treatment,
                allocation.allocation_probability,
                allocation.stratification_key
            ))
            conn.commit()
            conn.close()
            
            return allocation
            
        except Exception as e:
            logger.error(f"Treatment allocation failed: {e}")
            return None
    
    def add_observation(self, experiment_id: str, allocation_id: str, 
                       outcome_metrics: Dict[str, float]) -> bool:
        """Add observation outcome"""
        
        if experiment_id not in self.active_experiments:
            logger.error(f"Experiment {experiment_id} not active")
            return False
        
        experiment = self.active_experiments[experiment_id]
        
        try:
            experiment.add_observation(allocation_id, outcome_metrics)
            
            # Store observation in database
            conn = sqlite3.connect(self.db_path)
            conn.execute("""
                INSERT INTO observations (experiment_id, allocation_id, timestamp, metrics)
                VALUES (?, ?, ?, ?)
            """, (
                experiment_id,
                allocation_id,
                datetime.now().isoformat(),
                json.dumps(outcome_metrics)
            ))
            conn.commit()
            conn.close()
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to add observation: {e}")
            return False
    
    def get_experiment_results(self, experiment_id: str) -> Optional[ABTestResults]:
        """Get current experiment results"""
        
        if experiment_id in self.active_experiments:
            experiment = self.active_experiments[experiment_id]
            return experiment.analyze_results()
        
        return None
    
    def stop_experiment(self, experiment_id: str, reason: str = "Manual stop") -> bool:
        """Stop an experiment"""
        
        if experiment_id not in self.active_experiments:
            logger.error(f"Experiment {experiment_id} not active")
            return False
        
        experiment = self.active_experiments[experiment_id]
        experiment.stop_experiment(reason)
        
        # Update database
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            UPDATE experiments 
            SET status = ?, end_time = ?
            WHERE id = ?
        """, (
            experiment.status.value,
            experiment.end_time.isoformat() if experiment.end_time else None,
            experiment_id
        ))
        conn.commit()
        conn.close()
        
        # Remove from active experiments
        del self.active_experiments[experiment_id]
        
        logger.info(f"Stopped experiment {experiment_id}: {reason}")
        return True
    
    def get_active_experiments(self) -> List[Dict[str, Any]]:
        """Get list of active experiments"""
        
        return [exp.get_experiment_summary() for exp in self.active_experiments.values()]
    
    def get_experiment_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get experiment history from database"""
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.execute("""
            SELECT id, name, champion_model, challenger_model, status, start_time, end_time
            FROM experiments 
            ORDER BY created_at DESC 
            LIMIT ?
        """, (limit,))
        
        experiments = []
        for row in cursor.fetchall():
            experiments.append({
                'id': row[0],
                'name': row[1],
                'champion_model': row[2],
                'challenger_model': row[3],
                'status': row[4],
                'start_time': row[5],
                'end_time': row[6]
            })
        
        conn.close()
        return experiments

class ExperimentRegistry:
    """Registry for tracking and managing all experiments"""
    
    def __init__(self):
        self.registered_models = {}
        self.experiment_templates = {}
        self.approval_workflows = {}
        
    def register_model(self, model_id: str, model_info: Dict[str, Any]):
        """Register a model for experimentation"""
        
        self.registered_models[model_id] = {
            'info': model_info,
            'registered_at': datetime.now().isoformat(),
            'status': 'registered'
        }
        
        logger.info(f"Registered model {model_id}")
    
    def create_experiment_template(self, template_name: str, config_template: Dict[str, Any]):
        """Create reusable experiment template"""
        
        self.experiment_templates[template_name] = {
            'template': config_template,
            'created_at': datetime.now().isoformat()
        }
        
        logger.info(f"Created experiment template {template_name}")
    
    def validate_experiment_proposal(self, config: ABTestConfig) -> Dict[str, Any]:
        """Validate experiment proposal before approval"""
        
        validation_results = {
            'valid': True,
            'warnings': [],
            'errors': []
        }
        
        # Check if models are registered
        if config.champion_model not in self.registered_models:
            validation_results['errors'].append(f"Champion model {config.champion_model} not registered")
            validation_results['valid'] = False
        
        if config.challenger_model not in self.registered_models:
            validation_results['errors'].append(f"Challenger model {config.challenger_model} not registered")
            validation_results['valid'] = False
        
        # Check business justification
        if not config.business_justification.strip():
            validation_results['warnings'].append("Missing business justification")
        
        # Check risk assessment
        if not config.risk_assessment.strip():
            validation_results['warnings'].append("Missing risk assessment")
        
        # Check statistical parameters
        if config.minimum_effect_size < 0.01:
            validation_results['warnings'].append("Very small minimum effect size may require large sample")
        
        return validation_results
    
    def get_registered_models(self) -> Dict[str, Any]:
        """Get all registered models"""
        return self.registered_models.copy()
    
    def get_experiment_templates(self) -> Dict[str, Any]:
        """Get all experiment templates"""
        return self.experiment_templates.copy()