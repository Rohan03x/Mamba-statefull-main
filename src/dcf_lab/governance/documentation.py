"""
Regulatory Documentation Framework

This module implements comprehensive documentation and audit trail systems for regulatory compliance:
- Experiment and model versioning with complete lineage tracking
- Fold boundaries, embargo periods, and cross-validation documentation
- Parameter dumps with cryptographic hashing for integrity verification
- Seed tracking and reproducibility guarantees
- Data hash verification and change detection
- Regulatory reporting templates and automated compliance checks

Key features:
- Complete audit trails for all model development activities
- Cryptographic integrity verification of all artifacts
- Automated regulatory report generation
- Model lineage tracking with dependency graphs
- Data provenance and transformation documentation
- Compliance validation against industry standards (MiFID II, SR 11-7, etc.)
"""

import json
import hashlib
import uuid
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any
import pandas as pd
import numpy as np
import pickle
import sqlite3
import logging
from dataclasses import dataclass, field, asdict
from pathlib import Path
from enum import Enum
import os

logger = logging.getLogger(__name__)

class DocumentationType(Enum):
    """Types of regulatory documents"""
    MODEL_CARD = "model_card"
    EXPERIMENT_LOG = "experiment_log"
    DATA_LINEAGE = "data_lineage"
    VALIDATION_REPORT = "validation_report"
    RISK_ASSESSMENT = "risk_assessment"
    PERFORMANCE_MONITORING = "performance_monitoring"
    AUDIT_TRAIL = "audit_trail"
    COMPLIANCE_REPORT = "compliance_report"

class ComplianceFramework(Enum):
    """Regulatory compliance frameworks"""
    MIFID_II = "mifid_ii"
    SR_11_7 = "sr_11_7"  # Federal Reserve SR 11-7
    BASEL_III = "basel_iii"
    GDPR = "gdpr"
    SOX = "sarbanes_oxley"
    INTERNAL = "internal"

@dataclass
class DataHash:
    """Data integrity hash with metadata"""
    
    hash_value: str
    algorithm: str = "sha256"
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    size_bytes: int = 0
    row_count: Optional[int] = None
    column_count: Optional[int] = None
    schema_hash: Optional[str] = None

@dataclass
class FoldBoundary:
    """Cross-validation fold boundary documentation"""
    
    fold_id: str
    train_start_date: str
    train_end_date: str
    validation_start_date: str
    validation_end_date: str
    test_start_date: Optional[str] = None
    test_end_date: Optional[str] = None
    embargo_days: int = 0
    samples_train: int = 0
    samples_validation: int = 0
    samples_test: int = 0
    data_hash: Optional[DataHash] = None

@dataclass
class ParameterSnapshot:
    """Complete parameter state snapshot"""
    
    snapshot_id: str
    timestamp: str
    model_type: str
    hyperparameters: Dict[str, Any]
    feature_parameters: Dict[str, Any]
    preprocessing_parameters: Dict[str, Any]
    training_parameters: Dict[str, Any]
    parameter_hash: str
    random_seed: Optional[int] = None
    environment_hash: Optional[str] = None

@dataclass
class ExperimentMetadata:
    """Complete experiment documentation"""
    
    experiment_id: str
    experiment_name: str
    created_timestamp: str
    status: str  # running, completed, failed, cancelled
    created_by: str
    last_modified_by: str
    last_modified_timestamp: str
    
    # Experiment design
    objective: str = ""
    hypothesis: str = ""
    methodology: str = ""
    success_criteria: List[str] = field(default_factory=list)
    
    # Data documentation
    datasets_used: List[str] = field(default_factory=list)
    data_hashes: List[DataHash] = field(default_factory=list)
    fold_boundaries: List[FoldBoundary] = field(default_factory=list)
    
    # Model documentation
    model_versions: List[str] = field(default_factory=list)
    parameter_snapshots: List[ParameterSnapshot] = field(default_factory=list)
    
    # Results documentation
    metrics: Dict[str, float] = field(default_factory=dict)
    validation_results: Dict[str, Any] = field(default_factory=dict)
    statistical_tests: Dict[str, Any] = field(default_factory=dict)
    
    # Compliance
    compliance_frameworks: List[ComplianceFramework] = field(default_factory=list)
    approval_status: str = "pending"
    approver: Optional[str] = None
    approval_timestamp: Optional[str] = None
    
    # Audit trail
    change_log: List[Dict[str, Any]] = field(default_factory=list)

@dataclass
class ModelCard:
    """ML model card for regulatory documentation"""
    
    model_id: str
    model_name: str
    model_version: str
    created_timestamp: str
    
    # Model details
    model_type: str
    model_architecture: str
    training_data_description: str
    training_procedure: str
    evaluation_procedure: str
    
    # Performance metrics
    training_metrics: Dict[str, float]
    validation_metrics: Dict[str, float]
    test_metrics: Dict[str, float]
    out_of_sample_metrics: Dict[str, float]
    
    # Bias and fairness
    bias_assessment: Dict[str, Any]
    fairness_metrics: Dict[str, float]
    ethical_considerations: List[str]
    
    # Limitations and risks
    known_limitations: List[str]
    risk_assessment: Dict[str, Any]
    failure_modes: List[str]
    monitoring_recommendations: List[str]
    
    # Usage guidelines
    intended_use: str
    prohibited_uses: List[str]
    deployment_constraints: List[str]
    maintenance_requirements: List[str]
    
    # Regulatory compliance
    regulatory_approvals: List[str]
    compliance_statements: Dict[str, str]
    audit_information: Dict[str, Any]
    
    # Technical specifications
    feature_importance: Dict[str, float]
    feature_descriptions: Dict[str, str]
    data_requirements: Dict[str, Any]
    computational_requirements: Dict[str, Any]

class DataHasher:
    """Utility for computing and verifying data hashes"""
    
    @staticmethod
    def hash_dataframe(df: pd.DataFrame, algorithm: str = "sha256") -> DataHash:
        """Compute hash of pandas DataFrame"""
        
        # Convert DataFrame to consistent byte representation
        df_bytes = pickle.dumps(df.sort_index().sort_index(axis=1))
        
        hasher = hashlib.new(algorithm)
        hasher.update(df_bytes)
        hash_value = hasher.hexdigest()
        
        # Compute schema hash
        schema_str = json.dumps({
            'columns': list(df.columns),
            'dtypes': {col: str(dtype) for col, dtype in df.dtypes.items()},
            'shape': df.shape
        }, sort_keys=True)
        
        schema_hasher = hashlib.new(algorithm)
        schema_hasher.update(schema_str.encode())
        schema_hash = schema_hasher.hexdigest()
        
        return DataHash(
            hash_value=hash_value,
            algorithm=algorithm,
            size_bytes=df.memory_usage(deep=True).sum(),
            row_count=len(df),
            column_count=len(df.columns),
            schema_hash=schema_hash
        )
    
    @staticmethod
    def hash_file(file_path: str, algorithm: str = "sha256") -> DataHash:
        """Compute hash of file"""
        
        hasher = hashlib.new(algorithm)
        
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(8192), b""):
                hasher.update(chunk)
        
        hash_value = hasher.hexdigest()
        file_size = os.path.getsize(file_path)
        
        return DataHash(
            hash_value=hash_value,
            algorithm=algorithm,
            size_bytes=file_size
        )
    
    @staticmethod
    def hash_parameters(params: Dict[str, Any], algorithm: str = "sha256") -> str:
        """Compute hash of parameter dictionary"""
        
        # Convert to consistent JSON representation
        params_json = json.dumps(params, sort_keys=True, default=str)
        
        hasher = hashlib.new(algorithm)
        hasher.update(params_json.encode())
        
        return hasher.hexdigest()
    
    @staticmethod
    def verify_hash(data_hash: DataHash, data: Any) -> bool:
        """Verify data against stored hash"""
        
        if isinstance(data, pd.DataFrame):
            computed_hash = DataHasher.hash_dataframe(data, data_hash.algorithm)
        elif isinstance(data, (str, Path)) and Path(data).exists():
            computed_hash = DataHasher.hash_file(str(data), data_hash.algorithm)
        else:
            logger.error("Unsupported data type for hash verification")
            return False
        
        return computed_hash.hash_value == data_hash.hash_value

class FoldDocumenter:
    """Documents cross-validation fold boundaries and embargo periods"""
    
    def __init__(self, embargo_days: int = 1):
        self.embargo_days = embargo_days
        
    def document_time_series_folds(self, 
                                  dates: pd.DatetimeIndex,
                                  n_folds: int,
                                  validation_size: float = 0.2,
                                  test_size: float = 0.2) -> List[FoldBoundary]:
        """
        Document time series cross-validation fold boundaries
        
        Args:
            dates: Time series dates
            n_folds: Number of folds
            validation_size: Fraction for validation
            test_size: Fraction for test (optional)
            
        Returns:
            List of documented fold boundaries
        """
        
        fold_boundaries = []
        total_samples = len(dates)
        
        # Sort dates to ensure chronological order
        sorted_dates = dates.sort_values()
        
        for fold_idx in range(n_folds):
            # Calculate fold boundaries with walk-forward logic
            fold_end_idx = int((fold_idx + 1) * total_samples / n_folds)
            
            # Training data: all data up to fold boundary minus embargo
            train_end_idx = max(0, fold_end_idx - int(self.embargo_days * len(dates) / 
                                                     (dates.max() - dates.min()).days))
            
            # Validation data: after embargo period
            val_start_idx = train_end_idx + int(self.embargo_days * len(dates) / 
                                               (dates.max() - dates.min()).days)
            val_end_idx = min(len(dates), val_start_idx + int(validation_size * train_end_idx))
            
            # Test data: remaining data (if specified)
            test_start_idx = val_end_idx
            test_end_idx = min(len(dates), fold_end_idx) if test_size > 0 else None
            
            # Ensure indices are within bounds
            train_end_idx = min(train_end_idx, len(dates))
            val_start_idx = min(val_start_idx, len(dates))
            val_end_idx = min(val_end_idx, len(dates))
            if test_start_idx:
                test_start_idx = min(test_start_idx, len(dates))
            if test_end_idx:
                test_end_idx = min(test_end_idx, len(dates))
            
            # Create fold boundary with safe indexing
            def safe_date_access(idx, default_idx=0):
                """Safely access date index, return default if out of bounds"""
                if idx >= len(sorted_dates) or idx < 0:
                    return sorted_dates[min(default_idx, len(sorted_dates)-1)]
                return sorted_dates[idx]
            
            fold_boundary = FoldBoundary(
                fold_id=f"fold_{fold_idx:03d}",
                train_start_date=safe_date_access(0).isoformat(),
                train_end_date=safe_date_access(train_end_idx-1 if train_end_idx > 0 else 0).isoformat(),
                validation_start_date=safe_date_access(val_start_idx, -1).isoformat(),
                validation_end_date=safe_date_access(val_end_idx-1 if val_end_idx > val_start_idx else val_start_idx, -1).isoformat(),
                test_start_date=safe_date_access(test_start_idx).isoformat() if test_end_idx and test_start_idx < len(dates) else None,
                test_end_date=safe_date_access(test_end_idx-1 if test_end_idx and test_end_idx > test_start_idx else test_start_idx).isoformat() if test_end_idx else None,
                embargo_days=self.embargo_days,
                samples_train=min(train_end_idx, len(dates)),
                samples_validation=max(0, min(val_end_idx, len(dates)) - val_start_idx),
                samples_test=max(0, min(test_end_idx, len(dates)) - test_start_idx) if test_end_idx else 0
            )
            
            fold_boundaries.append(fold_boundary)
        
        return fold_boundaries
    
    def validate_fold_boundaries(self, fold_boundaries: List[FoldBoundary]) -> Dict[str, Any]:
        """Validate fold boundary consistency"""
        
        validation_results = {
            'valid': True,
            'issues': [],
            'statistics': {}
        }
        
        for i, fold in enumerate(fold_boundaries):
            # Check date ordering
            dates = [fold.train_start_date, fold.train_end_date, 
                    fold.validation_start_date, fold.validation_end_date]
            
            if fold.test_start_date:
                dates.extend([fold.test_start_date, fold.test_end_date])
            
            sorted_dates = sorted([d for d in dates if d])
            
            if dates != sorted_dates:
                validation_results['valid'] = False
                validation_results['issues'].append(f"Fold {i}: Dates not in chronological order")
            
            # Check embargo period
            train_end = pd.to_datetime(fold.train_end_date)
            val_start = pd.to_datetime(fold.validation_start_date)
            actual_embargo = (val_start - train_end).days
            
            if actual_embargo < fold.embargo_days:
                validation_results['valid'] = False
                validation_results['issues'].append(f"Fold {i}: Insufficient embargo period ({actual_embargo} < {fold.embargo_days})")
            
            # Check sample counts
            if fold.samples_train <= 0:
                validation_results['valid'] = False
                validation_results['issues'].append(f"Fold {i}: No training samples")
            
            if fold.samples_validation <= 0:
                validation_results['issues'].append(f"Fold {i}: No validation samples")
        
        # Calculate statistics
        validation_results['statistics'] = {
            'total_folds': len(fold_boundaries),
            'avg_train_samples': np.mean([f.samples_train for f in fold_boundaries]),
            'avg_validation_samples': np.mean([f.samples_validation for f in fold_boundaries]),
            'avg_test_samples': np.mean([f.samples_test for f in fold_boundaries if f.samples_test > 0]),
            'min_embargo_days': min(f.embargo_days for f in fold_boundaries),
            'max_embargo_days': max(f.embargo_days for f in fold_boundaries)
        }
        
        return validation_results

class ParameterDocumenter:
    """Documents model parameters and hyperparameters with versioning"""
    
    def __init__(self, storage_path: str = "parameter_snapshots"):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(exist_ok=True)
        
        # Database for parameter tracking
        self.db_path = self.storage_path / "parameters.db"
        self._init_database()
    
    def _init_database(self):
        """Initialize parameter tracking database"""
        
        conn = sqlite3.connect(self.db_path)
        
        conn.execute("""
            CREATE TABLE IF NOT EXISTS parameter_snapshots (
                snapshot_id TEXT PRIMARY KEY,
                timestamp TEXT NOT NULL,
                model_type TEXT NOT NULL,
                parameter_hash TEXT NOT NULL,
                file_path TEXT NOT NULL,
                random_seed INTEGER,
                environment_hash TEXT,
                created_by TEXT,
                experiment_id TEXT
            )
        """)
        
        conn.commit()
        conn.close()
    
    def create_snapshot(self, 
                       model_type: str,
                       hyperparameters: Dict[str, Any],
                       feature_parameters: Dict[str, Any] = None,
                       preprocessing_parameters: Dict[str, Any] = None,
                       training_parameters: Dict[str, Any] = None,
                       random_seed: Optional[int] = None,
                       experiment_id: Optional[str] = None,
                       created_by: str = "system") -> ParameterSnapshot:
        """Create a complete parameter snapshot"""
        
        # Generate unique snapshot ID
        snapshot_id = str(uuid.uuid4())
        timestamp = datetime.now(timezone.utc).isoformat()
        
        # Combine all parameters
        all_parameters = {
            'hyperparameters': hyperparameters or {},
            'feature_parameters': feature_parameters or {},
            'preprocessing_parameters': preprocessing_parameters or {},
            'training_parameters': training_parameters or {}
        }
        
        # Compute parameter hash
        parameter_hash = DataHasher.hash_parameters(all_parameters)
        
        # Create snapshot object
        snapshot = ParameterSnapshot(
            snapshot_id=snapshot_id,
            timestamp=timestamp,
            model_type=model_type,
            hyperparameters=hyperparameters or {},
            feature_parameters=feature_parameters or {},
            preprocessing_parameters=preprocessing_parameters or {},
            training_parameters=training_parameters or {},
            parameter_hash=parameter_hash,
            random_seed=random_seed,
            environment_hash=self._get_environment_hash()
        )
        
        # Save to file
        file_path = self.storage_path / f"{snapshot_id}.json"
        with open(file_path, 'w') as f:
            json.dump(asdict(snapshot), f, indent=2, default=str)
        
        # Save to database
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            INSERT INTO parameter_snapshots 
            (snapshot_id, timestamp, model_type, parameter_hash, file_path,
             random_seed, environment_hash, created_by, experiment_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            snapshot_id, timestamp, model_type, parameter_hash, str(file_path),
            random_seed, snapshot.environment_hash, created_by, experiment_id
        ))
        conn.commit()
        conn.close()
        
        logger.info(f"Created parameter snapshot {snapshot_id}")
        
        return snapshot
    
    def load_snapshot(self, snapshot_id: str) -> Optional[ParameterSnapshot]:
        """Load parameter snapshot by ID"""
        
        file_path = self.storage_path / f"{snapshot_id}.json"
        
        if not file_path.exists():
            logger.error(f"Snapshot file not found: {file_path}")
            return None
        
        try:
            with open(file_path, 'r') as f:
                snapshot_data = json.load(f)
            
            return ParameterSnapshot(**snapshot_data)
            
        except Exception as e:
            logger.error(f"Failed to load snapshot {snapshot_id}: {e}")
            return None
    
    def find_snapshots(self, 
                      model_type: Optional[str] = None,
                      parameter_hash: Optional[str] = None,
                      experiment_id: Optional[str] = None,
                      limit: int = 100) -> List[Dict[str, Any]]:
        """Find parameter snapshots matching criteria"""
        
        conn = sqlite3.connect(self.db_path)
        
        query = "SELECT * FROM parameter_snapshots WHERE 1=1"
        params = []
        
        if model_type:
            query += " AND model_type = ?"
            params.append(model_type)
        
        if parameter_hash:
            query += " AND parameter_hash = ?"
            params.append(parameter_hash)
        
        if experiment_id:
            query += " AND experiment_id = ?"
            params.append(experiment_id)
        
        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)
        
        cursor = conn.execute(query, params)
        columns = [description[0] for description in cursor.description]
        
        results = []
        for row in cursor.fetchall():
            results.append(dict(zip(columns, row)))
        
        conn.close()
        
        return results
    
    def _get_environment_hash(self) -> str:
        """Get hash of current environment state"""
        
        try:
            import sys
            import platform
            
            env_info = {
                'python_version': sys.version,
                'platform': platform.platform(),
                'packages': {}  # Would include package versions in practice
            }
            
            return DataHasher.hash_parameters(env_info)
            
        except Exception as e:
            logger.warning(f"Failed to compute environment hash: {e}")
            return "unknown"

class ExperimentDocumenter:
    """Documents complete experiments with all metadata"""
    
    def __init__(self, storage_path: str = "experiment_docs"):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(exist_ok=True)
        
        # Sub-directories for organization
        (self.storage_path / "experiments").mkdir(exist_ok=True)
        (self.storage_path / "model_cards").mkdir(exist_ok=True)
        (self.storage_path / "reports").mkdir(exist_ok=True)
        
        # Database for experiment tracking
        self.db_path = self.storage_path / "experiments.db"
        self._init_database()
        
        # Initialize component documenters
        self.fold_documenter = FoldDocumenter()
        self.parameter_documenter = ParameterDocumenter(
            str(self.storage_path / "parameters")
        )
    
    def _init_database(self):
        """Initialize experiment tracking database"""
        
        conn = sqlite3.connect(self.db_path)
        
        conn.execute("""
            CREATE TABLE IF NOT EXISTS experiments (
                experiment_id TEXT PRIMARY KEY,
                experiment_name TEXT NOT NULL,
                created_timestamp TEXT NOT NULL,
                status TEXT NOT NULL,
                objective TEXT,
                created_by TEXT,
                last_modified_by TEXT,
                last_modified_timestamp TEXT,
                file_path TEXT NOT NULL
            )
        """)
        
        conn.commit()
        conn.close()
    
    def create_experiment(self,
                         experiment_name: str,
                         objective: str,
                         hypothesis: str,
                         methodology: str,
                         success_criteria: List[str],
                         compliance_frameworks: List[ComplianceFramework] = None,
                         created_by: str = "system") -> ExperimentMetadata:
        """Create new experiment documentation"""
        
        experiment_id = str(uuid.uuid4())
        timestamp = datetime.now(timezone.utc).isoformat()
        
        experiment = ExperimentMetadata(
            experiment_id=experiment_id,
            experiment_name=experiment_name,
            created_timestamp=timestamp,
            status="created",
            objective=objective,
            hypothesis=hypothesis,
            methodology=methodology,
            success_criteria=success_criteria,
            datasets_used=[],
            data_hashes=[],
            fold_boundaries=[],
            model_versions=[],
            parameter_snapshots=[],
            metrics={},
            validation_results={},
            statistical_tests={},
            compliance_frameworks=compliance_frameworks or [ComplianceFramework.INTERNAL],
            approval_status="pending",
            created_by=created_by,
            last_modified_by=created_by,
            last_modified_timestamp=timestamp
        )
        
        # Save to file
        file_path = self.storage_path / "experiments" / f"{experiment_id}.json"
        self._save_experiment(experiment, file_path)
        
        # Save to database
        self._update_experiment_db(experiment, file_path)
        
        logger.info(f"Created experiment {experiment_id}: {experiment_name}")
        
        return experiment
    
    def update_experiment(self, experiment_id: str, updates: Dict[str, Any],
                         updated_by: str = "system") -> Optional[ExperimentMetadata]:
        """Update experiment with new information"""
        
        experiment = self.load_experiment(experiment_id)
        if not experiment:
            return None
        
        # Apply updates
        for key, value in updates.items():
            if hasattr(experiment, key):
                setattr(experiment, key, value)
        
        # Update metadata
        experiment.last_modified_by = updated_by
        experiment.last_modified_timestamp = datetime.now(timezone.utc).isoformat()
        
        # Add to change log
        change_entry = {
            'timestamp': experiment.last_modified_timestamp,
            'updated_by': updated_by,
            'changes': list(updates.keys()),
            'change_id': str(uuid.uuid4())
        }
        experiment.change_log.append(change_entry)
        
        # Save updates
        file_path = self.storage_path / "experiments" / f"{experiment_id}.json"
        self._save_experiment(experiment, file_path)
        self._update_experiment_db(experiment, file_path)
        
        logger.info(f"Updated experiment {experiment_id}")
        
        return experiment
    
    def load_experiment(self, experiment_id: str) -> Optional[ExperimentMetadata]:
        """Load experiment by ID"""
        
        file_path = self.storage_path / "experiments" / f"{experiment_id}.json"
        
        if not file_path.exists():
            logger.error(f"Experiment file not found: {file_path}")
            return None
        
        try:
            with open(file_path, 'r') as f:
                experiment_data = json.load(f)
            
            # Convert enum strings back to enums
            if 'compliance_frameworks' in experiment_data:
                experiment_data['compliance_frameworks'] = [
                    ComplianceFramework(fw) for fw in experiment_data['compliance_frameworks']
                ]
            
            return ExperimentMetadata(**experiment_data)
            
        except Exception as e:
            logger.error(f"Failed to load experiment {experiment_id}: {e}")
            return None
    
    def _save_experiment(self, experiment: ExperimentMetadata, file_path: Path):
        """Save experiment to JSON file"""
        
        # Convert enums to strings for JSON serialization
        experiment_dict = asdict(experiment)
        experiment_dict['compliance_frameworks'] = [
            fw.value for fw in experiment.compliance_frameworks
        ]
        
        with open(file_path, 'w') as f:
            json.dump(experiment_dict, f, indent=2, default=str)
    
    def _update_experiment_db(self, experiment: ExperimentMetadata, file_path: Path):
        """Update experiment in database"""
        
        conn = sqlite3.connect(self.db_path)
        
        conn.execute("""
            INSERT OR REPLACE INTO experiments 
            (experiment_id, experiment_name, created_timestamp, status,
             objective, created_by, last_modified_by, last_modified_timestamp, file_path)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            experiment.experiment_id,
            experiment.experiment_name,
            experiment.created_timestamp,
            experiment.status,
            experiment.objective,
            experiment.created_by,
            experiment.last_modified_by,
            experiment.last_modified_timestamp,
            str(file_path)
        ))
        
        conn.commit()
        conn.close()
    
    def create_model_card(self, experiment_id: str, model_id: str,
                         model_details: Dict[str, Any]) -> ModelCard:
        """Create comprehensive model card"""
        
        experiment = self.load_experiment(experiment_id)
        if not experiment:
            raise ValueError(f"Experiment {experiment_id} not found")
        
        timestamp = datetime.now(timezone.utc).isoformat()
        
        model_card = ModelCard(
            model_id=model_id,
            model_name=model_details.get('model_name', f"Model_{model_id[:8]}"),
            model_version=model_details.get('model_version', '1.0'),
            created_timestamp=timestamp,
            model_type=model_details.get('model_type', 'unknown'),
            model_architecture=model_details.get('model_architecture', ''),
            training_data_description=model_details.get('training_data_description', ''),
            training_procedure=model_details.get('training_procedure', ''),
            evaluation_procedure=model_details.get('evaluation_procedure', ''),
            training_metrics=model_details.get('training_metrics', {}),
            validation_metrics=model_details.get('validation_metrics', {}),
            test_metrics=model_details.get('test_metrics', {}),
            out_of_sample_metrics=model_details.get('out_of_sample_metrics', {}),
            bias_assessment=model_details.get('bias_assessment', {}),
            fairness_metrics=model_details.get('fairness_metrics', {}),
            ethical_considerations=model_details.get('ethical_considerations', []),
            known_limitations=model_details.get('known_limitations', []),
            risk_assessment=model_details.get('risk_assessment', {}),
            failure_modes=model_details.get('failure_modes', []),
            monitoring_recommendations=model_details.get('monitoring_recommendations', []),
            intended_use=model_details.get('intended_use', ''),
            prohibited_uses=model_details.get('prohibited_uses', []),
            deployment_constraints=model_details.get('deployment_constraints', []),
            maintenance_requirements=model_details.get('maintenance_requirements', []),
            regulatory_approvals=model_details.get('regulatory_approvals', []),
            compliance_statements=model_details.get('compliance_statements', {}),
            audit_information=model_details.get('audit_information', {}),
            feature_importance=model_details.get('feature_importance', {}),
            feature_descriptions=model_details.get('feature_descriptions', {}),
            data_requirements=model_details.get('data_requirements', {}),
            computational_requirements=model_details.get('computational_requirements', {})
        )
        
        # Save model card
        file_path = self.storage_path / "model_cards" / f"{model_id}.json"
        with open(file_path, 'w') as f:
            json.dump(asdict(model_card), f, indent=2, default=str)
        
        logger.info(f"Created model card for {model_id}")
        
        return model_card
    
    def generate_compliance_report(self, experiment_id: str,
                                  framework: ComplianceFramework) -> Dict[str, Any]:
        """Generate compliance report for specific framework"""
        
        experiment = self.load_experiment(experiment_id)
        if not experiment:
            raise ValueError(f"Experiment {experiment_id} not found")
        
        report = {
            'experiment_id': experiment_id,
            'framework': framework.value,
            'generated_timestamp': datetime.now(timezone.utc).isoformat(),
            'compliance_status': 'under_review',
            'sections': {}
        }
        
        if framework == ComplianceFramework.SR_11_7:
            report['sections'] = self._generate_sr11_7_report(experiment)
        elif framework == ComplianceFramework.MIFID_II:
            report['sections'] = self._generate_mifid_ii_report(experiment)
        else:
            report['sections'] = self._generate_internal_report(experiment)
        
        # Save report
        report_file = self.storage_path / "reports" / f"{experiment_id}_{framework.value}_report.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        return report
    
    def _generate_sr11_7_report(self, experiment: ExperimentMetadata) -> Dict[str, Any]:
        """Generate SR 11-7 compliance report"""
        
        return {
            'model_development': {
                'development_process_documented': len(experiment.change_log) > 0,
                'validation_performed': len(experiment.validation_results) > 0,
                'parameter_documentation': len(experiment.parameter_snapshots) > 0,
                'data_documentation': len(experiment.data_hashes) > 0
            },
            'model_validation': {
                'conceptual_soundness': experiment.objective != '',
                'process_verification': len(experiment.fold_boundaries) > 0,
                'outcome_analysis': len(experiment.metrics) > 0,
                'ongoing_monitoring': 'monitoring' in experiment.methodology.lower()
            },
            'governance': {
                'approval_process': experiment.approval_status != 'pending',
                'documentation_completeness': self._assess_documentation_completeness(experiment),
                'change_management': len(experiment.change_log) > 0
            }
        }
    
    def _generate_mifid_ii_report(self, experiment: ExperimentMetadata) -> Dict[str, Any]:
        """Generate MiFID II compliance report"""
        
        return {
            'algorithmic_trading_controls': {
                'pre_trade_controls_documented': 'pre-trade' in experiment.methodology.lower(),
                'risk_controls_implemented': 'risk' in experiment.objective.lower(),
                'testing_procedures_documented': len(experiment.validation_results) > 0
            },
            'record_keeping': {
                'parameter_records': len(experiment.parameter_snapshots) > 0,
                'audit_trail_complete': len(experiment.change_log) > 0,
                'data_lineage_documented': len(experiment.data_hashes) > 0
            }
        }
    
    def _generate_internal_report(self, experiment: ExperimentMetadata) -> Dict[str, Any]:
        """Generate internal compliance report"""
        
        return {
            'experiment_quality': {
                'objective_defined': experiment.objective != '',
                'hypothesis_stated': experiment.hypothesis != '',
                'methodology_documented': experiment.methodology != '',
                'success_criteria_defined': len(experiment.success_criteria) > 0
            },
            'technical_documentation': {
                'parameters_tracked': len(experiment.parameter_snapshots) > 0,
                'data_lineage_documented': len(experiment.data_hashes) > 0,
                'fold_boundaries_documented': len(experiment.fold_boundaries) > 0,
                'metrics_recorded': len(experiment.metrics) > 0
            },
            'reproducibility': {
                'seeds_documented': any(p.random_seed is not None for p in experiment.parameter_snapshots),
                'environment_tracked': any(p.environment_hash is not None for p in experiment.parameter_snapshots),
                'data_hashes_available': len(experiment.data_hashes) > 0
            }
        }
    
    def _assess_documentation_completeness(self, experiment: ExperimentMetadata) -> float:
        """Assess completeness of experiment documentation"""
        
        required_fields = [
            experiment.objective,
            experiment.hypothesis,
            experiment.methodology,
            len(experiment.success_criteria) > 0,
            len(experiment.parameter_snapshots) > 0,
            len(experiment.data_hashes) > 0,
            len(experiment.metrics) > 0
        ]
        
        completed_fields = sum(1 for field in required_fields if field)
        return completed_fields / len(required_fields)

class AuditTrailManager:
    """Manages comprehensive audit trails for regulatory compliance"""
    
    def __init__(self, storage_path: str = "audit_trails"):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(exist_ok=True)
        
        # Database for audit events
        self.db_path = self.storage_path / "audit_trail.db"
        self._init_database()
    
    def _init_database(self):
        """Initialize audit trail database"""
        
        conn = sqlite3.connect(self.db_path)
        
        conn.execute("""
            CREATE TABLE IF NOT EXISTS audit_events (
                event_id TEXT PRIMARY KEY,
                timestamp TEXT NOT NULL,
                event_type TEXT NOT NULL,
                user_id TEXT NOT NULL,
                resource_type TEXT NOT NULL,
                resource_id TEXT NOT NULL,
                action TEXT NOT NULL,
                details TEXT,
                ip_address TEXT,
                user_agent TEXT,
                session_id TEXT
            )
        """)
        
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_timestamp ON audit_events(timestamp)
        """)
        
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_user_id ON audit_events(user_id)
        """)
        
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_resource_id ON audit_events(resource_id)
        """)
        
        conn.commit()
        conn.close()
    
    def log_event(self, event_type: str, user_id: str, resource_type: str,
                  resource_id: str, action: str, details: Dict[str, Any] = None,
                  ip_address: str = None, user_agent: str = None,
                  session_id: str = None) -> str:
        """Log audit event"""
        
        event_id = str(uuid.uuid4())
        timestamp = datetime.now(timezone.utc).isoformat()
        
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            INSERT INTO audit_events 
            (event_id, timestamp, event_type, user_id, resource_type,
             resource_id, action, details, ip_address, user_agent, session_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            event_id, timestamp, event_type, user_id, resource_type,
            resource_id, action, json.dumps(details or {}),
            ip_address, user_agent, session_id
        ))
        conn.commit()
        conn.close()
        
        return event_id
    
    def get_audit_trail(self, resource_id: str = None, user_id: str = None,
                       start_date: str = None, end_date: str = None,
                       limit: int = 1000) -> List[Dict[str, Any]]:
        """Retrieve audit trail with filters"""
        
        conn = sqlite3.connect(self.db_path)
        
        query = "SELECT * FROM audit_events WHERE 1=1"
        params = []
        
        if resource_id:
            query += " AND resource_id = ?"
            params.append(resource_id)
        
        if user_id:
            query += " AND user_id = ?"
            params.append(user_id)
        
        if start_date:
            query += " AND timestamp >= ?"
            params.append(start_date)
        
        if end_date:
            query += " AND timestamp <= ?"
            params.append(end_date)
        
        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)
        
        cursor = conn.execute(query, params)
        columns = [description[0] for description in cursor.description]
        
        results = []
        for row in cursor.fetchall():
            event = dict(zip(columns, row))
            if event['details']:
                event['details'] = json.loads(event['details'])
            results.append(event)
        
        conn.close()
        
        return results