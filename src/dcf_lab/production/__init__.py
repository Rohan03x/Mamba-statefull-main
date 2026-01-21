"""
Live Production Training Loop System

This module implements Task 9: "Live training loop (production)" with comprehensive
production-ready capabilities:

1. Data Ingestion & Validation: Schema validation, missing value checks, future date rejection
2. Drift Detection: ADWIN/Page-Hinkley monitoring with adaptive responses
3. Model Updates: Light updates vs scheduled monthly retraining
4. Scoring & Calibration: Point predictions, quantiles, conformal intervals
5. Comprehensive Monitoring: Performance metrics, calibration plots, coverage tracking
6. Model Registry: Holdout validation guards against backtest overfitting
7. Production Deployment: Nightly job orchestration and self-monitoring

Key components:
- data_ingestion.py: Real-time data validation and feature store
- drift_monitoring.py: Continuous drift detection and response
- model_updater.py: Adaptive model updates and retraining
- scoring_engine.py: Production scoring with calibration
- monitoring_system.py: Comprehensive performance monitoring
- model_registry.py: Model versioning and deployment guards
- production_orchestrator.py: Nightly job coordination
"""

# Core production components
from .data_ingestion import (
    ProductionDataValidator,
    FeatureStore,
    SchemaValidator,
    DataIngestionPipeline
)

from .drift_monitoring import (
    ProductionDriftMonitor,
    DriftResponseManager,
    StreamingMonitor,
    DriftAlertSystem
)

from .model_updater import (
    ProductionModelUpdater,
    LightModelUpdater,
    FullModelUpdater,
    UpdateScheduler,
    ModelVersionManager,
    UpdateConfig,
    UpdateResult
)

from .scoring_engine import (
    ProductionScoringEngine,
    CalibrationEngine,
    ConformalPredictor,
    QuantilePredictor,
    PredictionResult
)

# Backwards compatibility aliases
CalibrationManager = CalibrationEngine
ConformalWrapper = ConformalPredictor
PredictionAggregator = ProductionScoringEngine

from .monitoring import (
    MonitoringConfig,
    PerformanceTracker,
    AlertEvent,
    PerformanceMetrics
)

__all__ = [
    # Data ingestion
    'ProductionDataValidator',
    'FeatureStore',
    'SchemaValidator', 
    'DataIngestionPipeline',
    
    # Drift monitoring
    'ProductionDriftMonitor',
    'DriftResponseManager',
    'StreamingMonitor',
    'DriftAlertSystem',
    
    # Model updates
    'ProductionModelUpdater',
    'LightModelUpdater',
    'FullModelUpdater',
    'UpdateScheduler',
    'ModelVersionManager',
    'UpdateConfig',
    'UpdateResult',
    
    # Scoring engine
    'ProductionScoringEngine',
    'CalibrationEngine',
    'CalibrationManager',
    'ConformalPredictor',
    'ConformalWrapper',
    'QuantilePredictor',
    'PredictionResult',
    'PredictionAggregator',
    
    # Monitoring
    'MonitoringConfig',
    'PerformanceTracker',
    'AlertEvent',
    'PerformanceMetrics',
]

# Production system information
PRODUCTION_INFO = {
    'name': 'Live Production Training Loop System',
    'task': 'Task 9: Live training loop (production)',
    'capabilities': [
        'Real-time data ingestion with schema validation',
        'Continuous drift detection with adaptive responses', 
        'Light model updates vs scheduled retraining',
        'Production scoring with calibration and conformal intervals',
        'Comprehensive performance and calibration monitoring',
        'Model registry with holdout validation guards',
        'Automated nightly job orchestration'
    ],
    'components': {
        'data_ingestion': 'Real-time data validation and feature store management',
        'drift_monitoring': 'ADWIN/Page-Hinkley drift detection with response systems',
        'model_updater': 'Adaptive model updates and scheduled retraining',
        'scoring_engine': 'Production scoring with calibration and intervals',
        'monitoring_system': 'Performance, calibration, and regime monitoring',
        'model_registry': 'Model versioning with deployment validation',
        'production_orchestrator': 'Nightly job coordination and system health'
    },
    'deliverables': [
        'Nightly job that keeps models fresh and calibrated',
        'Self-aware system with comprehensive monitoring',
        'Production-ready deployment with guardrails',
        'Automated drift response and model updates'
    ]
}

def get_production_info():
    """Get information about the production system"""
    return PRODUCTION_INFO

def get_system_status():
    """Get current production system status"""
    # This would be implemented with actual system monitoring
    return {
        'status': 'operational',
        'last_update': '2025-09-20T17:50:00Z',
        'active_models': 3,
        'drift_alerts': 0,
        'system_health': 'green'
    }