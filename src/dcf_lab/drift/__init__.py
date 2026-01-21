"""
Concept Drift Detection & Adaptive Learning Framework

This module provides comprehensive drift detection and adaptive learning capabilities
for financial ML systems, including:

1. Stream Monitors:
   - ADWIN (Adaptive Sliding Window) for automatic window sizing
   - Page-Hinkley test for persistent mean shift detection
   - Feature drift monitoring and correlation changes
   - Residual distribution monitoring

2. Drift Actions:
   - Light retrain with exponential recency weighting
   - Hard reset of regime/HMM models when major drift occurs
   - Online/streaming learners for incremental updates
   - Adaptive ensemble weight adjustment

3. Financial-Specific Features:
   - Market regime change detection
   - Volatility regime monitoring
   - Correlation structure drift analysis
   - News sentiment and microstructure drift

The framework integrates seamlessly with existing ML pipelines and provides
production-ready adaptive learning for dynamic financial markets.
"""

from .monitors import (
    ADWINMonitor,
    PageHinkleyMonitor,
    FeatureDriftMonitor,
    ResidualDriftMonitor,
    DriftAlert,
    DriftSeverity
)

from .adaptive import (
    AdaptiveLearner,
    LightRetrainer,
    OnlineUpdater,
    RegimeResetter,
    RecencyWeighter
)

from .streaming import (
    StreamingFramework,
    StreamingConfig,
    IncrementalUpdater,
    RiverIntegration
)

from .framework import (
    ConceptDriftFramework,
    DriftConfig,
    AdaptiveMLPipeline,
    DriftValidator
)

import os

__all__ = [
    # Monitors
    'ADWINMonitor',
    'PageHinkleyMonitor', 
    'FeatureDriftMonitor',
    'ResidualDriftMonitor',
    'DriftAlert',
    'DriftSeverity',
    
    # Adaptive Learning
    'AdaptiveLearner',
    'LightRetrainer',
    'OnlineUpdater',
    'RegimeResetter',
    'RecencyWeighter',
    
    # Streaming
    'StreamingFramework',
    'StreamingConfig',
    'IncrementalUpdater',
    'RiverIntegration',
    
    # Framework
    'ConceptDriftFramework',
    'DriftConfig',
    'AdaptiveMLPipeline',
    'DriftValidator'
]

# Framework version and metadata
__version__ = "1.0.0"
__author__ = "Financial ML Research Team"
__description__ = "Concept drift detection and adaptive learning for financial time series"

# Configuration defaults
DEFAULT_ADWIN_DELTA = 0.002  # Confidence parameter for ADWIN
DEFAULT_PH_THRESHOLD = 50.0  # Page-Hinkley threshold
DEFAULT_DRIFT_WINDOW = 100   # Minimum samples for drift detection
DEFAULT_RETRAIN_EPOCHS = 5   # Light retrain epochs

# Avoid noisy import-time output in production runs.
if os.getenv("DCF_DRIFT_SHOW_BANNER", "0").strip() not in {"", "0", "false", "False", "no", "NO"}:
    print("🚀 Concept Drift Detection & Adaptive Learning Framework")
    print("=" * 60)
    print("Framework for stream monitoring, drift detection, and adaptive model updates")
    print("Supports ADWIN, Page-Hinkley, light retraining, and online learning")
    print("Optimized for financial time series and dynamic market conditions")
    print(f"Version: {__version__}")
    print("Ready for Task 6 implementation! 🎯")
