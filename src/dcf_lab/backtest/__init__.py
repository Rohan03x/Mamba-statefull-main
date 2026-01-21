"""
Walk-Forward Backtest Framework

This module provides a comprehensive walk-forward backtesting framework
for financial machine learning models with sophisticated timing controls,
hyperparameter optimization, and comprehensive audit trails.

Key Components:
- BacktestConfig: Configuration management with cadence validation
- HyperparameterOptimizer: Optuna-based hyperparameter optimization
- BacktestArtifactManager: Comprehensive artifact storage and audit trails
- WalkForwardBacktester: Main orchestration class for backtesting

Features:
- Multiple cadences (daily, weekly, monthly, quarterly)
- Time-series aware hyperparameter optimization
- Complete data lineage tracking with cryptographic hashes
- Overfitting detection through statistical analysis
- Memory-efficient execution with checkpointing
- Regulatory-compliant audit trails

Usage:
    from dcf_lab.backtest import (
        create_default_backtest_config,
        BacktestArtifactManager,
        WalkForwardBacktester
    )
    
    # Create configuration
    config = create_default_backtest_config()
    
    # Set up artifact management
    artifact_manager = BacktestArtifactManager(config, "backtest_artifacts")
    
    # Create backtester
    backtester = WalkForwardBacktester(config, artifact_manager)
    
    # Run backtest
    results = backtester.run_backtest(data, targets)

Academic Foundation:
- Based on "Advances in Financial Machine Learning" by Marcos Lopez de Prado
- Implements proper time-series cross-validation to prevent lookahead bias
- Follows David H. Bailey's recommendations for walk-forward analysis
- Includes PBO (Probability of Backtest Overfitting) detection
"""

from .config import (
    BacktestConfig,
    DataWindowType,
    CadenceType,
    BacktestState,
    create_default_backtest_config,
    create_conservative_backtest_config,
    create_aggressive_backtest_config
)

from .hyperopt import (
    HyperparameterOptimizer,
    HyperparameterSpace,
    OptimizationResult
)

from .artifacts import (
    BacktestArtifactManager,
    BacktestStep,
    DataSnapshot,
    ModelSnapshot,
    PredictionSnapshot
)

from .backtester import WalkForwardBacktester

# Version information
__version__ = "1.0.0"
__author__ = "DCF Lab"
__description__ = "Sophisticated walk-forward backtesting framework for financial ML"

# Export main components
__all__ = [
    # Configuration
    "BacktestConfig",
    "DataWindowType", 
    "CadenceType",
    "BacktestState",
    "create_default_backtest_config",
    "create_conservative_backtest_config",
    "create_aggressive_backtest_config",
    
    # Hyperparameter optimization
    "HyperparameterOptimizer",
    "HyperparameterSpace",
    "OptimizationResult",
    
    # Artifact management
    "BacktestArtifactManager",
    "BacktestStep",
    "DataSnapshot",
    "ModelSnapshot", 
    "PredictionSnapshot",
    
    # Main backtester
    "WalkForwardBacktester",
    
    # Utility functions
    "get_framework_info",
    "validate_framework_setup",
    "print_framework_summary",
    
    # Metadata
    "__version__",
    "__author__",
    "__description__"
]


def get_framework_info() -> dict:
    """
    Get comprehensive information about the backtest framework
    
    Returns:
        Dictionary with framework details, capabilities, and usage examples
    """
    return {
        "name": "Walk-Forward Backtest Framework",
        "version": __version__,
        "description": __description__,
        "author": __author__,
        
        "capabilities": {
            "cadences": ["daily", "weekly", "monthly", "quarterly", "annually"],
            "data_windows": ["expanding", "rolling_2y", "rolling_3y", "rolling_5y"],
            "optimization": "Optuna-based hyperparameter optimization with pruning",
            "artifacts": "Complete audit trails with cryptographic verification",
            "overfitting_detection": "PBO and statistical overfitting analysis",
            "memory_management": "Efficient storage with compression and checkpointing"
        },
        
        "components": {
            "config": "Configuration management with validation",
            "hyperopt": "Time-series aware hyperparameter optimization", 
            "artifacts": "Comprehensive artifact storage and management",
            "backtester": "Main orchestration for walk-forward backtesting"
        },
        
        "academic_foundation": [
            "Advances in Financial Machine Learning (Lopez de Prado)",
            "Time-series cross-validation best practices",
            "Walk-forward analysis methodology (Bailey et al.)",
            "Probability of Backtest Overfitting detection"
        ],
        
        "usage_patterns": {
            "quick_start": "Use create_default_backtest_config() for standard setups",
            "conservative": "Use create_conservative_backtest_config() for robust backtesting",
            "aggressive": "Use create_aggressive_backtest_config() for fast iteration",
            "custom": "Create BacktestConfig() with specific parameters"
        }
    }


def validate_framework_setup() -> bool:
    """
    Validate that the framework is properly set up and all dependencies are available
    
    Returns:
        True if framework is ready to use, False otherwise
    """
    try:
        # Test imports
        from .config import create_default_backtest_config
        from .hyperopt import HyperparameterOptimizer
        from .artifacts import BacktestArtifactManager
        from .backtester import WalkForwardBacktester
        
        # Test basic functionality
        config = create_default_backtest_config()
        
        if not config.validate_timing():
            print("❌ Configuration validation failed")
            return False
        
        # Test required dependencies
        import optuna
        import pandas as pd
        import numpy as np
        import sklearn
        
        print("✅ Framework validation passed")
        return True
        
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        return False
    except Exception as e:
        print(f"❌ Framework validation failed: {e}")
        return False


def print_framework_summary():
    """Print a summary of the framework capabilities"""
    info = get_framework_info()
    
    print("=" * 70)
    print(f"{info['name']} v{info['version']}")
    print("=" * 70)
    print(f"📖 {info['description']}")
    print(f"👤 Author: {info['author']}")
    print()
    
    print("🎯 Key Capabilities:")
    for capability, description in info['capabilities'].items():
        print(f"  • {capability.title()}: {description}")
    print()
    
    print("🧩 Components:")
    for component, description in info['components'].items():
        print(f"  • {component}: {description}")
    print()
    
    print("📚 Academic Foundation:")
    for foundation in info['academic_foundation']:
        print(f"  • {foundation}")
    print()
    
    print("🚀 Usage Patterns:")
    for pattern, description in info['usage_patterns'].items():
        print(f"  • {pattern}: {description}")
    
    print("=" * 70)


if __name__ == "__main__":
    print_framework_summary()
    
    print("\n🔍 Validating framework setup...")
    if validate_framework_setup():
        print("✅ Framework is ready for use!")
    else:
        print("❌ Framework setup issues detected.")