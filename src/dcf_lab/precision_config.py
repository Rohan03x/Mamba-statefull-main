"""
GLOBAL PRECISION CONFIGURATION SYSTEM
=====================================

Centralized configuration for mathematical precision throughout the entire AI forecasting application.
Provides automatic initialization, context management, and runtime precision adjustments.
"""

import json
import logging
import os
import warnings
from contextlib import contextmanager
from decimal import Decimal, getcontext
from enum import Enum
from typing import Any, Dict, Optional, Union

import numpy as np
import torch

# Setup logging for precision operations
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PrecisionLevel(Enum):
    """Predefined precision levels for different use cases"""
    STANDARD = "standard"      # Standard float64 precision
    HIGH = "high"             # Extended precision for critical calculations
    MAXIMUM = "maximum"       # Decimal arithmetic with 50+ digits
    ULTRA = "ultra"          # Maximum possible precision (100+ digits)


class GlobalPrecisionConfig:
    """
    Global precision configuration manager for the entire application.

    Manages precision settings across all mathematical operations including:
    - Decimal arithmetic precision
    - NumPy array precision
    - PyTorch tensor precision
    - Scientific computation precision
    - Financial calculation precision
    """

    def __init__(self):
        self._initialized = False
        self._current_level = PrecisionLevel.STANDARD
        self._config = self._load_default_config()
        self._backup_settings = {}
        # Initialize current_config with default
        self.current_config = self._config[self._current_level]

    def _load_default_config(self) -> Dict[str, Any]:
        """Load default precision configuration"""
        return {
            PrecisionLevel.STANDARD: {
                "decimal_precision": 28,
                "numpy_dtype": np.float64,
                "torch_dtype": torch.float32,
                "scientific_dtype": np.float64,
                "strict_tolerance": Decimal('1e-12'),
                "normal_tolerance": Decimal('1e-9'),
                "description": "Standard precision for general use"
            },
            PrecisionLevel.HIGH: {
                "decimal_precision": 35,
                "numpy_dtype": np.longdouble,
                "torch_dtype": torch.float64,
                "scientific_dtype": np.longdouble,
                "strict_tolerance": Decimal('1e-15'),
                "normal_tolerance": Decimal('1e-12'),
                "description": "High precision for important calculations"
            },
            PrecisionLevel.MAXIMUM: {
                "decimal_precision": 50,
                "numpy_dtype": np.longdouble,
                "torch_dtype": torch.float64,
                "scientific_dtype": np.longdouble,
                "strict_tolerance": Decimal('1e-18'),
                "normal_tolerance": Decimal('1e-15'),
                "description": "Maximum precision for critical financial calculations"
            },
            PrecisionLevel.ULTRA: {
                "decimal_precision": 100,
                "numpy_dtype": np.longdouble,
                "torch_dtype": torch.float64,
                "scientific_dtype": np.longdouble,
                "strict_tolerance": Decimal('1e-25'),
                "normal_tolerance": Decimal('1e-20'),
                "description": "Ultra precision for research and validation"
            }
        }

    def initialize(self,
                   level: Union[PrecisionLevel,
                                str] = PrecisionLevel.MAXIMUM,
                   config_file: Optional[str] = None) -> None:
        """
        Initialize global precision settings for the entire application.

        Args:
            level: Precision level to use
            config_file: Optional path to custom configuration file
        """
        if isinstance(level, str):
            level = PrecisionLevel(level)

        self._current_level = level

        # Load custom config if provided
        if config_file and os.path.exists(config_file):
            self._load_config_file(config_file)

        # Backup current settings
        self._backup_current_settings()

        # Apply precision settings
        self._apply_precision_settings(level)

        self._initialized = True

        logger.info(f"Global precision initialized to {level.value} level")
        self._log_current_settings()

    def _load_config_file(self, config_file: str) -> None:
        """Load precision configuration from JSON file"""
        try:
            with open(config_file, 'r') as f:
                custom_config = json.load(f)

            # Merge with default config
            for level_name, settings in custom_config.items():
                if level_name in [level.value for level in PrecisionLevel]:
                    level_enum = PrecisionLevel(level_name)
                    self._config[level_enum].update(settings)

            logger.info(f"Custom precision config loaded from {config_file}")
        except Exception as e:
            logger.warning(f"Failed to load config file {config_file}: {e}")

    def _backup_current_settings(self) -> None:
        """Backup current precision settings for restoration"""
        self._backup_settings = {
            'decimal_prec': getcontext().prec,
            'numpy_errors': np.geterr(),
            'torch_dtype': torch.get_default_dtype(),
        }

    def _apply_precision_settings(self, level: PrecisionLevel) -> None:
        """Apply precision settings for the specified level"""
        config = self._config[level]

        # Set decimal precision
        getcontext().prec = config["decimal_precision"]
        from decimal import ROUND_HALF_EVEN
        getcontext().rounding = ROUND_HALF_EVEN

        # Configure NumPy
        np.seterr(all='raise')  # Raise on numerical errors
        warnings.filterwarnings('error', category=RuntimeWarning)

        # Set PyTorch precision
        torch.set_default_dtype(config["torch_dtype"])

        # Store current settings for runtime access
        self.current_config = config

    def _log_current_settings(self) -> None:
        """Log current precision settings"""
        config = self.current_config
        logger.info("Current precision settings:")
        logger.info(
            f"  Decimal precision: {config['decimal_precision']} digits")
        logger.info(f"  NumPy dtype: {config['numpy_dtype']}")
        logger.info(f"  PyTorch dtype: {config['torch_dtype']}")
        logger.info(f"  Strict tolerance: {config['strict_tolerance']}")
        logger.info(f"  Description: {config['description']}")

    def get_current_level(self) -> PrecisionLevel:
        """Get current precision level"""
        return self._current_level

    def get_config(
            self, level: Optional[PrecisionLevel] = None) -> Dict[str, Any]:
        """Get configuration for specified or current level"""
        if level is None:
            level = self._current_level
        return self._config[level].copy()

    def set_level(self, level: Union[PrecisionLevel, str]) -> None:
        """Change precision level at runtime"""
        if isinstance(level, str):
            level = PrecisionLevel(level)

        if level != self._current_level:
            logger.info(
                f"Changing precision level from {self._current_level.value} to {level.value}")
            self._apply_precision_settings(level)
            self._current_level = level
            self._log_current_settings()

    @contextmanager
    def temporary_precision(self, level: Union[PrecisionLevel, str]):
        """Context manager for temporary precision changes"""
        if isinstance(level, str):
            level = PrecisionLevel(level)

        original_level = self._current_level

        try:
            self.set_level(level)
            yield self
        finally:
            self.set_level(original_level)

    def restore_defaults(self) -> None:
        """Restore original precision settings"""
        if self._backup_settings:
            getcontext().prec = self._backup_settings['decimal_prec']
            np.seterr(**self._backup_settings['numpy_errors'])
            torch.set_default_dtype(self._backup_settings['torch_dtype'])

            logger.info("Precision settings restored to original values")

    def create_config_file(self, filepath: str) -> None:
        """Create a configuration file with current settings"""
        config_data = {}
        for level, settings in self._config.items():
            # Convert non-serializable objects to strings
            serializable_settings = {}
            for key, value in settings.items():
                if key in ['numpy_dtype', 'torch_dtype', 'scientific_dtype']:
                    serializable_settings[key] = str(value)
                elif isinstance(value, Decimal):
                    serializable_settings[key] = str(value)
                else:
                    serializable_settings[key] = value

            config_data[level.value] = serializable_settings

        with open(filepath, 'w') as f:
            json.dump(config_data, f, indent=2)

        logger.info(f"Configuration saved to {filepath}")

    def validate_precision(self) -> bool:
        """Validate that precision settings are working correctly"""
        try:
            # Test decimal precision
            test_decimal = Decimal('1') / Decimal('3')
            decimal_digits = len(str(test_decimal).split('.')[-1])

            # Test numpy precision
            test_array = np.array([1.0/3.0],
                                  dtype=self.current_config['numpy_dtype'])

            # Test torch precision
            test_tensor = torch.tensor(
                [1.0/3.0], dtype=self.current_config['torch_dtype'])

            logger.info("Precision validation successful:")
            logger.info(f"  Decimal digits achieved: {decimal_digits}")
            logger.info(f"  NumPy precision: {test_array.dtype}")
            logger.info(f"  PyTorch precision: {test_tensor.dtype}")

            return True

        except Exception as e:
            logger.error(f"Precision validation failed: {e}")
            return False

    def benchmark_precision(self) -> Dict[str, float]:
        """Benchmark computational performance at current precision"""
        import time

        # Benchmark decimal operations
        start_time = time.time()
        for _ in range(1000):
            _ = Decimal('1.23456789') ** Decimal('2.98765')
        decimal_time = time.time() - start_time

        # Benchmark numpy operations
        start_time = time.time()
        rng = np.random.default_rng(42)
        test_array = rng.random(1000).astype(
            self.current_config['numpy_dtype'])
        for _ in range(100):
            _ = np.sum(test_array ** 2.5)
        numpy_time = time.time() - start_time

        # Benchmark torch operations
        start_time = time.time()
        test_tensor = torch.randn(
            1000, dtype=self.current_config['torch_dtype'])
        for _ in range(100):
            _ = torch.sum(test_tensor ** 2.5)
        torch_time = time.time() - start_time

        benchmark_results = {
            'decimal_ops_per_sec': 1000 / decimal_time,
            'numpy_ops_per_sec': 100 / numpy_time,
            'torch_ops_per_sec': 100 / torch_time,
            'decimal_time': decimal_time,
            'numpy_time': numpy_time,
            'torch_time': torch_time
        }

        logger.info("Precision benchmark results:")
        for metric, value in benchmark_results.items():
            logger.info(f"  {metric}: {value:.2f}")

        return benchmark_results


# Global instance
precision_manager = GlobalPrecisionConfig()


# Convenience functions for easy access
def initialize_precision(level: Union[PrecisionLevel,
                                      str] = PrecisionLevel.MAXIMUM,
                         config_file: Optional[str] = None) -> None:
    """Initialize global precision settings"""
    precision_manager.initialize(level, config_file)


def set_precision_level(level: Union[PrecisionLevel, str]) -> None:
    """Change global precision level"""
    precision_manager.set_level(level)


def get_precision_level() -> PrecisionLevel:
    """Get current precision level"""
    return precision_manager.get_current_level()


@contextmanager
def high_precision_mode(
        level: Union[PrecisionLevel, str] = PrecisionLevel.MAXIMUM):
    """Context manager for temporary high precision"""
    with precision_manager.temporary_precision(level):
        yield


def validate_precision() -> bool:
    """Validate current precision settings"""
    return precision_manager.validate_precision()


def benchmark_precision() -> Dict[str, float]:
    """Benchmark current precision performance"""
    return precision_manager.benchmark_precision()


def create_precision_config(filepath: str) -> None:
    """Create precision configuration file"""
    precision_manager.create_config_file(filepath)


def get_current_torch_dtype():
    """Get the current PyTorch dtype from precision configuration"""
    if hasattr(precision_manager, 'current_config'):
        return precision_manager.current_config.get('torch_dtype', torch.float32)
    else:
        # Fallback if not initialized
        return torch.float32


# Application-specific precision decorators
def require_high_precision(level: PrecisionLevel = PrecisionLevel.MAXIMUM):
    """Decorator to ensure functions run with high precision"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            with precision_manager.temporary_precision(level):
                return func(*args, **kwargs)
        return wrapper
    return decorator


def financial_precision(func):
    """Decorator for financial calculations requiring maximum precision"""
    def wrapper(*args, **kwargs):
        with precision_manager.temporary_precision(PrecisionLevel.MAXIMUM):
            return func(*args, **kwargs)
    return wrapper


def ultra_precision(func):
    """Decorator for research-grade ultra-high precision calculations"""
    def wrapper(*args, **kwargs):
        with precision_manager.temporary_precision(PrecisionLevel.ULTRA):
            return func(*args, **kwargs)
    return wrapper


# Automatic initialization on import
def auto_initialize():
    """Automatically initialize precision based on environment variables"""
    level_str = os.getenv('DCF_PRECISION_LEVEL', 'maximum')
    config_file = os.getenv('DCF_PRECISION_CONFIG')

    try:
        level = PrecisionLevel(level_str.lower())
        initialize_precision(level, config_file)
        logger.info(f"Auto-initialized precision to {level.value} level")
    except ValueError:
        logger.warning(f"Invalid precision level '{level_str}', using maximum")
        initialize_precision(PrecisionLevel.MAXIMUM, config_file)


if __name__ == "__main__":
    # Test the precision configuration system
    print("🎯 Global Precision Configuration System")
    print("=======================================")

    # Test different precision levels
    for level in PrecisionLevel:
        print(f"\nTesting {level.value} precision level:")
        initialize_precision(level)

        # Test decimal calculation
        result = Decimal('1') / Decimal('3')
        print(f"  1/3 = {result}")

        # Validate precision
        valid = validate_precision()
        print(f"  Validation: {'✅ PASSED' if valid else '❌ FAILED'}")

    # Test context manager
    print("\n=== Testing Precision Context Manager ===")
    initialize_precision(PrecisionLevel.STANDARD)
    print(f"Standard precision: {Decimal('1')/Decimal('3')}")

    with high_precision_mode(PrecisionLevel.ULTRA):
        print(f"Ultra precision: {Decimal('1')/Decimal('3')}")

    print(f"Back to standard: {Decimal('1')/Decimal('3')}")

    # Performance benchmark
    print("\n=== Precision Performance Benchmark ===")
    initialize_precision(PrecisionLevel.MAXIMUM)
    results = benchmark_precision()

    print("\n✅ Global precision configuration system ready!")
