"""
PyTorch Dtype Consistency Module

This module ensures consistent tensor dtypes across all PyTorch models
in the ensemble to prevent Float/Double mismatches during training and inference.

Key Features:
- Global dtype configuration management
- Tensor casting utilities for model consistency
- Validation functions to catch dtype mismatches early
- Integration with ensemble forecaster components

Usage:
    import torch_dtype_guard
    torch_dtype_guard.set_global_dtype(torch.float64)
    
    # All subsequent tensor operations will use float64
"""

import torch
import logging
from typing import Optional

logger = logging.getLogger(__name__)

# Global dtype state
_GLOBAL_DTYPE = torch.float32
_DTYPE_INITIALIZED = False


def set_global_dtype(dtype: torch.dtype = torch.float64) -> None:
    """
    Set global PyTorch dtype for consistent tensor operations
    
    Args:
        dtype: Target PyTorch dtype (float32 or float64)
    """
    global _GLOBAL_DTYPE, _DTYPE_INITIALIZED
    
    _GLOBAL_DTYPE = dtype
    torch.set_default_dtype(dtype)
    _DTYPE_INITIALIZED = True
    
    logger.info(f"✅ Global PyTorch dtype set to {dtype}")


def get_global_dtype() -> torch.dtype:
    """Get the current global dtype"""
    return _GLOBAL_DTYPE


def ensure_tensor_dtype(tensor: torch.Tensor, 
                       target_dtype: Optional[torch.dtype] = None) -> torch.Tensor:
    """
    Ensure tensor has the correct dtype
    
    Args:
        tensor: Input tensor
        target_dtype: Target dtype (uses global if None)
        
    Returns:
        Tensor with correct dtype
    """
    if target_dtype is None:
        target_dtype = _GLOBAL_DTYPE
    
    if tensor.dtype != target_dtype:
        logger.debug(f"Converting tensor from {tensor.dtype} to {target_dtype}")
        return tensor.to(dtype=target_dtype)
    
    return tensor


def validate_model_dtypes(model: torch.nn.Module, 
                         expected_dtype: Optional[torch.dtype] = None) -> bool:
    """
    Validate that all model parameters have consistent dtypes
    
    Args:
        model: PyTorch model to validate
        expected_dtype: Expected dtype (uses global if None)
        
    Returns:
        True if all parameters have consistent dtypes
    """
    if expected_dtype is None:
        expected_dtype = _GLOBAL_DTYPE
    
    inconsistent_params = []
    
    for name, param in model.named_parameters():
        if param.dtype != expected_dtype:
            inconsistent_params.append((name, param.dtype))
    
    if inconsistent_params:
        logger.warning(f"Found {len(inconsistent_params)} parameters with inconsistent dtypes:")
        for name, dtype in inconsistent_params[:5]:  # Show first 5
            logger.warning(f"  {name}: {dtype} (expected {expected_dtype})")
        return False
    
    return True


def convert_model_dtype(model: torch.nn.Module, 
                       target_dtype: Optional[torch.dtype] = None) -> torch.nn.Module:
    """
    Convert all model parameters to target dtype
    
    Args:
        model: PyTorch model to convert
        target_dtype: Target dtype (uses global if None)
        
    Returns:
        Model with converted parameters
    """
    if target_dtype is None:
        target_dtype = _GLOBAL_DTYPE
    
    logger.info(f"Converting model to dtype {target_dtype}")
    return model.to(dtype=target_dtype)


def ensure_ensemble_consistency(*models: torch.nn.Module) -> None:
    """
    Ensure all ensemble models have consistent dtypes
    
    Args:
        *models: Variable number of PyTorch models
    """
    if not _DTYPE_INITIALIZED:
        set_global_dtype(torch.float64)  # Default to float64 for financial precision
    
    inconsistent_models = []
    
    for i, model in enumerate(models):
        if not validate_model_dtypes(model):
            inconsistent_models.append(i)
            logger.info(f"Converting model {i} to {_GLOBAL_DTYPE}")
            convert_model_dtype(model)
    
    if inconsistent_models:
        logger.info(f"✅ Fixed dtype consistency for {len(inconsistent_models)} models")
    else:
        logger.info("✅ All ensemble models have consistent dtypes")


class DtypeGuard:
    """Context manager for ensuring dtype consistency"""
    
    def __init__(self, dtype: torch.dtype = torch.float64):
        self.dtype = dtype
        self.original_dtype = None
    
    def __enter__(self):
        self.original_dtype = torch.get_default_dtype()
        set_global_dtype(self.dtype)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.original_dtype:
            torch.set_default_dtype(self.original_dtype)


# Convenience functions
def with_float64():
    """Context manager for float64 operations"""
    return DtypeGuard(torch.float64)


def with_float32():
    """Context manager for float32 operations"""
    return DtypeGuard(torch.float32)


# Initialize on import
if not _DTYPE_INITIALIZED:
    set_global_dtype(torch.float64)  # Default to high precision for financial modeling