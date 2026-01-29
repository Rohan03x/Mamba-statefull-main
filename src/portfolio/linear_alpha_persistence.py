"""Persistence for Linear Alpha Combiner state.

Handles saving and loading of trained linear model coefficients,
standardization statistics, and configuration to/from disk.

Storage format: JSON for portability and debuggability.
Location: artifacts/governance/linear_combiner/
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence

logger = logging.getLogger(__name__)

# Default storage directory
_LINEAR_COMBINER_DIR = Path("artifacts/governance/linear_combiner")


def _ensure_dir(path: Path) -> Path:
    """Ensure directory exists."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def _compute_universe_hash(symbols: Sequence[str]) -> str:
    """Compute hash of symbol universe for keying."""
    syms_str = ",".join(sorted(str(s).upper() for s in symbols))
    return hashlib.sha256(syms_str.encode()).hexdigest()[:12]


def _state_key(horizon: int, universe_hash: str) -> str:
    """Generate filename key for state."""
    return f"linear_combiner_h{horizon}_{universe_hash}"


def save_linear_combiner_state(
    state_dict: Dict,
    *,
    horizon: int,
    symbols: Sequence[str],
    storage_dir: Optional[Path] = None,
) -> Path:
    """Save LinearCombinerState to disk.
    
    Args:
        state_dict: Result of LinearCombinerState.to_dict()
        horizon: Prediction horizon
        symbols: Symbol universe
        storage_dir: Override storage directory
    
    Returns:
        Path to saved file
    """
    storage_dir = Path(storage_dir) if storage_dir else _LINEAR_COMBINER_DIR
    _ensure_dir(storage_dir)
    
    universe_hash = _compute_universe_hash(symbols)
    key = _state_key(horizon, universe_hash)
    
    # Add metadata
    state_dict["_meta"] = {
        "horizon": horizon,
        "universe_hash": universe_hash,
        "n_symbols": len(symbols),
        "symbols_sample": list(symbols)[:10],  # First 10 for reference
    }
    
    filepath = storage_dir / f"{key}.json"
    
    try:
        with open(filepath, "w") as f:
            json.dump(state_dict, f, indent=2, default=_json_default)
        logger.info("[linear.persist] Saved state to %s", filepath)
        return filepath
    except Exception as e:
        logger.error("[linear.persist] Failed to save: %s", e)
        raise


def load_linear_combiner_state(
    *,
    horizon: int,
    symbols: Sequence[str],
    storage_dir: Optional[Path] = None,
) -> Optional[Dict]:
    """Load LinearCombinerState from disk.
    
    Args:
        horizon: Prediction horizon
        symbols: Symbol universe (must match saved universe)
        storage_dir: Override storage directory
    
    Returns:
        State dict if found and valid, else None
    """
    storage_dir = Path(storage_dir) if storage_dir else _LINEAR_COMBINER_DIR
    
    universe_hash = _compute_universe_hash(symbols)
    key = _state_key(horizon, universe_hash)
    filepath = storage_dir / f"{key}.json"
    
    if not filepath.exists():
        logger.debug("[linear.persist] No saved state found at %s", filepath)
        return None
    
    try:
        with open(filepath, "r") as f:
            state_dict = json.load(f)
        
        # Validate universe hash
        meta = state_dict.get("_meta", {})
        if meta.get("universe_hash") != universe_hash:
            logger.warning(
                "[linear.persist] Universe hash mismatch: expected %s, got %s",
                universe_hash, meta.get("universe_hash")
            )
            return None
        
        logger.info("[linear.persist] Loaded state from %s", filepath)
        return state_dict
    
    except Exception as e:
        logger.error("[linear.persist] Failed to load: %s", e)
        return None


def delete_linear_combiner_state(
    *,
    horizon: int,
    symbols: Sequence[str],
    storage_dir: Optional[Path] = None,
) -> bool:
    """Delete saved LinearCombinerState.
    
    Returns:
        True if deleted, False if not found
    """
    storage_dir = Path(storage_dir) if storage_dir else _LINEAR_COMBINER_DIR
    
    universe_hash = _compute_universe_hash(symbols)
    key = _state_key(horizon, universe_hash)
    filepath = storage_dir / f"{key}.json"
    
    if filepath.exists():
        filepath.unlink()
        logger.info("[linear.persist] Deleted state at %s", filepath)
        return True
    
    return False


def list_linear_combiner_states(
    storage_dir: Optional[Path] = None,
) -> List[Dict]:
    """List all saved LinearCombinerState files.
    
    Returns:
        List of metadata dicts for each saved state
    """
    storage_dir = Path(storage_dir) if storage_dir else _LINEAR_COMBINER_DIR
    
    if not storage_dir.exists():
        return []
    
    results = []
    for filepath in storage_dir.glob("linear_combiner_*.json"):
        try:
            with open(filepath, "r") as f:
                state_dict = json.load(f)
            meta = state_dict.get("_meta", {})
            meta["filepath"] = str(filepath)
            meta["n_updates"] = state_dict.get("n_updates", 0)
            meta["r_squared"] = state_dict.get("r_squared", 0.0)
            results.append(meta)
        except Exception:
            continue
    
    return results


def _json_default(obj):
    """JSON serialization fallback for numpy types."""
    import numpy as np
    
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer, np.floating)):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
