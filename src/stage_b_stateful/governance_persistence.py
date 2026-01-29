"""
Governance Layer Persistence for Mamba Phase-2 Stateful Learning.

This module manages persistence for GOVERNANCE LAYERS that sit alongside
Mamba, NOT as feature families. These include:

1. MambaCalibrationTracker - Tracks Mamba (μ, σ²) vs realized returns
2. OnlineLearningState - Tracks online learning metrics and drift

IMPORTANT ARCHITECTURAL PRINCIPLE:
────────────────────────────────────────────────────────────────────────────
Calibration and Online Learning are NOT feature families. They are
governance layers that:
- Monitor Mamba model quality
- Gate learning decisions
- Track model drift
- Persist separately from prep_families output

They have their own directory structure:
    artifacts/governance/
        mamba_calibration/
            {model_id}_{horizon}_{symbol}.pkl
        online_learning/
            {model_id}_{horizon}_{symbol}.pkl
────────────────────────────────────────────────────────────────────────────

State Key Design:
    state_key = (model_id, horizon, symbol)

If horizon changes → key mismatch → clean initialization.
This prevents accidental state leakage between different horizon configs.

Copyright 2024-2025. All rights reserved.
"""

from __future__ import annotations

import hashlib
import logging
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Directory Structure
# ─────────────────────────────────────────────────────────────────────────────

GOVERNANCE_ROOT = Path("artifacts/governance")
MAMBA_CALIBRATION_DIR = GOVERNANCE_ROOT / "mamba_calibration"
ONLINE_LEARNING_DIR = GOVERNANCE_ROOT / "online_learning"

# Ensure directories exist
GOVERNANCE_ROOT.mkdir(parents=True, exist_ok=True)
MAMBA_CALIBRATION_DIR.mkdir(parents=True, exist_ok=True)
ONLINE_LEARNING_DIR.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# State Key Management
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class GovernanceStateKey:
    """
    Immutable state key for governance layer persistence.
    
    The key includes horizon to prevent accidental leakage:
    - If horizon changes → key mismatch → clean initialization
    - This is by design to avoid mixing state across different horizons
    """
    model_id: str
    horizon: int
    symbol: str
    
    def __post_init__(self):
        # Normalize symbol to uppercase
        object.__setattr__(self, 'symbol', str(self.symbol).upper())
    
    def to_tuple(self) -> Tuple[str, int, str]:
        """Return as tuple for hashing/comparison."""
        return (self.model_id, self.horizon, self.symbol)
    
    def to_filename(self, suffix: str = ".pkl") -> str:
        """Generate filename for persistence."""
        # Use hash for long model_ids
        if len(self.model_id) > 32:
            model_hash = hashlib.sha256(self.model_id.encode()).hexdigest()[:12]
        else:
            model_hash = self.model_id.replace("/", "_").replace(":", "_")
        return f"{model_hash}_h{self.horizon}_{self.symbol}{suffix}"
    
    def __str__(self) -> str:
        return f"GovernanceStateKey(model={self.model_id[:20]}..., horizon={self.horizon}, symbol={self.symbol})"


def make_state_key(
    model_id: str,
    horizon: int,
    symbol: str,
) -> GovernanceStateKey:
    """Factory for creating governance state keys."""
    return GovernanceStateKey(
        model_id=str(model_id),
        horizon=int(horizon),
        symbol=str(symbol).upper(),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Mamba Calibration Persistence
# ─────────────────────────────────────────────────────────────────────────────

def save_mamba_calibration_state(
    state_key: GovernanceStateKey,
    tracker_state: Dict[str, Any],
) -> bool:
    """
    Persist MambaCalibrationTracker state to disk.
    
    Args:
        state_key: Governance state key (model_id, horizon, symbol).
        tracker_state: Serializable tracker state dict.
    
    Returns:
        True if save succeeded, False otherwise.
    """
    filepath = MAMBA_CALIBRATION_DIR / state_key.to_filename()
    try:
        data = {
            "state_key": state_key.to_tuple(),
            "tracker_state": tracker_state,
        }
        with open(filepath, "wb") as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
        logger.debug("[governance.mamba_calib] saved state to %s", filepath.name)
        return True
    except Exception as e:
        logger.warning("[governance.mamba_calib] failed to save state: %s", e)
        return False


def load_mamba_calibration_state(
    state_key: GovernanceStateKey,
) -> Optional[Dict[str, Any]]:
    """
    Load MambaCalibrationTracker state from disk.
    
    Args:
        state_key: Governance state key (model_id, horizon, symbol).
    
    Returns:
        Tracker state dict if found and key matches, None otherwise.
        Returns None on key mismatch (horizon changed) → clean initialization.
    """
    filepath = MAMBA_CALIBRATION_DIR / state_key.to_filename()
    if not filepath.exists():
        logger.debug("[governance.mamba_calib] no saved state for %s", state_key)
        return None
    
    try:
        with open(filepath, "rb") as f:
            data = pickle.load(f)
        
        # Verify key matches (horizon check)
        saved_key = data.get("state_key")
        if saved_key != state_key.to_tuple():
            logger.warning(
                "[governance.mamba_calib] key mismatch (horizon changed?): "
                "saved=%s, requested=%s → clean initialization",
                saved_key, state_key.to_tuple()
            )
            return None
        
        logger.debug("[governance.mamba_calib] loaded state from %s", filepath.name)
        return data.get("tracker_state")
    except Exception as e:
        logger.warning("[governance.mamba_calib] failed to load state: %s", e)
        return None


def clear_mamba_calibration_state(state_key: GovernanceStateKey) -> bool:
    """Delete persisted state for a given key."""
    filepath = MAMBA_CALIBRATION_DIR / state_key.to_filename()
    if filepath.exists():
        try:
            filepath.unlink()
            logger.info("[governance.mamba_calib] cleared state for %s", state_key)
            return True
        except Exception as e:
            logger.warning("[governance.mamba_calib] failed to clear state: %s", e)
            return False
    return True


# ─────────────────────────────────────────────────────────────────────────────
# Online Learning Persistence
# ─────────────────────────────────────────────────────────────────────────────

def save_online_learning_state(
    state_key: GovernanceStateKey,
    learning_state: Dict[str, Any],
) -> bool:
    """
    Persist OnlineLearningSystem state to disk.
    
    Args:
        state_key: Governance state key (model_id, horizon, symbol).
        learning_state: Serializable learning system state dict.
    
    Returns:
        True if save succeeded, False otherwise.
    """
    filepath = ONLINE_LEARNING_DIR / state_key.to_filename()
    try:
        data = {
            "state_key": state_key.to_tuple(),
            "learning_state": learning_state,
        }
        with open(filepath, "wb") as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
        logger.debug("[governance.online_learning] saved state to %s", filepath.name)
        return True
    except Exception as e:
        logger.warning("[governance.online_learning] failed to save state: %s", e)
        return False


def load_online_learning_state(
    state_key: GovernanceStateKey,
) -> Optional[Dict[str, Any]]:
    """
    Load OnlineLearningSystem state from disk.
    
    Args:
        state_key: Governance state key (model_id, horizon, symbol).
    
    Returns:
        Learning state dict if found and key matches, None otherwise.
        Returns None on key mismatch (horizon changed) → clean initialization.
    """
    filepath = ONLINE_LEARNING_DIR / state_key.to_filename()
    if not filepath.exists():
        logger.debug("[governance.online_learning] no saved state for %s", state_key)
        return None
    
    try:
        with open(filepath, "rb") as f:
            data = pickle.load(f)
        
        # Verify key matches (horizon check)
        saved_key = data.get("state_key")
        if saved_key != state_key.to_tuple():
            logger.warning(
                "[governance.online_learning] key mismatch (horizon changed?): "
                "saved=%s, requested=%s → clean initialization",
                saved_key, state_key.to_tuple()
            )
            return None
        
        logger.debug("[governance.online_learning] loaded state from %s", filepath.name)
        return data.get("learning_state")
    except Exception as e:
        logger.warning("[governance.online_learning] failed to load state: %s", e)
        return None


def clear_online_learning_state(state_key: GovernanceStateKey) -> bool:
    """Delete persisted state for a given key."""
    filepath = ONLINE_LEARNING_DIR / state_key.to_filename()
    if filepath.exists():
        try:
            filepath.unlink()
            logger.info("[governance.online_learning] cleared state for %s", state_key)
            return True
        except Exception as e:
            logger.warning("[governance.online_learning] failed to clear state: %s", e)
            return False
    return True


# ─────────────────────────────────────────────────────────────────────────────
# Utility Functions
# ─────────────────────────────────────────────────────────────────────────────

def list_governance_states(layer: str = "all") -> Dict[str, list]:
    """
    List all persisted governance states.
    
    Args:
        layer: "mamba_calibration", "online_learning", or "all"
    
    Returns:
        Dict mapping layer name to list of state filenames.
    """
    result = {}
    
    if layer in ("all", "mamba_calibration"):
        result["mamba_calibration"] = sorted([
            f.name for f in MAMBA_CALIBRATION_DIR.glob("*.pkl")
        ])
    
    if layer in ("all", "online_learning"):
        result["online_learning"] = sorted([
            f.name for f in ONLINE_LEARNING_DIR.glob("*.pkl")
        ])
    
    return result


def clear_all_governance_states(layer: str = "all") -> int:
    """
    Clear all persisted governance states.
    
    Args:
        layer: "mamba_calibration", "online_learning", or "all"
    
    Returns:
        Number of files deleted.
    """
    count = 0
    
    if layer in ("all", "mamba_calibration"):
        for f in MAMBA_CALIBRATION_DIR.glob("*.pkl"):
            try:
                f.unlink()
                count += 1
            except Exception:
                pass
    
    if layer in ("all", "online_learning"):
        for f in ONLINE_LEARNING_DIR.glob("*.pkl"):
            try:
                f.unlink()
                count += 1
            except Exception:
                pass
    
    logger.info("[governance] cleared %d state files for layer=%s", count, layer)
    return count
