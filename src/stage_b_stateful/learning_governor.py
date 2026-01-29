"""
Learning Governor: Calibration-Driven Runtime Risk Control for Mamba Learning.

This module implements explicit, stricter learning governance based on the
MambaCalibrationTracker. It provides:

1. Three-Zone Policy (stable and auditable):
   - GREEN (cal >= 0.70): Allow slow updates (reduce frequency)
   - YELLOW (0.55 <= cal < 0.70): Allow normal incremental updates
   - RED (cal < 0.55): Freeze learning OR only allow safe recalibration

2. Drift-Triggered Micro-Updates:
   - Detect rapid calibration drops (Δcal < -0.10 over 5 sessions)
   - Trigger micro-update (small steps) instead of full retrain
   - Increase update cadence temporarily

3. Human-in-the-Loop Gate:
   - Auto-revert to last checkpoint if calibration worsens after update
   - Require manual "unlock" flag to resume updates after auto-revert

Copyright 2024-2026. All rights reserved.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Tuple

from collections import deque

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Configuration Constants
# ─────────────────────────────────────────────────────────────────────────────

# Three-zone calibration thresholds
CALIB_GREEN_THRESHOLD = float(os.environ.get("LEARNING_GOV_GREEN_THRESHOLD", "0.70"))
CALIB_YELLOW_THRESHOLD = float(os.environ.get("LEARNING_GOV_YELLOW_THRESHOLD", "0.55"))

# Drift detection: calibration drop over N sessions
DRIFT_WINDOW_SESSIONS = int(os.environ.get("LEARNING_GOV_DRIFT_WINDOW", "5"))
DRIFT_DROP_THRESHOLD = float(os.environ.get("LEARNING_GOV_DRIFT_DROP", "-0.10"))

# Learning rate multipliers per zone
LR_MULT_GREEN = float(os.environ.get("LEARNING_GOV_LR_GREEN", "0.5"))  # Slow updates
LR_MULT_YELLOW = float(os.environ.get("LEARNING_GOV_LR_YELLOW", "1.0"))  # Normal
LR_MULT_RED = float(os.environ.get("LEARNING_GOV_LR_RED", "0.0"))  # Frozen

# Update cadence multipliers per zone
CADENCE_MULT_GREEN = float(os.environ.get("LEARNING_GOV_CADENCE_GREEN", "2.0"))  # Double interval
CADENCE_MULT_YELLOW = float(os.environ.get("LEARNING_GOV_CADENCE_YELLOW", "1.0"))  # Normal
CADENCE_MULT_RED = float(os.environ.get("LEARNING_GOV_CADENCE_RED", "0.0"))  # No updates

# Micro-update configuration
MICRO_UPDATE_LR_MULT = float(os.environ.get("LEARNING_GOV_MICRO_LR", "0.25"))
MICRO_UPDATE_CADENCE_BOOST = int(os.environ.get("LEARNING_GOV_MICRO_CADENCE", "3"))

# Auto-revert threshold: if calibration drops by this much after update
AUTO_REVERT_THRESHOLD = float(os.environ.get("LEARNING_GOV_AUTO_REVERT", "-0.05"))


# ─────────────────────────────────────────────────────────────────────────────
# Data Structures
# ─────────────────────────────────────────────────────────────────────────────

class LearningZone(Enum):
    """Learning zone based on calibration score."""
    GREEN = "green"   # cal >= 0.70: slow updates allowed
    YELLOW = "yellow" # 0.55 <= cal < 0.70: normal updates
    RED = "red"       # cal < 0.55: learning frozen


class LearningDecision(Enum):
    """Learning governor decision for this session."""
    ALLOW_UPDATE = "allow_update"           # Normal update allowed
    ALLOW_MICRO_UPDATE = "allow_micro"      # Only micro-update allowed
    FREEZE_LEARNING = "freeze"              # No learning updates
    REQUIRE_UNLOCK = "require_unlock"       # Human unlock required


@dataclass
class LearningGovernorDecision:
    """Point-in-time decision from the Learning Governor."""
    
    date: str  # Session date YYYY-MM-DD
    day_idx: int  # OOS day index
    
    # Calibration state
    calibration_score: float  # Current calibration [0, 1]
    zone: LearningZone  # GREEN/YELLOW/RED
    
    # Decision
    decision: LearningDecision
    learning_rate_mult: float  # Multiplier for base learning rate
    cadence_mult: float  # Multiplier for update cadence (interval)
    
    # Drift detection
    drift_detected: bool = False
    drift_delta: float = 0.0  # Change in calibration over drift window
    
    # Human-in-the-loop state
    is_locked: bool = False  # True if human unlock required
    auto_reverted: bool = False  # True if we just reverted
    
    # Reason for decision
    reason: str = ""


@dataclass
class LearningGovernorState:
    """Serializable state for the Learning Governor."""
    
    calibration_history: "List[float]" = field(default_factory=list)  # noqa: F821
    decision_history: "List[Dict[str, Any]]" = field(default_factory=list)  # noqa: F821
    
    # Checkpoint tracking
    last_checkpoint_date: Optional[str] = None
    last_checkpoint_calib: float = 0.5
    
    # Lock state
    is_locked: bool = False
    lock_reason: Optional[str] = None
    lock_date: Optional[str] = None
    
    # Micro-update state
    micro_update_sessions_remaining: int = 0
    
    # Statistics
    total_updates: int = 0
    total_freezes: int = 0
    total_micro_updates: int = 0
    total_auto_reverts: int = 0


class LearningGovernor:
    """
    Calibration-driven runtime risk control for Mamba learning.
    
    Usage:
        governor = LearningGovernor(checkpoint_dir=Path("artifacts/checkpoints"))
        
        # During OOS loop:
        for i, day in enumerate(oos_days):
            # Get calibration from Mamba tracker
            calib_snapshot = mamba_calib_tracker.get_calibration()
            calib_score = calib_snapshot.calibration_overall
            
            # Get learning decision
            decision = governor.get_decision(
                date=str(day)[:10],
                day_idx=i,
                calibration_score=calib_score,
            )
            
            # Apply decision
            if decision.decision == LearningDecision.ALLOW_UPDATE:
                lr = base_lr * decision.learning_rate_mult
                # ... do update
            elif decision.decision == LearningDecision.FREEZE_LEARNING:
                # Skip update
                pass
            
            # After update, report outcome
            governor.report_update_outcome(
                date=str(day)[:10],
                pre_calib=calib_score,
                post_calib=new_calib_score,
            )
    """
    
    def __init__(
        self,
        *,
        checkpoint_dir: Optional[Path] = None,
        decision_log_path: Optional[Path] = None,
        green_threshold: float = CALIB_GREEN_THRESHOLD,
        yellow_threshold: float = CALIB_YELLOW_THRESHOLD,
        drift_window: int = DRIFT_WINDOW_SESSIONS,
        drift_drop_threshold: float = DRIFT_DROP_THRESHOLD,
    ) -> None:
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None
        self.decision_log_path = Path(decision_log_path) if decision_log_path else None
        
        self.green_threshold = float(green_threshold)
        self.yellow_threshold = float(yellow_threshold)
        self.drift_window = int(drift_window)
        self.drift_drop_threshold = float(drift_drop_threshold)
        
        # State
        self._state = LearningGovernorState()
        self._calib_buffer: Deque[float] = deque(maxlen=drift_window)
        
        # Load existing state if available
        self._load_state()
    
    def _get_zone(self, calibration_score: float) -> LearningZone:
        """Determine learning zone based on calibration score."""
        if calibration_score >= self.green_threshold:
            return LearningZone.GREEN
        elif calibration_score >= self.yellow_threshold:
            return LearningZone.YELLOW
        else:
            return LearningZone.RED
    
    def _detect_drift(self, calibration_score: float) -> Tuple[bool, float]:
        """Detect rapid calibration drop indicating drift."""
        if len(self._calib_buffer) < self.drift_window:
            return False, 0.0
        
        # Compute delta from window start to now
        old_calib = self._calib_buffer[0]
        delta = calibration_score - old_calib
        
        is_drift = delta < self.drift_drop_threshold
        return is_drift, delta
    
    def get_decision(
        self,
        *,
        date: str,
        day_idx: int,
        calibration_score: float,
    ) -> LearningGovernorDecision:
        """
        Get learning decision for this session.
        
        Args:
            date: Session date string (YYYY-MM-DD)
            day_idx: OOS day index
            calibration_score: Current Mamba calibration score [0, 1]
        
        Returns:
            LearningGovernorDecision with zone, decision, and multipliers.
        """
        # Update calibration buffer
        self._calib_buffer.append(calibration_score)
        self._state.calibration_history.append(calibration_score)
        
        # Determine zone
        zone = self._get_zone(calibration_score)
        
        # Detect drift
        drift_detected, drift_delta = self._detect_drift(calibration_score)
        
        # Check lock state
        if self._state.is_locked:
            decision = LearningDecision.REQUIRE_UNLOCK
            lr_mult = 0.0
            cadence_mult = 0.0
            reason = f"Locked since {self._state.lock_date}: {self._state.lock_reason}"
        
        # Check micro-update mode
        elif self._state.micro_update_sessions_remaining > 0:
            decision = LearningDecision.ALLOW_MICRO_UPDATE
            lr_mult = MICRO_UPDATE_LR_MULT
            cadence_mult = 1.0 / MICRO_UPDATE_CADENCE_BOOST  # Faster cadence
            self._state.micro_update_sessions_remaining -= 1
            reason = f"Micro-update mode ({self._state.micro_update_sessions_remaining} sessions remaining)"
        
        # Drift-triggered micro-update
        elif drift_detected and zone != LearningZone.RED:
            decision = LearningDecision.ALLOW_MICRO_UPDATE
            lr_mult = MICRO_UPDATE_LR_MULT
            cadence_mult = 1.0 / MICRO_UPDATE_CADENCE_BOOST
            self._state.micro_update_sessions_remaining = MICRO_UPDATE_CADENCE_BOOST
            self._state.total_micro_updates += 1
            reason = f"Drift detected (Δcal={drift_delta:.3f}), triggering micro-updates"
        
        # Zone-based decision
        elif zone == LearningZone.GREEN:
            decision = LearningDecision.ALLOW_UPDATE
            lr_mult = LR_MULT_GREEN
            cadence_mult = CADENCE_MULT_GREEN
            self._state.total_updates += 1
            reason = f"GREEN zone (cal={calibration_score:.3f} >= {self.green_threshold}): slow updates"
        
        elif zone == LearningZone.YELLOW:
            decision = LearningDecision.ALLOW_UPDATE
            lr_mult = LR_MULT_YELLOW
            cadence_mult = CADENCE_MULT_YELLOW
            self._state.total_updates += 1
            reason = f"YELLOW zone ({self.yellow_threshold} <= cal={calibration_score:.3f} < {self.green_threshold}): normal updates"
        
        else:  # RED
            decision = LearningDecision.FREEZE_LEARNING
            lr_mult = LR_MULT_RED
            cadence_mult = CADENCE_MULT_RED
            self._state.total_freezes += 1
            reason = f"RED zone (cal={calibration_score:.3f} < {self.yellow_threshold}): learning frozen"
        
        # Build decision object
        gov_decision = LearningGovernorDecision(
            date=date,
            day_idx=day_idx,
            calibration_score=calibration_score,
            zone=zone,
            decision=decision,
            learning_rate_mult=lr_mult,
            cadence_mult=cadence_mult,
            drift_detected=drift_detected,
            drift_delta=drift_delta,
            is_locked=self._state.is_locked,
            auto_reverted=False,
            reason=reason,
        )
        
        # Log decision
        self._log_decision(gov_decision)
        
        return gov_decision
    
    def report_update_outcome(
        self,
        *,
        date: str,
        pre_calib: float,
        post_calib: float,
        checkpoint_path: Optional[Path] = None,
    ) -> bool:
        """
        Report outcome of a learning update.
        
        If calibration worsened significantly, auto-revert and lock.
        
        Args:
            date: Session date
            pre_calib: Calibration before update
            post_calib: Calibration after update
            checkpoint_path: Path to checkpoint for potential revert
        
        Returns:
            True if update was accepted, False if auto-reverted.
        """
        delta = post_calib - pre_calib
        
        if delta < AUTO_REVERT_THRESHOLD:
            # Auto-revert trigger
            logger.warning(
                "[LearningGovernor] Calibration dropped %.3f after update (threshold=%.3f). "
                "Auto-reverting and locking.",
                delta, AUTO_REVERT_THRESHOLD
            )
            
            self._state.is_locked = True
            self._state.lock_reason = f"Auto-revert: calibration dropped {delta:.3f}"
            self._state.lock_date = date
            self._state.total_auto_reverts += 1
            
            # Log the revert
            self._log_auto_revert(date, pre_calib, post_calib, checkpoint_path)
            
            self._save_state()
            return False
        
        # Update accepted, save checkpoint reference
        if checkpoint_path is not None:
            self._state.last_checkpoint_date = date
            self._state.last_checkpoint_calib = post_calib
        
        self._save_state()
        return True
    
    def unlock(self, *, reason: str = "manual unlock") -> None:
        """
        Unlock learning after human review.
        
        Args:
            reason: Reason for unlocking (logged for audit)
        """
        logger.info("[LearningGovernor] Unlocking: %s", reason)
        self._state.is_locked = False
        self._state.lock_reason = None
        self._state.lock_date = None
        self._save_state()
    
    def get_state_summary(self) -> Dict[str, Any]:
        """Get summary of governor state for dashboards."""
        recent_calib = list(self._calib_buffer)
        return {
            "is_locked": self._state.is_locked,
            "lock_reason": self._state.lock_reason,
            "lock_date": self._state.lock_date,
            "current_zone": self._get_zone(recent_calib[-1]).value if recent_calib else "unknown",
            "recent_calibration": recent_calib,
            "micro_update_sessions_remaining": self._state.micro_update_sessions_remaining,
            "total_updates": self._state.total_updates,
            "total_freezes": self._state.total_freezes,
            "total_micro_updates": self._state.total_micro_updates,
            "total_auto_reverts": self._state.total_auto_reverts,
            "last_checkpoint_date": self._state.last_checkpoint_date,
            "last_checkpoint_calib": self._state.last_checkpoint_calib,
        }
    
    def _log_decision(self, decision: LearningGovernorDecision) -> None:
        """Log decision to file and decision history."""
        record = {
            "date": decision.date,
            "day_idx": decision.day_idx,
            "calibration_score": decision.calibration_score,
            "zone": decision.zone.value,
            "decision": decision.decision.value,
            "learning_rate_mult": decision.learning_rate_mult,
            "cadence_mult": decision.cadence_mult,
            "drift_detected": decision.drift_detected,
            "drift_delta": decision.drift_delta,
            "is_locked": decision.is_locked,
            "reason": decision.reason,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        
        self._state.decision_history.append(record)
        
        # Write to log file if configured
        if self.decision_log_path is not None:
            try:
                self.decision_log_path.parent.mkdir(parents=True, exist_ok=True)
                with open(self.decision_log_path, "a") as f:
                    f.write(json.dumps(record) + "\n")
            except Exception as e:
                logger.debug("[LearningGovernor] Failed to write decision log: %s", e)
        
        # Also log to standard logger
        logger.info(
            "[LearningGovernor] %s day=%d cal=%.3f zone=%s decision=%s lr_mult=%.2f: %s",
            decision.date, decision.day_idx, decision.calibration_score,
            decision.zone.value, decision.decision.value,
            decision.learning_rate_mult, decision.reason
        )
    
    def _log_auto_revert(
        self,
        date: str,
        pre_calib: float,
        post_calib: float,
        checkpoint_path: Optional[Path],
    ) -> None:
        """Log auto-revert event."""
        record = {
            "event": "auto_revert",
            "date": date,
            "pre_calib": pre_calib,
            "post_calib": post_calib,
            "delta": post_calib - pre_calib,
            "checkpoint_path": str(checkpoint_path) if checkpoint_path else None,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        
        if self.decision_log_path is not None:
            try:
                with open(self.decision_log_path, "a") as f:
                    f.write(json.dumps(record) + "\n")
            except Exception:
                pass
    
    def _load_state(self) -> None:
        """Load state from checkpoint directory."""
        if self.checkpoint_dir is None:
            return
        
        state_path = self.checkpoint_dir / "learning_governor_state.json"
        if not state_path.exists():
            return
        
        try:
            with open(state_path, "r") as f:
                data = json.load(f)
            
            self._state.is_locked = bool(data.get("is_locked", False))
            self._state.lock_reason = data.get("lock_reason")
            self._state.lock_date = data.get("lock_date")
            self._state.micro_update_sessions_remaining = int(data.get("micro_update_sessions_remaining", 0))
            self._state.total_updates = int(data.get("total_updates", 0))
            self._state.total_freezes = int(data.get("total_freezes", 0))
            self._state.total_micro_updates = int(data.get("total_micro_updates", 0))
            self._state.total_auto_reverts = int(data.get("total_auto_reverts", 0))
            self._state.last_checkpoint_date = data.get("last_checkpoint_date")
            self._state.last_checkpoint_calib = float(data.get("last_checkpoint_calib", 0.5))
            
            # Restore recent calibration buffer
            recent = data.get("recent_calibration", [])
            for c in recent[-self.drift_window:]:
                self._calib_buffer.append(float(c))
            
            logger.info("[LearningGovernor] Loaded state from %s", state_path)
        except Exception as e:
            logger.warning("[LearningGovernor] Failed to load state: %s", e)
    
    def _save_state(self) -> None:
        """Save state to checkpoint directory."""
        if self.checkpoint_dir is None:
            return
        
        try:
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
            state_path = self.checkpoint_dir / "learning_governor_state.json"
            
            data = {
                "is_locked": self._state.is_locked,
                "lock_reason": self._state.lock_reason,
                "lock_date": self._state.lock_date,
                "micro_update_sessions_remaining": self._state.micro_update_sessions_remaining,
                "total_updates": self._state.total_updates,
                "total_freezes": self._state.total_freezes,
                "total_micro_updates": self._state.total_micro_updates,
                "total_auto_reverts": self._state.total_auto_reverts,
                "last_checkpoint_date": self._state.last_checkpoint_date,
                "last_checkpoint_calib": self._state.last_checkpoint_calib,
                "recent_calibration": list(self._calib_buffer),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            
            with open(state_path, "w") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.debug("[LearningGovernor] Failed to save state: %s", e)
    
    def serialize(self) -> Dict[str, Any]:
        """Serialize governor state for checkpointing."""
        return {
            "calibration_history": self._state.calibration_history[-100:],
            "is_locked": self._state.is_locked,
            "lock_reason": self._state.lock_reason,
            "lock_date": self._state.lock_date,
            "micro_update_sessions_remaining": self._state.micro_update_sessions_remaining,
            "total_updates": self._state.total_updates,
            "total_freezes": self._state.total_freezes,
            "total_micro_updates": self._state.total_micro_updates,
            "total_auto_reverts": self._state.total_auto_reverts,
            "last_checkpoint_date": self._state.last_checkpoint_date,
            "last_checkpoint_calib": self._state.last_checkpoint_calib,
            "recent_calibration": list(self._calib_buffer),
        }
    
    @classmethod
    def deserialize(
        cls,
        state: Dict[str, Any],
        *,
        checkpoint_dir: Optional[Path] = None,
        decision_log_path: Optional[Path] = None,
    ) -> "LearningGovernor":
        """Restore governor from serialized state."""
        gov = cls(
            checkpoint_dir=checkpoint_dir,
            decision_log_path=decision_log_path,
        )
        
        gov._state.is_locked = bool(state.get("is_locked", False))
        gov._state.lock_reason = state.get("lock_reason")
        gov._state.lock_date = state.get("lock_date")
        gov._state.micro_update_sessions_remaining = int(state.get("micro_update_sessions_remaining", 0))
        gov._state.total_updates = int(state.get("total_updates", 0))
        gov._state.total_freezes = int(state.get("total_freezes", 0))
        gov._state.total_micro_updates = int(state.get("total_micro_updates", 0))
        gov._state.total_auto_reverts = int(state.get("total_auto_reverts", 0))
        gov._state.last_checkpoint_date = state.get("last_checkpoint_date")
        gov._state.last_checkpoint_calib = float(state.get("last_checkpoint_calib", 0.5))
        gov._state.calibration_history = list(state.get("calibration_history", []))
        
        recent = state.get("recent_calibration", [])
        for c in recent:
            gov._calib_buffer.append(float(c))
        
        return gov


def create_learning_governor(
    *,
    checkpoint_dir: Optional[Path] = None,
    decision_log_path: Optional[Path] = None,
) -> LearningGovernor:
    """
    Factory for creating a LearningGovernor.
    
    Args:
        checkpoint_dir: Directory for state persistence
        decision_log_path: Path for decision log (JSONL)
    
    Returns:
        Initialized LearningGovernor.
    """
    return LearningGovernor(
        checkpoint_dir=checkpoint_dir,
        decision_log_path=decision_log_path,
    )
