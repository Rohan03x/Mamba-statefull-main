"""
Risk Events Ledger — Daily diagnostic output for risk control events.

This module writes a structured log of all risk events during a backtest
or live trading session, enabling post-run analysis and debugging.

Format: JSONL (one JSON object per line)
Location: artifacts/risk_events/<symbol>_<horizon>_risk_events.jsonl

Copyright 2024-2026. All rights reserved.
"""

import json
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from collections import deque
import threading

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Event Types
# ─────────────────────────────────────────────────────────────────────────────

class EventType:
    """Risk event type constants."""
    # Intraday monitor
    INTRADAY_EMERGENCY = "INTRADAY_EMERGENCY"
    INTRADAY_ALERT = "INTRADAY_ALERT"
    
    # Emergency stop
    EMERGENCY_STOP = "EMERGENCY_STOP"
    
    # Input health
    INPUT_HEALTH_FAIL = "INPUT_HEALTH_FAIL"
    HYGIENE_LOW = "HYGIENE_LOW"
    ROLE_CTX_STALE = "ROLE_CTX_STALE"
    ELIGIBLE_MASK_COLLAPSE = "ELIGIBLE_MASK_COLLAPSE"
    
    # Output sanity
    OUTPUT_SANITY_FAIL = "OUTPUT_SANITY_FAIL"
    SIGMA_COLLAPSE = "SIGMA_COLLAPSE"
    Z_SATURATION = "Z_SATURATION"
    MU_CONSTANT = "MU_CONSTANT"
    SIGN_FLIP_EXTREME = "SIGN_FLIP_EXTREME"
    
    # Stress triggers
    DRAWDOWN_STRESS = "DRAWDOWN_STRESS"
    REALIZED_VOL_STRESS = "REALIZED_VOL_STRESS"
    
    # Kill switches
    ONE_DAY_LOSS_KILL = "ONE_DAY_LOSS_KILL"
    NAN_INF_DETECTED = "NAN_INF_DETECTED"
    
    # Calibration
    CALIBRATION_FAIL = "CALIBRATION_FAIL"
    CALIBRATION_DRIFT = "CALIBRATION_DRIFT"
    
    # Learning governor
    LEARNING_FROZEN = "LEARNING_FROZEN"
    LEARNING_LOCKED = "LEARNING_LOCKED"
    LEARNING_UNLOCKED = "LEARNING_UNLOCKED"
    LEARNING_UPDATE = "LEARNING_UPDATE"
    LEARNING_REVERT = "LEARNING_REVERT"
    
    # State transitions
    MODE_TRANSITION = "MODE_TRANSITION"
    COOLDOWN_START = "COOLDOWN_START"
    COOLDOWN_END = "COOLDOWN_END"
    
    # Session markers
    SESSION_START = "SESSION_START"
    SESSION_END = "SESSION_END"


class ActionTaken:
    """Action taken in response to event."""
    NONE = "NONE"
    LOG_ONLY = "LOG_ONLY"
    THROTTLE = "THROTTLE"
    FLATTEN = "FLATTEN"
    SAFE_FALLBACK = "SAFE_FALLBACK"
    EMERGENCY_STOP = "EMERGENCY_STOP"
    FREEZE_LEARNING = "FREEZE_LEARNING"
    MICRO_UPDATE = "MICRO_UPDATE"
    CHECKPOINT = "CHECKPOINT"


# ─────────────────────────────────────────────────────────────────────────────
# Event Record
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class RiskEvent:
    """Single risk event record."""
    
    # Required fields
    timestamp: str
    date: str
    day_idx: int
    event_type: str
    
    # Threshold info
    threshold: Optional[str] = None
    threshold_value: Optional[float] = None
    observed_value: Optional[float] = None
    threshold_crossed: bool = False
    
    # Action
    action_taken: str = ActionTaken.NONE
    exposure_scale: float = 1.0
    
    # Cooldown info
    cooldown_remaining: int = 0
    emergency_remaining: int = 0
    
    # Mode info
    mode_before: Optional[str] = None
    mode_after: Optional[str] = None
    
    # Additional context
    trigger_reason: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict for JSON serialization."""
        d = asdict(self)
        # Remove None values for cleaner output
        return {k: v for k, v in d.items() if v is not None}
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict())


# ─────────────────────────────────────────────────────────────────────────────
# Ledger Writer
# ─────────────────────────────────────────────────────────────────────────────

class RiskEventsLedger:
    """
    Risk events ledger for tracking and persisting risk control events.
    
    Features:
    - Thread-safe buffered writes
    - JSONL format for easy parsing
    - Automatic flushing on session end
    - Summary statistics
    
    Usage:
        ledger = RiskEventsLedger(path="artifacts/risk_events/AAPL_h63.jsonl")
        
        # Add events during trading
        ledger.add_event(
            date="2025-01-15",
            day_idx=10,
            event_type=EventType.DRAWDOWN_STRESS,
            threshold="DD > 8%",
            threshold_value=0.08,
            observed_value=0.092,
            threshold_crossed=True,
            action_taken=ActionTaken.THROTTLE,
            exposure_scale=0.7,
        )
        
        # Flush at end of session
        ledger.flush()
        
        # Get summary
        summary = ledger.get_summary()
    """
    
    def __init__(
        self,
        path: Union[str, Path],
        buffer_size: int = 100,
        auto_flush: bool = True,
    ):
        """
        Initialize ledger.
        
        Args:
            path: Output file path (JSONL format)
            buffer_size: Events to buffer before auto-flush
            auto_flush: Whether to auto-flush when buffer is full
        """
        self.path = Path(path)
        self.buffer_size = buffer_size
        self.auto_flush = auto_flush
        
        # Thread-safe buffer
        self._buffer: deque[RiskEvent] = deque()
        self._lock = threading.Lock()
        
        # Statistics
        self._event_counts: Dict[str, int] = {}
        self._action_counts: Dict[str, int] = {}
        self._total_events = 0
        
        # Ensure directory exists
        self.path.parent.mkdir(parents=True, exist_ok=True)
        
        logger.debug(f"RiskEventsLedger initialized: {self.path}")
    
    def add_event(
        self,
        date: str,
        day_idx: int,
        event_type: str,
        threshold: Optional[str] = None,
        threshold_value: Optional[float] = None,
        observed_value: Optional[float] = None,
        threshold_crossed: bool = False,
        action_taken: str = ActionTaken.NONE,
        exposure_scale: float = 1.0,
        cooldown_remaining: int = 0,
        emergency_remaining: int = 0,
        mode_before: Optional[str] = None,
        mode_after: Optional[str] = None,
        trigger_reason: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> RiskEvent:
        """
        Add a risk event to the ledger.
        
        Returns:
            The created RiskEvent
        """
        event = RiskEvent(
            timestamp=datetime.now().isoformat(),
            date=date,
            day_idx=day_idx,
            event_type=event_type,
            threshold=threshold,
            threshold_value=threshold_value,
            observed_value=observed_value,
            threshold_crossed=threshold_crossed,
            action_taken=action_taken,
            exposure_scale=exposure_scale,
            cooldown_remaining=cooldown_remaining,
            emergency_remaining=emergency_remaining,
            mode_before=mode_before,
            mode_after=mode_after,
            trigger_reason=trigger_reason,
            details=details or {},
        )
        
        with self._lock:
            self._buffer.append(event)
            self._total_events += 1
            self._event_counts[event_type] = self._event_counts.get(event_type, 0) + 1
            self._action_counts[action_taken] = self._action_counts.get(action_taken, 0) + 1
            
            if self.auto_flush and len(self._buffer) >= self.buffer_size:
                self._flush_unsafe()
        
        return event
    
    def add_mode_transition(
        self,
        date: str,
        day_idx: int,
        mode_before: str,
        mode_after: str,
        trigger_reason: str,
        details: Optional[Dict[str, Any]] = None,
    ) -> RiskEvent:
        """Add a mode transition event."""
        return self.add_event(
            date=date,
            day_idx=day_idx,
            event_type=EventType.MODE_TRANSITION,
            mode_before=mode_before,
            mode_after=mode_after,
            trigger_reason=trigger_reason,
            details=details,
        )
    
    def add_session_start(
        self,
        date: str,
        day_idx: int,
        mode: str,
        cooldown_remaining: int = 0,
        emergency_remaining: int = 0,
    ) -> RiskEvent:
        """Add session start marker."""
        return self.add_event(
            date=date,
            day_idx=day_idx,
            event_type=EventType.SESSION_START,
            mode_after=mode,
            cooldown_remaining=cooldown_remaining,
            emergency_remaining=emergency_remaining,
        )
    
    def add_session_end(
        self,
        date: str,
        day_idx: int,
        mode: str,
        cooldown_remaining: int = 0,
        emergency_remaining: int = 0,
        details: Optional[Dict[str, Any]] = None,
    ) -> RiskEvent:
        """Add session end marker."""
        return self.add_event(
            date=date,
            day_idx=day_idx,
            event_type=EventType.SESSION_END,
            mode_after=mode,
            cooldown_remaining=cooldown_remaining,
            emergency_remaining=emergency_remaining,
            details=details,
        )
    
    def flush(self) -> int:
        """
        Flush buffered events to file.
        
        Returns:
            Number of events flushed
        """
        with self._lock:
            return self._flush_unsafe()
    
    def _flush_unsafe(self) -> int:
        """Flush without lock (internal use only)."""
        if not self._buffer:
            return 0
        
        count = len(self._buffer)
        
        try:
            with open(self.path, "a") as f:
                while self._buffer:
                    event = self._buffer.popleft()
                    f.write(event.to_json() + "\n")
            
            logger.debug(f"Flushed {count} events to {self.path}")
        except Exception as e:
            logger.error(f"Failed to flush events: {e}")
            raise
        
        return count
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get summary statistics.
        
        Returns:
            Summary dict with counts and breakdown
        """
        return {
            "total_events": self._total_events,
            "events_by_type": dict(self._event_counts),
            "events_by_action": dict(self._action_counts),
            "path": str(self.path),
        }
    
    def get_events_by_type(self, event_type: str) -> List[RiskEvent]:
        """
        Get all buffered events of a specific type.
        
        Note: This only returns buffered events, not flushed ones.
        Use load_all_events() to get all events from file.
        """
        with self._lock:
            return [e for e in self._buffer if e.event_type == event_type]
    
    def load_all_events(self) -> List[RiskEvent]:
        """
        Load all events from file.
        
        Returns:
            List of all events (flushed + buffered)
        """
        events = []
        
        if self.path.exists():
            try:
                with open(self.path, "r") as f:
                    for line in f:
                        if line.strip():
                            data = json.loads(line)
                            events.append(RiskEvent(**data))
            except Exception as e:
                logger.error(f"Failed to load events: {e}")
        
        # Add buffered events
        with self._lock:
            events.extend(list(self._buffer))
        
        return events
    
    def clear(self):
        """Clear buffer and statistics (does not delete file)."""
        with self._lock:
            self._buffer.clear()
            self._event_counts.clear()
            self._action_counts.clear()
            self._total_events = 0


# ─────────────────────────────────────────────────────────────────────────────
# Daily Summary Generator
# ─────────────────────────────────────────────────────────────────────────────

def generate_daily_summary(
    events: List[RiskEvent],
    date: str,
) -> Dict[str, Any]:
    """
    Generate a daily summary from risk events.
    
    Args:
        events: List of events
        date: Date to summarize (YYYY-MM-DD)
    
    Returns:
        Summary dict
    """
    day_events = [e for e in events if e.date == date]
    
    if not day_events:
        return {"date": date, "total_events": 0}
    
    # Count by type
    type_counts: Dict[str, int] = {}
    action_counts: Dict[str, int] = {}
    
    for e in day_events:
        type_counts[e.event_type] = type_counts.get(e.event_type, 0) + 1
        action_counts[e.action_taken] = action_counts.get(e.action_taken, 0) + 1
    
    # Get mode at end of day
    session_end = [e for e in day_events if e.event_type == EventType.SESSION_END]
    final_mode = session_end[-1].mode_after if session_end else None
    
    # Check for severe events
    severe_types = {
        EventType.EMERGENCY_STOP,
        EventType.ONE_DAY_LOSS_KILL,
        EventType.NAN_INF_DETECTED,
        EventType.INTRADAY_EMERGENCY,
    }
    severe_events = [e for e in day_events if e.event_type in severe_types]
    
    return {
        "date": date,
        "total_events": len(day_events),
        "events_by_type": type_counts,
        "events_by_action": action_counts,
        "final_mode": final_mode,
        "severe_events": len(severe_events),
        "has_severe": len(severe_events) > 0,
    }


def generate_run_summary(
    ledger: RiskEventsLedger,
    output_path: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """
    Generate a full run summary.
    
    Args:
        ledger: RiskEventsLedger instance
        output_path: Optional path to write summary JSON
    
    Returns:
        Summary dict
    """
    events = ledger.load_all_events()
    
    if not events:
        return {"total_events": 0, "days": []}
    
    # Get unique dates
    dates = sorted(set(e.date for e in events))
    
    # Generate daily summaries
    daily_summaries = [generate_daily_summary(events, d) for d in dates]
    
    # Aggregate stats
    total_severe = sum(s["severe_events"] for s in daily_summaries)
    days_with_issues = sum(1 for s in daily_summaries if s["total_events"] > 1)
    
    # Mode transitions
    transitions = [e for e in events if e.event_type == EventType.MODE_TRANSITION]
    
    summary = {
        "run_summary": {
            "total_events": len(events),
            "total_days": len(dates),
            "days_with_issues": days_with_issues,
            "total_severe_events": total_severe,
            "mode_transitions": len(transitions),
        },
        "daily_summaries": daily_summaries,
        "source_file": str(ledger.path),
    }
    
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(summary, f, indent=2)
        logger.info(f"Run summary written to {output_path}")
    
    return summary


# ─────────────────────────────────────────────────────────────────────────────
# Factory Function
# ─────────────────────────────────────────────────────────────────────────────

def create_ledger(
    symbol: str,
    horizon: int,
    base_dir: Union[str, Path] = "cache/debugging/risk_events",
    run_id: Optional[str] = None,
) -> RiskEventsLedger:
    """
    Create a risk events ledger for a symbol/horizon.
    
    Args:
        symbol: Symbol (e.g., "AAPL")
        horizon: Horizon (e.g., 63)
        base_dir: Base directory for ledger files
        run_id: Optional run identifier
    
    Returns:
        RiskEventsLedger instance
    """
    base_dir = Path(base_dir)
    
    # Organize by symbol/horizon subdirectory
    symbol_dir = base_dir / symbol.upper() / f"h{horizon}"
    
    if run_id:
        filename = f"{run_id}_risk_events.jsonl"
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp}_risk_events.jsonl"
    
    path = symbol_dir / filename
    
    return RiskEventsLedger(path=path)


if __name__ == "__main__":
    # Demo usage
    ledger = create_ledger("AAPL", 63, run_id="demo")
    print(f"Ledger path: {ledger.path}")  # cache/debugging/risk_events/AAPL/h63/demo_risk_events.jsonl
    
    ledger.add_session_start("2025-01-15", 10, "NORMAL")
    ledger.add_event(
        date="2025-01-15",
        day_idx=10,
        event_type=EventType.DRAWDOWN_STRESS,
        threshold="DD > 8%",
        threshold_value=0.08,
        observed_value=0.092,
        threshold_crossed=True,
        action_taken=ActionTaken.THROTTLE,
        exposure_scale=0.7,
    )
    ledger.add_mode_transition(
        date="2025-01-15",
        day_idx=10,
        mode_before="NORMAL",
        mode_after="THROTTLE",
        trigger_reason="DRAWDOWN_STRESS",
    )
    ledger.add_session_end("2025-01-15", 10, "THROTTLE", cooldown_remaining=2)
    
    ledger.flush()
    
    summary = ledger.get_summary()
    print(f"Summary: {json.dumps(summary, indent=2)}")
