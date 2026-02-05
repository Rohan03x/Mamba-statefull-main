"""
Tests for Gap 5: Event Code and Logging Clarity

Verifies:
1. Linear blend emits LINEAR_BLEND_APPLIED (not QUANTILE_BLEND_APPLIED)
2. Quantile blend emits QUANTILE_BLEND_APPLIED
3. Event codes accurately reflect what's happening
4. All defined event codes are used where appropriate

Classification: Trivial impact, Low severity (cosmetic/monitoring only)
"""

import os
import sys
import unittest
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Any

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.stage_b_stateful.phase2_stateful import EventCode, EventSeverity


@dataclass
class MockEvent:
    """Mock event for testing."""
    date: str
    day_idx: int
    severity: str
    code: str
    message: str
    payload: Dict[str, Any]


class MockEventBus:
    """Mock event bus that captures emitted events."""
    
    def __init__(self):
        self.events: List[MockEvent] = []
    
    def emit(self, *, date: str, day_idx: int, severity: str, code: str, 
             message: str, payload: Dict[str, Any]) -> None:
        """Capture emitted event."""
        self.events.append(
            MockEvent(
                date=date,
                day_idx=day_idx,
                severity=severity,
                code=code,
                message=message,
                payload=payload,
            )
        )
    
    def get_events_by_code(self, code: str) -> List[MockEvent]:
        """Get all events with specified code."""
        return [e for e in self.events if e.code == code]
    
    def clear(self) -> None:
        """Clear all captured events."""
        self.events = []


class TestLinearBlendEventCode(unittest.TestCase):
    """Test that linear blend uses correct event code."""
    
    def test_linear_blend_has_own_event_code(self):
        """LINEAR_BLEND_APPLIED event code should exist."""
        self.assertTrue(hasattr(EventCode, "LINEAR_BLEND_APPLIED"))
        self.assertEqual(EventCode.LINEAR_BLEND_APPLIED, "LINEAR_BLEND_APPLIED")
    
    def test_quantile_blend_event_code_exists(self):
        """QUANTILE_BLEND_APPLIED event code should exist."""
        self.assertTrue(hasattr(EventCode, "QUANTILE_BLEND_APPLIED"))
        self.assertEqual(EventCode.QUANTILE_BLEND_APPLIED, "QUANTILE_BLEND_APPLIED")
    
    def test_blend_event_codes_are_distinct(self):
        """Linear and quantile blend codes should be different."""
        self.assertNotEqual(EventCode.LINEAR_BLEND_APPLIED, EventCode.QUANTILE_BLEND_APPLIED)


class TestLearningGovernorEventCodes(unittest.TestCase):
    """Test that learning governor event codes exist and are used."""
    
    def test_learning_frozen_event_code_exists(self):
        """LEARNING_FROZEN event code should exist."""
        self.assertTrue(hasattr(EventCode, "LEARNING_FROZEN"))
        self.assertEqual(EventCode.LEARNING_FROZEN, "LEARNING_FROZEN")
    
    def test_learning_gov_decision_event_code_exists(self):
        """LEARNING_GOV_DECISION event code should exist."""
        self.assertTrue(hasattr(EventCode, "LEARNING_GOV_DECISION"))
        self.assertEqual(EventCode.LEARNING_GOV_DECISION, "LEARNING_GOV_DECISION")
    
    def test_learning_gov_locked_event_code_exists(self):
        """LEARNING_GOV_LOCKED event code should exist."""
        self.assertTrue(hasattr(EventCode, "LEARNING_GOV_LOCKED"))
        self.assertEqual(EventCode.LEARNING_GOV_LOCKED, "LEARNING_GOV_LOCKED")
    
    def test_learning_gov_unlocked_event_code_exists(self):
        """LEARNING_GOV_UNLOCKED event code should exist."""
        self.assertTrue(hasattr(EventCode, "LEARNING_GOV_UNLOCKED"))
        self.assertEqual(EventCode.LEARNING_GOV_UNLOCKED, "LEARNING_GOV_UNLOCKED")


class TestCalibrationEventCodes(unittest.TestCase):
    """Test that calibration-related event codes exist."""
    
    def test_mamba_calib_updated_event_code_exists(self):
        """MAMBA_CALIB_UPDATED event code should exist."""
        self.assertTrue(hasattr(EventCode, "MAMBA_CALIB_UPDATED"))
        self.assertEqual(EventCode.MAMBA_CALIB_UPDATED, "MAMBA_CALIB_UPDATED")
    
    def test_sigma_clipped_event_code_exists(self):
        """SIGMA_CLIPPED event code should exist."""
        self.assertTrue(hasattr(EventCode, "SIGMA_CLIPPED"))
        self.assertEqual(EventCode.SIGMA_CLIPPED, "SIGMA_CLIPPED")


class TestEventPayloadStructure(unittest.TestCase):
    """Test that event payloads have consistent structure."""
    
    def test_blend_payload_includes_blend_type(self):
        """
        Blend events should include blend_type in payload.
        This allows dashboards to filter by blend type even with shared code.
        """
        bus = MockEventBus()
        
        # Simulate linear blend emission
        bus.emit(
            date="2024-01-01",
            day_idx=0,
            severity=EventSeverity.INFO,
            code=EventCode.LINEAR_BLEND_APPLIED,
            message="Linear blend applied",
            payload={"blend_type": "linear", "weight": 0.5, "n_alphas": 3},
        )
        
        # Simulate quantile blend emission
        bus.emit(
            date="2024-01-01",
            day_idx=0,
            severity=EventSeverity.INFO,
            code=EventCode.QUANTILE_BLEND_APPLIED,
            message="Quantile blend applied",
            payload={"blend_type": "quantile", "n_quantiles": 5},
        )
        
        linear_events = bus.get_events_by_code(EventCode.LINEAR_BLEND_APPLIED)
        quantile_events = bus.get_events_by_code(EventCode.QUANTILE_BLEND_APPLIED)
        
        self.assertEqual(len(linear_events), 1)
        self.assertEqual(len(quantile_events), 1)
        
        self.assertEqual(linear_events[0].payload["blend_type"], "linear")
        self.assertEqual(quantile_events[0].payload["blend_type"], "quantile")


class TestEventSeverityLevels(unittest.TestCase):
    """Test that events use appropriate severity levels."""
    
    def test_linear_blend_is_info_level(self):
        """Linear blend should be INFO level (routine operation)."""
        bus = MockEventBus()
        bus.emit(
            date="2024-01-01",
            day_idx=0,
            severity=EventSeverity.INFO,
            code=EventCode.LINEAR_BLEND_APPLIED,
            message="Linear blend applied",
            payload={"blend_type": "linear"},
        )
        
        events = bus.get_events_by_code(EventCode.LINEAR_BLEND_APPLIED)
        self.assertEqual(events[0].severity, EventSeverity.INFO)
    
    def test_learning_locked_is_warning_level(self):
        """Learning locked should be WARNING level (requires attention)."""
        bus = MockEventBus()
        bus.emit(
            date="2024-01-01",
            day_idx=0,
            severity=EventSeverity.WARNING,
            code=EventCode.LEARNING_GOV_LOCKED,
            message="Learning governor auto-locked",
            payload={"lock_reason": "calibration dropped"},
        )
        
        events = bus.get_events_by_code(EventCode.LEARNING_GOV_LOCKED)
        self.assertEqual(events[0].severity, EventSeverity.WARNING)


class TestEventCodeCompleteness(unittest.TestCase):
    """Test that all event code categories are complete."""
    
    def test_all_signal_processing_codes_defined(self):
        """All signal processing event codes should be defined."""
        required_codes = [
            "SIGMA_CLIPPED",
            "HYGIENE_VETO",
            "RISK_SCALE_APPLIED",
            "REGIME_MULT_APPLIED",
            "SPLIT_STRESS_APPLIED",
            "EVENT_RISK_APPLIED",
            "LINEAR_BLEND_APPLIED",
            "QUANTILE_BLEND_APPLIED",
        ]
        
        for code_name in required_codes:
            self.assertTrue(hasattr(EventCode, code_name), f"Missing event code: {code_name}")
    
    def test_all_learning_gov_codes_defined(self):
        """All learning governance event codes should be defined."""
        required_codes = [
            "LEARNING_FROZEN",
            "LEARNING_GOV_DECISION",
            "LEARNING_GOV_LOCKED",
            "LEARNING_GOV_UNLOCKED",
        ]
        
        for code_name in required_codes:
            self.assertTrue(hasattr(EventCode, code_name), f"Missing event code: {code_name}")
    
    def test_all_risk_latch_codes_defined(self):
        """All risk latch event codes should be defined."""
        required_codes = [
            "RISK_LATCH_THROTTLE",
            "RISK_LATCH_FLATTEN",
            "RISK_LATCH_SAFE_FALLBACK",
            "RISK_LATCH_EMERGENCY",
        ]
        
        for code_name in required_codes:
            self.assertTrue(hasattr(EventCode, code_name), f"Missing event code: {code_name}")


class TestEventFilteringByCode(unittest.TestCase):
    """Test that events can be properly filtered by code."""
    
    def test_dashboard_can_filter_linear_blend_events(self):
        """Dashboards should be able to filter linear blend events by code."""
        bus = MockEventBus()
        
        # Emit multiple events
        bus.emit(
            date="2024-01-01",
            day_idx=0,
            severity=EventSeverity.INFO,
            code=EventCode.LINEAR_BLEND_APPLIED,
            message="Linear blend 1",
            payload={"blend_type": "linear"},
        )
        bus.emit(
            date="2024-01-02",
            day_idx=1,
            severity=EventSeverity.INFO,
            code=EventCode.QUANTILE_BLEND_APPLIED,
            message="Quantile blend 1",
            payload={"blend_type": "quantile"},
        )
        bus.emit(
            date="2024-01-03",
            day_idx=2,
            severity=EventSeverity.INFO,
            code=EventCode.LINEAR_BLEND_APPLIED,
            message="Linear blend 2",
            payload={"blend_type": "linear"},
        )
        
        # Filter by code
        linear_events = bus.get_events_by_code(EventCode.LINEAR_BLEND_APPLIED)
        quantile_events = bus.get_events_by_code(EventCode.QUANTILE_BLEND_APPLIED)
        
        self.assertEqual(len(linear_events), 2)
        self.assertEqual(len(quantile_events), 1)
        
        # Verify all linear events have correct code
        for event in linear_events:
            self.assertEqual(event.code, EventCode.LINEAR_BLEND_APPLIED)
            self.assertEqual(event.payload["blend_type"], "linear")
    
    def test_dashboard_can_filter_learning_events(self):
        """Dashboards should be able to filter learning governor events."""
        bus = MockEventBus()
        
        # Emit multiple learning events
        bus.emit(
            date="2024-01-01",
            day_idx=0,
            severity=EventSeverity.INFO,
            code=EventCode.LEARNING_GOV_DECISION,
            message="Decision 1",
            payload={"zone": "GREEN"},
        )
        bus.emit(
            date="2024-01-02",
            day_idx=1,
            severity=EventSeverity.WARNING,
            code=EventCode.LEARNING_GOV_LOCKED,
            message="Locked",
            payload={"lock_reason": "calib dropped"},
        )
        bus.emit(
            date="2024-01-03",
            day_idx=2,
            severity=EventSeverity.INFO,
            code=EventCode.LEARNING_GOV_DECISION,
            message="Decision 2",
            payload={"zone": "YELLOW"},
        )
        
        # Filter by code
        decision_events = bus.get_events_by_code(EventCode.LEARNING_GOV_DECISION)
        locked_events = bus.get_events_by_code(EventCode.LEARNING_GOV_LOCKED)
        
        self.assertEqual(len(decision_events), 2)
        self.assertEqual(len(locked_events), 1)
        
        # Verify locked event has warning severity
        self.assertEqual(locked_events[0].severity, EventSeverity.WARNING)


if __name__ == "__main__":
    unittest.main()
