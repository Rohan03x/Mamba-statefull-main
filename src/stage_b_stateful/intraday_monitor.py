"""
Intraday Risk Monitor - Live Safety Layer for Phase-2.

This module provides:
1. IntradaySnapshot - Data structure for intraday risk metrics
2. IntradayRiskMonitor - Main monitor class that computes and persists snapshots
3. read_intraday_snapshot() - Function for Phase-2 to read latest snapshot

The monitor runs OUTSIDE the daily backtest loop (in live: as a separate process).
It writes a JSON file that the daily engine reads at the top of each day.

KEY RULE: Intraday monitor never touches Mamba. It only sets a risk latch
that overrides allocations.

Example usage (live monitor process):
    monitor = IntradayRiskMonitor(symbols=["AAPL", "NVDA", ...])
    while True:
        snapshot = monitor.update(prices_df, positions)
        if snapshot.should_emergency_stop():
            monitor.write_snapshot()  # Daily engine reads this
        time.sleep(30)  # 30-second intervals

Example usage (Phase-2 integration):
    snapshot = read_intraday_snapshot(snapshot_path)
    if snapshot is not None and snapshot.should_emergency_stop():
        w = np.zeros(n_assets)  # Force flatten
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Intraday Snapshot Data Structure
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class IntradaySnapshot:
    """
    Snapshot of intraday risk metrics.
    
    This is the output payload that the daily engine reads.
    """
    
    # Timestamp (ISO 8601 with timezone)
    ts: str = ""
    
    # Portfolio-level metrics
    portfolio_intraday_dd: float = 0.0      # Intraday drawdown (negative = loss)
    portfolio_intraday_pnl: float = 0.0     # Intraday PnL (cumulative since open)
    portfolio_intraday_high: float = 0.0    # Intraday high water mark
    portfolio_intraday_var99: float = 0.0   # 1-day VaR (99th percentile)
    
    # Volatility metrics
    realized_vol_30m_annual: float = 0.0    # 30-min rolling vol (annualized)
    realized_vol_5m_annual: float = 0.0     # 5-min rolling vol (annualized)
    
    # System health
    market_halt_flag: bool = False          # Exchange-wide halt detected
    data_feed_ok: bool = True               # Data feed is functioning
    data_staleness_seconds: float = 0.0     # Seconds since last data update
    
    # Risk latch recommendations
    recommended_mode: str = "NORMAL"        # NORMAL, THROTTLE, FLATTEN, EMERGENCY_STOP
    recommended_scale: float = 1.0          # Exposure scale (0-1)
    trigger_reason: str = ""                # Why the mode was triggered
    
    # Per-symbol metrics (optional, for debugging)
    symbol_intraday_pnl: Dict[str, float] = field(default_factory=lambda: {})
    symbol_intraday_vol: Dict[str, float] = field(default_factory=lambda: {})
    shocked_symbols: List[str] = field(default_factory=lambda: [])
    
    def __post_init__(self):
        if not self.ts:
            self.ts = datetime.now(timezone.utc).isoformat()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)
    
    def to_json(self) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict(), indent=2, default=str)
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "IntradaySnapshot":
        """Create from dictionary."""
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})
    
    @classmethod
    def from_json(cls, s: str) -> "IntradaySnapshot":
        """Deserialize from JSON string."""
        return cls.from_dict(json.loads(s))
    
    def should_emergency_stop(self) -> bool:
        """Check if snapshot indicates EMERGENCY_STOP."""
        return self.recommended_mode == "EMERGENCY_STOP"
    
    def should_flatten(self) -> bool:
        """Check if snapshot indicates FLATTEN or worse."""
        return self.recommended_mode in ("FLATTEN", "EMERGENCY_STOP", "SAFE_FALLBACK")
    
    def should_throttle(self) -> bool:
        """Check if snapshot indicates THROTTLE or worse."""
        return self.recommended_mode in ("THROTTLE", "FLATTEN", "EMERGENCY_STOP", "SAFE_FALLBACK")
    
    def get_exposure_scale(self) -> float:
        """Get recommended exposure scale."""
        if self.should_flatten():
            return 0.0
        return self.recommended_scale


# ─────────────────────────────────────────────────────────────────────────────
# Intraday Monitor Configuration
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class IntradayMonitorConfig:
    """Configuration for intraday risk monitor."""
    
    # Intraday DD kill threshold
    intraday_dd_kill_pct: float = 0.025     # -2.5% intraday DD → EMERGENCY_STOP
    intraday_dd_throttle_pct: float = 0.015 # -1.5% intraday DD → THROTTLE
    
    # Intraday vol spike thresholds
    vol_spike_kill_mult: float = 3.0        # rv_30m > 3x target_vol → FLATTEN
    vol_spike_throttle_mult: float = 2.0    # rv_30m > 2x target_vol → THROTTLE
    target_vol_annual: float = 0.15         # Target vol for comparison
    
    # Data feed thresholds
    data_stale_warn_seconds: float = 60.0   # Warn if data > 60s old
    data_stale_kill_seconds: float = 300.0  # Kill if data > 5min old
    
    # VaR thresholds
    var99_kill_pct: float = 0.05            # VaR99 > 5% → THROTTLE
    
    # Throttle scaling
    throttle_scale: float = 0.5             # Scale to 50% on throttle
    
    # Snapshot persistence
    snapshot_path: str = "artifacts/intraday_monitor/latest_snapshot.json"
    snapshot_history_dir: str = "artifacts/intraday_monitor/history"
    max_history_files: int = 1000


# ─────────────────────────────────────────────────────────────────────────────
# Intraday Risk Monitor
# ─────────────────────────────────────────────────────────────────────────────

class IntradayRiskMonitor:
    """
    Intraday risk monitor for live trading.
    
    This runs as a separate process and writes snapshots that
    the daily Phase-2 engine reads.
    """
    
    def __init__(
        self,
        symbols: List[str],
        config: Optional[IntradayMonitorConfig] = None,
        initial_equity: float = 1.0,
    ):
        """
        Initialize intraday monitor.
        
        Args:
            symbols: List of symbols to monitor
            config: Monitor configuration
            initial_equity: Starting equity for the day
        """
        self.symbols = list(symbols)
        self.n_assets = len(symbols)
        self.config = config or IntradayMonitorConfig()
        
        # State tracking
        self.initial_equity = initial_equity
        self.current_equity = initial_equity
        self.intraday_high = initial_equity
        self.last_update_time: Optional[datetime] = None
        
        # Price/return tracking (rolling windows)
        self._price_history: List[np.ndarray] = []
        self._return_history: List[np.ndarray] = []
        self._pnl_history: List[float] = []
        self._timestamps: List[datetime] = []
        
        # Current positions (weights)
        self._current_weights: Optional[np.ndarray] = None
        
        # Latest snapshot
        self._latest_snapshot: Optional[IntradaySnapshot] = None
        
        logger.info(
            "[IntradayMonitor] Initialized with %d symbols, DD_kill=%.1f%%, vol_kill=%.0fx",
            self.n_assets,
            self.config.intraday_dd_kill_pct * 100,
            self.config.vol_spike_kill_mult,
        )
    
    def set_positions(self, weights: np.ndarray) -> None:
        """
        Set current position weights.
        
        Args:
            weights: Position weights (sum to ~1.0)
        """
        self._current_weights = np.asarray(weights, dtype=float).copy()
    
    def update(
        self,
        prices: np.ndarray,
        timestamp: Optional[datetime] = None,
        market_halt: bool = False,
        data_feed_ok: bool = True,
    ) -> IntradaySnapshot:
        """
        Update monitor with latest prices.
        
        Args:
            prices: Current prices for all symbols
            timestamp: Current timestamp (defaults to now)
            market_halt: Whether market is halted
            data_feed_ok: Whether data feed is functioning
        
        Returns:
            Updated IntradaySnapshot
        """
        now = timestamp or datetime.now(timezone.utc)
        prices = np.asarray(prices, dtype=float)
        
        # Track price history
        self._price_history.append(prices.copy())
        self._timestamps.append(now)
        
        # Compute returns if we have history
        if len(self._price_history) >= 2:
            prev_prices = self._price_history[-2]
            returns = (prices - prev_prices) / (prev_prices + 1e-12)
            self._return_history.append(returns)
        
        # Compute portfolio PnL if we have positions
        pnl = 0.0
        if self._current_weights is not None and len(self._return_history) > 0:
            # Cumulative PnL since start of day
            cumulative_returns = np.zeros(self.n_assets)
            for r in self._return_history:
                cumulative_returns += r
            pnl = float(np.dot(self._current_weights, cumulative_returns))
        
        self._pnl_history.append(pnl)
        
        # Update equity tracking
        self.current_equity = self.initial_equity * (1.0 + pnl)
        self.intraday_high = max(self.intraday_high, self.current_equity)
        
        # Compute intraday drawdown
        intraday_dd = (self.current_equity - self.intraday_high) / self.intraday_high
        
        # Compute 30-min rolling vol (annualized)
        vol_30m = self._compute_rolling_vol(window_minutes=30, now=now)
        vol_5m = self._compute_rolling_vol(window_minutes=5, now=now)
        
        # Compute VaR99 (parametric, using recent vol)
        var99 = self._compute_var99(vol_30m)
        
        # Compute data staleness
        staleness = 0.0
        if self.last_update_time is not None:
            staleness = (now - self.last_update_time).total_seconds()
        self.last_update_time = now
        
        # Per-symbol metrics
        symbol_pnl: Dict[str, float] = {}
        symbol_vol: Dict[str, float] = {}
        shocked: List[str] = []
        
        if len(self._return_history) > 0:
            cumulative_returns = np.zeros(self.n_assets)
            for r in self._return_history:
                cumulative_returns += r
            
            for j, sym in enumerate(self.symbols):
                symbol_pnl[sym] = float(cumulative_returns[j])
                # Check for shocked symbols (> 3% move)
                if abs(cumulative_returns[j]) > 0.03:
                    shocked.append(sym)
        
        # Determine recommended mode
        mode, scale, reason = self._determine_mode(
            intraday_dd=intraday_dd,
            vol_30m=vol_30m,
            var99=var99,
            market_halt=market_halt,
            data_feed_ok=data_feed_ok,
            data_staleness=staleness,
        )
        
        # Build snapshot
        snapshot = IntradaySnapshot(
            ts=now.isoformat(),
            portfolio_intraday_dd=float(intraday_dd),
            portfolio_intraday_pnl=float(pnl),
            portfolio_intraday_high=float(self.intraday_high),
            portfolio_intraday_var99=float(var99),
            realized_vol_30m_annual=float(vol_30m),
            realized_vol_5m_annual=float(vol_5m),
            market_halt_flag=market_halt,
            data_feed_ok=data_feed_ok,
            data_staleness_seconds=float(staleness),
            recommended_mode=mode,
            recommended_scale=float(scale),
            trigger_reason=reason,
            symbol_intraday_pnl=symbol_pnl,
            symbol_intraday_vol=symbol_vol,
            shocked_symbols=shocked[:10],  # Top 10
        )
        
        self._latest_snapshot = snapshot
        return snapshot
    
    def _compute_rolling_vol(
        self,
        window_minutes: int,
        now: datetime,
    ) -> float:
        """Compute rolling volatility over a time window."""
        if len(self._return_history) < 2:
            return 0.0
        
        # Filter returns within window
        cutoff = now - pd.Timedelta(minutes=window_minutes)
        recent_returns: List[float] = []
        
        for ts, r in zip(self._timestamps[1:], self._return_history):
            if ts >= cutoff:
                # Portfolio return
                if self._current_weights is not None:
                    port_ret = float(np.dot(self._current_weights, r))
                else:
                    port_ret = float(np.mean(r))
                recent_returns.append(port_ret)
        
        if len(recent_returns) < 2:
            return 0.0
        
        # Compute vol (annualized assuming 1-minute bars → 252*6.5*60 per year)
        # Adjust based on actual bar frequency
        bar_freq_per_year = 252.0 * 6.5 * 60.0 / max(1.0, window_minutes / len(recent_returns))
        vol = float(np.std(recent_returns)) * np.sqrt(bar_freq_per_year)
        
        return vol
    
    def _compute_var99(self, vol_annual: float) -> float:
        """Compute 1-day VaR at 99th percentile (parametric)."""
        # Daily vol
        daily_vol = vol_annual / np.sqrt(252.0)
        # VaR99 = 2.33 * daily_vol (assuming normal distribution)
        return 2.33 * daily_vol
    
    def _determine_mode(
        self,
        intraday_dd: float,
        vol_30m: float,
        var99: float,
        market_halt: bool,
        data_feed_ok: bool,
        data_staleness: float,
    ) -> Tuple[str, float, str]:
        """
        Determine recommended risk mode based on metrics.
        
        Returns:
            (mode, scale, reason)
        """
        cfg = self.config
        
        # Priority 1: Market halt → EMERGENCY_STOP
        if market_halt:
            return ("EMERGENCY_STOP", 0.0, "Market halt detected")
        
        # Priority 2: Data feed failure → SAFE_FALLBACK
        if not data_feed_ok:
            return ("SAFE_FALLBACK", 0.0, "Data feed failure")
        
        if data_staleness >= cfg.data_stale_kill_seconds:
            return ("SAFE_FALLBACK", 0.0, f"Data stale: {data_staleness:.0f}s > {cfg.data_stale_kill_seconds:.0f}s")
        
        # Priority 3: Intraday DD kill → EMERGENCY_STOP
        if intraday_dd <= -cfg.intraday_dd_kill_pct:
            return (
                "EMERGENCY_STOP",
                0.0,
                f"Intraday DD kill: {intraday_dd:.2%} <= -{cfg.intraday_dd_kill_pct:.2%}",
            )
        
        # Priority 4: Vol spike kill → FLATTEN
        _target_daily = cfg.target_vol_annual / np.sqrt(252.0)
        _vol_30m_daily = vol_30m / np.sqrt(252.0)
        
        if vol_30m > 0 and vol_30m >= cfg.vol_spike_kill_mult * cfg.target_vol_annual:
            return (
                "FLATTEN",
                0.0,
                f"Vol spike kill: rv_30m={vol_30m:.1%} >= {cfg.vol_spike_kill_mult:.0f}x target",
            )
        
        # Priority 5: Intraday DD throttle
        if intraday_dd <= -cfg.intraday_dd_throttle_pct:
            return (
                "THROTTLE",
                cfg.throttle_scale,
                f"Intraday DD throttle: {intraday_dd:.2%} <= -{cfg.intraday_dd_throttle_pct:.2%}",
            )
        
        # Priority 6: Vol spike throttle
        if vol_30m > 0 and vol_30m >= cfg.vol_spike_throttle_mult * cfg.target_vol_annual:
            return (
                "THROTTLE",
                cfg.throttle_scale,
                f"Vol spike throttle: rv_30m={vol_30m:.1%} >= {cfg.vol_spike_throttle_mult:.0f}x target",
            )
        
        # Priority 7: VaR99 throttle
        if var99 >= cfg.var99_kill_pct:
            return (
                "THROTTLE",
                cfg.throttle_scale,
                f"VaR99 throttle: {var99:.2%} >= {cfg.var99_kill_pct:.2%}",
            )
        
        # Normal operation
        return ("NORMAL", 1.0, "")
    
    def write_snapshot(self, path: Optional[str] = None) -> None:
        """
        Write latest snapshot to file.
        
        Args:
            path: Path to write (defaults to config.snapshot_path)
        """
        if self._latest_snapshot is None:
            logger.warning("[IntradayMonitor] No snapshot to write")
            return
        
        path = path or self.config.snapshot_path
        
        # Ensure directory exists
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        
        # Write snapshot
        with open(path, "w") as f:
            f.write(self._latest_snapshot.to_json())
        
        logger.debug("[IntradayMonitor] Wrote snapshot to %s", path)
        
        # Also write to history
        self._write_history()
    
    def _write_history(self) -> None:
        """Write snapshot to history directory."""
        if self._latest_snapshot is None:
            return
        
        history_dir = Path(self.config.snapshot_history_dir)
        history_dir.mkdir(parents=True, exist_ok=True)
        
        # Filename with timestamp
        ts_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        history_path = history_dir / f"snapshot_{ts_str}.json"
        
        with open(history_path, "w") as f:
            f.write(self._latest_snapshot.to_json())
        
        # Cleanup old history files
        self._cleanup_history()
    
    def _cleanup_history(self) -> None:
        """Remove old history files."""
        history_dir = Path(self.config.snapshot_history_dir)
        if not history_dir.exists():
            return
        
        files = sorted(history_dir.glob("snapshot_*.json"))
        if len(files) > self.config.max_history_files:
            for f in files[:-self.config.max_history_files]:
                try:
                    f.unlink()
                except Exception:
                    pass
    
    def reset_day(self, new_initial_equity: float) -> None:
        """
        Reset for a new trading day.
        
        Args:
            new_initial_equity: Opening equity for the new day
        """
        self.initial_equity = new_initial_equity
        self.current_equity = new_initial_equity
        self.intraday_high = new_initial_equity
        self._price_history.clear()
        self._return_history.clear()
        self._pnl_history.clear()
        self._timestamps.clear()
        self._latest_snapshot = None
        logger.info("[IntradayMonitor] Reset for new day, equity=%.6f", new_initial_equity)
    
    @property
    def latest_snapshot(self) -> Optional[IntradaySnapshot]:
        """Get the latest snapshot."""
        return self._latest_snapshot


# ─────────────────────────────────────────────────────────────────────────────
# Snapshot Reader (for Phase-2 integration)
# ─────────────────────────────────────────────────────────────────────────────

def read_intraday_snapshot(
    path: str = "artifacts/intraday_monitor/latest_snapshot.json",
    max_age_seconds: float = 300.0,
) -> Optional[IntradaySnapshot]:
    """
    Read the latest intraday snapshot from file.
    
    This is called by Phase-2 at the top of each day to check
    if the intraday monitor has flagged any issues.
    
    Args:
        path: Path to snapshot file
        max_age_seconds: Max age of snapshot before it's considered stale
    
    Returns:
        IntradaySnapshot if valid and fresh, None otherwise
    """
    try:
        if not os.path.exists(path):
            return None
        
        # Check file age
        file_age = datetime.now().timestamp() - os.path.getmtime(path)
        if file_age > max_age_seconds:
            logger.debug(
                "[read_intraday_snapshot] Snapshot too old: %.0fs > %.0fs",
                file_age, max_age_seconds,
            )
            return None
        
        with open(path, "r") as f:
            content = f.read()
        
        snapshot = IntradaySnapshot.from_json(content)
        
        logger.debug(
            "[read_intraday_snapshot] Read snapshot: mode=%s, dd=%.2f%%, vol=%.1f%%",
            snapshot.recommended_mode,
            snapshot.portfolio_intraday_dd * 100,
            snapshot.realized_vol_30m_annual * 100,
        )
        
        return snapshot
    
    except Exception as e:
        logger.warning("[read_intraday_snapshot] Failed to read snapshot: %s", e)
        return None


def apply_intraday_snapshot_to_latch(
    snapshot: IntradaySnapshot,
    risk_latch: Any,  # RiskLatch type
) -> bool:
    """
    Apply intraday snapshot to risk latch.
    
    Args:
        snapshot: Intraday snapshot
        risk_latch: RiskLatch instance
    
    Returns:
        True if latch was modified
    """
    try:
        from src.stage_b_stateful.risk_latch import (
            RiskLatchMode,
            TriggerReason,
            TriggerEvent,
        )
        
        if snapshot.recommended_mode == "EMERGENCY_STOP":
            risk_latch._add_trigger(TriggerEvent(
                reason=TriggerReason.MANUAL_EMERGENCY,
                mode=RiskLatchMode.EMERGENCY_STOP,
                severity=1.0,
                message=f"Intraday monitor: {snapshot.trigger_reason}",
                details=snapshot.to_dict(),
                latch_sessions=1,
            ))
            return True
        
        elif snapshot.recommended_mode == "SAFE_FALLBACK":
            risk_latch._add_trigger(TriggerEvent(
                reason=TriggerReason.DATA_FAILURE,
                mode=RiskLatchMode.SAFE_FALLBACK,
                severity=1.0,
                message=f"Intraday monitor: {snapshot.trigger_reason}",
                details=snapshot.to_dict(),
                latch_sessions=1,
            ))
            return True
        
        elif snapshot.recommended_mode == "FLATTEN":
            risk_latch._add_trigger(TriggerEvent(
                reason=TriggerReason.VOL_KILL,
                mode=RiskLatchMode.FLATTEN,
                severity=1.0,
                message=f"Intraday monitor: {snapshot.trigger_reason}",
                details=snapshot.to_dict(),
                latch_sessions=1,
            ))
            return True
        
        elif snapshot.recommended_mode == "THROTTLE":
            risk_latch._add_trigger(TriggerEvent(
                reason=TriggerReason.VOL_STRESS,
                mode=RiskLatchMode.THROTTLE,
                severity=snapshot.recommended_scale,
                message=f"Intraday monitor: {snapshot.trigger_reason}",
                details=snapshot.to_dict(),
                latch_sessions=1,
            ))
            return True
        
        return False
    
    except Exception as e:
        logger.warning("[apply_intraday_snapshot_to_latch] Error: %s", e)
        return False


# ─────────────────────────────────────────────────────────────────────────────
# Example Live Monitor Loop
# ─────────────────────────────────────────────────────────────────────────────

def run_live_monitor_loop(
    symbols: List[str],
    get_prices_fn: "Callable[[], np.ndarray]",
    get_positions_fn: "Callable[[], np.ndarray]",
    interval_seconds: float = 30.0,
    config: Optional[IntradayMonitorConfig] = None,
) -> None:
    """
    Run live intraday monitor loop.
    
    This is a template for running the monitor in production.
    
    Args:
        symbols: List of symbols to monitor
        get_prices_fn: Function that returns current prices as np.ndarray
        get_positions_fn: Function that returns current weights as np.ndarray
        interval_seconds: Update interval
        config: Monitor configuration
    """
    import time
    
    monitor = IntradayRiskMonitor(symbols=symbols, config=config)
    
    logger.info("[run_live_monitor_loop] Starting intraday monitor loop")
    
    while True:
        try:
            # Get current data
            prices = get_prices_fn()
            positions = get_positions_fn()
            
            # Update monitor
            monitor.set_positions(positions)
            snapshot = monitor.update(
                prices=prices,
                data_feed_ok=True,
            )
            
            # Always write snapshot (so daily engine has latest)
            monitor.write_snapshot()
            
            # Log if non-normal
            if snapshot.recommended_mode != "NORMAL":
                logger.warning(
                    "[IntradayMonitor] %s: %s (dd=%.2f%%, vol=%.1f%%)",
                    snapshot.recommended_mode,
                    snapshot.trigger_reason,
                    snapshot.portfolio_intraday_dd * 100,
                    snapshot.realized_vol_30m_annual * 100,
                )
            
        except Exception as e:
            logger.error("[run_live_monitor_loop] Error: %s", e)
            # Write error snapshot
            try:
                error_snapshot = IntradaySnapshot(
                    data_feed_ok=False,
                    recommended_mode="SAFE_FALLBACK",
                    trigger_reason=f"Monitor error: {e}",
                )
                with open(config.snapshot_path if config else "artifacts/intraday_monitor/latest_snapshot.json", "w") as f:
                    f.write(error_snapshot.to_json())
            except Exception:
                pass
        
        time.sleep(interval_seconds)
