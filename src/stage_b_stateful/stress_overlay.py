"""
Scenario Stress Tests - On-the-fly Stress Overlays for Phase-2.

This module provides cheap, daily, powerful stress probes that convert
holdings into "what if" losses and scale weights down if thresholds
are breached.

Stress Tests:
A) k-sigma market shock (covariance-based)
   - Compute port_vol = sqrt(w' @ cov @ w)
   - stress_loss = k * port_vol
   - If stress_loss > cap → THROTTLE or FLATTEN

B) Correlation blow-up stress
   - Use corr_hhi (already computed in policy dims)
   - If corr_hhi > threshold AND vol rising → reduce gross

C) Sector/event concentration stress
   - If max sector weight > cap → throttle
   - If sector concentration rising sharply → throttle

Usage:
    stress_overlay = StressTestOverlay(config, n_assets)
    
    # After optimizer produces w_target, before final safety overlays:
    w_stressed, stress_result = stress_overlay.apply(
        w=w_target,
        cov=cov,
        vol_prev=vol_prev,
        vol_curr=vol_curr,
        corr_hhi=corr_hhi,
        sector_weights=sector_weights,
    )
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Stress Test Configuration
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class StressTestConfig:
    """Configuration for stress test overlays."""
    
    # k-sigma market shock
    shock_k_sigma: float = 3.0              # k-sigma shock multiplier
    shock_loss_throttle_pct: float = 0.08   # Throttle if stress_loss > 8%
    shock_loss_flatten_pct: float = 0.15    # Flatten if stress_loss > 15%
    shock_throttle_scale: float = 0.5       # Scale to 50% on throttle
    
    # Correlation blow-up
    corr_hhi_throttle: float = 0.4          # Throttle if corr_hhi > 0.4
    corr_hhi_flatten: float = 0.6           # Flatten if corr_hhi > 0.6
    corr_vol_rising_mult: float = 1.2       # "Vol rising" = curr > 1.2x prev
    corr_throttle_scale: float = 0.6        # Scale to 60% on corr stress
    
    # Sector concentration
    sector_max_weight: float = 0.30         # Max sector weight 30%
    sector_throttle_mult: float = 1.5       # Throttle if sector > 1.5x cap
    sector_rising_threshold: float = 0.05   # "Rising sharply" = +5% session
    sector_throttle_scale: float = 0.7      # Scale to 70% on sector stress
    
    # Combined stress
    max_combined_scale: float = 0.2         # Never scale below 20% from stress
    stress_enabled: bool = True             # Master switch


@dataclass
class StressTestResult:
    """Result of stress test overlay."""
    
    # Overall result
    triggered: bool = False
    final_scale: float = 1.0
    
    # Individual stress results
    shock_triggered: bool = False
    shock_scale: float = 1.0
    shock_loss_pct: float = 0.0
    
    corr_triggered: bool = False
    corr_scale: float = 1.0
    corr_hhi: float = 0.0
    vol_rising: bool = False
    
    sector_triggered: bool = False
    sector_scale: float = 1.0
    max_sector_weight: float = 0.0
    max_sector_name: str = ""
    
    # Messages
    messages: List[str] = field(default_factory=lambda: [])
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "triggered": self.triggered,
            "final_scale": float(self.final_scale),
            "shock": {
                "triggered": self.shock_triggered,
                "scale": float(self.shock_scale),
                "loss_pct": float(self.shock_loss_pct),
            },
            "correlation": {
                "triggered": self.corr_triggered,
                "scale": float(self.corr_scale),
                "hhi": float(self.corr_hhi),
                "vol_rising": self.vol_rising,
            },
            "sector": {
                "triggered": self.sector_triggered,
                "scale": float(self.sector_scale),
                "max_weight": float(self.max_sector_weight),
                "max_name": self.max_sector_name,
            },
            "messages": self.messages,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Stress Test Overlay
# ─────────────────────────────────────────────────────────────────────────────

class StressTestOverlay:
    """
    Apply stress test overlays to portfolio weights.
    
    This is called AFTER the optimizer produces w_target,
    BEFORE final safety overlays finalize it.
    """
    
    def __init__(
        self,
        config: Optional[StressTestConfig] = None,
        n_assets: int = 0,
    ):
        """
        Initialize stress test overlay.
        
        Args:
            config: Stress test configuration
            n_assets: Number of assets
        """
        self.config = config or StressTestConfig()
        self.n_assets = n_assets
        
        # Track previous values for "rising" detection
        self._prev_vol: Optional[float] = None
        self._prev_sector_weights: Optional[Dict[str, float]] = None
    
    def apply(
        self,
        w: np.ndarray,
        cov: Optional[np.ndarray] = None,
        vol_prev: Optional[float] = None,
        vol_curr: Optional[float] = None,
        corr_hhi: Optional[float] = None,
        sector_weights: Optional[Dict[str, float]] = None,
        symbols: Optional[List[str]] = None,
        group_by_symbol: Optional[Dict[str, str]] = None,
    ) -> Tuple[np.ndarray, StressTestResult]:
        """
        Apply all stress tests and return scaled weights.
        
        Args:
            w: Target weights from optimizer
            cov: Covariance matrix (n_assets x n_assets)
            vol_prev: Previous session's realized vol
            vol_curr: Current session's realized vol
            corr_hhi: Correlation HHI (0-1)
            sector_weights: Dict of sector -> total weight
            symbols: Symbol names (for sector calculation)
            group_by_symbol: Dict of symbol -> sector
        
        Returns:
            (scaled_weights, StressTestResult)
        """
        cfg = self.config
        result = StressTestResult()
        
        if not cfg.stress_enabled:
            return w.copy(), result
        
        w = np.asarray(w, dtype=float).copy()
        scales: List[float] = []
        
        # ─────────────────────────────────────────────────────────────────────
        # A) k-sigma market shock (covariance-based)
        # ─────────────────────────────────────────────────────────────────────
        if cov is not None:
            shock_scale, shock_loss = self._stress_k_sigma_shock(w, cov)
            result.shock_scale = shock_scale
            result.shock_loss_pct = shock_loss
            
            if shock_scale < 1.0:
                result.shock_triggered = True
                scales.append(shock_scale)
                result.messages.append(
                    f"k-sigma shock: {cfg.shock_k_sigma:.0f}σ loss = {shock_loss:.1%} → scale={shock_scale:.2f}"
                )
        
        # ─────────────────────────────────────────────────────────────────────
        # B) Correlation blow-up stress
        # ─────────────────────────────────────────────────────────────────────
        if corr_hhi is not None:
            vol_rising = False
            if vol_prev is not None and vol_curr is not None and vol_prev > 0:
                vol_rising = vol_curr > cfg.corr_vol_rising_mult * vol_prev
            
            result.corr_hhi = corr_hhi
            result.vol_rising = vol_rising
            
            corr_scale = self._stress_correlation_blowup(corr_hhi, vol_rising)
            result.corr_scale = corr_scale
            
            if corr_scale < 1.0:
                result.corr_triggered = True
                scales.append(corr_scale)
                result.messages.append(
                    f"Corr blowup: HHI={corr_hhi:.2f}, vol_rising={vol_rising} → scale={corr_scale:.2f}"
                )
        
        # ─────────────────────────────────────────────────────────────────────
        # C) Sector/event concentration stress
        # ─────────────────────────────────────────────────────────────────────
        computed_sector_weights = sector_weights
        if computed_sector_weights is None and group_by_symbol is not None and symbols is not None:
            # Compute sector weights from positions
            computed_sector_weights = self._compute_sector_weights(w, symbols, group_by_symbol)
        
        if computed_sector_weights is not None:
            sector_scale, max_w, max_name = self._stress_sector_concentration(
                computed_sector_weights
            )
            result.sector_scale = sector_scale
            result.max_sector_weight = max_w
            result.max_sector_name = max_name
            
            if sector_scale < 1.0:
                result.sector_triggered = True
                scales.append(sector_scale)
                result.messages.append(
                    f"Sector conc: {max_name}={max_w:.1%} > {cfg.sector_max_weight:.1%} → scale={sector_scale:.2f}"
                )
            
            # Update previous for next call
            self._prev_sector_weights = computed_sector_weights.copy()
        
        # Update previous vol for next call
        if vol_curr is not None:
            self._prev_vol = vol_curr
        
        # ─────────────────────────────────────────────────────────────────────
        # Combine scales
        # ─────────────────────────────────────────────────────────────────────
        if scales:
            # Take minimum of all scales
            final_scale = float(np.min(scales))
            # But never go below max_combined_scale
            final_scale = max(cfg.max_combined_scale, final_scale)
            
            result.triggered = True
            result.final_scale = final_scale
            
            # Apply scale
            w = w * final_scale
            
            logger.info(
                "[StressOverlay] Applied stress scale %.2f: %s",
                final_scale,
                " | ".join(result.messages),
            )
        else:
            result.final_scale = 1.0
        
        return w, result
    
    def _stress_k_sigma_shock(
        self,
        w: np.ndarray,
        cov: np.ndarray,
    ) -> Tuple[float, float]:
        """
        k-sigma market shock stress test.
        
        Computes:
            port_vol = sqrt(w' @ cov @ w)
            stress_loss = k * port_vol
        
        Returns:
            (scale, stress_loss)
        """
        cfg = self.config
        
        try:
            # Portfolio variance
            port_var = float(w @ cov @ w)
            if port_var <= 0:
                return 1.0, 0.0
            
            # Portfolio vol (daily)
            port_vol = np.sqrt(port_var)
            
            # k-sigma stress loss
            stress_loss = cfg.shock_k_sigma * port_vol
            
            # Determine scale
            if stress_loss >= cfg.shock_loss_flatten_pct:
                # Flatten
                return 0.0, stress_loss
            elif stress_loss >= cfg.shock_loss_throttle_pct:
                # Throttle
                return cfg.shock_throttle_scale, stress_loss
            else:
                return 1.0, stress_loss
        
        except Exception as e:
            logger.warning("[StressOverlay._stress_k_sigma_shock] Error: %s", e)
            return 1.0, 0.0
    
    def _stress_correlation_blowup(
        self,
        corr_hhi: float,
        vol_rising: bool,
    ) -> float:
        """
        Correlation blow-up stress test.
        
        Returns:
            scale (0-1)
        """
        cfg = self.config
        
        if corr_hhi >= cfg.corr_hhi_flatten:
            # Very high correlation → flatten
            return 0.0
        
        if corr_hhi >= cfg.corr_hhi_throttle:
            if vol_rising:
                # High correlation AND vol rising → throttle
                return cfg.corr_throttle_scale
            else:
                # High correlation but vol stable → mild throttle
                return (1.0 + cfg.corr_throttle_scale) / 2.0
        
        return 1.0
    
    def _stress_sector_concentration(
        self,
        sector_weights: Dict[str, float],
    ) -> Tuple[float, float, str]:
        """
        Sector concentration stress test.
        
        Returns:
            (scale, max_weight, max_sector_name)
        """
        cfg = self.config
        
        if not sector_weights:
            return 1.0, 0.0, ""
        
        # Find max sector
        max_sector = max(sector_weights.items(), key=lambda x: abs(x[1]))
        max_name = max_sector[0]
        max_weight = abs(max_sector[1])
        
        # Check if rising sharply
        rising_sharply = False
        if self._prev_sector_weights is not None and max_name in self._prev_sector_weights:
            prev_w = abs(self._prev_sector_weights.get(max_name, 0.0))
            if max_weight - prev_w >= cfg.sector_rising_threshold:
                rising_sharply = True
        
        # Determine scale
        if max_weight >= cfg.sector_max_weight * cfg.sector_throttle_mult:
            # Very high concentration → flatten
            return 0.0, max_weight, max_name
        elif max_weight >= cfg.sector_max_weight:
            if rising_sharply:
                # High and rising → throttle more
                return cfg.sector_throttle_scale * 0.8, max_weight, max_name
            else:
                # High but stable → throttle
                return cfg.sector_throttle_scale, max_weight, max_name
        
        return 1.0, max_weight, max_name
    
    def _compute_sector_weights(
        self,
        w: np.ndarray,
        symbols: List[str],
        group_by_symbol: Dict[str, str],
    ) -> Dict[str, float]:
        """Compute sector weights from position weights."""
        sector_weights: Dict[str, float] = {}
        
        for j, sym in enumerate(symbols):
            sector = group_by_symbol.get(str(sym).upper(), "UNKNOWN")
            if sector not in sector_weights:
                sector_weights[sector] = 0.0
            sector_weights[sector] += float(w[j])
        
        return sector_weights


# ─────────────────────────────────────────────────────────────────────────────
# Factory function
# ─────────────────────────────────────────────────────────────────────────────

def create_stress_overlay(
    cfg: Dict[str, Any],
    n_assets: int,
) -> StressTestOverlay:
    """
    Create StressTestOverlay from config dict.
    
    Config keys (all optional, with defaults):
        phase2_stress_enabled: bool = True
        phase2_stress_k_sigma: float = 3.0
        phase2_stress_loss_throttle: float = 0.03  # Daily vol units: 3% stress loss
        phase2_stress_loss_flatten: float = 0.05   # Daily vol units: 5% stress loss
        phase2_stress_shock_scale: float = 0.5
        phase2_stress_corr_hhi_throttle: float = 0.4
        phase2_stress_corr_hhi_flatten: float = 0.6
        phase2_stress_corr_scale: float = 0.6
        phase2_stress_sector_max: float = 0.30
        phase2_stress_sector_scale: float = 0.7
        phase2_stress_min_scale: float = 0.2
    """
    config = StressTestConfig(
        stress_enabled=bool(cfg.get("phase2_stress_enabled", True)),
        shock_k_sigma=float(cfg.get("phase2_stress_k_sigma", 3.0)),
        shock_loss_throttle_pct=float(cfg.get("phase2_stress_loss_throttle", 0.03)),
        shock_loss_flatten_pct=float(cfg.get("phase2_stress_loss_flatten", 0.05)),
        shock_throttle_scale=float(cfg.get("phase2_stress_shock_scale", 0.5)),
        corr_hhi_throttle=float(cfg.get("phase2_stress_corr_hhi_throttle", 0.4)),
        corr_hhi_flatten=float(cfg.get("phase2_stress_corr_hhi_flatten", 0.6)),
        corr_throttle_scale=float(cfg.get("phase2_stress_corr_scale", 0.6)),
        sector_max_weight=float(cfg.get("phase2_stress_sector_max", 0.30)),
        sector_throttle_scale=float(cfg.get("phase2_stress_sector_scale", 0.7)),
        max_combined_scale=float(cfg.get("phase2_stress_min_scale", 0.2)),
    )
    
    return StressTestOverlay(config=config, n_assets=n_assets)


# ─────────────────────────────────────────────────────────────────────────────
# Convenience function for Phase-2 integration
# ─────────────────────────────────────────────────────────────────────────────

def apply_stress_overlay(
    w: np.ndarray,
    cov: np.ndarray,
    vol_curr: float,
    vol_prev: float,
    corr_hhi: float,
    cfg: Dict[str, Any],
    symbols: Optional[List[str]] = None,
    group_by_symbol: Optional[Dict[str, str]] = None,
    stress_overlay: Optional[StressTestOverlay] = None,
) -> Tuple[np.ndarray, StressTestResult]:
    """
    Apply stress overlay to weights (convenience function).
    
    Args:
        w: Target weights
        cov: Covariance matrix
        vol_curr: Current realized vol
        vol_prev: Previous realized vol
        corr_hhi: Correlation HHI
        cfg: Config dict
        symbols: Symbol names
        group_by_symbol: Symbol → sector mapping
        stress_overlay: Existing overlay instance (reused for state)
    
    Returns:
        (scaled_weights, StressTestResult)
    """
    if stress_overlay is None:
        stress_overlay = create_stress_overlay(cfg, len(w))
    
    return stress_overlay.apply(
        w=w,
        cov=cov,
        vol_prev=vol_prev,
        vol_curr=vol_curr,
        corr_hhi=corr_hhi,
        symbols=symbols,
        group_by_symbol=group_by_symbol,
    )
