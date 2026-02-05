"""Feature Governance: Hard drop/mask enforcement based on Feature Atlas.

This module provides runtime enforcement of feature quality rules determined by
the Feature Atlas. It is applied at Track-C/Stage-B panel build time to:

1. DROP columns that are dead (variance < threshold, null% > threshold)
2. MASK columns to zero that are stubs (zero% > threshold but still have some variance)
3. FLAG columns with governance metadata for downstream auditing

Architecture:
    Feature Atlas (tools/build_feature_atlas.py)
        ↓ generates
    feature_atlas_drop_mask.json
        ↓ loaded by
    FeatureGovernance (this module)
        ↓ applied at
    Track-C build time / Stage-B panel builder

Usage:
    from src.features.feature_governance import FeatureGovernance
    
    gov = FeatureGovernance.load()
    panel = gov.apply(panel)  # Returns cleaned panel with governance flags

Copyright 2024-2026. All rights reserved.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, FrozenSet, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Default paths
REPO_ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS_ROOT = REPO_ROOT / "artifacts"
ATLAS_DIR = ARTIFACTS_ROOT / "feature_atlas"
DROP_MASK_PATH = ATLAS_DIR / "feature_atlas_drop_mask.json"

# Thresholds (defaults, can be overridden by atlas)
DEFAULT_VARIANCE_THRESHOLD = 1e-10
DEFAULT_ZERO_PCT_THRESHOLD = 0.95
DEFAULT_NULL_PCT_THRESHOLD = 0.50


@dataclass
class GovernanceConfig:
    """Configuration for feature governance enforcement."""
    
    # Columns to drop entirely
    drop_columns: FrozenSet[str] = field(default_factory=frozenset)
    
    # Columns to mask to zero (with governance flag)
    mask_columns: FrozenSet[str] = field(default_factory=frozenset)
    
    # Thresholds used
    variance_threshold: float = DEFAULT_VARIANCE_THRESHOLD
    zero_pct_threshold: float = DEFAULT_ZERO_PCT_THRESHOLD
    null_pct_threshold: float = DEFAULT_NULL_PCT_THRESHOLD
    
    # Metadata
    generated_at: str = ""
    source_path: str = ""
    
    # Runtime enforcement settings
    strict_mode: bool = False  # If True, fail on unknown columns
    log_actions: bool = True
    add_governance_flags: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "drop_columns": sorted(self.drop_columns),
            "mask_columns": sorted(self.mask_columns),
            "variance_threshold": self.variance_threshold,
            "zero_pct_threshold": self.zero_pct_threshold,
            "null_pct_threshold": self.null_pct_threshold,
            "generated_at": self.generated_at,
            "source_path": self.source_path,
        }


@dataclass
class GovernanceResult:
    """Result of applying governance to a panel."""
    
    n_columns_input: int = 0
    n_columns_output: int = 0
    n_dropped: int = 0
    n_masked: int = 0
    dropped_columns: List[str] = field(default_factory=list)
    masked_columns: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "n_columns_input": self.n_columns_input,
            "n_columns_output": self.n_columns_output,
            "n_dropped": self.n_dropped,
            "n_masked": self.n_masked,
            "dropped_columns": self.dropped_columns,
            "masked_columns": self.masked_columns,
        }


class FeatureGovernance:
    """Feature governance enforcement based on Feature Atlas quality metrics.
    
    This class provides runtime enforcement of feature quality rules:
    - Drops dead columns (near-zero variance, high null%)
    - Masks stub columns to zero (mostly zeros but have some signal)
    - Adds governance flags for downstream auditing
    
    The enforcement rules are determined by the Feature Atlas tool and
    stored in feature_atlas_drop_mask.json.
    """
    
    def __init__(self, config: GovernanceConfig):
        self.config = config
        self._drop_set: Set[str] = set(config.drop_columns)
        self._mask_set: Set[str] = set(config.mask_columns)
    
    @classmethod
    def load(
        cls,
        path: Optional[Path] = None,
        strict_mode: bool = False,
        log_actions: bool = True,
    ) -> "FeatureGovernance":
        """Load governance config from Feature Atlas drop_mask.json.
        
        Args:
            path: Path to feature_atlas_drop_mask.json. If None, uses default.
            strict_mode: If True, fail on unknown columns.
            log_actions: If True, log drop/mask actions.
        
        Returns:
            FeatureGovernance instance.
        """
        if path is None:
            path = DROP_MASK_PATH
        
        path = Path(path)
        
        if not path.exists():
            logger.warning(
                "No Feature Atlas drop_mask.json found at %s. "
                "Using empty governance (no columns will be dropped/masked). "
                "Run tools/build_feature_atlas.py to generate.",
                path,
            )
            return cls(GovernanceConfig(strict_mode=strict_mode, log_actions=log_actions))
        
        try:
            with open(path, "r") as f:
                data = json.load(f)
        except Exception as e:
            logger.error("Failed to load Feature Atlas drop_mask.json: %s", e)
            return cls(GovernanceConfig(strict_mode=strict_mode, log_actions=log_actions))
        
        thresholds = data.get("thresholds", {})
        
        config = GovernanceConfig(
            drop_columns=frozenset(data.get("drop_columns", [])),
            mask_columns=frozenset(data.get("mask_columns", [])),
            variance_threshold=float(thresholds.get("variance_threshold", DEFAULT_VARIANCE_THRESHOLD)),
            zero_pct_threshold=float(thresholds.get("zero_pct_threshold", DEFAULT_ZERO_PCT_THRESHOLD)),
            null_pct_threshold=float(thresholds.get("null_pct_threshold", DEFAULT_NULL_PCT_THRESHOLD)),
            generated_at=str(data.get("generated_at", "")),
            source_path=str(path),
            strict_mode=strict_mode,
            log_actions=log_actions,
        )
        
        logger.info(
            "Loaded Feature Governance: %d drop, %d mask columns from %s",
            len(config.drop_columns),
            len(config.mask_columns),
            path.name,
        )
        
        return cls(config)
    
    @classmethod
    def from_runtime_analysis(
        cls,
        panel: pd.DataFrame,
        variance_threshold: float = DEFAULT_VARIANCE_THRESHOLD,
        zero_pct_threshold: float = DEFAULT_ZERO_PCT_THRESHOLD,
        null_pct_threshold: float = DEFAULT_NULL_PCT_THRESHOLD,
    ) -> "FeatureGovernance":
        """Create governance config from runtime analysis of a panel.
        
        This is a fallback when Feature Atlas has not been run. It performs
        the same analysis but at runtime (slower).
        
        Args:
            panel: Feature panel to analyze.
            variance_threshold: Below this → dead column.
            zero_pct_threshold: Above this → stub column.
            null_pct_threshold: Above this → drop column.
        
        Returns:
            FeatureGovernance instance.
        """
        drop_cols: Set[str] = set()
        mask_cols: Set[str] = set()
        
        for col in panel.columns:
            if col in {"date", "symbol", "session", "row_id", "timestamp"}:
                continue
            
            col_data = pd.to_numeric(panel[col], errors="coerce")
            n_rows = len(col_data)
            n_null = int(col_data.isna().sum())
            null_pct = n_null / n_rows if n_rows > 0 else 1.0
            
            # High null → drop
            if null_pct > null_pct_threshold:
                drop_cols.add(col)
                continue
            
            # Compute variance on valid data
            valid_data = col_data.dropna()
            if len(valid_data) == 0:
                drop_cols.add(col)
                continue
            
            variance = float(valid_data.var())
            
            # Near-zero variance → dead → drop
            if variance < variance_threshold:
                drop_cols.add(col)
                continue
            
            # Check zero percentage
            zero_pct = (valid_data == 0).sum() / len(valid_data)
            
            # High zero% but has some variance → stub → mask
            if zero_pct > zero_pct_threshold:
                mask_cols.add(col)
        
        config = GovernanceConfig(
            drop_columns=frozenset(drop_cols),
            mask_columns=frozenset(mask_cols),
            variance_threshold=variance_threshold,
            zero_pct_threshold=zero_pct_threshold,
            null_pct_threshold=null_pct_threshold,
            generated_at="runtime",
            source_path="runtime_analysis",
            log_actions=True,
        )
        
        logger.info(
            "Runtime governance analysis: %d drop, %d mask columns",
            len(drop_cols),
            len(mask_cols),
        )
        
        return cls(config)
    
    def apply(
        self,
        panel: pd.DataFrame,
        inplace: bool = False,
    ) -> Tuple[pd.DataFrame, GovernanceResult]:
        """Apply governance rules to a feature panel.
        
        Args:
            panel: Feature panel DataFrame.
            inplace: If True, modify panel in place.
        
        Returns:
            Tuple of (cleaned_panel, GovernanceResult).
        """
        result = GovernanceResult(n_columns_input=len(panel.columns))
        
        if not inplace:
            panel = panel.copy()
        
        # Identify columns to drop (intersection with panel columns)
        panel_cols = set(panel.columns)
        to_drop = self._drop_set & panel_cols
        to_mask = self._mask_set & panel_cols
        
        # Drop columns
        if to_drop:
            panel.drop(columns=list(to_drop), inplace=True)
            result.dropped_columns = sorted(to_drop)
            result.n_dropped = len(to_drop)
            if self.config.log_actions:
                logger.info(
                    "GOVERNANCE: Dropped %d dead/stub columns (examples: %s)",
                    len(to_drop),
                    list(to_drop)[:5],
                )
        
        # Mask columns to zero
        if to_mask:
            for col in to_mask:
                if col in panel.columns:
                    panel[col] = 0.0
            result.masked_columns = sorted(to_mask)
            result.n_masked = len(to_mask)
            if self.config.log_actions:
                logger.info(
                    "GOVERNANCE: Masked %d stub columns to zero (examples: %s)",
                    len(to_mask),
                    list(to_mask)[:5],
                )
        
        # Add governance flags
        if self.config.add_governance_flags:
            panel.attrs["governance_applied"] = True
            panel.attrs["governance_n_dropped"] = result.n_dropped
            panel.attrs["governance_n_masked"] = result.n_masked
            panel.attrs["governance_source"] = self.config.source_path
        
        result.n_columns_output = len(panel.columns)
        
        return panel, result
    
    def should_drop(self, column: str) -> bool:
        """Check if a column should be dropped."""
        return column in self._drop_set
    
    def should_mask(self, column: str) -> bool:
        """Check if a column should be masked to zero."""
        return column in self._mask_set
    
    def get_usable_columns(self, columns: List[str]) -> List[str]:
        """Filter a list of columns to only usable ones.
        
        Removes columns that should be dropped. Columns that should be
        masked are kept (they will be zeroed at apply time).
        """
        return [c for c in columns if c not in self._drop_set]
    
    def __repr__(self) -> str:
        return (
            f"FeatureGovernance(drop={len(self._drop_set)}, "
            f"mask={len(self._mask_set)}, source={self.config.source_path})"
        )


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

_GLOBAL_GOVERNANCE: Optional[FeatureGovernance] = None


def get_governance() -> FeatureGovernance:
    """Get or create the global FeatureGovernance instance."""
    global _GLOBAL_GOVERNANCE
    if _GLOBAL_GOVERNANCE is None:
        _GLOBAL_GOVERNANCE = FeatureGovernance.load()
    return _GLOBAL_GOVERNANCE


def apply_governance(
    panel: pd.DataFrame,
    inplace: bool = False,
) -> Tuple[pd.DataFrame, GovernanceResult]:
    """Apply global feature governance to a panel.
    
    Convenience function that uses the global governance instance.
    """
    gov = get_governance()
    return gov.apply(panel, inplace=inplace)


def governance_filter_columns(columns: List[str]) -> List[str]:
    """Filter columns through governance (remove drop columns)."""
    gov = get_governance()
    return gov.get_usable_columns(columns)


# =============================================================================
# INTEGRATION HELPER FOR TRACK-C / STAGE-B
# =============================================================================

def apply_governance_to_panel_build(
    panel: pd.DataFrame,
    symbol: str,
    horizon: int,
    governance: Optional[FeatureGovernance] = None,
) -> pd.DataFrame:
    """Apply governance during panel build (Track-C/Stage-B integration point).
    
    This function should be called at the end of panel building, before
    the panel is used for training or inference.
    
    Args:
        panel: Built feature panel.
        symbol: Symbol being processed.
        horizon: Horizon in trading days.
        governance: Optional FeatureGovernance instance. If None, uses global.
    
    Returns:
        Cleaned panel with governance applied.
    """
    if governance is None:
        governance = get_governance()
    
    panel, result = governance.apply(panel, inplace=False)
    
    if result.n_dropped > 0 or result.n_masked > 0:
        logger.info(
            "[%s h%d] Governance: dropped=%d, masked=%d",
            symbol,
            horizon,
            result.n_dropped,
            result.n_masked,
        )
    
    # Add to panel attrs for provenance
    panel.attrs["governance_result"] = result.to_dict()
    
    return panel
