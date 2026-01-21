"""Aggregator-backed family modules for Stage A blending.

Each base feature family from the universal aggregator is exposed as a lightweight
``ModuleSignal`` emitter so Stage A can learn per-family weights alongside
specialised Hugging Face modules.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, MutableMapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

try:  # Prefer scikit-learn when available for SVD-based compression
    from sklearn.decomposition import TruncatedSVD
except Exception:  # pragma: no cover - optional dependency in some runtimes
    TruncatedSVD = None  # type: ignore

from ..signal_bus import ModuleSignal

try:
    from ...features.aggregator_panel import build_panel as default_build_panel
except Exception:  # pragma: no cover - aggregator may be optional in some tests
    default_build_panel = None  # type: ignore

LOGGER = logging.getLogger(__name__)

PanelBuilder = Callable[..., pd.DataFrame]

# Default list mirrors ``tools/generate_family_signals.DEFAULT_BASE_FAMILIES``.
DEFAULT_FAMILIES: Tuple[str, ...] = (
    "microstructure",
    "cross_asset",
    "macro_sector",
    "macro_enhanced",
    "garch_iv",
    "cboe_term",
    "correlation",
    "options",
    "short_interest",
    "subsidiary",
    "earnings",
    "dividends",
    "alternative_signals",
    "calibration",
    "online_learning",
    "ml_framework",
    "drift_monitor",
    "quantile_forecast",
    "arima_forecast",
    "multiasset",
    "regime",
    "options_anchoring",
    "tft_features",
    "fin_g2",
    "fin_g3",
    "finbert",
    "dcf",
    # HuggingFace-based modules
    "earnings_transcript_hf",  # ✨ Added: HF FinBERT earnings transcripts
    "doc_embedding_novelty_hf",  # ✨ Added: HF document embedding novelty
    "macro_tst_hf",  # ✨ Added: HF macro time-series transformer
)


def _sanitize_family_name(token: str) -> str:
    normalized = str(token or "").strip().lower()
    if not normalized:
        return "family"
    allowed = []
    for char in normalized:
        if char.isalnum() or char in {"_", "-"}:
            allowed.append(char)
        else:
            allowed.append("_")
    collapsed = "".join(allowed).strip("_")
    return collapsed or "family"


def _default_panel_builder() -> PanelBuilder:
    if default_build_panel is None:
        raise RuntimeError("aggregator_panel.build_panel is unavailable")
    return default_build_panel


def _resolve_lookup_window(horizon: int, minimum: int, multiplier: float) -> int:
    base = int(max(float(horizon) * multiplier, minimum))
    return max(base, minimum)


@dataclass
class FamilyModule:
    """Adapter that turns an aggregator family into a ``ModuleSignal``."""

    family: str
    panel_builder: Optional[PanelBuilder] = None
    lookback_days: int = 504
    min_history_days: int = 120
    min_features: int = 2
    history_multiplier: float = 40.0
    macro_lag_days: int = 1
    finbert_gap_thresh: int = 0
    n_components: int = 1
    svd_random_state: int = 13
    require_svd: bool = False
    metadata: MutableMapping[str, object] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        self.family = str(self.family or "").strip()
        if not self.family:
            raise ValueError("family module requires a non-empty family name")
        if self.panel_builder is None:
            self.panel_builder = _default_panel_builder()
        self.metadata = {
            "group": "family",
            "family": self.family,
            "builder": getattr(self.panel_builder, "__name__", "panel_builder"),
        }

    @property
    def name(self) -> str:
        return f"family_{_sanitize_family_name(self.family)}"

    def _fetch_panel(self, symbol: str, horizon: int) -> Optional[pd.DataFrame]:
        if self.panel_builder is None:
            LOGGER.debug("Panel builder missing for family %s", self.family)
            return None

        end_dt = pd.Timestamp.utcnow().normalize()
        window = _resolve_lookup_window(horizon, self.lookback_days, self.history_multiplier)
        start_dt = end_dt - pd.Timedelta(days=window + self.min_history_days)

        try:
            panel = self.panel_builder(  # type: ignore[misc]
                symbol=symbol,
                start=start_dt.strftime("%Y-%m-%d"),
                end=end_dt.strftime("%Y-%m-%d"),
                families=[self.family],
                macro_lag_days=int(self.macro_lag_days),
                finbert_gap_thresh=int(self.finbert_gap_thresh),
            )
        except TypeError:
            panel = self.panel_builder(  # type: ignore[misc]
                symbol=symbol,
                start=start_dt.strftime("%Y-%m-%d"),
                end=end_dt.strftime("%Y-%m-%d"),
                families=[self.family],
            )
        except Exception as exc:  # pragma: no cover - defensive logging
            LOGGER.warning("Family panel build failed for %s (%s): %s", symbol, self.family, exc)
            return None

        if panel is None or panel.empty:
            return None
        panel = panel.copy()
        panel.index = pd.to_datetime(panel.index, utc=False, errors="coerce").normalize()
        panel = panel.sort_index().dropna(how="all")
        return panel

    def _collapse_panel(self, panel: pd.DataFrame) -> Optional[pd.DataFrame]:
        target_prefix = f"{self.family}_"
        cols = [col for col in panel.columns if str(col).startswith(target_prefix)]
        if not cols:
            return None

        subset = panel[cols].replace([np.inf, -np.inf], np.nan)
        coverage = subset.notna().sum(axis=1)
        subset = subset.ffill().bfill().dropna(how="all")
        if subset.empty:
            return None

        # Remove columns that remain constant after filling to avoid singular matrices.
        # EXCEPT for legitimately static families (subsidiary, short_interest) where
        # constant values are expected (e.g., corporate structure, bi-weekly updates)
        STATIC_FAMILIES = {'subsidiary', 'short_interest'}
        
        if self.family in STATIC_FAMILIES:
            # For static families, use mean of raw values (no SVD needed)
            # These represent point-in-time characteristics that don't vary
            raw_mean = subset.mean(axis=1)
            # Normalize to [-1, 1] range based on the distribution
            raw_min, raw_max = raw_mean.min(), raw_mean.max()
            if raw_max > raw_min + 1e-9:
                # Values vary over time, normalize them
                scores = 2.0 * (raw_mean - raw_min) / (raw_max - raw_min) - 1.0
            else:
                # Truly static (all rows identical) - return small positive signal
                # Use the raw mean value itself (normalized by typical range)
                # For subsidiary: ~1.3, for short_interest: varies
                # Map to [-1, 1] based on whether value is above/below 0.5
                scores = pd.Series(np.tanh((raw_mean - 0.5) * 2.0), index=subset.index)
            
            # Confidence is based on data coverage
            coverage_ratio = (coverage.reindex(subset.index).fillna(0.0) / float(len(cols))).clip(0.0, 1.0)
            conf = np.clip(coverage_ratio.to_numpy() * 0.5, 0.0, 1.0)  # Static data gets moderate conf
            
            collapsed = pd.DataFrame({"score": scores, "conf": conf}, index=subset.index)
            return collapsed.sort_index()
        
        # For non-static families, continue with normal SVD-based collapse
        if self.family not in STATIC_FAMILIES:
            constant_mask = subset.apply(lambda col: col.nunique(dropna=True) <= 1)
            subset = subset.loc[:, ~constant_mask]
            if subset.empty:
                return None

        standardized = subset.copy()
        standardized = standardized - standardized.mean(axis=0)
        std = standardized.std(axis=0, ddof=0).replace(0.0, 1.0)
        standardized = (standardized / std).fillna(0.0)

        n_samples, n_features = standardized.shape
        if n_samples < max(5, self.min_history_days // 10):
            LOGGER.debug(
                "Insufficient sample history (%d) for family %s", n_samples, self.family
            )
            return None

        svd_components = min(max(1, self.n_components), n_features, n_samples)
        explained = 0.0
        if TruncatedSVD is not None and svd_components >= 1:
            try:
                svd = TruncatedSVD(n_components=svd_components, random_state=self.svd_random_state)
                transformed = svd.fit_transform(standardized)
                primary = transformed[:, 0]
                explained = float(svd.explained_variance_ratio_[0]) if svd.explained_variance_ratio_.size else 0.0
            except Exception as exc:  # pragma: no cover - defensive logging
                if self.require_svd:
                    LOGGER.warning("SVD failed for family %s: %s", self.family, exc)
                    return None
                LOGGER.debug("SVD failed for family %s, falling back to mean: %s", self.family, exc)
                primary = standardized.mean(axis=1).to_numpy()
        else:
            if self.require_svd:
                LOGGER.debug(
                    "TruncatedSVD unavailable but required for family %s", self.family
                )
                return None
            primary = standardized.mean(axis=1).to_numpy()

        if len(primary) != n_samples:
            return None

        primary_series = pd.Series(primary, index=standardized.index)
        centered = primary_series - primary_series.mean()
        scaled = centered / (primary_series.std(ddof=0) or 1.0)
        scores = np.tanh(scaled.clip(-6.0, 6.0))

        coverage_ratio = (coverage.reindex(standardized.index).fillna(0.0) / float(n_features)).clip(0.0, 1.0)
        base_conf = explained if explained > 0.0 else 0.35  # heuristic floor when SVD fails
        conf = np.clip(coverage_ratio.to_numpy() * base_conf, 0.0, 1.0)

        collapsed = pd.DataFrame({"score": scores, "conf": conf}, index=standardized.index)
        collapsed = collapsed.sort_index()
        return collapsed

    def emit_signal(self, symbol: str, horizon: int) -> ModuleSignal:
        panel = self._fetch_panel(symbol, int(horizon))
        if panel is None or panel.empty:
            LOGGER.debug("No panel data for family %s (%s)", self.family, symbol)
            empty = pd.DataFrame(columns=["score", "conf"])
            return ModuleSignal(name=self.name, horizon=int(horizon), df=empty, symbol=symbol)

        collapsed = self._collapse_panel(panel)
        if collapsed is None or collapsed.empty:
            LOGGER.debug("Failed to collapse panel for family %s (%s)", self.family, symbol)
            empty = pd.DataFrame(columns=["score", "conf"], index=panel.index)
            return ModuleSignal(name=self.name, horizon=int(horizon), df=empty, symbol=symbol)

        collapsed = collapsed.sort_index()
        return ModuleSignal(name=self.name, horizon=int(horizon), df=collapsed, symbol=symbol)


def build_family_modules(
    families: Optional[Sequence[str]] = None,
    *,
    panel_builder: Optional[PanelBuilder] = None,
    **kwargs,
) -> Dict[str, FamilyModule]:
    """Instantiate ``FamilyModule`` objects for each requested family."""

    selected: Iterable[str] = families or DEFAULT_FAMILIES
    builder = panel_builder or (default_build_panel if panel_builder is None else panel_builder)
    modules: Dict[str, FamilyModule] = {}
    for family in selected:
        try:
            module = FamilyModule(family=family, panel_builder=builder, **kwargs)
        except Exception as exc:
            LOGGER.warning("Skipping family module '%s': %s", family, exc)
            continue
        modules[module.name] = module
    return modules


def register_families_as_modules(
    families: Optional[Sequence[str]] = None,
    *,
    panel_builder: Optional[PanelBuilder] = None,
    register: bool = True,
    **kwargs,
) -> Tuple[str, ...]:
    """Register family modules with the universal aggregator.

    Args:
        families: Optional sequence of family names. Defaults to ``DEFAULT_FAMILIES``.
        panel_builder: Optional custom panel builder to avoid importing the heavy default.
        register: When ``True`` (default) modules are registered immediately via
            :func:`src.dcf_lab.universal_aggregator.register_module`. Set to ``False`` to
            obtain instantiated modules without registration.
        **kwargs: Additional keyword arguments forwarded to :class:`FamilyModule`.

    Returns:
        Tuple containing the names of registered modules (or instantiated module names when
        ``register`` is ``False``).
    """

    modules = build_family_modules(families, panel_builder=panel_builder, **kwargs)
    if not register:
        return tuple(modules.keys())

    from ..universal_aggregator import register_module  # Imported lazily to avoid cycles

    registered: list[str] = []
    for name, module in modules.items():
        try:
            register_module(name, module)
        except Exception as exc:  # pragma: no cover - defensive logging
            LOGGER.warning("Failed to register family module '%s': %s", name, exc)
            continue
        registered.append(name)
    return tuple(registered)


__all__ = [
    "DEFAULT_FAMILIES",
    "FamilyModule",
    "build_family_modules",
    "register_families_as_modules",
]
