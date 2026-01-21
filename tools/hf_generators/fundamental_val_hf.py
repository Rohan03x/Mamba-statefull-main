"""HF block generator for fundamental valuation dynamics."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional

from ._hf_block_common import (
    prepare_block_payload,
    should_update_block_cache,
    update_block_cache,
)

LOGGER = logging.getLogger(__name__)
BLOCK_NAME = "fundamental_val_hf"


def build(
    symbol: str,
    horizon: int,
    start: str,
    end: str,
    out_path: str,
    raw_source_cfg: Optional[Dict[str, Any]] = None,
    compute_cfg: Optional[Dict[str, Any]] = None,
) -> None:
    """Materialize fundamental_val_hf outputs from cached fundamental families."""
    LOGGER.info(
        "%s: building block signal for %s h%s [%s → %s]",
        BLOCK_NAME,
        symbol,
        horizon,
        start,
        end,
    )
    payload, cache_dir = prepare_block_payload(
        block_name=BLOCK_NAME,
        symbol=symbol,
        horizon=horizon,
        start=start,
        end=end,
        out_path=out_path,
        raw_source_cfg=raw_source_cfg,
        compute_cfg=compute_cfg,
    )
    out_file = Path(out_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    payload.to_parquet(out_file, index=False)
    if should_update_block_cache(out_file, raw_source_cfg, compute_cfg):
        try:
            update_block_cache(payload, cache_dir, symbol, horizon, BLOCK_NAME)
        except Exception as exc:
            LOGGER.warning("%s: block cache update skipped (%s)", BLOCK_NAME, exc)
    LOGGER.info("%s: wrote %d rows to %s", BLOCK_NAME, len(payload), out_file)
