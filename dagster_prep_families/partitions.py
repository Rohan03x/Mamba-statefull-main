from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

from dagster import MultiPartitionsDefinition, StaticPartitionsDefinition


def _split_csv(value: str) -> List[str]:
    return [s.strip() for s in (value or "").split(",") if s and s.strip()]


def _load_symbols() -> List[str]:
    """Resolve symbol universe for partitions.

    Priority:
    1) DAGSTER_SYMBOLS="AAPL,MSFT,..."
    2) DAGSTER_SYMBOLS_FILE=/path/to/file (one symbol per line)
    3) default to ["AAPL"]
    """

    env_csv = os.getenv("DAGSTER_SYMBOLS", "").strip()
    if env_csv:
        return [s.upper() for s in _split_csv(env_csv)]

    env_file = os.getenv("DAGSTER_SYMBOLS_FILE", "").strip()
    if env_file:
        p = Path(env_file)
        if p.exists():
            out: List[str] = []
            for line in p.read_text().splitlines():
                sym = line.strip()
                if not sym or sym.startswith("#"):
                    continue
                out.append(sym.upper())
            if out:
                return out

    return ["AAPL"]


def _load_horizons() -> List[str]:
    """Resolve horizon universe for partitions (as strings for StaticPartitionsDefinition)."""

    env_csv = os.getenv("DAGSTER_HORIZONS", "63").strip()
    vals = _split_csv(env_csv) or ["63"]
    out: List[str] = []
    for v in vals:
        try:
            out.append(str(int(v)))
        except Exception:
            continue
    return out or ["63"]


def get_symbol_partitions_def() -> StaticPartitionsDefinition:
    """Symbol-only partitions."""

    symbols = _load_symbols()
    return StaticPartitionsDefinition(symbols)


def get_symbol_horizon_partitions_def() -> MultiPartitionsDefinition:
    """Symbol × horizon partitions."""

    symbols = _load_symbols()
    horizons = _load_horizons()
    return MultiPartitionsDefinition(
        {
            "symbol": StaticPartitionsDefinition(symbols),
            "horizon": StaticPartitionsDefinition(horizons),
        }
    )


# Back-compat alias (older code expects this name).
def get_partitions_def() -> MultiPartitionsDefinition:
    return get_symbol_horizon_partitions_def()
