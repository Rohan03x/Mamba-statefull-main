from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .types import MergedParquetStore


@dataclass
class LocalMergedParquetStore(MergedParquetStore):
    """Loads merged parquets from the standard cache path.

    This is *research-only* and should not be used by production consumers.
    """

    root: Path = Path("cache/features")

    def load_panel(self, *, symbol: str, horizon: int) -> pd.DataFrame:
        sym = str(symbol).upper()
        p = Path(self.root) / f"{sym}_h{int(horizon)}_merged.parquet"
        if not p.exists():
            return pd.DataFrame()
        df = pd.read_parquet(p)
        # Normalize time index: merged parquets are expected to have a DatetimeIndex.
        if "date" in df.columns and not isinstance(df.index, pd.DatetimeIndex):
            df = df.copy()
            df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.tz_localize(None)
            df = df.set_index("date")
        if not isinstance(df.index, pd.DatetimeIndex):
            try:
                df.index = pd.to_datetime(df.index, errors="coerce").tz_localize(None)
            except Exception:
                pass
        return df.sort_index()
