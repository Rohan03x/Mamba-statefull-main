from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class GroupMapBuildResult:
    group_by_symbol: Dict[str, str]
    missing_symbols: Tuple[str, ...]
    used_key: str
    out_path: Optional[Path]


def _extract_symbol_and_group(
    payload: object,
    *,
    symbol_key: str = "Code",
    group_key_preference: Sequence[str] = ("GicSector", "Sector"),
) -> Optional[Tuple[str, str, str]]:
    if not isinstance(payload, dict):
        return None

    # EODHD profile schema: {"General": {...}}
    general = payload.get("General") if isinstance(payload.get("General"), dict) else None
    if not isinstance(general, dict):
        return None

    sym = general.get(symbol_key)
    if not sym:
        return None
    sym_u = str(sym).upper().strip()
    if not sym_u:
        return None

    for key in group_key_preference:
        val = general.get(key)
        if val is None:
            continue
        g = str(val).strip()
        if g:
            return sym_u, g, str(key)

    return None


def _extract_from_holdings_blocks(
    payload: object,
    *,
    symbol_key: str = "Code",
    group_key_preference: Sequence[str] = ("GicSector", "Sector"),
) -> Dict[str, Tuple[str, str]]:
    """Extract {SYMBOL: (group, key_used)} from holdings-like EODHD JSONs.

    Some EODHD cache payloads are ETF/fund snapshots that include holdings under
    keys like "Top_10_Holdings" and/or "Holdings" where each holding has
    {"Code": "AAPL", "Sector": "Technology", ...}.
    """

    if not isinstance(payload, dict):
        return {}

    out: Dict[str, Tuple[str, str]] = {}
    for block_key in ("Top_10_Holdings", "Holdings"):
        block = payload.get(block_key)
        if not isinstance(block, dict):
            continue
        for _k, v in block.items():
            if not isinstance(v, dict):
                continue
            sym = v.get(symbol_key)
            if not sym:
                continue
            sym_u = str(sym).upper().strip()
            if not sym_u or sym_u in out:
                continue
            for key in group_key_preference:
                val = v.get(key)
                if val is None:
                    continue
                g = str(val).strip()
                if g:
                    out[sym_u] = (g, str(key))
                    break
    return out


def build_symbol_group_map_from_eodhd_cache(
    *,
    cache_dir: Path,
    symbols: Sequence[str],
    out_path: Optional[Path] = None,
    group_key_preference: Sequence[str] = ("GicSector", "Sector"),
    auto_fetch_missing: bool = True,
) -> GroupMapBuildResult:
    """Build a symbol->group map from the local EODHD cache.

    Intended for Phase2 group caps.

    - First tries local cache files under `data/cache/eodhd`
    - If auto_fetch_missing=True (default), fetches missing symbols via EODHD API
    - Stops early once all requested `symbols` have been found.

    Output CSV schema if `out_path` is provided: columns `symbol,group`.
    """

    cache_dir = Path(cache_dir)
    want = {str(s).upper().strip() for s in symbols if str(s).strip()}
    if not want:
        return GroupMapBuildResult(group_by_symbol={}, missing_symbols=(), used_key="", out_path=out_path)

    found: Dict[str, str] = {}
    used_key = ""

    if cache_dir.exists():
        # Iterate deterministically through existing cache.
        for p in sorted(cache_dir.glob("*.json")):
            try:
                payload = json.loads(p.read_text())
            except Exception:
                continue

            extracted = _extract_symbol_and_group(payload, group_key_preference=group_key_preference)
            if extracted is not None:
                sym_u, grp, key_used = extracted
                if sym_u in want and sym_u not in found:
                    found[sym_u] = grp
                    if not used_key:
                        used_key = key_used

            # Fallback for ETF/holdings-style payloads.
            if len(found) < len(want):
                hold = _extract_from_holdings_blocks(payload, group_key_preference=group_key_preference)
                for sym_u, (grp, key_used) in hold.items():
                    if sym_u in want and sym_u not in found:
                        found[sym_u] = grp
                        if not used_key:
                            used_key = key_used

            if len(found) >= len(want):
                break

    missing = tuple(sorted(want - set(found.keys())))

    # Auto-fetch missing symbols from EODHD API
    if auto_fetch_missing and missing:
        logger.info(f"Group map: found {len(found)}/{len(want)} symbols in cache, fetching {len(missing)} from EODHD API")
        try:
            from src.data_sources.eodhd_provider import EODHDProvider
            
            eodhd = EODHDProvider()
            for sym in missing:
                try:
                    logger.debug(f"Fetching fundamentals for {sym}")
                    fundamentals = eodhd.get_fundamentals(sym)
                    if fundamentals:
                        # Extract group using same logic
                        extracted = _extract_symbol_and_group(fundamentals, group_key_preference=group_key_preference)
                        if extracted is not None:
                            sym_u, grp, key_used = extracted
                            if sym_u == sym:
                                found[sym] = grp
                                if not used_key:
                                    used_key = key_used
                                logger.info(f"Fetched {sym} → {grp} (via {key_used})")
                        else:
                            # Fallback for ETFs/special cases: use Type or default to "ETF"
                            general = fundamentals.get("General", {})
                            if isinstance(general, dict):
                                # Try to get meaningful group from Type (ETF, Fund, etc.)
                                type_val = general.get("Type", "")
                                if type_val:
                                    found[sym] = f"{type_val}"
                                    logger.info(f"Fetched {sym} → {type_val} (fallback: Type field)")
                                else:
                                    found[sym] = "Other"
                                    logger.info(f"Fetched {sym} → Other (fallback: no sector)")
                except Exception as e:
                    logger.warning(f"Failed to fetch fundamentals for {sym}: {e}")
                    continue
            
            # Recalculate missing after fetch
            missing = tuple(sorted(want - set(found.keys())))
            if missing:
                logger.warning(f"Group map still missing {len(missing)} symbols after fetch: {missing}")
        except Exception as e:
            logger.error(f"Failed to auto-fetch missing symbols from EODHD: {e}")

    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        # Write a minimal, stable CSV.
        lines = ["symbol,group\n"]
        for sym in sorted(found.keys()):
            g = str(found[sym]).replace("\n", " ").replace("\r", " ").strip()
            lines.append(f"{sym},{g}\n")
        out_path.write_text("".join(lines))

    return GroupMapBuildResult(
        group_by_symbol=dict(found),
        missing_symbols=missing,
        used_key=used_key,
        out_path=out_path,
    )
