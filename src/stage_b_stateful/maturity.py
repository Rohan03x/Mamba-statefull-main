"""Maturity gating utilities for leakage-safe walk-forward.

Non-negotiable rule (H=63 by default): labels for day t are only known at t+H.
So at any 'now' day T, supervised samples with sample_day in (T-H, T] are NOT
eligible for training/updates.
"""

from __future__ import annotations

import pandas as pd

from src.dcf_lab.utils.trading_calendar import add_sessions


def is_matured(sample_day: pd.Timestamp, now_day: pd.Timestamp, H: int = 63) -> bool:
    """Return True iff sample_day is label-matured as of now_day.

    Uses trading-session arithmetic (XNYS when available).

    Equivalent to: sample_day <= now_day - H (in sessions).
    """

    s = pd.Timestamp(sample_day).normalize()
    n = pd.Timestamp(now_day).normalize()
    cutoff = add_sessions(n, -int(H))
    return bool(s <= cutoff)


def maturity_cutoff(now_day: pd.Timestamp, H: int = 63) -> pd.Timestamp:
    """Latest sample_day eligible at now_day under horizon H (session-aware)."""

    n = pd.Timestamp(now_day).normalize()
    return add_sessions(n, -int(H))
