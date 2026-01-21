from __future__ import annotations

from typing import Dict

import pandas as pd

from .datasets import align_financials, make_supervised
from .ml_models import one_step_predictions, train_history_models


def infer_paths_sklearn(inc_df: pd.DataFrame,
                        bal_df: pd.DataFrame,
                        cf_df: pd.DataFrame,
                        years: int,
                        rev_col: str) -> Dict[str,
                                              list[float]]:
    """Infer per‑year paths using sklearn models trained on the company's own history.

    This is a pragmatic per-ticker method: build supervised datasets from history and
    predict next-period ratios/growth, then extend as flat paths.
    """
    df = align_financials(inc_df, bal_df, cf_df)
    if df.empty or rev_col not in inc_df.columns:
        return {}

    # Supervised datasets (predict next-period values)
    Xg, yg = make_supervised(
        df
        [["rev_growth", "gross_margin", "ebit_margin", "da_to_rev",
          "capex_to_rev", "nwc_to_rev"]],
        "rev_growth")
    Xm, ym = make_supervised(
        df
        [["gross_margin", "ebit_margin", "da_to_rev", "capex_to_rev",
          "nwc_to_rev", "rev_growth"]],
        "ebit_margin")
    Xc, yc = make_supervised(
        df
        [["capex_to_rev", "gross_margin", "ebit_margin",
          "nwc_to_rev", "rev_growth"]],
        "capex_to_rev")
    Xn, yn = make_supervised(
        df
        [["nwc_to_rev", "gross_margin", "ebit_margin",
          "capex_to_rev", "rev_growth"]],
        "nwc_to_rev")

    models = train_history_models(Xg, yg, Xm, ym, Xc, yc, Xn, yn)
    # Current state vector: use last row's features
    last = df.dropna().tail(1)
    if last.empty:
        return {}
    Xg0 = last[["rev_growth", "gross_margin", "ebit_margin",
                "da_to_rev", "capex_to_rev", "nwc_to_rev"]].values[0]
    Xm0 = last[["gross_margin", "ebit_margin", "da_to_rev",
                "capex_to_rev", "nwc_to_rev", "rev_growth"]].values[0]
    Xc0 = last[["capex_to_rev", "gross_margin",
                "ebit_margin", "nwc_to_rev", "rev_growth"]].values[0]
    Xn0 = last[["nwc_to_rev", "gross_margin", "ebit_margin",
                "capex_to_rev", "rev_growth"]].values[0]

    pred = one_step_predictions(models, Xg0, Xm0, Xc0, Xn0)
    # Build flat paths (could extend to iterative one-step ahead; start simple)
    g = float(
        pred.get(
            "rev_cagr",
            last["rev_growth"].item() if "rev_growth" in last else 0.05))
    em = float(
        pred.get(
            "ebit_margin",
            last["ebit_margin"].item() if "ebit_margin" in last else 0.15))
    cx = float(
        pred.get(
            "capex_to_rev",
            last["capex_to_rev"].item() if "capex_to_rev" in last else 0.05))
    nw = float(
        pred.get(
            "nwc_to_rev",
            last["nwc_to_rev"].item() if "nwc_to_rev" in last else 0.03))

    # gross margin remains aligned with last or implied by ebit + opex offset
    gm = float(last.get("gross_margin", pd.Series([0.4])).iloc[-1])
    return {
        "rev_growth_path": [g] * years,
        "gross_margin_path": [gm] * years,
        "ebit_margin_path": [em] * years,
        "da_to_rev_path": [float(last.get("da_to_rev", pd.Series([0.04])).iloc[-1])] * years,
        "capex_to_rev_path": [cx] * years,
        "nwc_to_rev_path": [nw] * years,
    }
