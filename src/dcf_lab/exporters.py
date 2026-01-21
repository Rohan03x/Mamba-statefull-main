import pandas as pd


def to_excel(
        path,
        ticker,
        is_df,
        bs_df,
        cf_df,
        summary,
        assumptions,
        mc=None,
        shares=None):
    """Legacy minimalist export retained for compatibility."""
    with pd.ExcelWriter(path, engine="xlsxwriter") as xw:
        pd.DataFrame([summary]).to_excel(xw, index=False, sheet_name="Summary")
        pd.DataFrame([assumptions["drivers"]]).to_excel(
            xw, index=False, sheet_name="Drivers")
        pd.DataFrame([assumptions["starting"]]).to_excel(
            xw, index=False, sheet_name="Starting")
        is_df.to_excel(xw, index=False, sheet_name="IncomeStatement")
        bs_df.to_excel(xw, index=False, sheet_name="BalanceSheet")
        cf_df.to_excel(xw, index=False, sheet_name="CashFlow")
        if mc is not None:
            pd.DataFrame({"equity_value_sim": mc}).to_excel(
                xw, index=False, sheet_name="MonteCarlo")


def export_workbook(
    path: str,
    ticker: str,
    inputs: dict,
    is_df: pd.DataFrame,
    bs_df: pd.DataFrame,
    cf_df: pd.DataFrame,
    dcf_summary: dict,
    sensitivity: pd.DataFrame | None = None,
    mc_samples: list[float] | None = None,
    news_items: list[dict] | None = None,
    data_sources: list[dict] | None = None,
    is_ml: pd.DataFrame | None = None,
    bs_ml: pd.DataFrame | None = None,
    cf_ml: pd.DataFrame | None = None,
):
    """Export a full, auditable workbook per acceptance criteria."""
    with pd.ExcelWriter(path, engine="xlsxwriter") as xw:
        pd.DataFrame([inputs]).to_excel(xw, index=False, sheet_name="Inputs")
        is_df.to_excel(xw, index=False, sheet_name="IS_Forecast")
        bs_df.to_excel(xw, index=False, sheet_name="BS_Forecast")
        cf_df.to_excel(xw, index=False, sheet_name="CF_Forecast")
        pd.DataFrame([dcf_summary]).to_excel(xw, index=False, sheet_name="DCF")
        if is_ml is not None and bs_ml is not None and cf_ml is not None:
            is_ml.to_excel(xw, index=False, sheet_name="IS_Forecast_ML")
            bs_ml.to_excel(xw, index=False, sheet_name="BS_Forecast_ML")
            cf_ml.to_excel(xw, index=False, sheet_name="CF_Forecast_ML")
        if sensitivity is not None:
            sensitivity.to_excel(xw, index=False, sheet_name="Sensitivity")
        if mc_samples is not None:
            pd.DataFrame({"equity_value": mc_samples}).to_excel(
                xw, index=False, sheet_name="MonteCarlo")
        if news_items is not None:
            pd.DataFrame(news_items).to_excel(
                xw, index=False, sheet_name="News")
        if data_sources is not None:
            pd.DataFrame(data_sources).to_excel(
                xw, index=False, sheet_name="Data_Sources")
