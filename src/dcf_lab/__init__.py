"""DCF lab package initialization."""

__version__ = '0.3.0'

# Temporarily commented out for auto_opt compatibility
# from .assumption_builder import infer_drivers
# from .forecast_model import ForecastInputs, build_forecast
# from .valuation import MonteCarloDCF


# Define load_financials stub (will be implemented fully in data_provider.py)
def load_financials(ticker, source='yahoo'):
    """
    Load financial data for a company

    Args:
        ticker: Company ticker symbol
        source: Data source (default: yahoo)

    Returns:
        Dictionary with financial data
    """
    from datetime import datetime, timedelta

    import pandas as pd

    # Generate sample financial data for demonstration
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365*3)

    # Create date range
    dates = pd.date_range(start=start_date, end=end_date, freq='Q')

    # Generate sample revenue data
    revenue = [100 + i * 5 + (i % 4) * 20 for i in range(len(dates))]

    # Generate sample income statement
    income_statement = pd.DataFrame({
        'revenue': revenue,
        'cogs': [r * 0.6 for r in revenue],
        'gross_profit': [r * 0.4 for r in revenue],
        'operating_expenses': [r * 0.2 for r in revenue],
        'operating_income': [r * 0.2 for r in revenue],
        'net_income': [r * 0.15 for r in revenue]
    }, index=dates)

    # Generate sample balance sheet
    balance_sheet = pd.DataFrame({
        'cash': [50 + i * 2 for i in range(len(dates))],
        'accounts_receivable': [30 + i for i in range(len(dates))],
        'inventory': [40 + i * 0.5 for i in range(len(dates))],
        'total_assets': [200 + i * 5 for i in range(len(dates))],
        'accounts_payable': [20 + i * 0.3 for i in range(len(dates))],
        'long_term_debt': [80 - i * 0.5 for i in range(len(dates))],
        'total_liabilities': [120 - i * 0.2 for i in range(len(dates))],
        'equity': [80 + i * 5.2 for i in range(len(dates))]
    }, index=dates)

    # Generate sample cash flow statement
    cash_flow = pd.DataFrame({
        'operating_cash_flow': [r * 0.18 for r in revenue],
        'capital_expenditures': [-r * 0.08 for r in revenue],
        'free_cash_flow': [r * 0.1 for r in revenue],
        'change_in_working_capital': [r * 0.02 for r in revenue]
    }, index=dates)

    return {
        'income_statement': income_statement,
        'balance_sheet': balance_sheet,
        'cash_flow': cash_flow,
        'ticker': ticker,
        'source': source
    }


# Temporarily commented out for auto_opt compatibility
# __all__ = [
#     "infer_drivers", 
#     "ForecastInputs",
#     "build_forecast",
#     "MonteCarloDCF",
#     "load_financials",
# ]
