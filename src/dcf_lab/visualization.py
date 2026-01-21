"""Plotting helpers using matplotlib for financial projections."""
from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd


def plot_revenue(is_df: pd.DataFrame) -> None:
    """Line chart of historical and projected revenue."""
    plt.figure()
    plt.plot(is_df["year"], is_df["revenue"], marker="o")
    plt.title("Projected Revenue")
    plt.xlabel("Year")
    plt.ylabel("Revenue")


def plot_margins(is_df: pd.DataFrame) -> None:
    """Plot EBITDA and EBIT margins over the forecast."""
    plt.figure()
    ebit_margin = is_df["ebit"] / is_df["revenue"]
    ebitda_margin = (is_df["ebit"] + is_df["d_and_a"]) / is_df["revenue"]
    plt.plot(is_df["year"], ebit_margin, label="EBIT")
    plt.plot(is_df["year"], ebitda_margin, label="EBITDA")
    plt.title("Margin Trends")
    plt.xlabel("Year")
    plt.ylabel("Margin")
    plt.legend()


def plot_fcff(cf_df: pd.DataFrame) -> None:
    """Bar chart of free cash flow to the firm."""
    plt.figure()
    plt.bar(cf_df["year"], cf_df["fcf"])
    plt.title("FCFF Projection")
    plt.xlabel("Year")
    plt.ylabel("FCFF")


def plot_capital_structure(bs_df: pd.DataFrame) -> None:
    """Pie chart of ending capital structure."""
    last = bs_df.iloc[-1]
    plt.figure()
    plt.pie([last["total_debt"], last["equity"]],
            labels=["Debt", "Equity"], autopct="%1.1f%%")
    plt.title("Capital Structure (Final Year)")


def plot_price_history(price_df: pd.DataFrame) -> None:
    """Plot historical stock price with moving averages."""
    plt.figure()
    plt.plot(price_df.index, price_df["Close"], label="Close")
    for window in (50, 200):
        if len(price_df) >= window:
            plt.plot(
                price_df["Close"].rolling(window).mean(),
                label=f"{window}d MA")
    plt.title("Historical Price")
    plt.legend()
    plt.xlabel("Date")
    plt.ylabel("Price")
