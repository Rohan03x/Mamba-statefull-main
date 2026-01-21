"""
TFT forecasting page for stock price prediction.
"""
from datetime import datetime, timedelta

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

from dcf_lab.features.indicators import add_technical_indicators
from dcf_lab.models.transformer_tft import TFTModel


def load_stock_data(
        symbol: str,
        start_date: str,
        end_date: str) -> pd.DataFrame:
    """Load stock data using yfinance."""
    df = yf.download(symbol, start=start_date, end=end_date)
    df = df.reset_index()
    df.columns = [col.lower().replace(' ', '_') for col in df.columns]
    return df


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare features for TFT model."""
    # Add time features
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    df['dayofweek'] = df['date'].dt.dayofweek

    # Add technical indicators
    df = add_technical_indicators(df)

    # Add time index
    df['time_idx'] = range(len(df))

    return df


def plot_predictions(
        actual: pd.Series,
        predictions: dict,
        dates: pd.Series) -> None:
    """Plot actual values and predictions with confidence intervals."""
    fig = go.Figure()

    # Plot actual values
    fig.add_trace(go.Scatter(
        x=dates,
        y=actual,
        name='Actual',
        line=dict(color='blue')
    ))

    # Plot median prediction
    fig.add_trace(go.Scatter(
        x=dates,
        y=predictions['q50'],
        name='Median Prediction',
        line=dict(color='red')
    ))

    # Add confidence interval
    fig.add_trace(go.Scatter(
        x=dates.tolist() + dates.tolist()[::-1],
        y=predictions['q90'].tolist() + predictions['q10'].tolist()[::-1],
        fill='tosel',
        fillcolor='rgba(255,0,0,0.2)',
        line=dict(color='rgba(255,0,0,0)'),
        name='80% Confidence Interval'
    ))

    fig.update_layout(
        title='Stock Price Forecasting with TFT Model',
        xaxis_title='Date',
        yaxis_title='Price',
        hovermode='x unified',
        showlegend=True
    )

    st.plotly_chart(fig, use_container_width=True)


def app():
    st.title("Stock Price Forecasting with TFT")
    st.write(
        "This page uses a Temporal Fusion Transformer (TFT) model for stock price prediction.")

    # Input parameters
    col1, col2 = st.columns(2)
    with col1:
        symbol = st.text_input("Stock Symbol", value="AAPL")
        lookback_years = st.number_input(
            "Training Data (years)", min_value=1, max_value=10, value=5)

    with col2:
        forecast_days = st.number_input(
            "Forecast Horizon (days)",
            min_value=1,
            max_value=90,
            value=30)
        training_ratio = st.slider(
            "Training/Validation Split",
            min_value=0.5,
            max_value=0.9,
            value=0.8)

    if st.button("Generate Forecast"):
        with st.spinner("Loading data..."):
            # Load data
            end_date = datetime.now()
            start_date = end_date - timedelta(days=365 * lookback_years)
            df = load_stock_data(symbol, start_date.strftime(
                '%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))

            if df.empty:
                st.error(f"No data found for symbol {symbol}")
                return

            # Prepare features
            df = prepare_features(df)
            target = 'adj_close'

            # Define feature groups
            static_categoricals = []
            static_reals = []
            time_varying_known_categoricals = ['month', 'dayofweek']
            time_varying_known_reals = ['year']
            time_varying_unknown_categoricals = []
            time_varying_unknown_reals = [
                'open', 'high', 'low', 'close', 'adj_close', 'volume',
                'rsi', 'macd', 'macd_signal', 'macd_hist',
                'bb_upper', 'bb_middle', 'bb_lower'
            ]

            # Create and train model
            model = TFTModel(
                hidden_size=32,
                attention_head_size=4,
                dropout=0.1,
                hidden_continuous_size=16,
                learning_rate=1e-3,
                batch_size=32,
                max_epochs=50,
                early_stopping_patience=10
            )

            with st.spinner("Training model..."):
                try:
                    model.fit(
                        df=df,
                        target=target,
                        time_idx='time_idx',
                        static_categoricals=static_categoricals,
                        static_reals=static_reals,
                        time_varying_known_categoricals=time_varying_known_categoricals,
                        time_varying_known_reals=time_varying_known_reals,
                        time_varying_unknown_categoricals=time_varying_unknown_categoricals,
                        time_varying_unknown_reals=time_varying_unknown_reals,
                        group_ids=['symbol'],
                        max_encoder_length=30,
                        max_prediction_length=forecast_days,
                        train_val_split=1-training_ratio)

                    # Generate predictions
                    predictions = model.predict(df.tail(30 + forecast_days))

                    # Plot results
                    actual = df[target].tail(30)
                    dates = df['date'].tail(30)
                    plot_predictions(actual, predictions, dates)

                    # Display metrics
                    last_price = df[target].iloc[-1]
                    predicted_price = predictions['q50'][-1]
                    change_pct = (
                        predicted_price - last_price) / last_price * 100

                    st.metric(
                        "Forecasted Price Change",
                        f"{change_pct:.2f}%",
                        delta=f"${predicted_price - last_price:.2f}"
                    )

                except Exception as e:
                    st.error(
                        f"Error during model training/prediction: {str(e)}")
                    raise e


if __name__ == "__main__":
    app()
