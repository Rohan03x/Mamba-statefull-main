"""
Relative Valuation ML Model

This module implements machine learning models for predicting appropriate multiples
based on fundamental factors and market conditions. It helps calibrate exit multiples
and comparative valuation to company-specific characteristics.
"""

from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Constants
MODEL_NOT_FITTED = "Model is not fitted yet"


class MultiplePredictorModel:
    """
    Model that predicts fair multiples based on company fundamentals

    This class implements various machine learning models to predict
    appropriate valuation multiples (EV/EBITDA, EV/Sales, P/E, etc.)
    based on company characteristics like growth rates, margins,
    capital efficiency, and macroeconomic factors.
    """

    def __init__(
        self,
        multiple_type: str = 'ev_to_ebitda',
        model_type: str = 'rf',
        params: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize a multiple predictor model

        Args:
            multiple_type: Type of multiple to predict ('ev_to_ebitda', 'ev_to_sales', 'pe', etc.)
            model_type: Type of model to use ('r', 'gbm', 'elasticnet')
            params: Parameters for the model
        """
        self.multiple_type = multiple_type
        self.model_type = model_type
        self.params = params or {}

        # Initialize the model
        if model_type == 'r':
            self.model = RandomForestRegressor(
                n_estimators=self.params.get('n_estimators', 100),
                max_depth=self.params.get('max_depth', 10),
                min_samples_leaf=self.params.get('min_samples_lea', 5),
                random_state=42
            )
        elif model_type == 'gbm':
            self.model = GradientBoostingRegressor(
                n_estimators=self.params.get('n_estimators', 100),
                learning_rate=self.params.get('learning_rate', 0.1),
                max_depth=self.params.get('max_depth', 3),
                random_state=42
            )
        elif model_type == 'elasticnet':
            self.model = ElasticNet(
                alpha=self.params.get('alpha', 1.0),
                l1_ratio=self.params.get('l1_ratio', 0.5),
                random_state=42
            )
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

        # Create a pipeline with scaling
        self.pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('model', self.model)
        ], memory=None)

        self.feature_importance_ = None
        self.is_fitted = False

    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'MultiplePredictorModel':
        """
        Fit the model to data

        Args:
            X: Features dataframe
            y: Target multiple values

        Returns:
            self
        """
        self.pipeline.fit(X, y)

        # Store feature importance if available
        if hasattr(self.model, 'feature_importances_'):
            self.feature_importance_ = pd.Series(
                self.model.feature_importances_,
                index=X.columns
            ).sort_values(ascending=False)

        self.is_fitted = True
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict multiples for given features

        Args:
            X: Features dataframe

        Returns:
            Predicted multiples
        """
        if not self.is_fitted:
            raise ValueError(MODEL_NOT_FITTED)

        return self.pipeline.predict(X)

    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """
        Evaluate model performance

        Args:
            X: Features dataframe
            y: True multiple values

        Returns:
            Dictionary of evaluation metrics
        """
        if not self.is_fitted:
            raise ValueError(MODEL_NOT_FITTED)

        y_pred = self.predict(X)

        return {
            'mse': mean_squared_error(y, y_pred),
            'rmse': np.sqrt(mean_squared_error(y, y_pred)),
            'r2': r2_score(y, y_pred),
            'mean_error': np.mean(y - y_pred),
            'median_error': np.median(y - y_pred)
        }

    def plot_feature_importance(self, top_n: int = 10) -> None:
        """
        Plot feature importance

        Args:
            top_n: Number of top features to plot
        """
        if not self.is_fitted:
            raise ValueError(MODEL_NOT_FITTED)

        if self.feature_importance_ is None:
            raise ValueError(
                "Feature importance is not available for this model")

        plt.figure(figsize=(10, 6))
        self.feature_importance_[:top_n].plot(kind='barh')
        plt.title(f'Top {top_n} features for {self.multiple_type} prediction')
        plt.tight_layout()
        plt.show()


class RelativeValuationModel:
    """
    End-to-end relative valuation model that predicts fair value

    This class combines multiple predictors for different multiples
    and provides a comprehensive valuation based on peer comparison
    and cross-sectional machine learning.
    """

    def __init__(self):
        """Initialize the relative valuation model"""
        self.multiple_models = {}
        self.multiple_weights = {}
        self.is_fitted = False

    def add_multiple_model(
        self,
        multiple_type: str,
        model: Optional[MultiplePredictorModel] = None,
        weight: float = 1.0
    ) -> None:
        """
        Add a multiple prediction model

        Args:
            multiple_type: Type of multiple ('ev_to_ebitda', 'ev_to_sales', etc.)
            model: Pre-configured MultiplePredictorModel (or None to create default)
            weight: Weight for this multiple in the ensemble
        """
        if model is None:
            model = MultiplePredictorModel(multiple_type=multiple_type)

        self.multiple_models[multiple_type] = model
        self.multiple_weights[multiple_type] = weight

    def fit(
        self,
        X: Dict[str, pd.DataFrame],
        y: Dict[str, pd.Series]
    ) -> 'RelativeValuationModel':
        """
        Fit all multiple models

        Args:
            X: Dictionary of features dataframes for each multiple type
            y: Dictionary of target multiple values for each multiple type

        Returns:
            self
        """
        for multiple_type, model in self.multiple_models.items():
            if multiple_type in X and multiple_type in y:
                model.fit(X[multiple_type], y[multiple_type])
            else:
                raise ValueError(
                    f"Missing data for multiple type: {multiple_type}")

        self.is_fitted = True
        return self

    def predict(
        self,
        X: Dict[str, pd.DataFrame],
        metrics: Dict[str, float]
    ) -> Dict[str, Any]:
        """
        Predict fair value based on multiple models

        Args:
            X: Dictionary of features dataframes for each multiple type
            metrics: Dictionary of financial metrics to apply multiples to
                (e.g., {'ev_to_ebitda': 1000, 'ev_to_sales': 5000})

        Returns:
            Dictionary with predicted multiples and values
        """
        if not self.is_fitted:
            raise ValueError(MODEL_NOT_FITTED)

        results = {}
        total_weight = sum(self.multiple_weights.values())
        weighted_ev = 0

        for multiple_type, model in self.multiple_models.items():
            if multiple_type in X and multiple_type in metrics:
                # Predict multiple
                multiple = float(model.predict(X[multiple_type])[0])

                # Calculate implied EV
                metric_value = metrics[multiple_type]
                implied_ev = multiple * metric_value

                # Store results
                weight = self.multiple_weights[multiple_type] / total_weight
                weighted_ev += implied_ev * weight

                results[multiple_type] = {
                    'predicted_multiple': multiple,
                    'metric_value': metric_value,
                    'implied_ev': implied_ev,
                    'weight': weight
                }

        # Add weighted enterprise value
        results['weighted_enterprise_value'] = weighted_ev

        return results

    def predict_equity_value(
        self,
        X: Dict[str, pd.DataFrame],
        metrics: Dict[str, float],
        net_debt: float,
        shares_outstanding: Optional[float] = None,
        minority_interest: float = 0.0,
        preferred_equity: float = 0.0
    ) -> Dict[str, Any]:
        """
        Predict equity value by adjusting enterprise value for net debt and other items

        Args:
            X: Dictionary of features dataframes for each multiple type
            metrics: Dictionary of financial metrics to apply multiples to
            net_debt: Net debt (total debt - cash)
            shares_outstanding: Number of shares outstanding
            minority_interest: Minority interest value
            preferred_equity: Preferred equity value

        Returns:
            Dictionary with valuation results including equity value
        """
        results = self.predict(X, metrics)

        # Calculate equity value
        ev = results['weighted_enterprise_value']
        equity_value = ev - net_debt - minority_interest - preferred_equity
        results['equity_value'] = equity_value

        # Calculate per-share value if shares outstanding provided
        if shares_outstanding and shares_outstanding > 0:
            results['per_share_value'] = equity_value / shares_outstanding

        return results


def prepare_multiple_features(
    company_data: Dict[str, Any],
    # peer_group: Optional[pd.DataFrame] = None,  # Removing unused parameter
    include_macro: bool = True
) -> Dict[str, pd.DataFrame]:
    """
    Prepare features for multiple prediction models

    Args:
        company_data: Dictionary with company metrics and data
        peer_group: Optional dataframe with peer group data for relative comparison
        include_macro: Whether to include macroeconomic features

    Returns:
        Dictionary of features dataframes for each multiple type
    """
    # Start with basic features common to most multiple types
    base_features = {
        'rev_growth_1yr': company_data.get('rev_growth_1yr', 0),
        'rev_growth_3yr': company_data.get('rev_growth_3yr', 0),
        'rev_growth_5yr': company_data.get('rev_growth_5yr', 0),
        'ebit_margin': company_data.get('ebit_margin', 0),
        'ebitda_margin': company_data.get('ebitda_margin', 0),
        'gross_margin': company_data.get('gross_margin', 0),
        'net_margin': company_data.get('net_margin', 0),
        'roe': company_data.get('roe', 0),
        'roic': company_data.get('roic', 0),
        'debt_to_ebitda': company_data.get('debt_to_ebitda', 0),
        'capex_to_revenue': company_data.get('capex_to_revenue', 0),
        'fcf_to_revenue': company_data.get('fcf_to_revenue', 0),
        'dividend_yield': company_data.get('dividend_yield', 0)
    }

    # Add macroeconomic features if requested
    if include_macro:
        base_features.update({'risk_free_rate': company_data.get(
            'risk_free_rate', 0.035),
            'market_risk_premium': company_data.get(
            'market_risk_premium', 0.055),
            'vix': company_data.get('vix', 15)})

    # Create specific feature sets for each multiple type
    features = {}

    # EV/EBITDA features
    ev_to_ebitda_features = base_features.copy()
    # Add any specific features for this multiple
    features['ev_to_ebitda'] = pd.DataFrame([ev_to_ebitda_features])

    # EV/Sales features
    ev_to_sales_features = base_features.copy()
    features['ev_to_sales'] = pd.DataFrame([ev_to_sales_features])

    # P/E features
    pe_features = base_features.copy()
    features['pe'] = pd.DataFrame([pe_features])

    # EV/EBIT features
    ev_to_ebit_features = base_features.copy()
    features['ev_to_ebit'] = pd.DataFrame([ev_to_ebit_features])

    return features


def train_multiple_models(
    peer_data: pd.DataFrame,
    target_multiples: List[str] = ['ev_to_ebitda', 'ev_to_sales', 'pe']
    # test_size: float = 0.2  # Removing unused parameter
) -> RelativeValuationModel:
    """
    Train multiple prediction models using peer group data

    Args:
        peer_data: DataFrame with peer company data
        target_multiples: List of multiple types to train models for
        test_size: Proportion of data to use for testing

    Returns:
        Fitted RelativeValuationModel
    """
    # Initialize the relative valuation model
    model = RelativeValuationModel()

    # Prepare data for each multiple type
    features_dict = {}
    y_dict = {}

    for multiple in target_multiples:
        # Get features for this multiple type
        feature_cols = [col for col in peer_data.columns if col !=
                        multiple and not pd.isnull(peer_data[col]).all()]
        X = peer_data[feature_cols]

        # Get target values
        y = peer_data[multiple].copy()

        # Store in dictionaries
        features_dict[multiple] = X
        y_dict[multiple] = y

        # Create and add model with appropriate weight
        multiple_model = MultiplePredictorModel(multiple_type=multiple)
        weight = 1.0  # Default weight

        # Adjust weights based on multiple type
        if multiple == 'ev_to_ebitda':
            weight = 2.0
        elif multiple == 'ev_to_ebit':
            weight = 1.5

        model.add_multiple_model(multiple, multiple_model, weight)

    # Fit all models
    model.fit(features_dict, y_dict)

    return model
