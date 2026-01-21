"""
Data hygiene utilities for AI price forecasting
"""

from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def drop_constant_features(
        df: pd.DataFrame,
        feature_columns: List[str],
        threshold: float = 0.01) -> List[str]:
    """
    Drop features that are constant or nearly constant

    Args:
        df: DataFrame with features
        feature_columns: List of feature column names
        threshold: Variance threshold below which features are considered constant

    Returns:
        Filtered list of feature columns
    """
    filtered_columns = []

    for col in feature_columns:
        if col in df.columns:
            # Check if feature has sufficient variance
            feature_std = df[col].std()
            feature_nunique = df[col].nunique()

            # Keep feature if it has variance and multiple unique values
            if feature_std > threshold and feature_nunique > 1:
                filtered_columns.append(col)
            else:
                print(f"Dropping constant/low-variance feature: {col} (std={feature_std:.6f}, unique={feature_nunique})")

    return filtered_columns


def align_news_to_market_sessions(
    df: pd.DataFrame,
    sentiment_columns: List[str]
) -> pd.DataFrame:
    """
    Align news sentiment to market sessions with proper forward/backward fill

    Args:
        df: DataFrame with sentiment features
        sentiment_columns: List of sentiment column names

    Returns:
        DataFrame with properly aligned sentiment features
    """
    df_aligned = df.copy()

    for col in sentiment_columns:
        if col in df_aligned.columns:
            # Forward fill first, then backward fill, then fallback to 0.0
            df_aligned[col] = df_aligned[col].ffill().bfill().fillna(0.0)

            # Ensure no infinite values
            df_aligned[col] = df_aligned[col].replace([np.inf, -np.inf], 0.0)

    return df_aligned


def create_robust_preprocessing_pipeline() -> object:
    """
    Create a robust preprocessing pipeline with proper imputation and scaling

    Returns:
        Preprocessing pipeline
    """
    return make_pipeline(
        SimpleImputer(
            strategy="median",
            add_indicator=False),
        # Impute missing values first
        StandardScaler(with_mean=True, with_std=True),  # Then scale features
        memory=None  # No caching needed
    )


def validate_feature_quality(
    df: pd.DataFrame,
    feature_columns: List[str]
) -> Tuple[List[str], List[str]]:
    """
    Validate feature quality and identify problematic features

    Args:
        df: DataFrame with features
        feature_columns: List of feature column names

    Returns:
        Tuple of (good features, problematic features)
    """
    good_features = []
    problematic_features = []

    for col in feature_columns:
        if col not in df.columns:
            problematic_features.append(f"{col} (missing)")
            continue

        feature_data = df[col]

        # Check for various issues
        nan_ratio = feature_data.isna().sum() / len(feature_data)
        inf_count = np.isinf(feature_data).sum()
        zero_ratio = (feature_data == 0).sum() / len(feature_data)
        std_value = feature_data.std()

        issues = []
        if nan_ratio > 0.5:
            issues.append(f"high_nan_ratio={nan_ratio:.2f}")
        if inf_count > 0:
            issues.append(f"infinite_values={inf_count}")
        if zero_ratio > 0.9:
            issues.append(f"mostly_zeros={zero_ratio:.2f}")
        if std_value < 1e-6:
            issues.append(f"low_variance={std_value:.6f}")

        if issues:
            problematic_features.append(f"{col} ({', '.join(issues)})")
        else:
            good_features.append(col)

    return good_features, problematic_features


def clean_and_prepare_features(
    df: pd.DataFrame,
    feature_columns: List[str],
    min_features: int = 3
) -> Tuple[pd.DataFrame, List[str], bool]:
    """
    Clean and prepare features with comprehensive data hygiene

    Args:
        df: DataFrame with features
        feature_columns: List of feature column names
        min_features: Minimum number of features required

    Returns:
        Tuple of (cleaned dataframe, final feature columns, success flag)
    """
    print("Starting feature cleaning and preparation...")

    # Step 1: Validate feature quality
    good_features, problematic_features = validate_feature_quality(
        df, feature_columns)

    if problematic_features:
        print("Problematic features identified:")
        for feat in problematic_features:
            print(f"  - {feat}")

    # Step 2: Drop constant/low-variance features
    good_features = drop_constant_features(df, good_features)

    # Step 3: Align sentiment features if present
    sentiment_cols = [col for col in good_features if 'sent' in col.lower()]
    if sentiment_cols:
        print(f"Aligning {len(sentiment_cols)} sentiment features to market sessions")
        df = align_news_to_market_sessions(df, sentiment_cols)

    # Step 4: Check if we have enough features
    if len(good_features) < min_features:
        print(f"Not enough good features: {len(good_features)} < {min_features}")
        return df, good_features, False

    print(f"Feature cleaning complete. Using {len(good_features)} features:")
    for feat in good_features:
        print(f"  - {feat}")

    return df, good_features, True
