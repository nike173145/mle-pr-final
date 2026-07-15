"""Feature engineering and preprocessing pipeline definitions."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from bank_recommender.constants import (
    CATEGORICAL_PROFILE_COLUMNS,
    DATE_COLUMN,
    MODEL_CATEGORICAL_COLUMNS,
    MODEL_FEATURE_COLUMNS,
    MODEL_NUMERIC_COLUMNS,
    NUMERIC_PROFILE_COLUMNS,
    PRODUCT_COLUMNS,
    RAW_FEATURE_COLUMNS,
)


def engineer_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Create temporal, portfolio, income, and tenure features from raw input."""
    result = frame.copy()
    for column in RAW_FEATURE_COLUMNS:
        if column not in result:
            result[column] = np.nan

    snapshot = pd.to_datetime(result[DATE_COLUMN], errors="coerce")
    result["snapshot_year"] = snapshot.dt.year.astype("float64")
    result["snapshot_month"] = snapshot.dt.month.astype("float64")

    for column in NUMERIC_PROFILE_COLUMNS:
        result[column] = pd.to_numeric(result[column], errors="coerce")
    result["age"] = result["age"].where(result["age"].between(18, 100))
    result["antiguedad"] = result["antiguedad"].where(result["antiguedad"].ge(0))
    result["renta"] = result["renta"].where(result["renta"].ge(0))

    for product in PRODUCT_COLUMNS:
        result[product] = (
            pd.to_numeric(result[product], errors="coerce").fillna(0).clip(0, 1)
        )
    result["product_count"] = result[PRODUCT_COLUMNS].sum(axis=1)
    result["tenure_years"] = result["antiguedad"] / 12.0
    result["log_income"] = np.log1p(result["renta"].clip(lower=0))

    for column in CATEGORICAL_PROFILE_COLUMNS:
        result[column] = result[column].astype("object")
        result.loc[pd.isna(result[column]), column] = np.nan
        result.loc[~pd.isna(result[column]), column] = (
            result.loc[~pd.isna(result[column]), column].astype(str).str.strip()
        )
    return result[MODEL_FEATURE_COLUMNS]


def make_preprocessor() -> ColumnTransformer:
    """Create a preprocessing graph fitted exclusively on training data."""
    numeric = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            ("scaler", StandardScaler()),
        ]
    )
    categorical = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="constant", fill_value="__MISSING__"),
            ),
            (
                "one_hot",
                OneHotEncoder(
                    handle_unknown="infrequent_if_exist",
                    min_frequency=20,
                    sparse_output=True,
                ),
            ),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric, MODEL_NUMERIC_COLUMNS),
            ("categorical", categorical, MODEL_CATEGORICAL_COLUMNS),
        ],
        remainder="drop",
        sparse_threshold=0.3,
    )
