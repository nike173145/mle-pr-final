"""Serializable multi-label ranking model and popularity baseline."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.linear_model import SGDClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

from bank_recommender.constants import PRODUCT_COLUMNS, RANDOM_SEED
from bank_recommender.features import engineer_features, make_preprocessor


def _owned_matrix(frame: pd.DataFrame) -> np.ndarray:
    owned = frame.reindex(columns=PRODUCT_COLUMNS, fill_value=0)
    return (
        owned.apply(pd.to_numeric, errors="coerce")
        .fillna(0)
        .clip(0, 1)
        .to_numpy(dtype=bool)
    )


def ranked_recommendations(
    scores: np.ndarray,
    frame: pd.DataFrame,
    k: int,
) -> list[list[dict[str, Any]]]:
    """Rank scores deterministically while filtering products already owned."""
    if not 1 <= k <= len(PRODUCT_COLUMNS):
        raise ValueError(f"k must be between 1 and {len(PRODUCT_COLUMNS)}")
    values = np.asarray(scores, dtype=float).copy()
    if values.ndim == 1:
        values = values.reshape(1, -1)
    if values.shape != (len(frame), len(PRODUCT_COLUMNS)):
        raise ValueError("Score matrix shape does not match input and product catalog")
    values[~np.isfinite(values)] = 0.0
    values[_owned_matrix(frame)] = -np.inf

    result: list[list[dict[str, Any]]] = []
    for row in values:
        ordered = np.argsort(-row, kind="stable")
        recommendations: list[dict[str, Any]] = []
        for index in ordered:
            if not np.isfinite(row[index]):
                continue
            recommendations.append(
                {
                    "product": PRODUCT_COLUMNS[index],
                    "score": round(float(row[index]), 8),
                    "rank": len(recommendations) + 1,
                }
            )
            if len(recommendations) == k:
                break
        result.append(recommendations)
    return result


class PopularityRecommender(BaseEstimator):
    """Global acquisition-frequency baseline."""

    product_names = PRODUCT_COLUMNS
    model_version = "popularity-v1"

    def fit(self, features: pd.DataFrame, targets: pd.DataFrame):
        del features
        values = targets[PRODUCT_COLUMNS].to_numpy(dtype=float)
        self.label_prior_ = (values.sum(axis=0) + 1.0) / (len(values) + 2.0)
        return self

    def predict_scores(self, features: pd.DataFrame) -> np.ndarray:
        if not hasattr(self, "label_prior_"):
            raise RuntimeError("The baseline is not fitted")
        return np.tile(self.label_prior_, (len(features), 1))

    def recommend(
        self, features: pd.DataFrame, k: int = 7
    ) -> list[list[dict[str, Any]]]:
        return ranked_recommendations(self.predict_scores(features), features, k)


class BankProductRecommender(BaseEstimator):
    """One-vs-rest linear ranking model with a popularity prior."""

    product_names = PRODUCT_COLUMNS

    def __init__(
        self,
        *,
        alpha: float = 0.0001,
        max_iter: int = 60,
        prior_weight: float = 0.25,
        prior_correction: bool = False,
        random_seed: int = RANDOM_SEED,
        model_version: str = "sgd-ovr-v1",
    ) -> None:
        self.alpha = alpha
        self.max_iter = max_iter
        self.prior_weight = prior_weight
        self.prior_correction = prior_correction
        self.random_seed = random_seed
        self.model_version = model_version

    def _make_pipeline(self) -> Pipeline:
        classifier = OneVsRestClassifier(
            SGDClassifier(
                loss="log_loss",
                penalty="l2",
                alpha=self.alpha,
                max_iter=self.max_iter,
                tol=1e-3,
                class_weight=None,
                random_state=self.random_seed,
                average=True,
            ),
            n_jobs=1,
        )
        return Pipeline(
            steps=[
                (
                    "feature_engineering",
                    FunctionTransformer(engineer_features, validate=False),
                ),
                ("preprocessing", make_preprocessor()),
                ("classifier", classifier),
            ]
        )

    def fit(self, features: pd.DataFrame, targets: pd.DataFrame):
        target_values = targets[PRODUCT_COLUMNS].to_numpy(dtype=np.int8)
        self.label_prior_ = (target_values.sum(axis=0) + 1.0) / (
            len(target_values) + 2.0
        )
        self.pipeline_ = self._make_pipeline()
        self.pipeline_.fit(features, target_values)
        return self

    def predict_scores(self, features: pd.DataFrame) -> np.ndarray:
        if not hasattr(self, "pipeline_"):
            raise RuntimeError("The recommender is not fitted")
        probabilities = np.asarray(self.pipeline_.predict_proba(features), dtype=float)
        probabilities = probabilities.reshape(len(features), len(PRODUCT_COLUMNS))
        prior = np.tile(self.label_prior_, (len(features), 1))
        if self.prior_correction:
            # Optional correction for artifacts trained with an effective 0.5
            # label prior. Current project artifacts use unweighted log loss,
            # so this compatibility branch is disabled by default.
            clipped = np.clip(probabilities, 1e-7, 1.0 - 1e-7)
            numerator = clipped * prior
            denominator = numerator + (1.0 - clipped) * (1.0 - prior)
            probabilities = np.divide(
                numerator,
                denominator,
                out=np.zeros_like(numerator),
                where=denominator > 0,
            )
        weight = float(np.clip(self.prior_weight, 0.0, 1.0))
        return (1.0 - weight) * probabilities + weight * prior

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        """Scikit-compatible prediction method used by MLflow signatures."""
        return self.predict_scores(features)

    def recommend(
        self, features: pd.DataFrame, k: int = 7
    ) -> list[list[dict[str, Any]]]:
        return ranked_recommendations(self.predict_scores(features), features, k)
