from __future__ import annotations

import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning

from bank_recommender.constants import (
    CATEGORICAL_PROFILE_COLUMNS,
    DATE_COLUMN,
    NUMERIC_PROFILE_COLUMNS,
    PRODUCT_COLUMNS,
)
from bank_recommender.model import BankProductRecommender, ranked_recommendations


def synthetic_features(rows: int = 72) -> pd.DataFrame:
    frame = pd.DataFrame({DATE_COLUMN: ["2016-03-28"] * rows})
    for column in NUMERIC_PROFILE_COLUMNS:
        frame[column] = np.arange(rows) % 17
    frame["age"] = 20 + np.arange(rows) % 60
    frame["renta"] = 20_000 + np.arange(rows) * 100
    for column in CATEGORICAL_PROFILE_COLUMNS:
        frame[column] = np.where(np.arange(rows) % 2, "A", "B")
    for index, product in enumerate(PRODUCT_COLUMNS):
        frame[product] = (np.arange(rows) % (index + 3) == 0).astype(int)
    return frame


def synthetic_targets(rows: int = 72) -> pd.DataFrame:
    targets = pd.DataFrame(0, index=range(rows), columns=PRODUCT_COLUMNS)
    for index, product in enumerate(PRODUCT_COLUMNS):
        targets.loc[index % rows, product] = 1
        targets.loc[(index + 24) % rows, product] = 1
    return targets


def test_ranked_recommendations_are_unique_and_exclude_owned() -> None:
    frame = synthetic_features(1)
    frame.loc[:, PRODUCT_COLUMNS] = 0
    scores = np.arange(len(PRODUCT_COLUMNS), dtype=float).reshape(1, -1)
    owned = PRODUCT_COLUMNS[-1]
    frame.loc[0, owned] = 1

    recommendations = ranked_recommendations(scores, frame, k=7)[0]

    names = [item["product"] for item in recommendations]
    assert len(names) == len(set(names)) == 7
    assert owned not in names
    assert [item["rank"] for item in recommendations] == list(range(1, 8))


def test_model_serialization_round_trip(tmp_path) -> None:
    features = synthetic_features()
    targets = synthetic_targets()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=ConvergenceWarning)
        model = BankProductRecommender(max_iter=10).fit(features, targets)
    before = model.predict_scores(features.iloc[:2])
    path = tmp_path / "model.joblib"

    joblib.dump(model, path)
    restored = joblib.load(path)
    after = restored.predict_scores(features.iloc[:2])

    np.testing.assert_allclose(before, after)
    assert restored.recommend(features.iloc[:2], k=3) == model.recommend(
        features.iloc[:2], k=3
    )
