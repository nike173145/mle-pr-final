import numpy as np
import pandas as pd

from bank_recommender.constants import PRODUCT_COLUMNS
from bank_recommender.metrics import average_precision_at_k, evaluate_ranking


def test_average_precision_at_k_uses_rank_and_multiple_hits() -> None:
    actual = np.array([1, 0, 1, 0])
    predicted = np.array([0, 1, 2])

    assert average_precision_at_k(actual, predicted) == (1.0 + 2 / 3) / 2


def test_ranking_metrics_filter_owned_products() -> None:
    targets = pd.DataFrame(0, index=range(1), columns=PRODUCT_COLUMNS)
    targets.loc[0, PRODUCT_COLUMNS[1]] = 1
    scores = np.zeros((1, len(PRODUCT_COLUMNS)))
    scores[0, 0] = 1.0
    scores[0, 1] = 0.9
    owned = pd.DataFrame(0, index=range(1), columns=PRODUCT_COLUMNS)
    owned.loc[0, PRODUCT_COLUMNS[0]] = 1

    metrics = evaluate_ranking(targets, scores, owned, k=1)

    assert metrics["map_at_1"] == 1.0
    assert metrics["precision_at_1"] == 1.0
    assert metrics["recall_at_1"] == 1.0


def test_all_owned_catalog_does_not_leak_masked_products_into_coverage() -> None:
    targets = pd.DataFrame(0, index=range(1), columns=PRODUCT_COLUMNS)
    scores = np.ones((1, len(PRODUCT_COLUMNS)))
    owned = pd.DataFrame(1, index=range(1), columns=PRODUCT_COLUMNS)

    metrics = evaluate_ranking(targets, scores, owned, k=7)

    assert metrics["catalog_coverage_at_7"] == 0.0
    assert metrics["precision_at_7"] == 0.0
