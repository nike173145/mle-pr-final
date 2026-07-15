"""Ranking metrics for multi-label next-product recommendation."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from bank_recommender.constants import PRODUCT_COLUMNS


def top_k_indices(
    scores: np.ndarray,
    owned: np.ndarray | None = None,
    k: int = 7,
) -> np.ndarray:
    """Return stable top-k indices after optionally masking current products."""
    values = np.asarray(scores, dtype=float).copy()
    if owned is not None:
        values[np.asarray(owned, dtype=bool)] = -np.inf
    result = np.full((len(values), k), -1, dtype=int)
    for row_index, row in enumerate(values):
        ordered = np.argsort(-row, kind="stable")
        available = ordered[np.isfinite(row[ordered])][:k]
        result[row_index, : len(available)] = available
    return result


def average_precision_at_k(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Average precision for one customer, zero when no purchase occurred."""
    positives = set(np.flatnonzero(actual))
    if not positives:
        return 0.0
    score = 0.0
    hits = 0
    for rank, product_index in enumerate(predicted, start=1):
        if int(product_index) in positives:
            hits += 1
            score += hits / rank
    return score / min(len(positives), len(predicted))


def evaluate_ranking(
    targets: pd.DataFrame | np.ndarray,
    scores: np.ndarray,
    owned: pd.DataFrame | np.ndarray | None = None,
    k: int = 7,
) -> dict[str, float]:
    """Calculate MAP, precision, recall, coverage, and per-label PR-AUC."""
    truth = (
        targets[PRODUCT_COLUMNS].to_numpy(dtype=np.int8)
        if isinstance(targets, pd.DataFrame)
        else np.asarray(targets, dtype=np.int8)
    )
    current = None
    if isinstance(owned, pd.DataFrame):
        current = owned[PRODUCT_COLUMNS].to_numpy(dtype=bool)
    elif owned is not None:
        current = np.asarray(owned, dtype=bool)
    predictions = top_k_indices(scores, current, k)

    ap_values = [
        average_precision_at_k(actual, predicted)
        for actual, predicted in zip(truth, predictions, strict=True)
    ]
    valid_predictions = predictions >= 0
    safe_predictions = np.where(valid_predictions, predictions, 0)
    hits = (
        np.take_along_axis(truth, safe_predictions, axis=1) * valid_predictions
    ).sum(axis=1)
    positive_counts = truth.sum(axis=1)
    recalls = np.divide(
        hits,
        positive_counts,
        out=np.zeros_like(hits, dtype=float),
        where=positive_counts > 0,
    )
    pr_auc_values: list[float] = []
    for index in range(truth.shape[1]):
        if np.unique(truth[:, index]).size > 1:
            pr_auc_values.append(
                average_precision_score(truth[:, index], scores[:, index])
            )
    return {
        f"map_at_{k}": float(np.mean(ap_values)),
        f"precision_at_{k}": float(np.mean(hits / k)),
        f"recall_at_{k}": float(np.mean(recalls)),
        f"catalog_coverage_at_{k}": float(
            np.unique(predictions[predictions >= 0]).size / len(PRODUCT_COLUMNS)
        ),
        "customers_with_purchase_rate": float(np.mean(positive_counts > 0)),
        "macro_pr_auc": float(np.mean(pr_auc_values)) if pr_auc_values else 0.0,
    }
