"""Model loading, feature-frame construction, and output normalization."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from .schemas import PRODUCT_NAMES, PredictionRequest, Recommendation


class ModelLoadError(RuntimeError):
    """Raised when a serialized artifact cannot serve recommendations."""


class PredictionError(RuntimeError):
    """Raised when model inference fails or returns an invalid payload."""


def requests_to_frame(requests: Sequence[PredictionRequest]) -> pd.DataFrame:
    """Convert validated requests into the feature frame expected by the model."""

    rows: list[dict[str, Any]] = []
    for request in requests:
        row = request.model_dump(
            mode="json",
            exclude={"current_products", "top_k"},
        )
        owned = set(request.current_products)
        row.update({product: int(product in owned) for product in PRODUCT_NAMES})
        rows.append(row)

    return pd.DataFrame(rows)


class ModelPredictor:
    """Adapter for joblib artifacts exposing `recommend` or `predict_scores`."""

    def __init__(self, artifact: Any, model_path: str | Path = "<in-memory>"):
        self._artifact = artifact
        self.model_path = str(model_path)
        self.model_version = self._validate_version(artifact)
        self.product_names = self._validate_product_names(artifact)

    @classmethod
    def load(cls, model_path: str | Path) -> ModelPredictor:
        """Load and validate one joblib model artifact."""

        path = Path(model_path).expanduser()
        if not path.is_file():
            raise ModelLoadError(f"Model artifact does not exist: {path}")
        try:
            artifact = joblib.load(path)
        except Exception as error:
            raise ModelLoadError(
                f"Could not load model artifact {path}: {error}"
            ) from error
        try:
            return cls(artifact=artifact, model_path=path)
        except ModelLoadError:
            raise
        except Exception as error:
            raise ModelLoadError(
                f"Model artifact {path} has an invalid contract: {error}"
            ) from error

    @staticmethod
    def _validate_version(artifact: Any) -> str:
        version = getattr(artifact, "model_version", None)
        if version is None or not str(version).strip():
            raise ModelLoadError("Model artifact must define a non-empty model_version")
        return str(version)

    @staticmethod
    def _validate_product_names(artifact: Any) -> tuple[str, ...]:
        names = getattr(artifact, "product_names", None)
        if names is None:
            raise ModelLoadError("Model artifact must define product_names")
        try:
            normalized = tuple(str(name) for name in names)
        except TypeError as error:
            raise ModelLoadError(
                "product_names must be an iterable of strings"
            ) from error

        if len(normalized) != len(set(normalized)):
            raise ModelLoadError("product_names must not contain duplicates")
        if set(normalized) != set(PRODUCT_NAMES):
            missing = sorted(set(PRODUCT_NAMES) - set(normalized))
            unexpected = sorted(set(normalized) - set(PRODUCT_NAMES))
            raise ModelLoadError(
                "product_names must contain the 24 Santander products; "
                f"missing={missing}, unexpected={unexpected}"
            )
        if not callable(getattr(artifact, "recommend", None)) and not callable(
            getattr(artifact, "predict_scores", None)
        ):
            raise ModelLoadError(
                "Model artifact must implement recommend(frame, k) or predict_scores(frame)"
            )
        return normalized

    def recommend_many(
        self, requests: Sequence[PredictionRequest]
    ) -> list[list[Recommendation]]:
        """Return normalized, ownership-safe recommendations for every request."""

        if not requests:
            return []

        frame = requests_to_frame(requests)
        try:
            if callable(getattr(self._artifact, "recommend", None)):
                raw_batches = self._recommend_with_artifact(frame)
            else:
                raw_batches = self._recommend_from_scores(frame)
        except PredictionError:
            raise
        except Exception as error:
            raise PredictionError(f"Model inference failed: {error}") from error

        if len(raw_batches) != len(requests):
            raise PredictionError(
                "Model returned an unexpected number of recommendation rows: "
                f"expected {len(requests)}, got {len(raw_batches)}"
            )

        return [
            self._normalize_row(raw, request.current_products, request.top_k)
            for raw, request in zip(raw_batches, requests, strict=True)
        ]

    def _recommend_with_artifact(
        self, frame: pd.DataFrame
    ) -> list[list[Mapping[str, Any]]]:
        # Ask for the full catalog so the service-side ownership guard can still
        # fill top_k if an artifact accidentally includes an owned product.
        raw = self._artifact.recommend(frame, k=len(self.product_names))
        if not isinstance(raw, list | tuple):
            raise PredictionError("recommend() must return a list of rows")

        raw_list = list(raw)
        if len(frame) == 1 and (not raw_list or isinstance(raw_list[0], Mapping)):
            raw_list = [raw_list]

        batches: list[list[Mapping[str, Any]]] = []
        for row in raw_list:
            if not isinstance(row, list | tuple):
                raise PredictionError(
                    "Each recommend() output row must be a list of dictionaries"
                )
            batches.append(list(row))
        return batches

    def _recommend_from_scores(self, frame: pd.DataFrame) -> list[list[dict[str, Any]]]:
        scores = self._artifact.predict_scores(frame)
        if isinstance(scores, pd.DataFrame):
            if set(self.product_names).issubset(scores.columns):
                matrix = scores.loc[:, self.product_names].to_numpy()
            else:
                matrix = scores.to_numpy()
        else:
            matrix = np.asarray(scores)

        if matrix.ndim == 1 and len(frame) == 1:
            matrix = matrix.reshape(1, -1)
        expected_shape = (len(frame), len(self.product_names))
        if matrix.shape != expected_shape:
            raise PredictionError(
                f"predict_scores() returned shape {matrix.shape}; expected {expected_shape}"
            )

        batches: list[list[dict[str, Any]]] = []
        for score_row in matrix:
            ranked_indices = sorted(
                range(len(self.product_names)),
                key=lambda index: float(score_row[index]),
                reverse=True,
            )
            batches.append(
                [
                    {
                        "product": self.product_names[index],
                        "score": float(score_row[index]),
                    }
                    for index in ranked_indices
                ]
            )
        return batches

    def _normalize_row(
        self,
        raw_recommendations: Sequence[Mapping[str, Any]],
        current_products: Sequence[str],
        top_k: int,
    ) -> list[Recommendation]:
        owned = set(current_products)
        seen: set[str] = set()
        normalized: list[Recommendation] = []

        for entry in raw_recommendations:
            if not isinstance(entry, Mapping):
                raise PredictionError("Each recommendation must be a dictionary")

            product_value = next(
                (
                    entry[key]
                    for key in ("product", "product_name", "name", "item")
                    if key in entry
                ),
                None,
            )
            if product_value is None:
                raise PredictionError("A recommendation is missing its product name")
            product = str(product_value)
            if product not in self.product_names:
                raise PredictionError(f"Model returned an unknown product: {product}")
            if product in owned or product in seen:
                continue

            score_value = next(
                (
                    entry[key]
                    for key in ("score", "probability", "value")
                    if key in entry
                ),
                0.0,
            )
            try:
                score = float(score_value)
            except (TypeError, ValueError) as error:
                raise PredictionError(
                    f"Model returned a non-numeric score for {product}"
                ) from error
            if not math.isfinite(score):
                raise PredictionError(
                    f"Model returned a non-finite score for {product}"
                )

            seen.add(product)
            normalized.append(
                Recommendation(
                    product=product,
                    score=score,
                    rank=len(normalized) + 1,
                )
            )
            if len(normalized) == top_k:
                break

        # This assertion is intentionally independent from model-side masking.
        if any(rec.product in owned for rec in normalized):
            raise PredictionError("Owned products passed the service-side safety check")
        return normalized
