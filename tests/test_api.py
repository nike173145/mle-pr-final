"""Isolated API and predictor tests that do not require a model artifact."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from services.recomendation_service.app.main import create_app
from services.recomendation_service.app.predictor import ModelLoadError, ModelPredictor
from services.recomendation_service.app.schemas import PRODUCT_NAMES, PredictionRequest


class FakeRecommendationArtifact:
    """Deterministic artifact that deliberately returns owned products first."""

    model_version = "test-model-v1"
    product_names = PRODUCT_NAMES

    def __init__(self) -> None:
        self.last_frame: pd.DataFrame | None = None
        self.last_k: int | None = None

    def recommend(self, frame: pd.DataFrame, k: int) -> list[list[dict[str, Any]]]:
        self.last_frame = frame.copy()
        self.last_k = k
        rows: list[list[dict[str, Any]]] = []
        for _, feature_row in frame.iterrows():
            owned = [name for name in PRODUCT_NAMES if feature_row[name] == 1]
            ordered = owned + [name for name in PRODUCT_NAMES if name not in owned]
            rows.append(
                [
                    {"product": name, "score": 1.0 - index / 100.0}
                    for index, name in enumerate(ordered[:k])
                ]
            )
        return rows


class FakeScoreArtifact:
    """Minimal artifact exercising the predict_scores compatibility path."""

    model_version = "score-model-v1"
    product_names = PRODUCT_NAMES

    def predict_scores(self, frame: pd.DataFrame) -> np.ndarray:
        base = np.arange(len(PRODUCT_NAMES), dtype=float)
        return np.tile(base, (len(frame), 1))


@pytest.fixture
def sample_payload() -> dict[str, Any]:
    """Return one complete, valid API payload."""

    return {
        "fecha_dato": "2016-05-28",
        "ncodpers": 123456,
        "age": 42,
        "antiguedad": 84,
        "renta": 72_500.0,
        "pais_residencia": "ES",
        "sexo": "H",
        "ind_empleado": "N",
        "ind_actividad_cliente": 1,
        "segmento": "02 - PARTICULARES",
        "current_products": ["ind_ahor_fin_ult1"],
        "top_k": 3,
    }


@pytest.fixture
def fake_service() -> Iterator[tuple[TestClient, FakeRecommendationArtifact]]:
    """Run the application with an in-memory fake artifact."""

    artifact = FakeRecommendationArtifact()
    predictor = ModelPredictor(artifact)
    api = create_app(predictor_loader=lambda _: predictor)
    with TestClient(api) as client:
        yield client, artifact


def test_health_reports_loaded_model(fake_service: tuple[TestClient, Any]) -> None:
    client, _ = fake_service

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "model_version": "test-model-v1",
        "product_count": 24,
    }


def test_predict_builds_feature_frame_and_filters_owned_product(
    fake_service: tuple[TestClient, FakeRecommendationArtifact],
    sample_payload: dict[str, Any],
) -> None:
    client, artifact = fake_service

    response = client.post("/predict", json=sample_payload)

    assert response.status_code == 200
    body = response.json()
    assert body["customer_id"] == sample_payload["ncodpers"]
    assert body["model_version"] == "test-model-v1"
    assert [item["product"] for item in body["recommendations"]] == list(
        PRODUCT_NAMES[1:4]
    )
    assert [item["rank"] for item in body["recommendations"]] == [1, 2, 3]
    assert all(
        item["product"] not in sample_payload["current_products"]
        for item in body["recommendations"]
    )

    assert artifact.last_frame is not None
    assert artifact.last_frame.loc[0, "fecha_dato"] == "2016-05-28"
    assert artifact.last_frame.loc[0, "ind_ahor_fin_ult1"] == 1
    assert artifact.last_frame.loc[0, "ind_aval_fin_ult1"] == 0
    assert set(PRODUCT_NAMES).issubset(artifact.last_frame.columns)
    assert artifact.last_k == 24


def test_predict_batch_preserves_request_order(
    fake_service: tuple[TestClient, Any],
    sample_payload: dict[str, Any],
) -> None:
    client, _ = fake_service
    second = {
        **sample_payload,
        "ncodpers": 654321,
        "age": 31,
        "current_products": ["ind_aval_fin_ult1"],
        "top_k": 2,
    }

    response = client.post(
        "/predict/batch",
        json={"requests": [sample_payload, second]},
    )

    assert response.status_code == 200
    predictions = response.json()["predictions"]
    assert [item["customer_id"] for item in predictions] == [123456, 654321]
    assert len(predictions[0]["recommendations"]) == 3
    assert len(predictions[1]["recommendations"]) == 2
    assert "ind_aval_fin_ult1" not in {
        item["product"] for item in predictions[1]["recommendations"]
    }


@pytest.mark.parametrize(
    ("update", "message"),
    [
        ({"top_k": 0}, "greater than or equal to 1"),
        (
            {"current_products": ["ind_cco_fin_ult1", "ind_cco_fin_ult1"]},
            "must not contain duplicates",
        ),
        ({"current_products": ["not_a_product"]}, "Input should be"),
    ],
)
def test_predict_rejects_invalid_payloads(
    fake_service: tuple[TestClient, Any],
    sample_payload: dict[str, Any],
    update: dict[str, Any],
    message: str,
) -> None:
    client, _ = fake_service

    response = client.post("/predict", json={**sample_payload, **update})

    assert response.status_code == 422
    assert message in response.text


def test_unavailable_model_returns_503(sample_payload: dict[str, Any]) -> None:
    def unavailable_loader(_: str) -> Any:
        raise ModelLoadError("test artifact is missing")

    api = create_app(predictor_loader=unavailable_loader)
    with TestClient(api) as client:
        health = client.get("/health")
        prediction = client.post("/predict", json=sample_payload)

    assert health.status_code == 503
    assert "test artifact is missing" in health.json()["detail"]
    assert prediction.status_code == 503
    assert "Model is unavailable" in prediction.json()["detail"]


def test_predict_scores_artifact_is_supported(sample_payload: dict[str, Any]) -> None:
    predictor = ModelPredictor(FakeScoreArtifact())
    request = PredictionRequest.model_validate(sample_payload)

    recommendations = predictor.recommend_many([request])[0]

    assert len(recommendations) == sample_payload["top_k"]
    assert recommendations[0].product == PRODUCT_NAMES[-1]
    assert all(
        recommendation.product not in sample_payload["current_products"]
        for recommendation in recommendations
    )


def test_metrics_expose_required_names_and_labels(
    fake_service: tuple[TestClient, Any],
    sample_payload: dict[str, Any],
) -> None:
    client, _ = fake_service
    prediction = client.post("/predict", json=sample_payload)
    assert prediction.status_code == 200

    response = client.get("/metrics")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/plain")
    metrics = response.text
    assert "bank_api_requests_total" in metrics
    assert 'endpoint="/predict"' in metrics
    assert 'method="POST"' in metrics
    assert 'status_code="200"' in metrics
    assert "bank_api_request_duration_seconds" in metrics
    assert "bank_predictions_total" in metrics
    assert "bank_recommendations_total" in metrics
    assert 'product="ind_aval_fin_ult1"' in metrics
    assert "bank_input_age_bucket" in metrics
    assert "bank_input_income_bucket" in metrics
    assert "bank_recommendation_score_bucket" in metrics
    assert "bank_empty_recommendations_total" in metrics
    assert "bank_owned_products_filtered_total" in metrics
    assert "bank_prediction_failures_total" in metrics


def test_all_owned_products_produce_a_safe_empty_list(
    fake_service: tuple[TestClient, Any],
    sample_payload: dict[str, Any],
) -> None:
    client, _ = fake_service
    payload = {
        **sample_payload,
        "current_products": list(PRODUCT_NAMES),
        "top_k": 7,
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 200
    assert response.json()["recommendations"] == []


def test_unknown_routes_use_a_bounded_metrics_label(
    fake_service: tuple[TestClient, Any],
) -> None:
    client, _ = fake_service

    missing = client.get("/customers/123456/not-found")
    metrics = client.get("/metrics").text

    assert missing.status_code == 404
    assert 'endpoint="__unmatched__"' in metrics
    assert 'endpoint="/customers/123456/not-found"' not in metrics
