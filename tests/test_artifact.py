from __future__ import annotations

import hashlib
import json
from pathlib import Path

from services.recomendation_service.app.predictor import ModelPredictor
from services.recomendation_service.app.schemas import PredictionRequest

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_tracked_artifact_matches_model_card_and_predicts() -> None:
    artifact_path = PROJECT_ROOT / "ml_models" / "model.joblib"
    metadata = json.loads(
        (PROJECT_ROOT / "reports" / "model_metadata.json").read_text(encoding="utf-8")
    )
    request_payload = json.loads(
        (PROJECT_ROOT / "examples" / "predict_request.json").read_text(encoding="utf-8")
    )

    digest = hashlib.sha256(artifact_path.read_bytes()).hexdigest()
    assert digest == metadata["artifact_sha256"]

    predictor = ModelPredictor.load(artifact_path)
    request = PredictionRequest.model_validate(request_payload)
    recommendations = predictor.recommend_many([request])[0]

    assert predictor.model_version == metadata["model_version"]
    assert len(recommendations) == request.top_k
    assert not set(request.current_products).intersection(
        recommendation.product for recommendation in recommendations
    )
