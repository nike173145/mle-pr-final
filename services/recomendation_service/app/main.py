"""FastAPI application serving Santander product recommendations."""

from __future__ import annotations

import logging
import os
import time
from collections.abc import Callable, Sequence
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Any

from fastapi import Depends, FastAPI, HTTPException, Request, Response, status
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from pydantic import ValidationError

from .predictor import ModelPredictor, PredictionError
from .schemas import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    HealthResponse,
    PredictionRequest,
    PredictionResponse,
    Recommendation,
)

logger = logging.getLogger(__name__)
DEFAULT_MODEL_PATH = "ml_models/model.joblib"

BANK_API_REQUESTS = Counter(
    "bank_api_requests_total",
    "Number of HTTP requests handled by the bank recommendation API.",
    ("method", "endpoint", "status_code"),
)
BANK_API_REQUEST_DURATION = Histogram(
    "bank_api_request_duration_seconds",
    "HTTP request duration for the bank recommendation API.",
    ("method", "endpoint"),
)
BANK_PREDICTIONS = Counter(
    "bank_predictions_total",
    "Number of client profiles successfully scored.",
)
BANK_RECOMMENDATIONS = Counter(
    "bank_recommendations_total",
    "Number of products returned by the recommendation API.",
    ("product",),
)
BANK_RECOMMENDATION_SCORE = Histogram(
    "bank_recommendation_score",
    "Distribution of recommendation ranking scores.",
    buckets=(0, 0.001, 0.005, 0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 1.0),
)
BANK_EMPTY_RECOMMENDATIONS = Counter(
    "bank_empty_recommendations_total",
    "Number of scored profiles for which no new product could be returned.",
)
BANK_OWNED_PRODUCTS_FILTERED = Counter(
    "bank_owned_products_filtered_total",
    "Number of owned catalog products excluded from candidate rankings.",
)
BANK_PREDICTION_FAILURES = Counter(
    "bank_prediction_failures_total",
    "Number of inference failures grouped into bounded operational reasons.",
    ("reason",),
)
BANK_INPUT_AGE = Histogram(
    "bank_input_age",
    "Age distribution of successfully scored client profiles.",
    buckets=(18, 25, 35, 45, 55, 65, 75, 90, 120),
)
BANK_INPUT_INCOME = Histogram(
    "bank_input_income",
    "Income distribution of successfully scored client profiles.",
    buckets=(0, 20_000, 40_000, 60_000, 100_000, 200_000, 500_000, 1_000_000),
)

PredictorLoader = Callable[[str | Path], Any]


def get_predictor(request: Request) -> Any:
    """Return the lifespan-managed predictor or a clear readiness error."""

    predictor = getattr(request.app.state, "predictor", None)
    if predictor is None:
        reason = getattr(
            request.app.state,
            "model_error",
            "Model loading has not completed.",
        )
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Model is unavailable: {reason}",
        )
    return predictor


PredictorDependency = Annotated[Any, Depends(get_predictor)]


def _sanitize_again(
    recommendations: Sequence[Any],
    request: PredictionRequest,
) -> list[Recommendation]:
    """Apply a second ownership, duplicate, and top-k guard at the HTTP boundary."""

    owned = set(request.current_products)
    seen: set[str] = set()
    safe: list[Recommendation] = []
    for raw in recommendations:
        try:
            recommendation = (
                raw
                if isinstance(raw, Recommendation)
                else Recommendation.model_validate(raw)
            )
        except ValidationError as error:
            raise PredictionError(
                f"Predictor returned an invalid recommendation: {error}"
            ) from error
        if recommendation.product in owned or recommendation.product in seen:
            continue
        seen.add(recommendation.product)
        safe.append(
            Recommendation(
                product=recommendation.product,
                score=recommendation.score,
                rank=len(safe) + 1,
            )
        )
        if len(safe) == request.top_k:
            break
    return safe


def _predict_or_503(
    predictor: Any,
    requests: Sequence[PredictionRequest],
) -> list[PredictionResponse]:
    """Run inference and translate operational failures into HTTP 503."""

    try:
        raw_batches = predictor.recommend_many(requests)
        if len(raw_batches) != len(requests):
            raise PredictionError(
                "Predictor returned an unexpected number of result rows"
            )

        responses: list[PredictionResponse] = []
        for payload, raw_recommendations in zip(requests, raw_batches, strict=True):
            safe_recommendations = _sanitize_again(raw_recommendations, payload)
            responses.append(
                PredictionResponse(
                    customer_id=payload.ncodpers,
                    model_version=str(predictor.model_version),
                    recommendations=safe_recommendations,
                )
            )
    except PredictionError as error:
        BANK_PREDICTION_FAILURES.labels(reason="prediction_error").inc()
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Prediction is unavailable: {error}",
        ) from error
    except HTTPException:
        raise
    except Exception as error:
        BANK_PREDICTION_FAILURES.labels(reason="unexpected_error").inc()
        logger.exception("Unexpected prediction failure")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Prediction is unavailable due to an internal model error.",
        ) from error

    BANK_PREDICTIONS.inc(len(requests))
    for payload, response in zip(requests, responses, strict=True):
        BANK_INPUT_AGE.observe(payload.age)
        BANK_INPUT_INCOME.observe(payload.renta)
        BANK_OWNED_PRODUCTS_FILTERED.inc(len(payload.current_products))
        if not response.recommendations:
            BANK_EMPTY_RECOMMENDATIONS.inc()
        for recommendation in response.recommendations:
            BANK_RECOMMENDATIONS.labels(product=recommendation.product).inc()
            BANK_RECOMMENDATION_SCORE.observe(recommendation.score)
    return responses


def create_app(predictor_loader: PredictorLoader | None = None) -> FastAPI:
    """Build an application, optionally injecting a predictor loader for tests."""

    loader = predictor_loader or ModelPredictor.load

    @asynccontextmanager
    async def lifespan(api: FastAPI):
        model_path = os.getenv("MODEL_PATH", DEFAULT_MODEL_PATH)
        api.state.model_path = model_path
        api.state.predictor = None
        api.state.model_error = "Model loading has not completed."
        try:
            api.state.predictor = loader(model_path)
            api.state.model_error = None
            logger.info("Loaded recommendation model from %s", model_path)
        except Exception as error:
            api.state.model_error = str(error)
            logger.error("Could not load recommendation model: %s", error)
        yield
        api.state.predictor = None

    api = FastAPI(
        title="Santander Product Recommendation API",
        description="Recommend new banking products from a monthly client snapshot.",
        version="1.0.0",
        lifespan=lifespan,
    )

    @api.middleware("http")
    async def observe_http(request: Request, call_next: Callable[..., Any]) -> Response:
        started_at = time.perf_counter()
        response_status = status.HTTP_500_INTERNAL_SERVER_ERROR
        try:
            response = await call_next(request)
            response_status = response.status_code
            return response
        finally:
            route = request.scope.get("route")
            endpoint = getattr(route, "path", "__unmatched__")
            method = request.method.upper()
            BANK_API_REQUESTS.labels(
                method=method,
                endpoint=endpoint,
                status_code=str(response_status),
            ).inc()
            BANK_API_REQUEST_DURATION.labels(
                method=method,
                endpoint=endpoint,
            ).observe(time.perf_counter() - started_at)

    @api.get("/health", response_model=HealthResponse)
    def health(predictor: PredictorDependency) -> HealthResponse:
        """Report readiness only after a valid model artifact is loaded."""

        return HealthResponse(
            status="ok",
            model_version=str(predictor.model_version),
            product_count=len(predictor.product_names),
        )

    @api.post("/predict", response_model=PredictionResponse)
    def predict(
        payload: PredictionRequest,
        predictor: PredictorDependency,
    ) -> PredictionResponse:
        """Recommend products for one validated client profile."""

        return _predict_or_503(predictor, [payload])[0]

    @api.post("/predict/batch", response_model=BatchPredictionResponse)
    def predict_batch(
        payload: BatchPredictionRequest,
        predictor: PredictorDependency,
    ) -> BatchPredictionResponse:
        """Recommend products for up to one hundred client profiles."""

        return BatchPredictionResponse(
            predictions=_predict_or_503(predictor, payload.requests)
        )

    @api.get("/metrics", include_in_schema=False)
    def metrics() -> Response:
        """Expose Prometheus metrics in the text exposition format."""

        return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)

    return api


app = create_app()
