"""Command-line training, temporal validation, artifact export, and MLflow logging."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from bank_recommender.constants import PRODUCT_COLUMNS, RANDOM_SEED
from bank_recommender.data import build_temporal_pairs, read_sampled_snapshots
from bank_recommender.metrics import evaluate_ranking
from bank_recommender.model import BankProductRecommender, PopularityRecommender


def _json_default(value: Any):
    if isinstance(value, np.integer | np.floating):
        return value.item()
    if isinstance(value, pd.Timestamp | pd.Period):
        return str(value)
    raise TypeError(f"Cannot serialize {type(value)!r}")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default) + "\n",
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _flatten_metrics(
    comparison: dict[str, dict[str, float]],
) -> dict[str, float]:
    return {
        f"{experiment}.{metric}": value
        for experiment, values in comparison.items()
        for metric, value in values.items()
    }


def _log_mlflow(
    *,
    tracking_uri: str,
    experiment_name: str,
    registered_model_name: str | None,
    model: BankProductRecommender,
    model_path: Path,
    features: pd.DataFrame,
    parameters: dict[str, Any],
    comparison: dict[str, dict[str, float]],
    metadata_path: Path,
) -> str:
    import mlflow
    import mlflow.sklearn
    from mlflow.models import infer_signature

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
    input_example = features.head(5).copy()
    predictions = model.predict_scores(input_example)
    signature = infer_signature(input_example, predictions)
    with mlflow.start_run(run_name="final-sgd-ovr") as run:
        mlflow.log_params(parameters)
        mlflow.log_metrics(_flatten_metrics(comparison))
        mlflow.log_artifact(str(metadata_path), artifact_path="reports")
        mlflow.log_artifact(str(model_path), artifact_path="service-artifact")
        log_kwargs: dict[str, Any] = {}
        if registered_model_name:
            log_kwargs["registered_model_name"] = registered_model_name
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            signature=signature,
            input_example=input_example,
            code_paths=["src"],
            **log_kwargs,
        )
        return run.info.run_id


def run_training(
    *,
    data_path: str | Path,
    artifact_path: str | Path = "ml_models/model.joblib",
    report_dir: str | Path = "reports",
    sample_fraction: float = 0.02,
    validation_period: str = "2016-04",
    random_seed: int = RANDOM_SEED,
    alpha: float = 0.0001,
    max_iter: int = 60,
    k: int = 7,
    tracking_uri: str | None = None,
    experiment_name: str = "bank-product-recommendation",
    registered_model_name: str | None = None,
) -> dict[str, Any]:
    """Train, evaluate on an untouched month, refit, and save the model."""
    random.seed(random_seed)
    np.random.seed(random_seed)

    snapshots = read_sampled_snapshots(
        data_path,
        sample_fraction=sample_fraction,
        random_seed=random_seed,
    )
    pairs = build_temporal_pairs(snapshots)
    x_train, y_train, x_valid, y_valid = pairs.split(validation_period)

    baseline = PopularityRecommender().fit(x_train, y_train)
    baseline_scores = baseline.predict_scores(x_valid)
    comparison: dict[str, dict[str, float]] = {
        "popularity": evaluate_ranking(y_valid, baseline_scores, x_valid, k=k)
    }

    candidate = BankProductRecommender(
        alpha=alpha,
        max_iter=max_iter,
        prior_weight=0.0,
        random_seed=random_seed,
    ).fit(x_train, y_train)
    raw_scores = candidate.predict_scores(x_valid)
    prior_scores = baseline.predict_scores(x_valid)
    weights = [0.0, 0.25, 0.5, 0.75]
    for weight in weights:
        blended = (1.0 - weight) * raw_scores + weight * prior_scores
        comparison[f"sgd_prior_{weight:.2f}"] = evaluate_ranking(
            y_valid, blended, x_valid, k=k
        )

    primary_metric = f"map_at_{k}"
    learned_experiments = [name for name in comparison if name.startswith("sgd_")]
    selected_experiment = max(
        learned_experiments,
        key=lambda name: comparison[name][primary_metric],
    )
    selected_weight = float(selected_experiment.rsplit("_", 1)[1])

    all_features = pd.concat([x_train, x_valid], ignore_index=True)
    all_targets = pd.concat([y_train, y_valid], ignore_index=True)
    final_model = BankProductRecommender(
        alpha=alpha,
        max_iter=max_iter,
        prior_weight=selected_weight,
        random_seed=random_seed,
        model_version="sgd-ovr-v1",
    ).fit(all_features, all_targets)

    artifact = Path(artifact_path)
    artifact.parent.mkdir(parents=True, exist_ok=True)
    temporary_artifact = artifact.with_suffix(artifact.suffix + ".tmp")
    joblib.dump(final_model, temporary_artifact, compress=3)
    temporary_artifact.replace(artifact)

    report_path = Path(report_dir)
    metadata_path = report_path / "model_metadata.json"
    periods = pairs.periods.astype(str)
    metadata: dict[str, Any] = {
        "model_version": final_model.model_version,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "task": "rank products newly acquired in the next calendar month",
        "target_definition": "max(product[t+1] - product[t], 0)",
        "validation_period": validation_period,
        "training_period_min": periods.min(),
        "training_period_max": periods.max(),
        "forecast_snapshot": str(snapshots["fecha_dato"].max().date()),
        "forecast_month": str(snapshots["fecha_dato"].max().to_period("M") + 1),
        "sample_fraction": sample_fraction,
        "random_seed": random_seed,
        "snapshot_rows": len(snapshots),
        "training_pairs": len(x_train),
        "validation_pairs": len(x_valid),
        "refit_pairs": len(all_features),
        "selected_experiment": selected_experiment,
        "selected_prior_weight": selected_weight,
        "primary_metric": primary_metric,
        "product_columns": PRODUCT_COLUMNS,
        "validation_metrics": comparison[selected_experiment],
        "baseline_metrics": comparison["popularity"],
        "new_product_events_train": {
            product: int(y_train[product].sum()) for product in PRODUCT_COLUMNS
        },
    }
    final_model.training_metadata = metadata
    # Save once more so the service artifact carries the complete model card.
    joblib.dump(final_model, temporary_artifact, compress=3)
    temporary_artifact.replace(artifact)
    metadata["artifact_sha256"] = _sha256(artifact)
    _write_json(metadata_path, metadata)
    _write_json(report_path / "model_comparison.json", comparison)

    parameters = {
        "sample_fraction": sample_fraction,
        "validation_period": validation_period,
        "random_seed": random_seed,
        "alpha": alpha,
        "max_iter": max_iter,
        "top_k": k,
        "selected_prior_weight": selected_weight,
        "train_pairs": len(x_train),
        "validation_pairs": len(x_valid),
    }
    run_id = None
    if tracking_uri:
        run_id = _log_mlflow(
            tracking_uri=tracking_uri,
            experiment_name=experiment_name,
            registered_model_name=registered_model_name,
            model=final_model,
            model_path=artifact,
            features=all_features,
            parameters=parameters,
            comparison=comparison,
            metadata_path=metadata_path,
        )
    summary = {
        "artifact_path": str(artifact),
        "metadata_path": str(metadata_path),
        "selected_experiment": selected_experiment,
        "validation_metrics": comparison[selected_experiment],
        "baseline_metrics": comparison["popularity"],
        "mlflow_run_id": run_id,
    }
    _write_json(report_path / "training_summary.json", summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default=os.getenv("DATA_PATH", "train_ver2.csv"))
    parser.add_argument("--artifact", default="ml_models/model.joblib")
    parser.add_argument("--report-dir", default="reports")
    parser.add_argument("--sample-fraction", type=float, default=0.02)
    parser.add_argument("--validation-period", default="2016-04")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--alpha", type=float, default=0.0001)
    parser.add_argument("--max-iter", type=int, default=60)
    parser.add_argument("--top-k", type=int, default=7)
    parser.add_argument("--tracking-uri", default=os.getenv("MLFLOW_TRACKING_URI"))
    parser.add_argument("--experiment-name", default="bank-product-recommendation")
    parser.add_argument("--registered-model-name", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = run_training(
        data_path=args.data,
        artifact_path=args.artifact,
        report_dir=args.report_dir,
        sample_fraction=args.sample_fraction,
        validation_period=args.validation_period,
        random_seed=args.seed,
        alpha=args.alpha,
        max_iter=args.max_iter,
        k=args.top_k,
        tracking_uri=args.tracking_uri,
        experiment_name=args.experiment_name,
        registered_model_name=args.registered_model_name,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
