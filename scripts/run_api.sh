#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export MODEL_PATH="${MODEL_PATH:-${PROJECT_ROOT}/ml_models/model.joblib}"
export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

cd "${PROJECT_ROOT}"
exec uvicorn services.recomendation_service.app.main:app \
  --host "${API_HOST:-0.0.0.0}" \
  --port "${API_PORT:-8000}"
