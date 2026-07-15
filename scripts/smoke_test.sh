#!/usr/bin/env bash
set -euo pipefail

BASE_URL="${BASE_URL:-http://127.0.0.1:8000}"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

curl --fail --silent --show-error "${BASE_URL}/health"
curl --fail --silent --show-error \
  -X POST \
  -H "Content-Type: application/json" \
  --data @"${PROJECT_ROOT}/examples/predict_request.json" \
  "${BASE_URL}/predict"
curl --fail --silent --show-error "${BASE_URL}/metrics" >/dev/null
