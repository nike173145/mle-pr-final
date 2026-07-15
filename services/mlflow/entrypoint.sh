#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

if [[ -f "${PROJECT_ROOT}/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "${PROJECT_ROOT}/.env"
  set +a
fi

die() {
  printf 'MLflow configuration error: %s\n' "$1" >&2
  exit 2
}

require_env() {
  local variable_name="$1"
  [[ -n "${!variable_name:-}" ]] || die "${variable_name} is required; copy .env.example to .env and set it"
}

validate_config() {
  local variable_name
  for variable_name in \
    MLFLOW_BACKEND_STORE_URI \
    MLFLOW_ARTIFACTS_DESTINATION \
    MLFLOW_S3_ENDPOINT_URL \
    AWS_DEFAULT_REGION \
    AWS_ACCESS_KEY_ID \
    AWS_SECRET_ACCESS_KEY; do
    require_env "${variable_name}"
  done

  case "${MLFLOW_BACKEND_STORE_URI}" in
    postgresql://*|postgresql+psycopg2://*) ;;
    *) die "MLFLOW_BACKEND_STORE_URI must use an external postgresql:// or postgresql+psycopg2:// URI" ;;
  esac

  case "${MLFLOW_ARTIFACTS_DESTINATION}" in
    s3:///*) die "MLFLOW_ARTIFACTS_DESTINATION must include a bucket name" ;;
    s3://?*) ;;
    *) die "MLFLOW_ARTIFACTS_DESTINATION must use an s3:// URI" ;;
  esac

  case "${MLFLOW_S3_ENDPOINT_URL}" in
    https://?*) ;;
    *) die "MLFLOW_S3_ENDPOINT_URL must be an https:// URL supplied by the external S3 provider" ;;
  esac

  case "${MLFLOW_S3_IGNORE_TLS:-false}" in
    true|false) ;;
    *) die "MLFLOW_S3_IGNORE_TLS must be true or false" ;;
  esac

  [[ "${MLFLOW_PORT:-5000}" =~ ^[0-9]+$ ]] || die "MLFLOW_PORT must be numeric"
  local port_number=$((10#${MLFLOW_PORT}))
  (( port_number >= 1 && port_number <= 65535 )) || die "MLFLOW_PORT must be between 1 and 65535"
}

check_connections() {
  local python_bin="${PYTHON_BIN:-python}"
  command -v "${python_bin}" >/dev/null 2>&1 || die "${python_bin} executable is not installed"
  "${python_bin}" "${PROJECT_ROOT}/services/mlflow/check_storage.py"
}

MLFLOW_HOST="${MLFLOW_HOST:-0.0.0.0}"
MLFLOW_PORT="${MLFLOW_PORT:-5000}"
MLFLOW_S3_IGNORE_TLS="${MLFLOW_S3_IGNORE_TLS:-false}"
export MLFLOW_S3_IGNORE_TLS
validate_config

case "${1:-}" in
  --check-config)
    [[ $# -eq 1 ]] || die "--check-config does not accept additional arguments"
    printf 'MLflow configuration syntax is valid (PostgreSQL backend and S3 artifact store).\n'
    exit 0
    ;;
  --check)
    [[ $# -eq 1 ]] || die "--check does not accept additional arguments"
    check_connections
    printf 'MLflow external storage connectivity is valid.\n'
    exit 0
    ;;
  '') ;;
  *) die "unknown argument: $1" ;;
esac

command -v mlflow >/dev/null 2>&1 || die "mlflow executable is not installed"
check_connections

server_args=(
  server
  --host "${MLFLOW_HOST}"
  --port "${MLFLOW_PORT}"
  --backend-store-uri "${MLFLOW_BACKEND_STORE_URI}"
  --serve-artifacts
  --artifacts-destination "${MLFLOW_ARTIFACTS_DESTINATION}"
)

if [[ -n "${MLFLOW_PROMETHEUS_DIR:-}" ]]; then
  mkdir -p "${MLFLOW_PROMETHEUS_DIR}"
  server_args+=(--expose-prometheus "${MLFLOW_PROMETHEUS_DIR}")
fi

exec mlflow "${server_args[@]}"
