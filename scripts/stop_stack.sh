#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_FILE="${ENV_FILE:-${PROJECT_ROOT}/.env}"
COMPOSE_FILE="${PROJECT_ROOT}/services/docker-compose.yaml"

if [[ ! -f "${ENV_FILE}" ]]; then
  printf 'Configuration file not found: %s\n' "${ENV_FILE}" >&2
  exit 2
fi

exec docker compose \
  --env-file "${ENV_FILE}" \
  -f "${COMPOSE_FILE}" \
  down "$@"
