FROM python:3.11-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    HOME=/tmp \
    MLFLOW_PROMETHEUS_DIR=/tmp/mlflow-prometheus

WORKDIR /app

COPY services/mlflow/requirements.txt /tmp/mlflow-requirements.txt
RUN python -m pip install --no-cache-dir -r /tmp/mlflow-requirements.txt \
    && groupadd --gid 10001 mlflow \
    && useradd --uid 10001 --gid 10001 --no-create-home --home-dir /tmp --shell /usr/sbin/nologin mlflow

COPY --chown=10001:10001 services/mlflow/entrypoint.sh /app/services/mlflow/entrypoint.sh
COPY --chown=10001:10001 services/mlflow/check_storage.py /app/services/mlflow/check_storage.py
RUN chmod 0555 /app/services/mlflow/entrypoint.sh

USER 10001:10001
EXPOSE 5000

HEALTHCHECK --interval=15s --timeout=5s --start-period=20s --retries=5 \
  CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:5000/health', timeout=3).read()"]

ENTRYPOINT ["/app/services/mlflow/entrypoint.sh"]
