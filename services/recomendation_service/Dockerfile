FROM python:3.11-slim-bookworm

ARG APP_UID=10001
ARG APP_GID=10001

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app:/app/src \
    MODEL_PATH=/app/ml_models/model.joblib

WORKDIR /app

RUN groupadd --gid "${APP_GID}" bank-api \
    && useradd \
        --uid "${APP_UID}" \
        --gid "${APP_GID}" \
        --create-home \
        --shell /usr/sbin/nologin \
        bank-api

COPY requirements-runtime.txt ./requirements-runtime.txt
RUN python -m pip install -r requirements-runtime.txt

COPY --chown=bank-api:bank-api services ./services
COPY --chown=bank-api:bank-api src ./src
COPY --chown=bank-api:bank-api ml_models ./ml_models

USER bank-api

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3).read()"]

CMD ["uvicorn", "services.recomendation_service.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
