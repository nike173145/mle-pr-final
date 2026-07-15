"""Fail-fast connectivity checks for MLflow's external state stores."""

from __future__ import annotations

import os
import sys
from urllib.parse import urlparse


def _fail(service: str, error: Exception) -> None:
    """Report a useful error class without leaking credentials from an exception."""
    print(
        f"{service} connectivity check failed ({type(error).__name__}); "
        "verify credentials, TLS settings, network access, and provider status.",
        file=sys.stderr,
    )
    raise SystemExit(3)


def _check_postgres() -> None:
    try:
        from sqlalchemy import create_engine, text

        engine = create_engine(
            os.environ["MLFLOW_BACKEND_STORE_URI"],
            connect_args={"connect_timeout": 10},
            pool_pre_ping=True,
        )
        try:
            with engine.connect() as connection:
                result = connection.execute(text("SELECT 1")).scalar_one()
                if result != 1:
                    raise RuntimeError("unexpected SELECT 1 result")
        finally:
            engine.dispose()
    except Exception as error:
        _fail("PostgreSQL", error)
    print("PostgreSQL connectivity: OK")


def _check_s3() -> None:
    try:
        import boto3
        from botocore.config import Config

        artifact_uri = urlparse(os.environ["MLFLOW_ARTIFACTS_DESTINATION"])
        verify: bool | str = os.environ.get("MLFLOW_S3_IGNORE_TLS", "false") != "true"
        if verify and os.environ.get("AWS_CA_BUNDLE"):
            verify = os.environ["AWS_CA_BUNDLE"]

        s3 = boto3.client(
            "s3",
            endpoint_url=os.environ["MLFLOW_S3_ENDPOINT_URL"],
            region_name=os.environ["AWS_DEFAULT_REGION"],
            verify=verify,
            config=Config(
                connect_timeout=10,
                read_timeout=10,
                retries={"max_attempts": 2, "mode": "standard"},
            ),
        )
        bucket = artifact_uri.netloc
        prefix = artifact_uri.path.lstrip("/")
        s3.head_bucket(Bucket=bucket)
        s3.list_objects_v2(Bucket=bucket, Prefix=prefix, MaxKeys=1)
    except Exception as error:
        _fail("S3", error)
    print("S3 connectivity: OK")


def main() -> None:
    _check_postgres()
    _check_s3()


if __name__ == "__main__":
    main()
