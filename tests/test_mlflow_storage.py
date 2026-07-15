from __future__ import annotations

import sys
from types import ModuleType
from typing import Any

import pytest

from services.mlflow import check_storage


class _ScalarResult:
    def scalar_one(self) -> int:
        return 1


class _Connection:
    def __enter__(self) -> _Connection:
        return self

    def __exit__(self, *_: object) -> None:
        return None

    def execute(self, statement: str) -> _ScalarResult:
        assert statement == "SELECT 1"
        return _ScalarResult()


class _Engine:
    disposed = False

    def connect(self) -> _Connection:
        return _Connection()

    def dispose(self) -> None:
        self.disposed = True


def test_postgres_check_executes_read_only_probe(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    engine = _Engine()
    sqlalchemy = ModuleType("sqlalchemy")

    def create_engine(uri: str, **kwargs: Any) -> _Engine:
        assert uri.startswith("postgresql+psycopg2://")
        assert kwargs["connect_args"] == {"connect_timeout": 10}
        assert kwargs["pool_pre_ping"] is True
        return engine

    sqlalchemy.create_engine = create_engine  # type: ignore[attr-defined]
    sqlalchemy.text = lambda value: value  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "sqlalchemy", sqlalchemy)
    monkeypatch.setenv(
        "MLFLOW_BACKEND_STORE_URI",
        "postgresql+psycopg2://user:secret@db.example/mlflow?sslmode=require",
    )

    check_storage._check_postgres()

    assert engine.disposed is True
    assert capsys.readouterr().out == "PostgreSQL connectivity: OK\n"


def test_s3_check_targets_configured_bucket_and_prefix(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    calls: list[tuple[str, dict[str, Any]]] = []

    class FakeS3:
        def head_bucket(self, **kwargs: Any) -> None:
            calls.append(("head", kwargs))

        def list_objects_v2(self, **kwargs: Any) -> None:
            calls.append(("list", kwargs))

    boto3 = ModuleType("boto3")

    def client(service: str, **kwargs: Any) -> FakeS3:
        assert service == "s3"
        assert kwargs["endpoint_url"] == "https://s3.example.com"
        assert kwargs["region_name"] == "region-1"
        assert kwargs["verify"] is True
        return FakeS3()

    boto3.client = client  # type: ignore[attr-defined]
    botocore = ModuleType("botocore")
    botocore_config = ModuleType("botocore.config")
    botocore_config.Config = lambda **kwargs: kwargs  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "boto3", boto3)
    monkeypatch.setitem(sys.modules, "botocore", botocore)
    monkeypatch.setitem(sys.modules, "botocore.config", botocore_config)
    monkeypatch.setenv(
        "MLFLOW_ARTIFACTS_DESTINATION",
        "s3://mlflow-bucket/bank-recommender",
    )
    monkeypatch.setenv("MLFLOW_S3_ENDPOINT_URL", "https://s3.example.com")
    monkeypatch.setenv("AWS_DEFAULT_REGION", "region-1")
    monkeypatch.setenv("MLFLOW_S3_IGNORE_TLS", "false")

    check_storage._check_s3()

    assert calls == [
        ("head", {"Bucket": "mlflow-bucket"}),
        (
            "list",
            {"Bucket": "mlflow-bucket", "Prefix": "bank-recommender", "MaxKeys": 1},
        ),
    ]
    assert capsys.readouterr().out == "S3 connectivity: OK\n"


def test_connectivity_error_does_not_leak_exception_message(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit, match="3"):
        check_storage._fail(
            "PostgreSQL",
            RuntimeError("postgresql://user:super-secret@db.example/mlflow"),
        )

    message = capsys.readouterr().err
    assert "RuntimeError" in message
    assert "super-secret" not in message
    assert "postgresql://" not in message
