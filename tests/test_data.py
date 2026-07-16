from __future__ import annotations

import pandas as pd
import pytest

from bank_recommender.constants import (
    CATEGORICAL_PROFILE_COLUMNS,
    DATE_COLUMN,
    ID_COLUMN,
    NUMERIC_PROFILE_COLUMNS,
    PRODUCT_COLUMNS,
)
from bank_recommender.data import build_temporal_pairs, read_snapshots


def make_snapshot(customer: int, date: str, **products: int) -> dict:
    row = {ID_COLUMN: customer, DATE_COLUMN: date}
    row.update({column: 0 for column in NUMERIC_PROFILE_COLUMNS})
    row.update({column: None for column in CATEGORICAL_PROFILE_COLUMNS})
    row.update({column: 0 for column in PRODUCT_COLUMNS})
    row.update(products)
    return row


def test_target_contains_additions_but_not_product_removals() -> None:
    product = PRODUCT_COLUMNS[0]
    frame = pd.DataFrame(
        [
            make_snapshot(1, "2016-01-28", **{product: 0}),
            make_snapshot(1, "2016-02-28", **{product: 1}),
            make_snapshot(1, "2016-03-28", **{product: 0}),
        ]
    )

    pairs = build_temporal_pairs(frame)

    assert pairs.periods.astype(str).tolist() == ["2016-01", "2016-02"]
    assert pairs.targets[product].tolist() == [1, 0]


def test_missing_calendar_month_is_not_paired() -> None:
    frame = pd.DataFrame(
        [
            make_snapshot(1, "2016-01-28"),
            make_snapshot(1, "2016-03-28", **{PRODUCT_COLUMNS[1]: 1}),
            make_snapshot(2, "2016-01-28"),
            make_snapshot(2, "2016-02-28"),
        ]
    )

    pairs = build_temporal_pairs(frame)

    assert pairs.features[ID_COLUMN].tolist() == [2]


def test_duplicate_snapshot_uses_last_record() -> None:
    product = PRODUCT_COLUMNS[2]
    frame = pd.DataFrame(
        [
            make_snapshot(1, "2016-01-28", **{product: 0}),
            make_snapshot(1, "2016-01-28", **{product: 1}),
            make_snapshot(1, "2016-02-28", **{product: 1}),
        ]
    )

    pairs = build_temporal_pairs(frame)

    assert len(pairs.features) == 1
    assert pairs.targets.loc[0, product] == 0


def test_read_snapshots_loads_all_rows_in_one_dataset(
    tmp_path,
) -> None:
    frame = pd.DataFrame(
        [
            make_snapshot(1, "2016-01-28"),
            make_snapshot(1, "2016-02-28"),
            make_snapshot(2, "2016-01-28"),
        ]
    )
    path = tmp_path / "snapshots.csv"
    frame.to_csv(path, index=False)

    snapshots = read_snapshots(path)

    assert len(snapshots) == 3
    assert snapshots[ID_COLUMN].tolist() == [1, 1, 2]


def test_read_snapshots_uses_one_pandas_read(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frame = pd.DataFrame(
        [
            make_snapshot(1, "2016-01-28"),
            make_snapshot(1, "2016-02-28"),
        ]
    )
    path = tmp_path / "snapshots.csv"
    path.write_text("placeholder", encoding="utf-8")
    calls: list[dict] = []

    def fake_read_csv(csv_path, **kwargs):
        assert csv_path == path
        calls.append(kwargs)
        return frame.copy()

    monkeypatch.setattr("bank_recommender.data.pd.read_csv", fake_read_csv)

    snapshots = read_snapshots(path)

    assert len(snapshots) == 2
    assert len(calls) == 1
    assert calls[0]["low_memory"] is False


def test_read_snapshots_can_filter_dates_after_full_read(tmp_path) -> None:
    frame = pd.DataFrame(
        [
            make_snapshot(1, "2016-01-28"),
            make_snapshot(1, "2016-02-28"),
            make_snapshot(2, "2016-02-28"),
        ]
    )
    path = tmp_path / "snapshots.csv"
    frame.to_csv(path, index=False)

    snapshots = read_snapshots(path, dates=["2016-02-28"])

    assert len(snapshots) == 2
    assert snapshots[DATE_COLUMN].dt.strftime("%Y-%m-%d").unique().tolist() == [
        "2016-02-28"
    ]
