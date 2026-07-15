from __future__ import annotations

import pandas as pd

from bank_recommender.constants import (
    CATEGORICAL_PROFILE_COLUMNS,
    DATE_COLUMN,
    ID_COLUMN,
    NUMERIC_PROFILE_COLUMNS,
    PRODUCT_COLUMNS,
)
from bank_recommender.data import build_temporal_pairs, stable_customer_sample


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


def test_customer_sampling_is_stable_across_rows_and_seeded() -> None:
    ids = pd.Series([1, 1, 2, 2, 3, 3])
    first = stable_customer_sample(ids, sample_fraction=0.5, random_seed=42)
    second = stable_customer_sample(ids, sample_fraction=0.5, random_seed=42)

    assert first.equals(second)
    assert first.iloc[0] == first.iloc[1]
    assert first.iloc[2] == first.iloc[3]
