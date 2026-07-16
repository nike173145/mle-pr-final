"""Full-dataset loading and leakage-safe temporal target construction."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from bank_recommender.constants import (
    CATEGORICAL_PROFILE_COLUMNS,
    DATE_COLUMN,
    ID_COLUMN,
    NUMERIC_PROFILE_COLUMNS,
    PRODUCT_COLUMNS,
    RAW_FEATURE_COLUMNS,
)


@dataclass(frozen=True)
class TemporalPairs:
    """Features at month t and product additions at exactly month t + 1."""

    features: pd.DataFrame
    targets: pd.DataFrame
    periods: pd.Series

    def split(
        self, validation_period: str = "2016-04"
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Return chronological train and validation frames."""
        boundary = pd.Period(validation_period, freq="M")
        train_mask = self.periods < boundary
        valid_mask = self.periods == boundary
        if not train_mask.any():
            raise ValueError("No training pairs precede the validation period")
        if not valid_mask.any():
            raise ValueError("No pairs belong to the validation period")
        return (
            self.features.loc[train_mask].reset_index(drop=True),
            self.targets.loc[train_mask].reset_index(drop=True),
            self.features.loc[valid_mask].reset_index(drop=True),
            self.targets.loc[valid_mask].reset_index(drop=True),
        )


def normalise_snapshots(frame: pd.DataFrame) -> pd.DataFrame:
    """Coerce raw Santander values to a compact, consistent schema."""
    missing = sorted({ID_COLUMN, *RAW_FEATURE_COLUMNS} - set(frame.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    result = frame[[ID_COLUMN, *RAW_FEATURE_COLUMNS]].copy()
    result[ID_COLUMN] = pd.to_numeric(result[ID_COLUMN], errors="coerce")
    result[DATE_COLUMN] = pd.to_datetime(result[DATE_COLUMN], errors="coerce")
    result = result.dropna(subset=[ID_COLUMN, DATE_COLUMN])
    result[ID_COLUMN] = result[ID_COLUMN].astype("int64")

    for column in NUMERIC_PROFILE_COLUMNS:
        result[column] = pd.to_numeric(result[column], errors="coerce").astype(
            "float32"
        )
    for column in CATEGORICAL_PROFILE_COLUMNS:
        result[column] = (
            result[column]
            .astype("string")
            .str.strip()
            .replace({"": pd.NA, "NA": pd.NA})
        )
    for column in PRODUCT_COLUMNS:
        result[column] = (
            pd.to_numeric(result[column], errors="coerce")
            .where(lambda values: values.isin([0, 1]))
            .astype("Int8")
        )

    return (
        result.sort_values([ID_COLUMN, DATE_COLUMN], kind="stable")
        .drop_duplicates([ID_COLUMN, DATE_COLUMN], keep="last")
        .reset_index(drop=True)
    )


def read_snapshots(
    csv_path: str | Path,
    dates: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Load the complete CSV in one read and return all requested snapshots."""
    path = Path(csv_path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Dataset not found at {path}. Place train_ver2.csv there or pass --data."
        )
    selected_dates = set(dates) if dates is not None else None
    usecols = [ID_COLUMN, *RAW_FEATURE_COLUMNS]
    product_dtypes = {column: "float32" for column in PRODUCT_COLUMNS}
    frame = pd.read_csv(
        path,
        usecols=usecols,
        dtype=product_dtypes,
        na_values=["NA", " NA", ""],
        skipinitialspace=True,
        low_memory=False,
    )
    if selected_dates is not None:
        frame = frame[frame[DATE_COLUMN].isin(selected_dates)]
    if frame.empty:
        raise ValueError("The dataset contains no requested rows")
    return normalise_snapshots(frame)


def build_temporal_pairs(snapshots: pd.DataFrame) -> TemporalPairs:
    """Build targets from adjacent, fully observed product snapshots."""
    data = normalise_snapshots(snapshots)
    complete_products = data[PRODUCT_COLUMNS].notna().all(axis=1)
    data = data.loc[complete_products].reset_index(drop=True)
    if data.empty:
        raise ValueError("No snapshots have all product flags available")
    data["_period"] = data[DATE_COLUMN].dt.to_period("M")
    data["_next_period"] = data["_period"] + 1

    future = data[[ID_COLUMN, "_period", *PRODUCT_COLUMNS]].rename(
        columns={
            "_period": "_target_period",
            **{column: f"{column}__next" for column in PRODUCT_COLUMNS},
        }
    )
    paired = data.merge(
        future,
        left_on=[ID_COLUMN, "_next_period"],
        right_on=[ID_COLUMN, "_target_period"],
        how="inner",
        validate="one_to_one",
        sort=False,
    )
    if paired.empty:
        raise ValueError("No adjacent customer-month pairs were found")

    targets = pd.DataFrame(index=paired.index)
    for product in PRODUCT_COLUMNS:
        targets[product] = (
            (paired[f"{product}__next"] - paired[product]).clip(lower=0).astype("int8")
        )

    features = paired[[ID_COLUMN, *RAW_FEATURE_COLUMNS]].copy()
    periods = paired["_period"].copy()
    features.index = pd.RangeIndex(len(features))
    targets.index = features.index
    periods.index = features.index
    return TemporalPairs(features=features, targets=targets, periods=periods)


def latest_snapshot(snapshots: pd.DataFrame) -> pd.DataFrame:
    """Return one most-recent row per customer for next-month scoring."""
    data = normalise_snapshots(snapshots)
    max_date = data[DATE_COLUMN].max()
    return data.loc[data[DATE_COLUMN].eq(max_date)].reset_index(drop=True)
