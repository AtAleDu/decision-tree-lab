"""Loading, cleaning, splitting and CSV export helpers for Adult dataset."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


FEATURE_COLUMNS = [
    "age",
    "workclass",
    "fnlwgt",
    "education",
    "education-num",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "capital-gain",
    "capital-loss",
    "hours-per-week",
    "native-country",
]
TARGET_COLUMN = "income"
ALL_COLUMNS = FEATURE_COLUMNS + [TARGET_COLUMN]

NUMERIC_COLUMNS = [
    "age",
    "fnlwgt",
    "education-num",
    "capital-gain",
    "capital-loss",
    "hours-per-week",
]
CATEGORICAL_COLUMNS = [column for column in FEATURE_COLUMNS if column not in NUMERIC_COLUMNS]


def load_adult_dataset(path: str | Path, is_test: bool = False) -> pd.DataFrame:
    """Load Adult data/test file and normalize missing values, types and labels."""

    path = Path(path)
    skiprows = 1 if is_test else 0
    frame = pd.read_csv(
        path,
        header=None,
        names=ALL_COLUMNS,
        skiprows=skiprows,
        sep=",",
        skipinitialspace=True,
        na_values=["?"],
    )

    frame = frame.dropna().reset_index(drop=True)
    frame[TARGET_COLUMN] = frame[TARGET_COLUMN].astype(str).str.strip().str.rstrip(".")

    for column in NUMERIC_COLUMNS:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    for column in CATEGORICAL_COLUMNS:
        frame[column] = frame[column].astype(str).str.strip()

    frame = frame.dropna().reset_index(drop=True)
    return frame


def split_features_target(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Return X and y without leaking target values into features."""

    return frame[FEATURE_COLUMNS].copy(), frame[TARGET_COLUMN].copy()


def stratified_train_test_split(
    frame: pd.DataFrame,
    train_size: float,
    random_state: int = 42,
    target_column: str = TARGET_COLUMN,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Simple reproducible stratified split implemented with pandas/numpy."""

    if not 0 < train_size < 1:
        raise ValueError("train_size must be between 0 and 1 for a random split")

    rng = np.random.default_rng(random_state)
    train_parts: list[pd.DataFrame] = []
    test_parts: list[pd.DataFrame] = []

    for _, group in frame.groupby(target_column, sort=False):
        indices = group.index.to_numpy()
        rng.shuffle(indices)
        train_count = int(round(len(indices) * train_size))
        train_parts.append(frame.loc[indices[:train_count]])
        test_parts.append(frame.loc[indices[train_count:]])

    train = pd.concat(train_parts).sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    test = pd.concat(test_parts).sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    return train, test


def export_clean_csv(
    train_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    output_dir: str | Path,
) -> None:
    """Save cleaned CSV files so preprocessing can be inspected separately."""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    train_frame.to_csv(output_path / "adult_train_clean.csv", index=False)
    test_frame.to_csv(output_path / "adult_test_clean.csv", index=False)


def describe_class_distribution(labels: Iterable[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for label in labels:
        counts[label] = counts.get(label, 0) + 1
    return counts
