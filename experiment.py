"""Run all laboratory experiments and save results."""

from __future__ import annotations

import csv
import time
from pathlib import Path

import pandas as pd

from src.data_loader import (
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    CATEGORICAL_COLUMNS,
    export_clean_csv,
    load_adult_dataset,
    split_features_target,
    stratified_train_test_split,
)
from src.decision_tree import DecisionTreeClassifierScratch
from src.metrics import calculate_binary_metrics
from src.visualization import plot_metrics, visualize_tree


PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data"
RESULTS_DIR = PROJECT_DIR / "results"
TREES_DIR = RESULTS_DIR / "trees"
CSV_DIR = RESULTS_DIR / "csv"

CRITERIA = ["information_gain", "gain_ratio", "gini"]
TRAIN_SIZES = [0.6, 0.7, 0.8, 0.9]


def run_single_experiment(
    train_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    criterion: str,
    max_depth: int = 8,
    min_samples_split: int = 50,
) -> dict[str, object]:
    """Fit a tree and return model, metrics and structural statistics."""

    X_train, y_train = split_features_target(train_frame)
    X_test, y_test = split_features_target(test_frame)

    tree = DecisionTreeClassifierScratch(
        criterion=criterion,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
    )
    started_at = time.perf_counter()
    tree.fit(X_train, y_train)
    training_time = time.perf_counter() - started_at

    predictions = tree.predict(X_test)
    metrics = calculate_binary_metrics(y_test.tolist(), predictions)
    result = {
        "criterion": criterion,
        "training_time": training_time,
        "nodes": tree.count_nodes(),
        "leaves": tree.count_leaves(),
        "depth": tree.depth(),
        **metrics.as_dict(),
        "tree": tree,
    }
    return result


def save_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def export_encoded_csv(train_frame: pd.DataFrame, test_frame: pd.DataFrame) -> None:
    """Save additional CSV files with one-hot encoded categorical attributes."""

    CSV_DIR.mkdir(parents=True, exist_ok=True)
    combined = pd.concat(
        [
            train_frame.assign(_source="train"),
            test_frame.assign(_source="test"),
        ],
        ignore_index=True,
    )
    encoded = pd.get_dummies(
        combined,
        columns=CATEGORICAL_COLUMNS,
        prefix=CATEGORICAL_COLUMNS,
        dtype=int,
    )
    train_encoded = encoded[encoded["_source"] == "train"].drop(columns=["_source"])
    test_encoded = encoded[encoded["_source"] == "test"].drop(columns=["_source"])
    ordered_columns = [column for column in train_encoded.columns if column != TARGET_COLUMN] + [TARGET_COLUMN]
    train_encoded[ordered_columns].to_csv(CSV_DIR / "adult_train_encoded.csv", index=False)
    test_encoded[ordered_columns].to_csv(CSV_DIR / "adult_test_encoded.csv", index=False)


def print_table(title: str, rows: list[dict[str, object]], columns: list[str]) -> None:
    print(f"\n{title}")
    widths = {column: max(len(column), *(len(format_value(row.get(column, ''))) for row in rows)) for column in columns}
    header = " | ".join(column.ljust(widths[column]) for column in columns)
    print(header)
    print("-" * len(header))
    for row in rows:
        print(" | ".join(format_value(row.get(column, "")).ljust(widths[column]) for column in columns))


def format_value(value: object) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    TREES_DIR.mkdir(parents=True, exist_ok=True)
    CSV_DIR.mkdir(parents=True, exist_ok=True)

    train_official = load_adult_dataset(DATA_DIR / "adult.data", is_test=False)
    test_official = load_adult_dataset(DATA_DIR / "adult.test", is_test=True)

    export_clean_csv(train_official, test_official, CSV_DIR)
    export_encoded_csv(train_official, test_official)

    criteria_rows: list[dict[str, object]] = []
    for criterion in CRITERIA:
        result = run_single_experiment(train_official, test_official, criterion=criterion)
        tree = result.pop("tree")
        visualize_tree(
            tree.root_,
            TREES_DIR / f"{criterion}.png",
            max_depth=4,
            title=f"{criterion}: first 4 levels only",
        )
        criteria_rows.append(result)

    save_csv(
        RESULTS_DIR / "criteria_metrics.csv",
        criteria_rows,
        [
            "criterion",
            "training_time",
            "nodes",
            "leaves",
            "depth",
            "accuracy",
            "precision",
            "recall",
            "f1",
            "tp",
            "tn",
            "fp",
            "fn",
        ],
    )

    split_rows: list[dict[str, object]] = []
    for train_size in TRAIN_SIZES:
        train_frame, test_frame = stratified_train_test_split(
            train_official,
            train_size=train_size,
            random_state=42,
        )
        result = run_single_experiment(train_frame, test_frame, criterion="information_gain")
        result.pop("tree")
        split_rows.append(
            {
                "train_size": train_size,
                "test_size": round(1.0 - train_size, 1),
                **result,
            }
        )

    save_csv(
        RESULTS_DIR / "metrics.csv",
        split_rows,
        [
            "train_size",
            "test_size",
            "accuracy",
            "precision",
            "recall",
            "f1",
            "training_time",
            "nodes",
            "leaves",
            "depth",
            "tp",
            "tn",
            "fp",
            "fn",
            "criterion",
        ],
    )
    plot_metrics(split_rows, RESULTS_DIR / "metrics.png")

    print_table(
        "Experiment 1: official adult.test",
        criteria_rows,
        ["criterion", "training_time", "nodes", "leaves", "depth", "accuracy", "precision", "recall", "f1"],
    )
    print_table(
        "Experiment 2: stratified splits from adult.data",
        split_rows,
        ["train_size", "test_size", "training_time", "accuracy", "precision", "recall", "f1"],
    )
    print("\nSaved:")
    print(f"- {RESULTS_DIR / 'criteria_metrics.csv'}")
    print(f"- {RESULTS_DIR / 'metrics.csv'}")
    print(f"- {RESULTS_DIR / 'metrics.png'}")
    print(f"- {TREES_DIR}")
    print(f"- {CSV_DIR} (cleaned and one-hot encoded CSV files)")


if __name__ == "__main__":
    main()
