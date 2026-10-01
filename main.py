"""Command line interface for fitting a scratch decision tree on Adult data."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from src.data_loader import (
    load_adult_dataset,
    split_features_target,
    stratified_train_test_split,
)
from src.decision_tree import DecisionTreeClassifierScratch
from src.metrics import calculate_binary_metrics
from src.visualization import visualize_tree


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Decision tree classification for UCI Adult dataset.")
    parser.add_argument("--data", default="data/adult.data", help="Path to adult.data")
    parser.add_argument("--test-data", default=None, help="Optional path to official adult.test")
    parser.add_argument(
        "--criterion",
        default="information_gain",
        choices=["information_gain", "gain_ratio", "gini"],
        help="Split criterion",
    )
    parser.add_argument("--train-size", type=float, default=0.8, help="Train share when --test-data is not provided")
    parser.add_argument("--max-depth", type=int, default=8, help="Maximum tree depth")
    parser.add_argument("--min-samples-split", type=int, default=50, help="Minimum samples required to split a node")
    parser.add_argument("--random-state", type=int, default=42, help="Seed for reproducible split")
    parser.add_argument("--tree-image", default="results/trees/tree.png", help="Where to save tree visualization")
    parser.add_argument("--visualization-depth", type=int, default=4, help="How many tree levels to draw")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_path = Path(args.data)
    train_frame = load_adult_dataset(data_path, is_test=False)

    if args.test_data:
        test_frame = load_adult_dataset(args.test_data, is_test=True)
    else:
        train_frame, test_frame = stratified_train_test_split(
            train_frame,
            train_size=args.train_size,
            random_state=args.random_state,
        )

    X_train, y_train = split_features_target(train_frame)
    X_test, y_test = split_features_target(test_frame)

    tree = DecisionTreeClassifierScratch(
        criterion=args.criterion,
        max_depth=args.max_depth,
        min_samples_split=args.min_samples_split,
    )

    started_at = time.perf_counter()
    tree.fit(X_train, y_train)
    training_time = time.perf_counter() - started_at

    predictions = tree.predict(X_test)
    metrics = calculate_binary_metrics(y_test.tolist(), predictions)

    visualize_tree(
        tree.root_,
        args.tree_image,
        max_depth=args.visualization_depth,
        title=f"{args.criterion}, first {args.visualization_depth} levels",
    )

    print(f"Criterion: {args.criterion}")
    print(f"Training time: {training_time:.3f} sec")
    print(f"Nodes: {tree.count_nodes()}, leaves: {tree.count_leaves()}, depth: {tree.depth()}")
    print(
        "Accuracy={accuracy:.4f}, Precision={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f}".format(
            **metrics.as_dict()
        )
    )
    print(f"TP={metrics.tp}, TN={metrics.tn}, FP={metrics.fp}, FN={metrics.fn}")
    print(f"Tree image: {args.tree_image}")


if __name__ == "__main__":
    main()
