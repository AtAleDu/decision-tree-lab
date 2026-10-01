"""Educational decision tree classifier implemented from scratch."""

from __future__ import annotations

from dataclasses import dataclass, field
from math import log2
from typing import Any, Literal

import numpy as np
import pandas as pd

Criterion = Literal["information_gain", "gain_ratio", "gini"]


@dataclass
class TreeNode:
    """A node of the decision tree."""

    samples_count: int
    class_counts: dict[str, int]
    predicted_class: str
    depth: int
    criterion_value: float = 0.0
    feature: str | None = None
    threshold: float | None = None
    category: Any | None = None
    is_categorical: bool = False
    left: "TreeNode | None" = None
    right: "TreeNode | None" = None

    @property
    def is_leaf(self) -> bool:
        return self.left is None and self.right is None

    def condition_text(self) -> str:
        if self.feature is None:
            return "leaf"
        if self.is_categorical:
            return f"{self.feature} == {self.category}"
        return f"{self.feature} <= {self.threshold:.4g}"


@dataclass(frozen=True)
class SplitCandidate:
    feature: str
    score: float
    left_indices: np.ndarray
    right_indices: np.ndarray
    is_categorical: bool
    threshold: float | None = None
    category: Any | None = None
    weighted_impurity: float = 0.0


class DecisionTreeClassifierScratch:
    """Binary decision tree classifier with information gain, gain ratio and gini.

    The implementation intentionally avoids sklearn.tree. Numeric attributes are
    split by a threshold; categorical attributes are split as "value vs all other
    values". For unknown categorical values during prediction, the classifier
    falls back to the majority class stored in the current node.
    """

    def __init__(
        self,
        criterion: Criterion = "information_gain",
        max_depth: int = 8,
        min_samples_split: int = 50,
        min_gain: float = 1e-9,
        max_numeric_thresholds: int | None = 128,
    ) -> None:
        if criterion not in {"information_gain", "gain_ratio", "gini"}:
            raise ValueError(f"Unsupported criterion: {criterion}")
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_gain = min_gain
        self.max_numeric_thresholds = max_numeric_thresholds
        self.root_: TreeNode | None = None
        self.feature_types_: dict[str, str] = {}
        self.classes_: list[str] = []

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "DecisionTreeClassifierScratch":
        """Build the tree from feature matrix X and target labels y."""

        if len(X) != len(y):
            raise ValueError("X and y must have the same number of rows")
        X = X.reset_index(drop=True)
        y = y.reset_index(drop=True).astype(str)
        self.classes_ = sorted(y.unique().tolist())
        self.feature_types_ = {
            column: ("numeric" if pd.api.types.is_numeric_dtype(X[column]) else "categorical")
            for column in X.columns
        }
        self.root_ = self._build_node(X, y, depth=0)
        return self

    def predict(self, X: pd.DataFrame) -> list[str]:
        """Predict class labels for each row in X."""

        if self.root_ is None:
            raise RuntimeError("Tree is not fitted")
        return [self._predict_row(row, self.root_) for _, row in X.iterrows()]

    def count_nodes(self) -> int:
        return self._count_nodes(self._require_root())

    def count_leaves(self) -> int:
        return self._count_leaves(self._require_root())

    def depth(self) -> int:
        return self._depth(self._require_root())

    def _require_root(self) -> TreeNode:
        if self.root_ is None:
            raise RuntimeError("Tree is not fitted")
        return self.root_

    def _build_node(self, X: pd.DataFrame, y: pd.Series, depth: int) -> TreeNode:
        class_counts = y.value_counts().to_dict()
        predicted_class = max(class_counts.items(), key=lambda item: (item[1], item[0]))[0]
        node = TreeNode(
            samples_count=len(y),
            class_counts={str(label): int(count) for label, count in class_counts.items()},
            predicted_class=str(predicted_class),
            depth=depth,
        )

        if (
            len(class_counts) == 1
            or depth >= self.max_depth
            or len(y) < self.min_samples_split
        ):
            return node

        split = self._find_best_split(X, y)
        if split is None or split.score <= self.min_gain:
            return node

        node.feature = split.feature
        node.threshold = split.threshold
        node.category = split.category
        node.is_categorical = split.is_categorical
        node.criterion_value = split.score
        node.left = self._build_node(X.iloc[split.left_indices].reset_index(drop=True), y.iloc[split.left_indices].reset_index(drop=True), depth + 1)
        node.right = self._build_node(X.iloc[split.right_indices].reset_index(drop=True), y.iloc[split.right_indices].reset_index(drop=True), depth + 1)
        return node

    def _find_best_split(self, X: pd.DataFrame, y: pd.Series) -> SplitCandidate | None:
        best: SplitCandidate | None = None
        for feature in X.columns:
            if self.feature_types_[feature] == "numeric":
                candidates = self._numeric_splits(X[feature], y, feature)
            else:
                candidates = self._categorical_splits(X[feature], y, feature)
            for candidate in candidates:
                if best is None or candidate.score > best.score:
                    best = candidate
        return best

    def _numeric_splits(
        self,
        values: pd.Series,
        y: pd.Series,
        feature: str,
    ) -> list[SplitCandidate]:
        raw_values = values.to_numpy(dtype=float)
        labels = y.to_numpy()
        order = np.argsort(raw_values, kind="mergesort")
        sorted_values = raw_values[order]
        sorted_labels = labels[order]

        possible_positions = np.flatnonzero(
            (sorted_values[:-1] != sorted_values[1:]) & (sorted_labels[:-1] != sorted_labels[1:])
        ) + 1
        if possible_positions.size == 0:
            return []

        if self.max_numeric_thresholds and possible_positions.size > self.max_numeric_thresholds:
            selected = np.linspace(0, possible_positions.size - 1, self.max_numeric_thresholds, dtype=int)
            possible_positions = possible_positions[selected]

        class_to_index = {label: index for index, label in enumerate(self.classes_)}
        encoded = np.array([class_to_index[label] for label in sorted_labels], dtype=int)
        one_hot = np.zeros((len(encoded), len(self.classes_)), dtype=int)
        one_hot[np.arange(len(encoded)), encoded] = 1
        cumulative_counts = np.cumsum(one_hot, axis=0)
        total_counts = cumulative_counts[-1]

        best_position: int | None = None
        best_score = -1.0
        best_weighted_impurity = 0.0
        for position in possible_positions:
            left_counts = cumulative_counts[position - 1]
            right_counts = total_counts - left_counts
            score, weighted_impurity = self._score_from_counts(left_counts, right_counts)
            if score > best_score:
                best_score = score
                best_weighted_impurity = weighted_impurity
                best_position = int(position)

        if best_position is None:
            return []

        threshold = (sorted_values[best_position - 1] + sorted_values[best_position]) / 2.0
        left_mask = raw_values <= threshold
        left_indices = np.flatnonzero(left_mask)
        right_indices = np.flatnonzero(~left_mask)
        return [
            SplitCandidate(
                feature=feature,
                score=float(best_score),
                left_indices=left_indices,
                right_indices=right_indices,
                is_categorical=False,
                threshold=float(threshold),
                weighted_impurity=float(best_weighted_impurity),
            )
        ]

    def _categorical_splits(
        self,
        values: pd.Series,
        y: pd.Series,
        feature: str,
    ) -> list[SplitCandidate]:
        best_category: Any | None = None
        best_score = -1.0
        best_weighted_impurity = 0.0
        unique_values = values.dropna().unique()
        raw_values = values.to_numpy()
        total_counts = np.array([(y == label).sum() for label in self.classes_], dtype=int)
        for category in unique_values:
            left_mask = raw_values == category
            left_count = int(left_mask.sum())
            if left_count == 0 or left_count == len(values):
                continue
            left_counts = np.array([((y == label) & left_mask).sum() for label in self.classes_], dtype=int)
            right_counts = total_counts - left_counts
            score, weighted_impurity = self._score_from_counts(left_counts, right_counts)
            if score > best_score:
                best_score = score
                best_weighted_impurity = weighted_impurity
                best_category = category

        if best_category is None:
            return []

        left_mask = raw_values == best_category
        left_indices = np.flatnonzero(left_mask)
        right_indices = np.flatnonzero(~left_mask)
        return [
            SplitCandidate(
                feature=feature,
                score=float(best_score),
                left_indices=left_indices,
                right_indices=right_indices,
                is_categorical=True,
                category=best_category,
                weighted_impurity=float(best_weighted_impurity),
            )
        ]

    def _score_from_counts(self, left_counts: np.ndarray, right_counts: np.ndarray) -> tuple[float, float]:
        parent_counts = left_counts + right_counts
        left_total = int(left_counts.sum())
        right_total = int(right_counts.sum())
        parent_total = left_total + right_total
        left_weight = left_total / parent_total
        right_weight = right_total / parent_total

        if self.criterion == "gini":
            parent_impurity = self._gini_from_counts(parent_counts)
            weighted_impurity = left_weight * self._gini_from_counts(left_counts) + right_weight * self._gini_from_counts(right_counts)
            # We maximize Gini gain: parent impurity minus weighted child impurity.
            return parent_impurity - weighted_impurity, weighted_impurity

        parent_entropy = self._entropy_from_counts(parent_counts)
        weighted_impurity = left_weight * self._entropy_from_counts(left_counts) + right_weight * self._entropy_from_counts(right_counts)
        information_gain = parent_entropy - weighted_impurity
        if self.criterion == "gain_ratio":
            split_information = self._split_information([left_total, right_total])
            return (information_gain / split_information if split_information else 0.0), weighted_impurity
        return information_gain, weighted_impurity

    def _predict_row(self, row: pd.Series, node: TreeNode) -> str:
        while not node.is_leaf:
            assert node.feature is not None
            value = row[node.feature]
            if node.is_categorical:
                # Unknown categories are handled by staying with the node majority.
                if value == node.category:
                    next_node = node.left
                else:
                    next_node = node.right
            else:
                next_node = node.left if float(value) <= float(node.threshold) else node.right
            if next_node is None:
                return node.predicted_class
            node = next_node
        return node.predicted_class

    @staticmethod
    def _entropy(labels: np.ndarray) -> float:
        _, counts = np.unique(labels, return_counts=True)
        return DecisionTreeClassifierScratch._entropy_from_counts(counts)

    @staticmethod
    def _gini(labels: np.ndarray) -> float:
        _, counts = np.unique(labels, return_counts=True)
        return DecisionTreeClassifierScratch._gini_from_counts(counts)

    @staticmethod
    def _entropy_from_counts(counts: np.ndarray) -> float:
        total = counts.sum()
        if total == 0:
            return 0.0
        probabilities = counts / total
        return float(-np.sum([p * log2(p) for p in probabilities if p > 0]))

    @staticmethod
    def _gini_from_counts(counts: np.ndarray) -> float:
        total = counts.sum()
        if total == 0:
            return 0.0
        probabilities = counts / counts.sum()
        return float(1.0 - np.sum(probabilities**2))

    @staticmethod
    def _split_information(sizes: list[int]) -> float:
        total = sum(sizes)
        result = 0.0
        for size in sizes:
            if size:
                probability = size / total
                result -= probability * log2(probability)
        return result

    def _count_nodes(self, node: TreeNode) -> int:
        return 1 + (self._count_nodes(node.left) if node.left else 0) + (self._count_nodes(node.right) if node.right else 0)

    def _count_leaves(self, node: TreeNode) -> int:
        if node.is_leaf:
            return 1
        return (self._count_leaves(node.left) if node.left else 0) + (self._count_leaves(node.right) if node.right else 0)

    def _depth(self, node: TreeNode) -> int:
        if node.is_leaf:
            return node.depth
        return max(self._depth(child) for child in [node.left, node.right] if child is not None)
