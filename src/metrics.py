"""Binary classification metrics implemented without sklearn."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


POSITIVE_CLASS = ">50K"


@dataclass(frozen=True)
class BinaryMetrics:
    """Container for confusion matrix values and derived metrics."""

    accuracy: float
    precision: float
    recall: float
    f1: float
    tp: int
    tn: int
    fp: int
    fn: int

    def as_dict(self) -> dict[str, float | int]:
        return {
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "tp": self.tp,
            "tn": self.tn,
            "fp": self.fp,
            "fn": self.fn,
        }


def _safe_divide(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def calculate_binary_metrics(
    y_true: Iterable[str],
    y_pred: Iterable[str],
    positive_class: str = POSITIVE_CLASS,
) -> BinaryMetrics:
    """Calculate accuracy, precision, recall, F1 and confusion matrix values."""

    tp = tn = fp = fn = 0
    for actual, predicted in zip(y_true, y_pred, strict=True):
        if actual == positive_class and predicted == positive_class:
            tp += 1
        elif actual != positive_class and predicted != positive_class:
            tn += 1
        elif actual != positive_class and predicted == positive_class:
            fp += 1
        elif actual == positive_class and predicted != positive_class:
            fn += 1

    total = tp + tn + fp + fn
    accuracy = _safe_divide(tp + tn, total)
    precision = _safe_divide(tp, tp + fp)
    recall = _safe_divide(tp, tp + fn)
    f1 = _safe_divide(2 * precision * recall, precision + recall)

    return BinaryMetrics(
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1=f1,
        tp=tp,
        tn=tn,
        fp=fp,
        fn=fn,
    )
