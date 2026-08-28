"""
test_metrics.py
"""

import numpy as np
import pytest

from src.evaluation.metrics import compute_binary_metrics


def test_perfect_separation_gives_perfect_scores():
    y_true = np.array([0, 0, 0, 1, 1])
    scores = np.array([0.1, 0.2, 0.3, 0.9, 0.95])  # جدایی کامل بین دو کلاس
    metrics = compute_binary_metrics(y_true, scores, threshold=0.5)

    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["f1_score"] == 1.0
    assert metrics["roc_auc"] == 1.0


def test_confusion_matrix_counts_are_consistent_with_totals():
    y_true = np.array([0, 0, 1, 1])
    scores = np.array([0.1, 0.6, 0.2, 0.9])  # یک FP و یک FN عمدی
    metrics = compute_binary_metrics(y_true, scores, threshold=0.5)

    total = (
        metrics["true_negatives"] + metrics["false_positives"]
        + metrics["false_negatives"] + metrics["true_positives"]
    )
    assert total == len(y_true)
    assert metrics["false_positives"] == 1
    assert metrics["false_negatives"] == 1


def test_single_class_labels_raise_error():
    with pytest.raises(ValueError):
        compute_binary_metrics(np.array([0, 0, 0]), np.array([0.1, 0.2, 0.3]), threshold=0.5)


def test_mismatched_lengths_raise_error():
    with pytest.raises(ValueError):
        compute_binary_metrics(np.array([0, 1]), np.array([0.1, 0.2, 0.3]), threshold=0.5)
