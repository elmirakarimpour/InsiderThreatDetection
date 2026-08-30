"""
test_statistical_tests.py
"""

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from src.evaluation.statistical_tests import bootstrap_confidence_interval, delong_test, mcnemar_test


def _make_labels_and_scores():
    rng = np.random.default_rng(0)
    y_true = np.concatenate([np.zeros(500), np.ones(50)]).astype(np.int8)
    return rng, y_true


def test_delong_identical_models_gives_high_p_value():
    rng, y_true = _make_labels_and_scores()
    scores = rng.random(550)
    result = delong_test(y_true, scores, scores.copy())
    assert result["p_value"] > 0.9
    assert result["auc_difference"] == pytest.approx(0.0, abs=1e-9)


def test_delong_clearly_different_models_gives_low_p_value():
    rng, y_true = _make_labels_and_scores()
    random_scores = rng.random(550)
    good_scores = y_true.astype(float) * 3 + rng.random(550) * 0.5

    result = delong_test(y_true, good_scores, random_scores)
    assert result["p_value"] < 0.01
    assert result["auc_a"] == pytest.approx(roc_auc_score(y_true, good_scores), abs=1e-6)
    assert result["auc_b"] == pytest.approx(roc_auc_score(y_true, random_scores), abs=1e-6)


def test_delong_requires_both_classes():
    with pytest.raises(ValueError):
        delong_test(np.zeros(10), np.random.rand(10), np.random.rand(10))


def test_bootstrap_confidence_interval_contains_point_estimate():
    rng, y_true = _make_labels_and_scores()
    scores = y_true.astype(float) * 2 + rng.random(550) * 0.5

    ci = bootstrap_confidence_interval(y_true, scores, roc_auc_score, n_bootstrap=200, seed=1)
    assert ci["lower_bound"] <= ci["point_estimate"] <= ci["upper_bound"]
    assert 0.0 <= ci["lower_bound"] <= 1.0
    assert 0.0 <= ci["upper_bound"] <= 1.0


def test_mcnemar_identical_predictions_gives_p_value_one():
    _, y_true = _make_labels_and_scores()
    predictions = (np.random.default_rng(0).random(550) > 0.5).astype(np.int8)
    result = mcnemar_test(y_true, predictions, predictions.copy())
    assert result["p_value"] == 1.0
    assert result["only_a_correct"] == 0
    assert result["only_b_correct"] == 0


def test_mcnemar_asymmetric_disagreement_gives_low_p_value():
    y_true = np.array([1] * 50 + [0] * 50, dtype=np.int8)
    # مدل A تقریباً همیشه درست، مدل B تقریباً همیشه غلط -> اختلاف نامتقارن بزرگ
    predictions_a = y_true.copy()
    predictions_b = 1 - y_true

    result = mcnemar_test(y_true, predictions_a, predictions_b)
    assert result["p_value"] < 0.01
