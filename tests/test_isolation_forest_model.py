"""
test_isolation_forest_model.py
"""

import numpy as np
import pytest

from src.model.isolation_forest_model import IsolationForestModel


def test_flatten_converts_3d_sequences_to_2d():
    sequences = np.zeros((10, 7, 4))  # (batch, time, features)
    flattened = IsolationForestModel._flatten(sequences)
    assert flattened.shape == (10, 28)  # 7 * 4 = 28


def test_score_before_fit_raises_error():
    model = IsolationForestModel()
    with pytest.raises(RuntimeError):
        model.score(np.zeros((2, 7, 3)))


def test_fit_score_pipeline_returns_expected_shape():
    rng = np.random.default_rng(42)
    train_sequences = rng.normal(loc=0.0, scale=1.0, size=(200, 7, 5))
    eval_sequences = rng.normal(loc=0.0, scale=1.0, size=(10, 7, 5))

    model = IsolationForestModel(n_estimators=50, random_state=42)
    scores = model.fit_score(train_sequences, eval_sequences)

    assert scores.shape == (10,)
    assert np.isfinite(scores).all()


def test_obvious_outlier_gets_higher_score_than_normal_data():
    """
    داده‌ی train همه حول صفر است. یک نمونه‌ی eval که مقادیرش خیلی بزرگ
    است (مثلاً ۵۰) باید امتیاز ناهنجاری بیشتری از یک نمونه‌ی عادی بگیرد.
    """
    rng = np.random.default_rng(0)
    train_sequences = rng.normal(loc=0.0, scale=1.0, size=(300, 7, 4))

    normal_sample = rng.normal(loc=0.0, scale=1.0, size=(1, 7, 4))
    outlier_sample = np.full((1, 7, 4), 50.0)
    eval_sequences = np.concatenate([normal_sample, outlier_sample], axis=0)

    model = IsolationForestModel(n_estimators=100, random_state=42)
    scores = model.fit_score(train_sequences, eval_sequences)

    assert scores[1] > scores[0]  # outlier (index=1) باید امتیاز بالاتری بگیرد
