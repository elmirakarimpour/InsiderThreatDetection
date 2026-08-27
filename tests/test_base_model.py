"""
test_base_model.py

چون AnomalyDetectionModel یک کلاس انتزاعی است (نمی‌شود مستقیم ازش
شیء ساخت)، برای تستش یک مدل ساختگی و بسیار ساده (DummyMeanModel)
می‌سازیم که فقط میانگین هر توالی را به‌عنوان anomaly score برمی‌گرداند.
هدف تست این نیست که این مدل "دقیق" باشد، هدف این است که مطمئن شویم
قرارداد پایه (fit/score/fit_score) درست کار می‌کند.
"""

import numpy as np
import pytest

from src.model.base_model import AnomalyDetectionModel


class DummyMeanModel(AnomalyDetectionModel):
    """ساده‌ترین مدل ممکن: anomaly score = میانگین مقادیر توالی. فقط برای تست."""

    name = "dummy_mean_model"

    def fit(self, train_sequences):
        return self  # این مدل ساختگی چیزی یاد نمی‌گیرد، فقط قرارداد را برآورده می‌کند

    def score(self, sequences):
        return sequences.mean(axis=(1, 2))


def test_cannot_instantiate_abstract_base_directly():
    """نباید بشود مستقیماً از AnomalyDetectionModel شیء ساخت، چون متدهایش پیاده‌سازی نشده‌اند."""
    with pytest.raises(TypeError):
        AnomalyDetectionModel()


def test_concrete_subclass_fit_score_pipeline_returns_expected_shape():
    train_sequences = np.random.rand(10, 7, 3).astype(np.float32)
    eval_sequences = np.random.rand(4, 7, 3).astype(np.float32)

    model = DummyMeanModel()
    scores = model.fit_score(train_sequences, eval_sequences)

    assert scores.shape == (4,)
    assert model.name == "dummy_mean_model"
