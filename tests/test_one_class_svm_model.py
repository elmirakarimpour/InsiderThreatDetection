"""
test_one_class_svm_model.py
"""

import numpy as np
import pytest

from src.model.one_class_svm_model import OneClassSVMModel


def test_score_before_fit_raises_error():
    model = OneClassSVMModel()
    with pytest.raises(RuntimeError):
        model.score(np.zeros((2, 7, 3)))


def test_fit_score_pipeline_returns_expected_shape():
    rng = np.random.default_rng(42)
    train_sequences = rng.normal(loc=0.0, scale=1.0, size=(150, 7, 5))
    eval_sequences = rng.normal(loc=0.0, scale=1.0, size=(10, 7, 5))

    model = OneClassSVMModel(nu=0.05)
    scores = model.fit_score(train_sequences, eval_sequences)

    assert scores.shape == (10,)
    assert np.isfinite(scores).all()


def test_obvious_outlier_gets_higher_score_than_normal_data():
    rng = np.random.default_rng(0)
    train_sequences = rng.normal(loc=0.0, scale=1.0, size=(200, 7, 4))

    normal_sample = rng.normal(loc=0.0, scale=1.0, size=(1, 7, 4))
    outlier_sample = np.full((1, 7, 4), 50.0)
    eval_sequences = np.concatenate([normal_sample, outlier_sample], axis=0)

    model = OneClassSVMModel(nu=0.05)
    scores = model.fit_score(train_sequences, eval_sequences)

    assert scores[1] > scores[0]
