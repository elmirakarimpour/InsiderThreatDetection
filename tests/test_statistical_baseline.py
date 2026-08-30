"""
test_statistical_baseline.py
"""

import numpy as np

from src.model.statistical_baseline import StatisticalBaselineModel


def test_fit_score_pipeline_runs_without_error():
    rng = np.random.default_rng(42)
    train_sequences = rng.normal(loc=0.0, scale=1.0, size=(50, 7, 4)).astype(np.float32)
    eval_sequences = rng.normal(loc=0.0, scale=1.0, size=(5, 7, 4)).astype(np.float32)

    model = StatisticalBaselineModel()
    scores = model.fit_score(train_sequences, eval_sequences)

    assert scores.shape == (5,)
    assert np.isfinite(scores).all()


def test_score_before_fit_raises_error():
    model = StatisticalBaselineModel()
    try:
        model.score(np.zeros((2, 7, 3), dtype=np.float32))
        assert False, "انتظار می‌رفت RuntimeError بدهد"
    except RuntimeError:
        pass


def test_sample_far_from_training_mean_gets_higher_score():
    rng = np.random.default_rng(0)
    train_sequences = rng.normal(loc=0.0, scale=1.0, size=(100, 7, 4)).astype(np.float32)

    normal_sample = rng.normal(loc=0.0, scale=1.0, size=(1, 7, 4)).astype(np.float32)
    outlier_sample = np.full((1, 7, 4), 50.0, dtype=np.float32)
    eval_sequences = np.concatenate([normal_sample, outlier_sample], axis=0)

    model = StatisticalBaselineModel()
    scores = model.fit_score(train_sequences, eval_sequences)

    assert scores[1] > scores[0]
