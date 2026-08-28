"""
test_hyperparameter_config.py
"""

import numpy as np
import pytest

from src.training.hyperparameter_config import run_hyperparameter_sweep


def _make_tiny_data():
    rng = np.random.default_rng(0)
    X_train = rng.random((30, 7, 4)).astype(np.float32)
    X_val = rng.random((10, 7, 4)).astype(np.float32)
    y_val = np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 1], dtype=np.int8)
    return X_train, X_val, y_val


def test_sweep_returns_one_result_per_config_sorted_by_auc():
    X_train, X_val, y_val = _make_tiny_data()
    tiny_search_space = [
        {"gamma": 0.1, "pooling_strategy": "mean"},
        {"gamma": 0.5, "pooling_strategy": "max"},
    ]
    results = run_hyperparameter_sweep(
        X_train, X_val, y_val,
        search_space=tiny_search_space,
        fixed_kwargs=dict(n_clusters=2, encoder_size=8, latent_size=4, pretrain_epochs=1, fine_tune_epochs=1, batch_size=8),
    )

    assert len(results) == 2
    # باید نزولی مرتب شده باشد
    assert results[0]["roc_auc"] >= results[1]["roc_auc"]
    assert all("roc_auc" in r for r in results)


def test_sweep_raises_error_when_no_positive_validation_samples():
    X_train, X_val, _ = _make_tiny_data()
    y_val_all_zero = np.zeros(10, dtype=np.int8)
    with pytest.raises(ValueError):
        run_hyperparameter_sweep(X_train, X_val, y_val_all_zero)
