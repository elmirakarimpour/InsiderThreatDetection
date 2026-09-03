"""
test_hyperparameter_config.py
"""

import numpy as np
import pytest

from src.training.hyperparameter_config import run_hyperparameter_sweep


def _make_tiny_data():
    rng = np.random.default_rng(0)
    X_train = rng.random((30, 7, 4)).astype(np.float32)
    X_val = rng.random((40, 7, 4)).astype(np.float32)
    y_val = np.zeros(40, dtype=np.int8)
    # چند نمونه‌ی مثبت پخش‌شده تا بعد از تقسیم به دو نیمه، هر نیمه حداقل یکی داشته باشد
    y_val[[3, 10, 18, 25, 30, 37]] = 1
    return X_train, X_val, y_val


def test_sweep_returns_one_result_per_config_sorted_by_f1():
    X_train, X_val, y_val = _make_tiny_data()
    tiny_search_space = [
        {"gamma": 0.1, "pooling_strategy": "mean", "n_clusters": 2},
        {"gamma": 0.5, "pooling_strategy": "max", "n_clusters": 2},
    ]
    results = run_hyperparameter_sweep(
        X_train, X_val, y_val,
        search_space=tiny_search_space,
        fixed_kwargs=dict(encoder_size=8, latent_size=4, pretrain_epochs=1, fine_tune_epochs=1, batch_size=8),
        seed=1,
    )

    assert len(results) == 2
    assert results[0]["f1"] >= results[1]["f1"]  # نزولی بر اساس F1
    assert all("roc_auc" in r and "f1" in r and "threshold" in r for r in results)


def test_sweep_raises_error_when_no_positive_validation_samples():
    X_train, X_val, _ = _make_tiny_data()
    y_val_all_zero = np.zeros(40, dtype=np.int8)
    with pytest.raises(ValueError):
        run_hyperparameter_sweep(X_train, X_val, y_val_all_zero)
