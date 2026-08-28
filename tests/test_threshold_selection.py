"""
test_threshold_selection.py
"""

import numpy as np
import pytest

from src.evaluation.threshold_selection import select_threshold


def test_quantile_threshold_flags_expected_top_fraction():
    # ۱۰۰۰ امتیاز یکنواخت بین ۰ و ۱؛ آستانه‌ی ۰.۹۹ باید تقریباً روی ۰.۹۹ بیفتد
    scores = np.linspace(0, 1, 1000)
    threshold = select_threshold(scores, method="quantile", quantile=0.99)
    assert 0.97 < threshold <= 1.0


def test_mad_threshold_is_robust_to_extreme_outliers():
    """چند outlier خیلی بزرگ نباید threshold مبتنی بر MAD را بی‌اندازه جابه‌جا کند."""
    normal_scores = np.random.default_rng(0).normal(loc=0.05, scale=0.01, size=1000)
    scores_with_outliers = np.concatenate([normal_scores, [50.0, 60.0, 70.0]])

    threshold = select_threshold(scores_with_outliers, method="mad", mad_multiplier=3.5)
    # چون MAD مقاوم است، آستانه باید نزدیک توزیع اصلی بماند، نه به‌سمت outlierها کشیده شود
    assert threshold < 1.0


def test_invalid_method_raises_error():
    with pytest.raises(ValueError):
        select_threshold(np.array([0.1, 0.2, 0.3]), method="not_a_real_method")


def test_invalid_quantile_raises_error():
    with pytest.raises(ValueError):
        select_threshold(np.array([0.1, 0.2, 0.3]), method="quantile", quantile=1.5)


def test_empty_scores_raise_error():
    with pytest.raises(ValueError):
        select_threshold(np.array([]))
