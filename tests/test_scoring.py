"""
test_scoring.py

مهم‌ترین تست این فایل (test_max_pooling_detects_single_day_spike_that_mean_pooling_misses)
دقیقاً همان فرضیه‌ای را که چند مرحله قبل درباره‌ی رقیق‌شدن سیگنال مطرح
شد، با یک مثال عددی ساده ثابت می‌کند.
"""

import numpy as np
import pytest

from src.evaluation.scoring import compute_per_timestep_error, pool_temporal_errors


def test_per_timestep_error_averages_only_over_feature_axis():
    sequences = np.zeros((2, 3, 4))  # (batch=2, time=3, features=4)
    reconstructed = np.zeros((2, 3, 4))
    reconstructed[0, 1, :] = 1.0  # فقط روز دوم از پنجره‌ی اول کاملاً اشتباه بازسازی شده

    result = compute_per_timestep_error(sequences, reconstructed)
    assert result.shape == (2, 3)  # محور فیچر باید حذف شده باشد، محور زمان بماند
    assert result[0, 1] == pytest.approx(1.0)
    assert result[0, 0] == pytest.approx(0.0)


def test_mean_pooling_dilutes_single_day_spike():
    """
    یک پنجره با ۶ روز کاملاً عادی (خطای ۰.۰۱) و ۱ روز خیلی مشکوک (خطای ۱.۰).
    mean-pooling باید این سیگنال تیز را به شدت کاهش دهد.
    """
    per_timestep_error = np.array([[0.01, 0.01, 0.01, 1.0, 0.01, 0.01, 0.01]])
    mean_score = pool_temporal_errors(per_timestep_error, strategy="mean")
    assert mean_score[0] < 0.2  # سیگنال تیز (۱.۰) در میانگین ۷ روزه تقریباً گم می‌شود


def test_max_pooling_detects_single_day_spike_that_mean_pooling_misses():
    """
    تست اصلی: نشان می‌دهد max-pooling دقیقاً همان روز مشکوک را حفظ می‌کند،
    در حالی که mean-pooling آن را رقیق می‌کند — همان یافته‌ی جلسه‌ی قبل.
    """
    per_timestep_error = np.array([[0.01, 0.01, 0.01, 1.0, 0.01, 0.01, 0.01]])

    mean_score = pool_temporal_errors(per_timestep_error, strategy="mean")
    max_score = pool_temporal_errors(per_timestep_error, strategy="max")

    assert max_score[0] == pytest.approx(1.0)
    assert max_score[0] > mean_score[0] * 4  # اختلاف بزرگ و معنادار بین دو استراتژی


def test_topk_pooling_is_between_mean_and_max():
    per_timestep_error = np.array([[0.01, 0.01, 0.01, 1.0, 0.01, 0.01, 0.01]])

    mean_score = pool_temporal_errors(per_timestep_error, strategy="mean")[0]
    max_score = pool_temporal_errors(per_timestep_error, strategy="max")[0]
    topk_score = pool_temporal_errors(per_timestep_error, strategy="topk", top_k=2)[0]

    assert mean_score < topk_score < max_score


def test_invalid_strategy_raises_error():
    per_timestep_error = np.zeros((1, 7))
    with pytest.raises(ValueError):
        pool_temporal_errors(per_timestep_error, strategy="not_a_real_strategy")
