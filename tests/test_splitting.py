"""
test_splitting.py
"""

import numpy as np
import pytest

from src.data.splitting import build_chronological_split


def test_split_masks_cover_every_record_exactly_once():
    dates = np.arange(1, 11)  # ۱۰ تاریخ یکتا: ۱ تا ۱۰
    result = build_chronological_split(dates, train_fraction=0.6, validation_fraction=0.2)

    total_covered = result["train_mask"].sum() + result["validation_mask"].sum() + result["test_mask"].sum()
    assert total_covered == len(dates)
    # هیچ رکوردی نباید هم‌زمان در دو ماسک باشد
    assert not np.any(result["train_mask"] & result["validation_mask"])
    assert not np.any(result["validation_mask"] & result["test_mask"])


def test_split_respects_chronological_order():
    """هیچ تاریخی در train نباید از هیچ تاریخی در validation جدیدتر باشد (و همین‌طور validation نسبت به test)."""
    dates = np.arange(1, 11)
    result = build_chronological_split(dates, train_fraction=0.6, validation_fraction=0.2)

    max_train_date = dates[result["train_mask"]].max()
    min_validation_date = dates[result["validation_mask"]].min()
    max_validation_date = dates[result["validation_mask"]].max()
    min_test_date = dates[result["test_mask"]].min()

    assert max_train_date < min_validation_date
    assert max_validation_date < min_test_date


def test_split_fraction_sizes_are_approximately_correct():
    dates = np.arange(1, 101)  # ۱۰۰ تاریخ یکتا، برای چک دقیق‌تر نسبت‌ها
    result = build_chronological_split(dates, train_fraction=0.6, validation_fraction=0.2)

    assert 55 <= result["train_mask"].sum() <= 65
    assert 15 <= result["validation_mask"].sum() <= 25
    assert 15 <= result["test_mask"].sum() <= 25


def test_too_few_unique_dates_raises_error():
    with pytest.raises(ValueError):
        build_chronological_split(np.array([1, 1, 2]), train_fraction=0.6, validation_fraction=0.2)


def test_invalid_fractions_raise_error():
    dates = np.arange(1, 11)
    with pytest.raises(ValueError):
        build_chronological_split(dates, train_fraction=0.7, validation_fraction=0.5)  # مجموع > ۱
