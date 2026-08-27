"""
test_sequence_builder.py

تست‌های واحد (unit tests) برای sequence_builder.py

چرا این فایل جدا از sequence_builder.py است؟
چون در یک پروژه‌ی تمیز، کد اصلی (منطق) و کد تست باید از هم جدا باشند.
این‌طوری هروقت جای دیگری از پروژه را تغییر دادی، کافیست این فایل را
دوباره اجرا کنی تا مطمئن شوی sequence_builder.py هنوز درست کار می‌کند.

نحوه‌ی اجرا (از ریشه‌ی پروژه، توی ترمینال PyCharm):
    pytest tests/test_sequence_builder.py -v
"""

import numpy as np
import pandas as pd
import pytest

from src.data.sequence_builder import build_sequence_windows


def _make_fake_daily_profiles():
    """یک دیتافریم ساختگی کوچک می‌سازد: آلیس با ۱۰ روز (یک روز تهدید)، باب با ۵ روز."""
    return pd.DataFrame({
        "user": ["alice"] * 10 + ["bob"] * 5,
        "date_only": list(pd.date_range("2011-01-01", periods=10)) +
                     list(pd.date_range("2011-01-01", periods=5)),
        "logon_total": np.arange(15),
        "file_usb_count": np.zeros(15, dtype=int),
        # روز پنجم آلیس (index=4) عمداً تهدید علامت خورده
        "is_threat": [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    })


def test_output_shape_matches_expected_window_count():
    """آلیس ۱۰ روز دارد -> باید دقیقاً ۴ پنجره‌ی ۷روزه بسازد (10 - 7 + 1)."""
    daily_profiles = _make_fake_daily_profiles()
    result = build_sequence_windows(
        daily_profiles, feature_columns=["logon_total", "file_usb_count"], sequence_length=7,
    )
    assert result["features"].shape == (4, 7, 2)


def test_users_with_too_few_days_are_skipped():
    """باب فقط ۵ روز دارد (کمتر از ۷) -> هیچ پنجره‌ای از باب نباید ساخته شود."""
    daily_profiles = _make_fake_daily_profiles()
    result = build_sequence_windows(
        daily_profiles, feature_columns=["logon_total", "file_usb_count"], sequence_length=7,
    )
    assert "bob" not in result["window_user"]


def test_any_day_positive_labeling_rule():
    """اگر حتی یک روز از ۷ روز پنجره تهدید باشد، کل پنجره باید برچسب ۱ بگیرد."""
    daily_profiles = _make_fake_daily_profiles()
    result = build_sequence_windows(
        daily_profiles, feature_columns=["logon_total", "file_usb_count"], sequence_length=7,
    )
    # روز تهدید (index=4) داخل هر ۴ پنجره‌ی ممکن آلیس قرار می‌گیرد
    assert result["eval_only_label_any"].tolist() == [1, 1, 1, 1]


def test_full_window_label_preserves_exact_day_position():
    """label_full باید دقیقاً بگوید کدام روز از پنجره تهدید بوده، نه فقط اینکه پنجره مثبت است یا نه."""
    daily_profiles = _make_fake_daily_profiles()
    result = build_sequence_windows(
        daily_profiles, feature_columns=["logon_total", "file_usb_count"], sequence_length=7,
    )
    first_window_labels = result["eval_only_label_full"][0]
    assert first_window_labels.tolist() == [0, 0, 0, 0, 1, 0, 0]


def test_label_column_in_features_raises_error():
    """قفل ایمنی: اگر کسی به‌اشتباه ستون برچسب را در feature_columns بگذارد، باید فوراً خطا بدهد."""
    daily_profiles = _make_fake_daily_profiles()
    with pytest.raises(ValueError):
        build_sequence_windows(
            daily_profiles,
            feature_columns=["logon_total", "is_threat"],
            sequence_length=7,
        )
