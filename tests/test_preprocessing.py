"""
test_preprocessing.py
"""

import numpy as np
import pandas as pd
import pytest

from src.data.preprocessing import TabularPreprocessor, classify_feature_columns


def _make_fake_training_df():
    return pd.DataFrame({
        "logon_total": [1, 5, 9, 3, 7],       # پیوسته: بیش از ۲ مقدار یکتا
        "is_usb_transfer": [0, 1, 0, 0, 1],   # دودویی: فقط ۰ و ۱
        "department": ["it", "hr", "it", "finance", "hr"],
    })


def test_classify_feature_columns_splits_continuous_and_binary_correctly():
    training_df = _make_fake_training_df()
    result = classify_feature_columns(
        training_df,
        candidate_columns=["logon_total", "is_usb_transfer", "department"],
        categorical_columns=["department"],
    )
    assert result["continuous"] == ["logon_total"]
    assert result["binary"] == ["is_usb_transfer"]
    assert result["categorical"] == ["department"]


def test_transform_before_fit_raises_error():
    preprocessor = TabularPreprocessor()
    with pytest.raises(RuntimeError):
        preprocessor.transform(_make_fake_training_df())


def test_transform_output_shape_matches_expected_columns():
    training_df = _make_fake_training_df()
    preprocessor = TabularPreprocessor().fit(
        training_df,
        continuous_columns=["logon_total"],
        binary_columns=["is_usb_transfer"],
        categorical_columns=["department"],
    )
    feature_matrix, feature_names = preprocessor.transform(training_df)
    # ۱ ستون پیوسته + ۱ ستون دودویی + ۳ دسته‌ی department (it, hr, finance) = ۵ ستون
    assert feature_matrix.shape == (5, 5)
    assert len(feature_names) == 5


def test_scaler_uses_only_train_statistics_no_leakage():
    """
    تست حیاتی ضد-leakage: preprocessor فقط روی train فیت می‌شود.
    وقتی روی داده‌ی test (که مقداری کاملاً خارج از رنج train دارد)
    transform می‌کنیم، نباید هیچ خطایی بدهد و نباید مقیاس‌بندی از نو
    بر اساس test محاسبه شود — باید همان مقیاس train را به کار ببرد.
    """
    training_df = _make_fake_training_df()
    preprocessor = TabularPreprocessor().fit(
        training_df,
        continuous_columns=["logon_total"],
        binary_columns=["is_usb_transfer"],
        categorical_columns=["department"],
    )

    # داده‌ی تست با یک مقدار خیلی بزرگ‌تر از هرچیزی که train دیده بود
    test_df = pd.DataFrame({
        "logon_total": [500],
        "is_usb_transfer": [1],
        "department": ["it"],
    })
    test_matrix, _ = preprocessor.transform(test_df)

    # اگر مقیاس‌بندی دوباره روی test فیت می‌شد، این مقدار همیشه ۰ می‌بود
    # (چون تنها مقدار موجود است). چون از مقیاس train استفاده می‌کنیم،
    # باید یک عدد بزرگ و غیرصفر باشد.
    assert test_matrix[0, 0] != 0.0


def test_unseen_categorical_value_does_not_crash():
    """اگر دپارتمانی در test دیده شود که در train وجود نداشت، نباید کرش کند (handle_unknown='ignore')."""
    training_df = _make_fake_training_df()
    preprocessor = TabularPreprocessor().fit(
        training_df,
        continuous_columns=["logon_total"],
        binary_columns=["is_usb_transfer"],
        categorical_columns=["department"],
    )
    test_df = pd.DataFrame({
        "logon_total": [4],
        "is_usb_transfer": [0],
        "department": ["legal"],  # این دپارتمان در train نبود
    })
    feature_matrix, _ = preprocessor.transform(test_df)
    assert feature_matrix.shape == (1, 5)
