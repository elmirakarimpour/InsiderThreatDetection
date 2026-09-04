"""
preprocessing.py

مسئولیت: تبدیل ستون‌های خام پروفایل روزانه (اعداد پیوسته + اعداد
دودویی + دسته‌ای مثل دپارتمان) به یک ماتریس عددی آماده برای مدل.

قانون طلایی ضدـ leakage در این فایل:
    fit فقط یک‌بار و فقط روی داده‌ی TRAIN صدا زده می‌شود.
    transform روی train، validation، و test هرکدام جداگانه صدا زده می‌شود.
اگر این دو کار را در یک تابع ادغام کنیم، ریسک اینکه یک روز به‌اشتباه
روی کل داده (شامل validation/test) fit شود بیشتر می‌شود — برای همین
عمداً در قالب یک کلاس با دو متد جدا نوشته شده، نه یک تابع.
"""

import numpy as np
from sklearn.preprocessing import OneHotEncoder, RobustScaler


def classify_feature_columns(training_df, candidate_columns, categorical_columns):
    """
    ستون‌های نامزد را بر اساس آمار داده‌ی TRAIN (نه کل داده) به سه گروه تقسیم می‌کند:
        - continuous: بیش از ۲ مقدار یکتا دارند (مثل logon_total)
        - binary: فقط ۰ یا ۱ هستند (مثل is_usb_transfer)
        - categorical: رشته‌ای هستند (مثل department)

    چرا این تصمیم‌گیری هم فقط از train است؟ چون حتی «این ستون پیوسته
    است یا دودویی» یک تصمیم آماری است — اگر بر اساس val/test هم تصمیم
    بگیریم، یک نوع نشت اطلاعات ظریف رخ داده است.
    """
    categorical_columns = list(categorical_columns)
    numeric_candidates = [c for c in candidate_columns if c not in categorical_columns]

    continuous_columns = [
        column for column in numeric_candidates
        if training_df[column].nunique(dropna=False) > 2
    ]
    binary_columns = [column for column in numeric_candidates if column not in continuous_columns]

    return {
        "continuous": continuous_columns,
        "binary": binary_columns,
        "categorical": categorical_columns,
    }


class TabularPreprocessor:
    """
    شیء preprocessing با الگوی fit/transform (دقیقاً مثل scikit-learn).

    نحوه‌ی استفاده‌ی درست:
        preprocessor = TabularPreprocessor()
        preprocessor.fit(training_df, continuous_cols, binary_cols, categorical_cols)  # فقط روی train
        X_train, feature_names = preprocessor.transform(training_df)
        X_val, _   = preprocessor.transform(validation_df)   # از همون scaler/encoder فیت‌شده استفاده می‌کند
        X_test, _  = preprocessor.transform(test_df)
    """

    def __init__(self):
        self._continuous_columns = None
        self._binary_columns = None
        self._categorical_columns = None
        self._scaler = None
        self._encoder = None
        self._is_fitted = False

    def fit(self, training_df, continuous_columns, binary_columns, categorical_columns):
        self._continuous_columns = list(continuous_columns)
        self._binary_columns = list(binary_columns)
        self._categorical_columns = list(categorical_columns)

        # RobustScaler به‌جای StandardScaler انتخاب شده چون به outlierها حساس‌تر نیست
        # (میانه و IQR را به‌جای میانگین و انحراف‌معیار استفاده می‌کند)
        self._scaler = RobustScaler()
        self._scaler.fit(training_df[self._continuous_columns])

        self._encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype=np.float32)
        self._encoder.fit(training_df[self._categorical_columns].astype(str))

        self._is_fitted = True
        return self

    def transform(self, profile_df):
        if not self._is_fitted:
            raise RuntimeError(
                "قبل از transform باید fit صدا زده شود — و fit فقط باید روی داده‌ی train اجرا شود."
            )

        scaled_continuous = self._scaler.transform(profile_df[self._continuous_columns]).astype(np.float32)
        binary_values = profile_df[self._binary_columns].to_numpy(dtype=np.float32)
        encoded_categorical = self._encoder.transform(
            profile_df[self._categorical_columns].astype(str)
        ).astype(np.float32)

        feature_matrix = np.concatenate([scaled_continuous, binary_values, encoded_categorical], axis=1)
        feature_names = (
            self._continuous_columns
            + self._binary_columns
            + self._encoder.get_feature_names_out(self._categorical_columns).tolist()
        )
        return feature_matrix, feature_names
