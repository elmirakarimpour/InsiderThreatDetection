"""
sequence_builder.py

مسئولیت این فایل: تبدیل پروفایل‌های روزانه‌ی هر کاربر به پنجره‌های زمانی
(sequence) با طول ثابت، به‌همراه اطلاعات لازم برای split کردن و ارزیابی.

--------------------------------------------------------------------------
قانون طلایی این پروژه (رعایت اجباری در همه‌ی فایل‌ها):
مدل هرگز، تحت هیچ شرایطی، نباید ستون برچسب (label) را در ورودی خودش ببیند.
برای همین در این فایل، خروجی‌های مربوط به برچسب همیشه با پیشوند
eval_only_  شروع می‌شوند — این پیشوند یک یادآوری بصری دائمی است که
این مقادیر فقط برای مرحله‌ی ارزیابی (بعد از training) استفاده می‌شوند.
--------------------------------------------------------------------------
"""

import numpy as np
import pandas as pd


def build_sequence_windows(
    daily_profiles: pd.DataFrame,
    feature_columns: list,
    sequence_length: int = 7,
    user_col: str = "user",
    date_col: str = "date_only",
    label_col: str = "is_threat",
):
    """
    پروفایل‌های روزانه‌ی هر کاربر را به پنجره‌های sequence_length روزه تبدیل می‌کند.

    ورودی‌ها:
        daily_profiles: جدولی که هر سطرش یک "کاربر-روز" است (خروجی feature_engineering)
        feature_columns: اسم دقیق ستون‌هایی که باید وارد مدل بشن (فقط اعداد رفتاری، بدون لیبل)
        sequence_length: طول هر پنجره به روز (پیش‌فرض ۷)
        user_col, date_col, label_col: اسم ستون‌های شناسه، تاریخ، و برچسب

    خروجی: یک دیکشنری با این کلیدها:
        "features"              -> آرایه‌ی numpy شکل (N, sequence_length, n_features)
                                    این تنها چیزی است که وارد مدل می‌شود.
        "window_user"           -> شناسه‌ی کاربر هر پنجره (برای گزارش‌گیری)
        "window_end_date"       -> تاریخ آخرین روز هر پنجره (برای تقسیم Train/Val/Test)
        "eval_only_label_any"   -> ۱ اگر حداقل یک روز از ۷ روز پنجره تهدید باشد، وگرنه ۰
        "eval_only_label_full"  -> آرایه‌ی (N, sequence_length) با برچسب دقیق هر روز پنجره

    نکته‌ی مهم: label_col هرگز داخل feature_columns نباید باشد.
    این تابع این را با یک assert صریح چک می‌کند تا هرگونه اشتباه
    فوراً با خطا مشخص شود، نه در نتایج نهایی گم شود.
    """
    # قدم ۱: قفل ایمنی — جلوگیری از leakage در سطح کد، نه فقط کامنت
    if label_col in feature_columns:
        raise ValueError(
            f"'{label_col}' نباید داخل feature_columns باشد. "
            "این ستون فقط برای ارزیابی استفاده می‌شود، نه ورودی مدل."
        )

    # قدم ۲: مرتب‌سازی بر اساس کاربر و تاریخ (پیش‌نیاز اجباری sliding window)
    sorted_profiles = daily_profiles.sort_values([user_col, date_col]).reset_index(drop=True)

    all_feature_windows = []
    all_users = []
    all_end_dates = []
    all_label_any = []
    all_label_full = []

    # قدم ۳: برای هر کاربر جداگانه پنجره‌های لغزان می‌سازیم
    # (چرا جداگانه؟ چون نمی‌خواهیم آخرین روزهای کاربر A با اولین روزهای
    #  کاربر B در یک پنجره قاطی بشن — هر پنجره باید متعلق به یک کاربر باشد)
    for user_id, user_rows in sorted_profiles.groupby(user_col, sort=False):
        if len(user_rows) < sequence_length:
            continue  # این کاربر به اندازه‌ی کافی روز ثبت‌شده ندارد

        user_features = user_rows[feature_columns].to_numpy(dtype=np.float32)
        user_labels = user_rows[label_col].to_numpy(dtype=np.int8)
        user_dates = user_rows[date_col].to_numpy()

        # sliding_window_view: ترفند NumPy برای ساخت پنجره‌های لغزان بدون حلقه‌ی دستی روی هر پنجره
        feature_windows = np.lib.stride_tricks.sliding_window_view(
            user_features, window_shape=sequence_length, axis=0
        )
        # شکل خروجی sliding_window_view برای فیچرها: (num_windows, n_features, sequence_length)
        # ما شکل (num_windows, sequence_length, n_features) را می‌خواهیم، پس جابه‌جا می‌کنیم:
        feature_windows = np.swapaxes(feature_windows, 1, 2)

        label_windows = np.lib.stride_tricks.sliding_window_view(
            user_labels, window_shape=sequence_length
        )  # شکل: (num_windows, sequence_length)

        window_end_dates = user_dates[sequence_length - 1:]

        all_feature_windows.append(feature_windows)
        all_label_full.append(label_windows)
        all_label_any.append(label_windows.max(axis=1))  # هر روز تهدید=۱ کافیست
        all_end_dates.append(window_end_dates)
        all_users.append(np.full(len(window_end_dates), user_id))

    if not all_feature_windows:
        raise ValueError("هیچ کاربری به اندازه‌ی کافی روز ثبت‌شده نداشت.")

    return {
        "features": np.concatenate(all_feature_windows, axis=0),
        "window_user": np.concatenate(all_users, axis=0),
        "window_end_date": np.concatenate(all_end_dates, axis=0),
        "eval_only_label_any": np.concatenate(all_label_any, axis=0),
        "eval_only_label_full": np.concatenate(all_label_full, axis=0),
    }

