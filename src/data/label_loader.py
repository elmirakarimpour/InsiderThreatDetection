"""
label_loader.py

مسئولیت: خواندن insiders.csv (که بازه‌ی زمانی دقیق هر رخداد تهدید
داخلی واقعی را مشخص می‌کند) و تبدیل آن به یک برچسب ۰/۱ به‌ازای هر
«کاربر-روز» — دقیقاً هم‌طول با daily_profiles.

یادآوری قانون طلایی پروژه: خروجی این تابع (eval_only_label) هرگز
نباید به‌عنوان یکی از feature_columns وارد preprocessing یا مدل شود.
این برچسب فقط برای مرحله‌ی ارزیابی (بعد از training) استفاده می‌شود.
"""

import numpy as np
import pandas as pd


def attach_ground_truth_labels(
    daily_profiles: pd.DataFrame,
    insiders_file_path: str,
    dataset_version: str = "5.2",
    user_col: str = "user",
    date_col: str = "date_only",
) -> np.ndarray:
    """
    برای هر سطر daily_profiles (یک کاربر در یک روز)، اگر آن تاریخ داخل
    بازه‌ی زمانی یک رخداد ثبت‌شده در insiders.csv برای همان کاربر باشد،
    برچسب ۱ می‌گیرد، وگرنه ۰.

    خروجی: آرایه‌ی numpy به طول daily_profiles — قرار است در قدم بعد
    به یک ستون به اسم eval_only_label تبدیل شود (نه یک ستون فیچر).
    """
    insiders_data = pd.read_csv(insiders_file_path)
    required_columns = {"dataset", user_col, "start", "end"}
    missing = required_columns.difference(insiders_data.columns)
    if missing:
        raise ValueError(f"فایل insiders.csv این ستون‌ها را کم دارد: {sorted(missing)}")

    dataset_mask = insiders_data["dataset"].astype(str).str.contains(str(dataset_version), regex=False, na=False)
    relevant_intervals = insiders_data.loc[dataset_mask].copy()

    for column in ["start", "end"]:
        relevant_intervals[column] = pd.to_datetime(relevant_intervals[column], errors="coerce").dt.normalize()
    relevant_intervals = relevant_intervals.dropna(subset=["start", "end"])

    profile_dates = pd.to_datetime(daily_profiles[date_col], errors="coerce").dt.normalize()
    labels = pd.Series(0, index=daily_profiles.index, dtype=np.int8)

    for interval in relevant_intervals.itertuples(index=False):
        is_threat_day = (
            daily_profiles[user_col].eq(getattr(interval, user_col))
            & profile_dates.ge(interval.start)
            & profile_dates.le(interval.end)
        )
        labels.loc[is_threat_day] = 1

    return labels.to_numpy(dtype=np.int8)
