"""
splitting.py

مسئولیت: تقسیم داده به train/validation/test بر اساس زمان، نه رندوم.

چرا نه رندوم؟ چون داده‌ی ما سری زمانیه. اگه رندوم تقسیم کنیم، ممکنه
یه پنجره از هفته‌ی ۱۰ تو train باشه ولی پنجره‌ی همون کاربر از هفته‌ی ۵
تو test بیفته — یعنی مدل موقع validation/test داره از «آینده»ی خودش
که در واقعیت هنوز اتفاق نیفتاده استفاده می‌کنه (leakage زمانی).

چرا بر اساس «تاریخ‌های یکتا» تقسیم می‌کنیم نه بر اساس «تعداد رکورد»؟
چون تعداد پنجره‌ها در هر روز یکسان نیست (بعضی روزها کاربرهای بیشتری
فعالیت دارن). اگه بر اساس تعداد رکورد ۶۰٪/۲۰٪/۲۰٪ تقسیم کنیم، ممکنه
نقطه‌ی برش دقیق روی زمان درستی نیفتد. برای همین اول تاریخ‌های یکتا را
مرتب می‌کنیم، بعد ۶۰٪ اول را train، ۲۰٪ بعدی را validation، ۲۰٪ آخر
را test در نظر می‌گیریم.
"""

import numpy as np


def build_chronological_split(dates: np.ndarray, train_fraction: float = 0.6, validation_fraction: float = 0.2):
    """
    ورودی: آرایه‌ای از تاریخ‌ها، یکی به‌ازای هر پنجره (همان window_end_date
    که از sequence_builder می‌آید).

    خروجی: دیکشنری شامل سه ماسک بولین (train_mask/validation_mask/test_mask)
    به همان طول dates، به‌علاوه‌ی دو تاریخ برش برای مستندسازی گزارش.
    """
    if not 0 < train_fraction < 1 or not 0 < validation_fraction < 1:
        raise ValueError("train_fraction و validation_fraction باید بین صفر و یک باشند.")
    if train_fraction + validation_fraction >= 1:
        raise ValueError("مجموع train_fraction و validation_fraction باید کمتر از یک باشد تا برای test جا بماند.")

    unique_dates = np.sort(np.unique(dates))
    if len(unique_dates) < 3:
        raise ValueError("برای یک تقسیم سه‌بخشی معنادار، حداقل به ۳ تاریخ یکتا نیاز است.")

    train_cutoff_index = max(int(len(unique_dates) * train_fraction) - 1, 0)
    validation_cutoff_index = max(
        int(len(unique_dates) * (train_fraction + validation_fraction)) - 1,
        train_cutoff_index + 1,
    )
    validation_cutoff_index = min(validation_cutoff_index, len(unique_dates) - 1)

    train_end_date = unique_dates[train_cutoff_index]
    validation_end_date = unique_dates[validation_cutoff_index]

    train_mask = dates <= train_end_date
    validation_mask = (dates > train_end_date) & (dates <= validation_end_date)
    test_mask = dates > validation_end_date

    return {
        "train_mask": train_mask,
        "validation_mask": validation_mask,
        "test_mask": test_mask,
        "train_end_date": train_end_date,
        "validation_end_date": validation_end_date,
    }
