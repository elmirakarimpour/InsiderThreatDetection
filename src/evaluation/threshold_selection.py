"""
threshold_selection.py

مسئولیت: تبدیل یک عدد anomaly score پیوسته به یک تصمیم قطعی «تهدید» یا
«عادی»، با انتخاب یک نقطه‌ی برش (threshold).

قانون طلایی این فایل: threshold همیشه فقط از روی امتیازهای validation
انتخاب می‌شود، هرگز از روی test. اگر از روی test انتخاب می‌شد، یعنی
داریم از جواب امتحان برای طراحی سوال امتحان استفاده می‌کنیم — یک نوع
نشت اطلاعات (leakage) که اعتبار علمی نتیجه را زیر سوال می‌برد.
"""

import numpy as np


def select_threshold(validation_scores: np.ndarray, method: str = "quantile", quantile: float = 0.995, mad_multiplier: float = 3.5):
    """
    یک آستانه‌ی تصمیم‌گیری از روی امتیازهای validation انتخاب می‌کند
    (بدون استفاده از هیچ لیبلی — چون این آستانه باید در دنیای واقعی هم
    قابل‌محاسبه باشد، جایی که لیبل واقعی معمولاً در دسترس نیست).

        "quantile" -> فقط بالاترین (1-quantile) درصد امتیازها را «مشکوک» علامت می‌زند
        "mad"      -> میانه + چند برابر انحراف مطلق از میانه (به outlier حساس نیست)
    """
    scores = np.asarray(validation_scores, dtype=float)
    scores = scores[np.isfinite(scores)]
    if scores.size == 0:
        raise ValueError("validation_scores باید حداقل یک مقدار معتبر داشته باشد.")

    if method == "quantile":
        if not 0.0 < quantile < 1.0:
            raise ValueError("quantile باید بین صفر و یک باشد.")
        return float(np.quantile(scores, quantile))

    if method == "mad":
        median_score = float(np.median(scores))
        robust_scale = 1.4826 * float(np.median(np.abs(scores - median_score)))
        return median_score + mad_multiplier * robust_scale

    raise ValueError(f"روش نامعتبر: '{method}'. باید 'quantile' یا 'mad' باشد.")
