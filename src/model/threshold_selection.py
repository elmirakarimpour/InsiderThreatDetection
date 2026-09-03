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
from sklearn.metrics import fbeta_score


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


def select_threshold_by_fbeta(
    validation_labels: np.ndarray,
    validation_scores: np.ndarray,
    beta: float = 2.0,
    n_candidates: int = 200,
) -> dict:
    """
    برخلاف select_threshold (که کاملاً بدون لیبل کار می‌کند)، این تابع
    از لیبل‌های validation برای پیدا کردن threshold-ای استفاده می‌کند
    که F-beta را روی همان validation بیشینه کند.

    نکته‌ی مهم درباره‌ی «بدون‌نظارت باقی‌ماندن»: مدل هنوز هیچ لیبلی حین
    fit نمی‌بیند — این فقط یک قدم calibration بعد از training است،
    دقیقاً مثل هر threshold-selection دیگری، فقط این‌بار هوشمندتر.

    beta > 1 یعنی وزن بیشتر به Recall نسبت به Precision داده می‌شود —
    منطقی برای تشخیص تهدید داخلی، چون هزینه‌ی از‌دست‌دادن یک تهدید واقعی
    (false negative) معمولاً بیشتر از هزینه‌ی یک هشدار اشتباه است.
    """
    validation_labels = np.asarray(validation_labels, dtype=np.int8)
    validation_scores = np.asarray(validation_scores, dtype=float)
    if len(np.unique(validation_labels)) < 2:
        raise ValueError("validation باید هم نمونه‌ی عادی و هم تهدید داشته باشد.")

    candidate_thresholds = np.quantile(validation_scores, np.linspace(0.50, 0.999, n_candidates))

    best_threshold = candidate_thresholds[0]
    best_fbeta = -1.0
    for threshold in candidate_thresholds:
        predicted = (validation_scores >= threshold).astype(np.int8)
        score = fbeta_score(validation_labels, predicted, beta=beta, zero_division=0)
        if score > best_fbeta:
            best_fbeta = score
            best_threshold = threshold

    return {"threshold": float(best_threshold), "validation_fbeta": float(best_fbeta), "beta": beta}
