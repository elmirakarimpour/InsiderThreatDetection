"""
statistical_tests.py

مسئولیت: پاسخ به دقیقاً همان چیزی که استاد خواسته بود — «اختلاف
معمولاً معنادار است» یعنی چه؟ این فایل سه ابزار آماری استاندارد
می‌دهد تا بشود با عدد (نه با حدس) گفت که آیا برتری یک مدل نسبت به
مدل دیگر واقعی است یا احتمالاً نتیجه‌ی شانس بوده.

    delong_test          -> مقایسه‌ی دو ROC-AUC از روی همان نمونه‌ها (paired)
    bootstrap_confidence_interval -> بازه‌ی اطمینان برای هر معیاری (Recall، F1 و...)
    mcnemar_test          -> مقایسه‌ی دو مجموعه تصمیم قطعی (۰/۱) از روی همان نمونه‌ها
"""

import numpy as np
from scipy import stats


def _compute_midrank(scores: np.ndarray) -> np.ndarray:
    """رتبه‌ی میانی (midrank) هر امتیاز را حساب می‌کند — پایه‌ی محاسبات DeLong."""
    sorted_indices = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[sorted_indices]
    n = len(scores)
    ranks = np.zeros(n)

    i = 0
    while i < n:
        j = i
        while j < n and sorted_scores[j] == sorted_scores[i]:
            j += 1
        ranks[i:j] = 0.5 * (i + j - 1) + 1
        i = j

    midranks = np.empty(n)
    midranks[sorted_indices] = ranks
    return midranks


def _fast_delong_auc_variance(scores_a: np.ndarray, scores_b: np.ndarray, y_true: np.ndarray):
    """
    محاسبه‌ی AUC و کوواریانس دو مدل با روش DeLong (نسخه‌ی ساده‌شده).
    scores_a و scores_b باید روی دقیقاً همان نمونه‌ها (y_true) محاسبه شده باشند.
    """
    positive_mask = y_true == 1
    negative_mask = y_true == 0
    n_positive = positive_mask.sum()
    n_negative = negative_mask.sum()

    aucs = []
    v10_all, v01_all = [], []
    for scores in (scores_a, scores_b):
        positive_scores = scores[positive_mask]
        negative_scores = scores[negative_mask]

        tx = _compute_midrank(positive_scores)
        ty = _compute_midrank(negative_scores)
        tz = _compute_midrank(scores)

        auc = (tz[positive_mask].sum() / n_positive - (n_positive + 1) / 2) / n_negative
        v01 = (tz[positive_mask] - tx) / n_negative
        v10 = 1.0 - (tz[negative_mask] - ty) / n_positive

        aucs.append(auc)
        v01_all.append(v01)
        v10_all.append(v10)

    v01_all = np.vstack(v01_all)
    v10_all = np.vstack(v10_all)
    s01 = np.cov(v01_all) / n_positive
    s10 = np.cov(v10_all) / n_negative
    covariance_matrix = s01 + s10

    return np.array(aucs), covariance_matrix


def delong_test(y_true: np.ndarray, scores_a: np.ndarray, scores_b: np.ndarray) -> dict:
    """
    تست DeLong برای مقایسه‌ی دو ROC-AUC که روی دقیقاً همان مجموعه‌ی
    نمونه‌ها (همان y_true) محاسبه شده‌اند — یعنی برای مقایسه‌ی دو مدل
    مختلف روی همان test set، نه دو دیتاست جدا.

    خروجی شامل p-value است: اگر کمتر از ۰.۰۵ باشد، اختلاف AUC دو مدل
    از نظر آماری معنادار در نظر گرفته می‌شود.
    """
    y_true = np.asarray(y_true, dtype=np.int8)
    if len(np.unique(y_true)) < 2:
        raise ValueError("برای تست DeLong، هم نمونه‌ی عادی و هم تهدید لازم است.")

    aucs, covariance_matrix = _fast_delong_auc_variance(
        np.asarray(scores_a, dtype=float), np.asarray(scores_b, dtype=float), y_true
    )
    auc_difference = aucs[0] - aucs[1]
    variance_of_difference = covariance_matrix[0, 0] + covariance_matrix[1, 1] - 2 * covariance_matrix[0, 1]

    if variance_of_difference <= 0:
        # هر دو مدل تقریباً یکسان رفتار کرده‌اند؛ اختلاف معنادار نیست
        return {"auc_a": float(aucs[0]), "auc_b": float(aucs[1]), "auc_difference": float(auc_difference), "z_score": 0.0, "p_value": 1.0}

    z_score = auc_difference / np.sqrt(variance_of_difference)
    p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))

    return {
        "auc_a": float(aucs[0]),
        "auc_b": float(aucs[1]),
        "auc_difference": float(auc_difference),
        "z_score": float(z_score),
        "p_value": float(p_value),
    }


def bootstrap_confidence_interval(
    y_true: np.ndarray,
    scores: np.ndarray,
    metric_fn,
    n_bootstrap: int = 1000,
    confidence_level: float = 0.95,
    seed: int = 42,
) -> dict:
    """
    بازه‌ی اطمینان یک معیار دلخواه (مثلاً ROC-AUC یا F1 در یک threshold
    ثابت) را با نمونه‌گیری مجدد (bootstrap) از داده تخمین می‌زند.

    metric_fn باید تابعی باشد که (y_true, scores) می‌گیرد و یک عدد برمی‌گرداند.
    """
    y_true = np.asarray(y_true, dtype=np.int8)
    scores = np.asarray(scores, dtype=float)
    rng = np.random.default_rng(seed)
    n_samples = len(y_true)

    bootstrap_estimates = []
    for _ in range(n_bootstrap):
        sample_indices = rng.integers(0, n_samples, size=n_samples)
        y_sample = y_true[sample_indices]
        scores_sample = scores[sample_indices]
        if len(np.unique(y_sample)) < 2:
            continue  # این نمونه فقط یک کلاس دارد، معیارهایی مثل AUC تعریف‌نشده می‌شوند
        bootstrap_estimates.append(metric_fn(y_sample, scores_sample))

    if not bootstrap_estimates:
        raise ValueError("هیچ نمونه‌ی bootstrap معتبری (شامل هر دو کلاس) ساخته نشد.")

    bootstrap_estimates = np.array(bootstrap_estimates)
    alpha = 1.0 - confidence_level
    lower_bound = float(np.percentile(bootstrap_estimates, 100 * alpha / 2))
    upper_bound = float(np.percentile(bootstrap_estimates, 100 * (1 - alpha / 2)))

    return {
        "point_estimate": float(metric_fn(y_true, scores)),
        "confidence_level": confidence_level,
        "lower_bound": lower_bound,
        "upper_bound": upper_bound,
        "n_bootstrap_used": len(bootstrap_estimates),
    }


def mcnemar_test(y_true: np.ndarray, predictions_a: np.ndarray, predictions_b: np.ndarray) -> dict:
    """
    تست McNemar برای مقایسه‌ی دو تصمیم قطعی (۰/۱، بعد از اعمال threshold)
    از روی همان نمونه‌ها. برخلاف DeLong (که روی امتیاز پیوسته کار می‌کند)،
    این تست دقیقاً می‌پرسد: «در مواردی که دو مدل با هم اختلاف نظر داشتند
    (یکی درست، دیگری غلط)، آیا این اختلاف به‌طور نامتقارن به نفع یک مدل
    بوده؟»
    """
    y_true = np.asarray(y_true, dtype=np.int8)
    predictions_a = np.asarray(predictions_a, dtype=np.int8)
    predictions_b = np.asarray(predictions_b, dtype=np.int8)

    correct_a = (predictions_a == y_true)
    correct_b = (predictions_b == y_true)

    # فقط مواردی که دو مدل با هم اختلاف نظر داشتند اهمیت دارند
    only_a_correct = int(np.sum(correct_a & ~correct_b))
    only_b_correct = int(np.sum(correct_b & ~correct_a))

    contingency_sum = only_a_correct + only_b_correct
    if contingency_sum == 0:
        return {"only_a_correct": 0, "only_b_correct": 0, "statistic": 0.0, "p_value": 1.0}

    # تصحیح پیوستگی (continuity correction) طبق فرم استاندارد McNemar
    statistic = (abs(only_a_correct - only_b_correct) - 1) ** 2 / contingency_sum
    p_value = 1 - stats.chi2.cdf(statistic, df=1)

    return {
        "only_a_correct": only_a_correct,
        "only_b_correct": only_b_correct,
        "statistic": float(statistic),
        "p_value": float(p_value),
    }
