"""
hyperparameter_config.py

مسئولیت: امتحان‌کردن سیستماتیک چند ترکیب مختلف از hyperparameterهای
مدل اصلی (gamma، pooling_strategy، n_clusters) و رتبه‌بندی‌شان — تا
به‌جای حدس‌زدن دستی، بهترین ترکیب با شواهد انتخاب شود.

نکته‌ی مهم درباره‌ی معیار رتبه‌بندی: صرفاً ROC-AUC کافی نیست (طبق
شواهدی که در مقایسه‌ی DEC در برابر no-DEC دیدیم — AUC یکسان بود ولی
F1 در نقطه‌ی عملیاتی واقعی فرق معناداری داشت). برای همین، validation
به دو نیمه تقسیم می‌شود:
    - نیمه‌ی اول: برای انتخاب threshold با F-beta (calibration)
    - نیمه‌ی دوم: برای گزارش F1 نهایی آن پیکربندی (بدون overfit به
      همان داده‌ای که threshold از رویش انتخاب شده)

این مرحله فقط از validation استفاده می‌کند، هرگز از test.
"""

import numpy as np
from sklearn.metrics import f1_score, roc_auc_score

from src.evaluation.threshold_selection import select_threshold_by_fbeta
from src.model.bilstm_dec import BiLSTMDECModel

# ترکیب‌های پیش‌فرض برای امتحان — قابل تغییر بر اساس زمان و منابع در دسترس
DEFAULT_SEARCH_SPACE = [
    {"gamma": 0.5, "pooling_strategy": "max", "n_clusters": 3},   # پیکربندی فعلی، برای مقایسه
    {"gamma": 0.5, "pooling_strategy": "max", "n_clusters": 5},
    {"gamma": 0.5, "pooling_strategy": "max", "n_clusters": 8},
    {"gamma": 0.5, "pooling_strategy": "max", "n_clusters": 2},
    {"gamma": 0.3, "pooling_strategy": "max", "n_clusters": 5},
    {"gamma": 0.8, "pooling_strategy": "max", "n_clusters": 5},
    {"gamma": 0.5, "pooling_strategy": "topk", "n_clusters": 5, "top_k": 3},
]


def run_hyperparameter_sweep(
    X_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    search_space: list = None,
    fixed_kwargs: dict = None,
    fbeta: float = 2.0,
    seed: int = 42,
) -> list:
    """
    برای هر ترکیب، یک مدل تازه می‌سازد، روی X_train آموزش می‌دهد، و روی
    دو نیمه‌ی validation (calibration/report) ارزیابی می‌کند.

    خروجی: لیستی از دیکشنری‌ها شامل پیکربندی + roc_auc + F1 گزارش‌شده،
    مرتب‌شده از بهترین به بدترین بر اساس F1.
    """
    search_space = search_space if search_space is not None else DEFAULT_SEARCH_SPACE
    fixed_kwargs = fixed_kwargs or {}

    if y_val.sum() == 0:
        raise ValueError("y_val باید حداقل یک نمونه‌ی تهدید داشته باشد.")

    # validation را به دو نیمه تقسیم می‌کنیم: یکی برای calibration، یکی برای گزارش F1
    rng = np.random.default_rng(seed)
    shuffled_indices = rng.permutation(len(X_val))
    half = len(shuffled_indices) // 2
    calibration_indices = shuffled_indices[:half]
    report_indices = shuffled_indices[half:]

    X_val_calibration, y_val_calibration = X_val[calibration_indices], y_val[calibration_indices]
    X_val_report, y_val_report = X_val[report_indices], y_val[report_indices]

    if y_val_calibration.sum() == 0 or y_val_report.sum() == 0:
        raise ValueError("بعد از تقسیم validation به دو نیمه، حداقل یک نیمه هیچ نمونه‌ی تهدیدی ندارد.")

    results = []
    for config in search_space:
        model_kwargs = {**fixed_kwargs, **config}
        model = BiLSTMDECModel(**model_kwargs)
        model.fit(X_train)

        calibration_scores = model.score(X_val_calibration)
        report_scores = model.score(X_val_report)

        auc = roc_auc_score(y_val_report, report_scores)
        threshold = select_threshold_by_fbeta(y_val_calibration, calibration_scores, beta=fbeta)["threshold"]
        predicted = (report_scores >= threshold).astype(np.int8)
        f1 = f1_score(y_val_report, predicted, zero_division=0)

        results.append({**config, "roc_auc": float(auc), "f1": float(f1), "threshold": float(threshold)})
        print(f"[Sweep] {config} -> ROC-AUC: {auc:.4f} | F1: {f1:.4f}")

    return sorted(results, key=lambda r: r["f1"], reverse=True)
