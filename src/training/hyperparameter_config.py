"""
hyperparameter_config.py

مسئولیت: امتحان‌کردن سیستماتیک چند ترکیب مختلف از hyperparameterهای
مدل اصلی (gamma، pooling_strategy، n_clusters) روی validation، و
رتبه‌بندی‌شان بر اساس ROC-AUC — تا به‌جای حدس‌زدن دستی (مثل نسخه‌ی
قبلی پروژه که gamma=0.10 را بدون آزمایش انتخاب کرده بود)، بهترین
ترکیب با شواهد انتخاب شود.

نکته‌ی مهم: این مرحله فقط از validation استفاده می‌کند، هرگز از test —
چون test باید فقط یک‌بار، در پایان، برای گزارش نهایی دست‌نخورده بماند.
"""

import numpy as np
from sklearn.metrics import roc_auc_score

from src.model.bilstm_dec import BiLSTMDECModel

# ترکیب‌های پیش‌فرض برای امتحان — قابل تغییر بر اساس زمان و منابع در دسترس
DEFAULT_SEARCH_SPACE = [
    {"gamma": 0.1, "pooling_strategy": "mean"},   # پیکربندی نسخه‌ی قبلی پروژه، برای مقایسه
    {"gamma": 0.5, "pooling_strategy": "mean"},
    {"gamma": 0.1, "pooling_strategy": "max"},
    {"gamma": 0.5, "pooling_strategy": "max"},
    {"gamma": 1.0, "pooling_strategy": "max"},
    {"gamma": 0.5, "pooling_strategy": "topk"},
]


def run_hyperparameter_sweep(
    X_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    search_space: list = None,
    fixed_kwargs: dict = None,
) -> list:
    """
    برای هر ترکیب در search_space، یک مدل جدید می‌سازد، روی X_train
    آموزش می‌دهد، روی X_val امتیاز می‌گیرد، و ROC-AUC را حساب می‌کند.

    خروجی: لیستی از دیکشنری‌ها (هرکدام شامل پیکربندی + roc_auc)، مرتب‌شده
    از بهترین به بدترین.

    fixed_kwargs: پارامترهایی که در همه‌ی آزمایش‌ها ثابت می‌مانند (مثل
    n_clusters یا pretrain_epochs)، تا search_space فقط روی چیزهایی که
    واقعاً می‌خواهیم مقایسه کنیم تمرکز کند.
    """
    search_space = search_space if search_space is not None else DEFAULT_SEARCH_SPACE
    fixed_kwargs = fixed_kwargs or {}

    if y_val.sum() == 0:
        raise ValueError("y_val باید حداقل یک نمونه‌ی تهدید داشته باشد تا ROC-AUC معنا داشته باشد.")

    results = []
    for config in search_space:
        model_kwargs = {**fixed_kwargs, **config}
        model = BiLSTMDECModel(**model_kwargs)
        validation_scores = model.fit_score(X_train, X_val)
        auc = roc_auc_score(y_val, validation_scores)
        results.append({**config, "roc_auc": float(auc)})
        print(f"[Sweep] {config} -> ROC-AUC: {auc:.4f}")

    return sorted(results, key=lambda r: r["roc_auc"], reverse=True)
