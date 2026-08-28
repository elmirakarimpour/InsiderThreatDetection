"""
metrics.py

مسئولیت: محاسبه‌ی معیارهای استاندارد ارزیابی (همان‌هایی که استادت
مشخصاً خواسته بود: Precision، Recall، F1، ROC-AUC، PR-AUC، و
Confusion Matrix) بعد از اینکه یک threshold مشخص انتخاب شد.
"""

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)


def compute_binary_metrics(y_true: np.ndarray, anomaly_scores: np.ndarray, threshold: float) -> dict:
    """
    با داشتن لیبل واقعی، امتیاز ناهنجاری، و یک آستانه‌ی ثابت (که باید
    از قبل و فقط از روی validation انتخاب شده باشد)، تمام معیارهای
    استاندارد ارزیابی را حساب می‌کند.
    """
    y_true = np.asarray(y_true, dtype=np.int8)
    anomaly_scores = np.asarray(anomaly_scores, dtype=float)
    if len(y_true) != len(anomaly_scores):
        raise ValueError("y_true و anomaly_scores باید طول یکسان داشته باشند.")
    if len(np.unique(y_true)) < 2:
        raise ValueError("برای محاسبه‌ی این معیارها، هم نمونه‌ی عادی و هم نمونه‌ی تهدید لازم است.")

    predicted_labels = (anomaly_scores >= threshold).astype(np.int8)
    true_negatives, false_positives, false_negatives, true_positives = confusion_matrix(
        y_true, predicted_labels, labels=[0, 1]
    ).ravel()
    precision, recall, f1_score, _ = precision_recall_fscore_support(
        y_true, predicted_labels, average="binary", zero_division=0
    )

    return {
        "threshold": float(threshold),
        "roc_auc": float(roc_auc_score(y_true, anomaly_scores)),
        "pr_auc": float(average_precision_score(y_true, anomaly_scores)),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1_score),
        "false_positive_rate": float(false_positives / (false_positives + true_negatives)),
        "true_negatives": int(true_negatives),
        "false_positives": int(false_positives),
        "false_negatives": int(false_negatives),
        "true_positives": int(true_positives),
    }
