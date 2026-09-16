"""
src/evaluation/temporal_evidence.py
=====================================
لایه‌ی شاهد زمانی: امتیازهای تک‌پنجره‌ای (مثلاً contrastive_score) را با
تاریخچه‌ی هر کاربر ترکیب می‌کند تا امتیاز نهایی پایدارتر و دقیق‌تری
بسازد. این ماژول، مدل نهایی قفل‌شده‌ی پروژه است.

نتیجه‌ی تأییدشده با سه seed مستقل (42, 123, 7):
  AUC: 0.9476 - 0.9611  (max-min = 0.0135، پایدار)
  F1:  0.5113 - 0.5879  (میانگین ≈ 0.54، همه بالای 0.50)
در مقابل امتیاز تک‌پنجره‌ای (contrastive_score تنها): AUC≈0.875, F1≈0.30-0.45

مهم‌ترین ویژگی، به‌طور پیوسته در هر سه seed: max_jump (بزرگ‌ترین جهش
ناگهانی امتیاز نسبت به تاریخچه‌ی اخیر کاربر).

قانون leakage: fit این مدل همیشه فقط روی validation انجام می‌شود؛
ویژگی‌های شاهد زمانی هم فقط از پنجره‌های *قبل* از پنجره‌ی جاری ساخته
می‌شوند (rolling با closed="left").
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve

LOOKBACK_DAYS = 30
MIN_HISTORY_WINDOWS = 3
FEATURE_COLUMNS = ["score", "mean_score", "max_score", "std_score", "max_jump"]


def build_temporal_evidence_features(
    scores: np.ndarray,
    users: np.ndarray,
    dates: np.ndarray,
    labels: np.ndarray,
    lookback_days: int = LOOKBACK_DAYS,
) -> pd.DataFrame:
    """برای هر پنجره، ویژگی‌های شاهد زمانی را فقط از پنجره‌های قبلیِ همان کاربر می‌سازد."""
    frame = pd.DataFrame(
        {"user": users, "date": pd.to_datetime(dates), "score": scores, "label": labels}
    ).sort_values(["user", "date"]).reset_index(drop=True)

    evidence_frames = []
    for _, user_group in frame.groupby("user", sort=False):
        user_group = user_group.set_index("date")
        rolling = user_group["score"].rolling(f"{lookback_days}D", closed="left")
        user_group["mean_score"] = rolling.mean()
        user_group["max_score"] = rolling.max()
        user_group["std_score"] = rolling.std()
        user_group["history_count"] = rolling.count()
        user_group["max_jump"] = user_group["score"].diff().rolling(f"{lookback_days}D", closed="left").max()
        evidence_frames.append(user_group.reset_index())

    result = pd.concat(evidence_frames, ignore_index=True)
    result[["mean_score", "max_score", "std_score", "max_jump"]] = result[
        ["mean_score", "max_score", "std_score", "max_jump"]
    ].fillna(0.0)
    return result


def has_sufficient_history(evidence_frame: pd.DataFrame, min_history: int = MIN_HISTORY_WINDOWS) -> pd.Series:
    return evidence_frame["history_count"].fillna(0) >= min_history


def fit_temporal_evidence_model(
    evidence_frame: pd.DataFrame, usable_mask: pd.Series
) -> LogisticRegression:
    """فقط روی ردیف‌های usable (باید همیشه validation باشد، هرگز test) fit می‌کند."""
    rows = evidence_frame[usable_mask]
    model = LogisticRegression(max_iter=1000, class_weight="balanced")
    model.fit(rows[FEATURE_COLUMNS], rows["label"])
    return model


def score_with_temporal_evidence(model: LogisticRegression, evidence_frame: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(evidence_frame[FEATURE_COLUMNS])[:, 1]


def best_f1_threshold(labels: np.ndarray, scores: np.ndarray) -> tuple[float, float]:
    """جست‌وجوی کارآمد (O(n log n)) روی تمام آستانه‌های ممکن. فقط روی validation صدا زده شود."""
    precisions, recalls, thresholds = precision_recall_curve(labels, scores)
    f1_curve = 2 * precisions * recalls / (precisions + recalls + 1e-12)
    best_index = int(np.argmax(f1_curve[:-1]))
    return float(thresholds[best_index]), float(f1_curve[best_index])


def compute_fixed_temporal_combination(evidence_frame: pd.DataFrame) -> np.ndarray:
    """
    ترکیب ثابت (بدون هیچ پارامتر یادگرفته‌شده از لیبل): score + mean_score + max_jump.

    این فرمول در آزمایش سه‌seed (فایل 18_unsupervised_temporal_combination.py) با
    رگرسیون لجستیک نظارت‌شده مقایسه شد:
      AUC: 0.9474 (فرمول ثابت) در برابر 0.9525 (LR)
      F1:  0.5565 (فرمول ثابت) در برابر 0.5402 (LR) — فرمول ثابت حتی بهتر بود

    از آن‌جا که این فرمول هیچ پارامتری با لیبل fit نمی‌کند (فقط جمع سه ویژگی‌ای
    که خودشان کاملاً با آمار غلتان بدون‌نظارت ساخته شده‌اند)، این تابع اکنون
    روش قفل‌شده‌ی پیش‌فرض پروژه است، نه fit_temporal_evidence_model.
    تنها جایی که لیبل استفاده می‌شود، انتخاب threshold روی validation است.
    """
    return (evidence_frame["score"] + evidence_frame["mean_score"] + evidence_frame["max_jump"]).to_numpy()