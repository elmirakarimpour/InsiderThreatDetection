"""
test_label_loader.py
"""

import numpy as np
import pandas as pd

from src.data.label_loader import attach_ground_truth_labels


def test_attach_ground_truth_labels_marks_correct_interval(tmp_path):
    # یک insiders.csv ساختگی: alice از ۳ تا ۵ ژانویه ۲۰۱۱ رفتار مخرب داشته
    insiders_csv = tmp_path / "insiders.csv"
    pd.DataFrame({
        "dataset": ["r5.2"],
        "user": ["alice"],
        "start": ["2011-01-03"],
        "end": ["2011-01-05"],
    }).to_csv(insiders_csv, index=False)

    daily_profiles = pd.DataFrame({
        "user": ["alice"] * 7 + ["bob"] * 7,
        "date_only": list(pd.date_range("2011-01-01", periods=7)) * 2,
    })

    labels = attach_ground_truth_labels(daily_profiles, str(insiders_csv), dataset_version="5.2")

    alice_labels = labels[:7]
    bob_labels = labels[7:]

    # روزهای ۱،۲،۶،۷ آلیس باید ۰ باشند؛ روزهای ۳،۴،۵ باید ۱ باشند
    assert alice_labels.tolist() == [0, 0, 1, 1, 1, 0, 0]
    # باب اصلاً در insiders.csv نیست، همه‌ی روزهایش باید ۰ باشد
    assert bob_labels.tolist() == [0] * 7


def test_attach_ground_truth_labels_ignores_other_dataset_versions(tmp_path):
    """اگر رخدادی مربوط به نسخه‌ی دیگری از دیتاست (مثلاً r4.2) باشد، نباید روی r5.2 اثر بگذارد."""
    insiders_csv = tmp_path / "insiders.csv"
    pd.DataFrame({
        "dataset": ["r4.2"],
        "user": ["alice"],
        "start": ["2011-01-03"],
        "end": ["2011-01-05"],
    }).to_csv(insiders_csv, index=False)

    daily_profiles = pd.DataFrame({
        "user": ["alice"] * 3,
        "date_only": pd.date_range("2011-01-03", periods=3),
    })
    labels = attach_ground_truth_labels(daily_profiles, str(insiders_csv), dataset_version="5.2")
    assert labels.tolist() == [0, 0, 0]
