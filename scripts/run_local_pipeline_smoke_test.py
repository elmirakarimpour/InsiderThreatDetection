"""
run_local_pipeline_smoke_test.py

هدف: اجرای کل مسیر feature_engineering -> sequence_builder -> splitting
-> preprocessing روی یک نمونه‌ی کوچک از داده‌ی واقعی (نه کل ۳۷ گیگ)،
فقط برای مطمئن شدن از اینکه همه‌چیز درست به‌هم وصل می‌شود.

این اسکریپت را از ریشه‌ی پروژه اجرا کن:
    python scripts/run_local_pipeline_smoke_test.py

عمداً sample_nrows=200_000 گذاشته شده تا اجرا سریع باشد (چند دقیقه،
نه چند ساعت). وقتی همه‌چیز درست کار کرد، این عدد را برای اجرای نهایی
کامل حذف می‌کنیم (یا روی Colab با داده‌ی کامل اجرا می‌کنیم).
"""

from pathlib import Path

import sys
from pathlib import Path

# مسیر ریشه‌ی پروژه را صریح به پایتون معرفی می‌کنیم — بدون این خط،
# وقتی این فایل مستقیم اجرا شود (نه از طریق pytest)، پایتون پوشه‌ی
# src را پیدا نمی‌کند، چون فقط پوشه‌ی خود این اسکریپت را می‌شناسد.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from src.data.feature_engineering import (
    add_ldap_features,
    add_psychometric_features,
    build_multisource_behavioral_profiles,
)
from src.data.label_loader import attach_ground_truth_labels
from src.data.preprocessing import TabularPreprocessor, classify_feature_columns
from src.data.sequence_builder import build_sequence_windows
from src.data.splitting import build_chronological_split
from src.model.bilstm_dec import BiLSTMDECModel

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "r5.2"
ANSWERS_PATH = PROJECT_ROOT / "data" / "raw" / "answers"  # پوشه‌ی answers کنار r5.2 است، نه داخلش
SAMPLE_ROWS_PER_LOG_FILE = None  # فقط برای این تست سریع، نه اجرای نهایی
CATEGORICAL_COLUMNS = ["department", "role"]


def main():
    print("--- Step 1: Building daily behavioral profiles (small sample) ---")
    daily_profiles = build_multisource_behavioral_profiles(RAW_DATA_PATH, sample_nrows=SAMPLE_ROWS_PER_LOG_FILE)
    daily_profiles = add_psychometric_features(daily_profiles, RAW_DATA_PATH / "psychometric.csv")
    daily_profiles = add_ldap_features(daily_profiles, RAW_DATA_PATH / "LDAP" / "2010-01.csv")
    daily_profiles["is_threat"] = attach_ground_truth_labels(
        daily_profiles, ANSWERS_PATH / "insiders.csv", dataset_version="5.2"
    )
    print(f"User-day records: {len(daily_profiles):,}")
    print(f"Real threat days in this sample: {int(daily_profiles['is_threat'].sum())}")

    print("\n--- Step 2: Chronological split on daily data (before windowing) ---")
    daily_profiles["date_only"] = pd.to_datetime(daily_profiles["date_only"])
    daily_split = build_chronological_split(daily_profiles["date_only"].to_numpy())
    print(f"Train ends: {daily_split['train_end_date']} | Validation ends: {daily_split['validation_end_date']}")

    print("\n--- Step 3: Preprocessing (fit only on train days) ---")
    candidate_columns = [c for c in daily_profiles.columns if c not in {"user", "date_only", "is_threat"}]
    column_groups = classify_feature_columns(
        training_df=daily_profiles.loc[daily_split["train_mask"]],
        candidate_columns=candidate_columns,
        categorical_columns=[c for c in CATEGORICAL_COLUMNS if c in candidate_columns],
    )
    preprocessor = TabularPreprocessor().fit(
        training_df=daily_profiles.loc[daily_split["train_mask"]],
        continuous_columns=column_groups["continuous"],
        binary_columns=column_groups["binary"],
        categorical_columns=column_groups["categorical"],
    )
    feature_matrix, feature_names = preprocessor.transform(daily_profiles)
    print(f"Numeric feature matrix shape: {feature_matrix.shape} | Feature count: {len(feature_names)}")

    print("\n--- Step 4: Building numeric DataFrame and windowing ---")
    numeric_daily_df = pd.DataFrame(feature_matrix, columns=feature_names)
    numeric_daily_df["user"] = daily_profiles["user"].to_numpy()
    numeric_daily_df["date_only"] = daily_profiles["date_only"].to_numpy()
    numeric_daily_df["is_threat"] = daily_profiles["is_threat"].to_numpy()

    windows = build_sequence_windows(numeric_daily_df, feature_columns=feature_names, sequence_length=7)
    print(f"Sequence array shape: {windows['features'].shape}")

    window_split = build_chronological_split(windows["window_end_date"])
    print(f"Windows -> train: {window_split['train_mask'].sum():,} | "
          f"validation: {window_split['validation_mask'].sum():,} | test: {window_split['test_mask'].sum():,}")

    print("\n--- Pipeline smoke test completed, no errors ---")

    print("\n--- Step 5: First real model run (a rough first signal, not the final tuned version) ---")
    from sklearn.metrics import roc_auc_score

    X_train = windows["features"][window_split["train_mask"]]
    X_val = windows["features"][window_split["validation_mask"]]
    y_val = windows["eval_only_label_any"][window_split["validation_mask"]]

    if y_val.sum() == 0:
        print(
            "No threat windows found in validation for this small sample "
            "(sample_nrows=200,000). Either increase sample_nrows or run "
            "directly on Colab with the full dataset."
        )
    else:
        print(f"Threat windows in validation: {int(y_val.sum())} out of {len(y_val):,}")
        model = BiLSTMDECModel(
            n_clusters=3, encoder_size=32, latent_size=16,
            pretrain_epochs=3, fine_tune_epochs=3, batch_size=256,
            gamma=0.5, pooling_strategy="max",
        )
        val_scores = model.fit_score(X_train, X_val)
        auc = roc_auc_score(y_val, val_scores)
        print(f"Initial ROC-AUC (no hyperparameter tuning yet): {auc:.4f}")
        print("This is only a rough first signal — gamma/pooling/n_clusters are not tuned yet.")


if __name__ == "__main__":
    main()