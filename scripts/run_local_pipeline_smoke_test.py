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
from src.evaluation.metrics import compute_binary_metrics
from src.evaluation.threshold_selection import select_threshold
from src.model.bilstm_dec import BiLSTMDECModel

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "r5.2"
ANSWERS_PATH = PROJECT_ROOT / "data" / "raw" / "answers"  # پوشه‌ی answers کنار r5.2 است، نه داخلش
SAMPLE_ROWS_PER_LOG_FILE = None  # None یعنی کل دیتاست خوانده شود، نه فقط یک نمونه‌ی کوچک
CATEGORICAL_COLUMNS = ["department", "role"]
WINDOWS_CACHE_PATH = PROJECT_ROOT / "data" / "processed" / "windows_cache.npz"


def build_or_load_windows():
    """
    اگر پنجره‌های ساخته‌شده قبلاً روی دیسک ذخیره شده باشند، مستقیم از
    آنجا می‌خواند (چند ثانیه). در غیر این صورت، کل مسیر سنگین (خواندن
    ۳۷ گیگ خام، ویژگی‌سازی، عددی‌سازی، windowing) را یک‌بار اجرا می‌کند
    و نتیجه را برای اجراهای بعدی ذخیره می‌کند — تا دیگر هیچ‌وقت مجبور
    نباشیم برای امتحان‌کردن یک hyperparameter یا threshold جدید،
    دوباره کل داده‌ی خام را از اول پردازش کنیم.
    """
    if WINDOWS_CACHE_PATH.exists():
        print(f"Loading cached windows from {WINDOWS_CACHE_PATH} (skipping raw data processing) ...")
        cached = np.load(WINDOWS_CACHE_PATH, allow_pickle=True)
        return {
            "features": cached["features"],
            "window_user": cached["window_user"],
            "window_end_date": cached["window_end_date"],
            "eval_only_label_any": cached["eval_only_label_any"],
            "eval_only_label_full": cached["eval_only_label_full"],
            "feature_names": cached["feature_names"],
        }

    print("--- Step 1: Building daily behavioral profiles (full dataset) ---")
    daily_profiles = build_multisource_behavioral_profiles(RAW_DATA_PATH, sample_nrows=SAMPLE_ROWS_PER_LOG_FILE)
    daily_profiles = add_psychometric_features(daily_profiles, RAW_DATA_PATH / "psychometric.csv")
    daily_profiles = add_ldap_features(daily_profiles, RAW_DATA_PATH / "LDAP" / "2010-01.csv")
    daily_profiles["is_threat"] = attach_ground_truth_labels(
        daily_profiles, ANSWERS_PATH / "insiders.csv", dataset_version="5.2"
    )
    print(f"User-day records: {len(daily_profiles):,}")
    print(f"Real threat days: {int(daily_profiles['is_threat'].sum())}")

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

    WINDOWS_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(WINDOWS_CACHE_PATH, feature_names=np.array(feature_names), **windows)
    print(f"Windows cached to {WINDOWS_CACHE_PATH} for future runs.")

    return windows


def main():
    windows = build_or_load_windows()
    window_split = build_chronological_split(windows["window_end_date"])
    print(f"Windows -> train: {window_split['train_mask'].sum():,} | "
          f"validation: {window_split['validation_mask'].sum():,} | test: {window_split['test_mask'].sum():,}")

    print("\n--- Pipeline smoke test completed, no errors ---")

    print("\n--- Step 5: First real model run (a rough first signal, not the final tuned version) ---")
    X_train = windows["features"][window_split["train_mask"]]
    X_val = windows["features"][window_split["validation_mask"]]
    X_test = windows["features"][window_split["test_mask"]]
    y_val = windows["eval_only_label_any"][window_split["validation_mask"]]
    y_test = windows["eval_only_label_any"][window_split["test_mask"]]

    if y_val.sum() == 0 or y_test.sum() == 0:
        print(
            "Not enough threat windows in validation or test for this sample. "
            "Either increase sample_nrows or run directly on Colab with the full dataset."
        )
        return

    print(f"Threat windows -> validation: {int(y_val.sum())}/{len(y_val):,} | "
          f"test: {int(y_test.sum())}/{len(y_test):,}")

    scores_cache_path = PROJECT_ROOT / "data" / "processed" / "scores_cache_gamma0.5_max_k3.npz"
    if scores_cache_path.exists():
        print(f"Loading cached model scores from {scores_cache_path} (skipping training entirely) ...")
        cached_scores = np.load(scores_cache_path)
        validation_scores = cached_scores["validation_scores"]
        test_scores = cached_scores["test_scores"]
    else:
        model = BiLSTMDECModel(
            n_clusters=3, encoder_size=32, latent_size=16,
            pretrain_epochs=3, fine_tune_epochs=3, batch_size=256,
            gamma=0.5, pooling_strategy="max",
        )
        model.fit(X_train)
        validation_scores = model.score(X_val)
        test_scores = model.score(X_test)

        scores_cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(scores_cache_path, validation_scores=validation_scores, test_scores=test_scores)
        print(f"Model scores cached to {scores_cache_path} for future threshold tuning.")

    from sklearn.metrics import roc_auc_score, average_precision_score
    print(f"\nThreshold-independent metrics -> ROC-AUC: {roc_auc_score(y_test, test_scores):.4f} | "
          f"PR-AUC: {average_precision_score(y_test, test_scores):.4f}")

    print("\n--- Trying multiple threshold quantiles (single trained model, no retraining) ---")
    for quantile in [0.995, 0.99, 0.98, 0.97, 0.95, 0.93, 0.90, 0.85, 0.80, 0.70, 0.60, 0.50]:
        threshold = select_threshold(validation_scores, method="quantile", quantile=quantile)
        metrics = compute_binary_metrics(y_test, test_scores, threshold=threshold)
        print(f"quantile={quantile} | threshold={threshold:.4f} | "
              f"Precision={metrics['precision']:.4f} | Recall={metrics['recall']:.4f} | "
              f"F1={metrics['f1_score']:.4f} | FPR={metrics['false_positive_rate']:.4f}")

    print("This is only a rough first signal — gamma/pooling/n_clusters are not tuned yet.")


if __name__ == "__main__":
    main()