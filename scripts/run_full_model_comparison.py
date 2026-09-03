"""
run_full_model_comparison.py

هدف: آموزش و ارزیابی همه‌ی مدل‌ها (اصلی + baselineها) روی همان پنجره‌های
کش‌شده، تحت دقیقاً همان پروتکل (همان split، همان threshold rule)، و
مقایسه‌ی آماری مدل اصلی با هرکدام از baselineها.

این اسکریپت را از ریشه‌ی پروژه اجرا کن:
    python scripts/run_full_model_comparison.py

هشدار زمانی: چون ۹ مدل جدا باید train شوند، این اجرا ممکن است به‌طور
کلی از یک تا چند ساعت طول بکشد (روی CPU لپ‌تاپ، بدون GPU).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from src.data.splitting import build_chronological_split
from src.evaluation.metrics import compute_binary_metrics
from src.evaluation.statistical_tests import delong_test
from src.evaluation.threshold_selection import select_threshold, select_threshold_by_fbeta
from src.model.bilstm_autoencoder_no_dec import BiLSTMAutoencoderNoDECModel
from src.model.bilstm_dec import BiLSTMDECModel
from src.model.deep_svdd_model import DeepSVDDModel
from src.model.gru_autoencoder import GRUAutoencoderModel
from src.model.isolation_forest_model import IsolationForestModel
from src.model.lstm_autoencoder import LSTMAutoencoderModel
from src.model.one_class_svm_model import OneClassSVMModel
from src.model.simple_autoencoder import SimpleAutoencoderModel
from src.model.statistical_baseline import StatisticalBaselineModel
from src.model.bilstm_memory_autoencoder import BiLSTMMemoryAutoencoderModel

PROJECT_ROOT = Path(__file__).resolve().parent.parent
WINDOWS_CACHE_PATH = PROJECT_ROOT / "data" / "processed" / "windows_cache.npz"
SCORES_DIR = PROJECT_ROOT / "data" / "processed" / "model_scores"
FAST_EPOCHS = 5  # بودجه‌ی آموزش ثابت و برابر برای همه‌ی مدل‌های عصبی، به‌خاطر محدودیت CPU
ONE_CLASS_SVM_TRAIN_SUBSAMPLE = 20_000  # SVM با کرنل RBF روی کل train عملاً قابل‌اجرا نیست
RANDOM_SEED = 42

# نام مدل -> تابعی که یک نمونه‌ی جدید از آن مدل با تنظیمات ثابت می‌سازد
MODEL_BUILDERS = {
    "bilstm_dec (main)": lambda: BiLSTMDECModel(
        n_clusters=3, encoder_size=32, latent_size=16,
        pretrain_epochs=3, fine_tune_epochs=3, batch_size=256,
        gamma=0.5, pooling_strategy="max",
    ),
    "bilstm_autoencoder_no_dec": lambda: BiLSTMAutoencoderNoDECModel(
        encoder_size=32, latent_size=16, epochs=FAST_EPOCHS, batch_size=256, pooling_strategy="max",
    ),
    "lstm_autoencoder": lambda: LSTMAutoencoderModel(
        encoder_size=32, latent_size=16, epochs=FAST_EPOCHS, batch_size=256, pooling_strategy="max",
    ),
    "gru_autoencoder": lambda: GRUAutoencoderModel(
        encoder_size=32, latent_size=16, epochs=FAST_EPOCHS, batch_size=256, pooling_strategy="max",
    ),
    "simple_autoencoder": lambda: SimpleAutoencoderModel(
        hidden_size=128, latent_size=16, epochs=FAST_EPOCHS, batch_size=256, pooling_strategy="max",
    ),
    "deep_svdd": lambda: DeepSVDDModel(
        encoder_size=32, embedding_size=16, epochs=FAST_EPOCHS, batch_size=256,
    ),
    "statistical_baseline": lambda: StatisticalBaselineModel(pooling_strategy="max"),
    "isolation_forest": lambda: IsolationForestModel(n_estimators=200, random_state=RANDOM_SEED),
    "one_class_svm": lambda: OneClassSVMModel(nu=0.01),

    "bilstm_memory_autoencoder": lambda: BiLSTMMemoryAutoencoderModel(
        encoder_size=32, latent_size=16, memory_size=50,
        epochs=FAST_EPOCHS, batch_size=256, pooling_strategy="max",
    ),
}


def load_cached_windows():
    if not WINDOWS_CACHE_PATH.exists():
        raise FileNotFoundError(
            f"Cache not found at {WINDOWS_CACHE_PATH}. "
            "Run scripts/run_local_pipeline_smoke_test.py once first."
        )
    cached = np.load(WINDOWS_CACHE_PATH, allow_pickle=True)
    return cached["features"], cached["window_end_date"], cached["eval_only_label_any"]


def get_or_compute_scores(model_name, build_model, X_train, X_val, X_test):
    """امتیازهای هر مدل را کش می‌کند تا اگر اسکریپت دوباره اجرا شد، train دوباره لازم نباشد."""
    cache_path = SCORES_DIR / f"{model_name.replace(' ', '_').replace('(', '').replace(')', '')}.npz"
    if cache_path.exists():
        print(f"[{model_name}] Loading cached scores ...")
        cached = np.load(cache_path)
        return cached["validation_scores"], cached["test_scores"]

    print(f"[{model_name}] Training ...")
    model = build_model()

    train_data = X_train
    if model_name == "one_class_svm" and len(X_train) > ONE_CLASS_SVM_TRAIN_SUBSAMPLE:
        rng = np.random.default_rng(RANDOM_SEED)
        subsample_indices = rng.choice(len(X_train), size=ONE_CLASS_SVM_TRAIN_SUBSAMPLE, replace=False)
        train_data = X_train[subsample_indices]
        print(f"[{model_name}] Subsampled train set to {ONE_CLASS_SVM_TRAIN_SUBSAMPLE:,} windows (SVM does not scale to full data).")

    model.fit(train_data)
    validation_scores = model.score(X_val)
    test_scores = model.score(X_test)

    SCORES_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, validation_scores=validation_scores, test_scores=test_scores)
    print(f"[{model_name}] Done. Scores cached to {cache_path}")

    return validation_scores, test_scores


def main():
    print("Loading cached windows ...")
    features, window_end_date, labels = load_cached_windows()
    split = build_chronological_split(window_end_date)

    X_train = features[split["train_mask"]]
    X_val = features[split["validation_mask"]]
    X_test = features[split["test_mask"]]
    y_val = labels[split["validation_mask"]]
    y_test = labels[split["test_mask"]]

    print(f"Train: {len(X_train):,} | Validation: {len(X_val):,} ({int(y_val.sum())} threats) | "
          f"Test: {len(X_test):,} ({int(y_test.sum())} threats)\n")

    all_results = {}
    all_test_scores = {}

    for model_name, build_model in MODEL_BUILDERS.items():
        validation_scores, test_scores = get_or_compute_scores(model_name, build_model, X_train, X_val, X_test)
        fbeta_result = select_threshold_by_fbeta(y_val, validation_scores, beta=2.0)
        threshold = fbeta_result["threshold"]
        metrics = compute_binary_metrics(y_test, test_scores, threshold=threshold)
        all_results[model_name] = metrics
        all_test_scores[model_name] = test_scores
        print(f"[{model_name}] ROC-AUC={metrics['roc_auc']:.4f} | Precision={metrics['precision']:.4f} | "
              f"Recall={metrics['recall']:.4f} | F1={metrics['f1_score']:.4f}\n")

    print("\n=== Final comparison table (test split, threshold from validation quantile=0.95) ===")
    header = f"{'Model':<28} {'ROC-AUC':>8} {'PR-AUC':>8} {'Precision':>10} {'Recall':>8} {'F1':>8}"
    print(header)
    print("-" * len(header))
    for model_name, metrics in sorted(all_results.items(), key=lambda item: item[1]["roc_auc"], reverse=True):
        print(f"{model_name:<28} {metrics['roc_auc']:>8.4f} {metrics['pr_auc']:>8.4f} "
              f"{metrics['precision']:>10.4f} {metrics['recall']:>8.4f} {metrics['f1_score']:>8.4f}")

    print("\n=== Statistical significance: main model vs each baseline (DeLong test on ROC-AUC) ===")
    main_scores = all_test_scores["bilstm_dec (main)"]
    for model_name, test_scores in all_test_scores.items():
        if model_name == "bilstm_dec (main)":
            continue
        result = delong_test(y_test, main_scores, test_scores)
        significance = "SIGNIFICANT (p<0.05)" if result["p_value"] < 0.05 else "not significant"
        print(f"main vs {model_name:<28} AUC diff={result['auc_difference']:+.4f} "
              f"p-value={result['p_value']:.4g} -> {significance}")


if __name__ == "__main__":
    main()