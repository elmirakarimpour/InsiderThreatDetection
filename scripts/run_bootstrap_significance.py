"""
run_bootstrap_significance.py

هدف: با استفاده از امتیازهای از‌قبل‌کش‌شده (بدون نیاز به train دوباره)،
بررسی می‌کند آیا تفاوت F1 بین bilstm_dec و bilstm_autoencoder_no_dec
(نزدیک‌ترین رقیب) واقعاً معنادار است یا فقط نتیجه‌ی یک اجرای شانسی بوده.

این اسکریپت را از ریشه‌ی پروژه اجرا کن:
    python scripts/run_bootstrap_significance.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from sklearn.metrics import f1_score

from src.data.splitting import build_chronological_split
from src.evaluation.statistical_tests import bootstrap_metric_difference
from src.evaluation.threshold_selection import select_threshold_by_fbeta

PROJECT_ROOT = Path(__file__).resolve().parent.parent
WINDOWS_CACHE_PATH = PROJECT_ROOT / "data" / "processed" / "windows_cache.npz"
SCORES_DIR = PROJECT_ROOT / "data" / "processed" / "model_scores"


def load_model_scores(model_name):
    cache_path = SCORES_DIR / f"{model_name}.npz"
    if not cache_path.exists():
        raise FileNotFoundError(f"Scores not found for {model_name} at {cache_path}. Run the comparison script first.")
    cached = np.load(cache_path)
    return cached["validation_scores"], cached["test_scores"]


def main():
    cached_windows = np.load(WINDOWS_CACHE_PATH, allow_pickle=True)
    labels = cached_windows["eval_only_label_any"]
    split = build_chronological_split(cached_windows["window_end_date"])
    y_val = labels[split["validation_mask"]]
    y_test = labels[split["test_mask"]]

    main_val_scores, main_test_scores = load_model_scores("bilstm_dec_main")
    rival_val_scores, rival_test_scores = load_model_scores("bilstm_autoencoder_no_dec")

    main_threshold = select_threshold_by_fbeta(y_val, main_val_scores, beta=2.0)["threshold"]
    rival_threshold = select_threshold_by_fbeta(y_val, rival_val_scores, beta=2.0)["threshold"]
    print(f"main threshold: {main_threshold:.4f} | rival (no-DEC) threshold: {rival_threshold:.4f}")

    result = bootstrap_metric_difference(
        y_test, main_test_scores, rival_test_scores,
        threshold_a=main_threshold, threshold_b=rival_threshold,
        metric_fn=f1_score, n_bootstrap=2000,
    )

    print("\n=== Bootstrap comparison: bilstm_dec (main) vs bilstm_autoencoder_no_dec, F1 ===")
    print(f"F1 difference (main - no_dec): {result['point_difference']:+.4f}")
    print(f"95% CI: [{result['lower_bound']:+.4f}, {result['upper_bound']:+.4f}]")
    print(f"CI excludes zero (statistically real difference): {result['excludes_zero']}")
    print(f"Fraction of bootstrap resamples where main <= no_dec: {result['fraction_bootstrap_non_positive']:.3f}")


if __name__ == "__main__":
    main()
