"""
run_hyperparameter_sweep.py

هدف: اجرای واقعی run_hyperparameter_sweep (از hyperparameter_config.py)
روی پنجره‌های از قبل کش‌شده (windows_cache.npz) — بدون نیاز به خواندن
دوباره‌ی ۳۷ گیگ داده‌ی خام.

این اسکریپت را از ریشه‌ی پروژه اجرا کن:
    python scripts/run_hyperparameter_sweep.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from src.data.splitting import build_chronological_split
from src.training.hyperparameter_config import DEFAULT_SEARCH_SPACE, run_hyperparameter_sweep

PROJECT_ROOT = Path(__file__).resolve().parent.parent
WINDOWS_CACHE_PATH = PROJECT_ROOT / "data" / "processed" / "windows_cache.npz"


def main():
    if not WINDOWS_CACHE_PATH.exists():
        raise FileNotFoundError(
            f"Cache not found at {WINDOWS_CACHE_PATH}. "
            "Run scripts/run_local_pipeline_smoke_test.py once first to build it."
        )

    print(f"Loading cached windows from {WINDOWS_CACHE_PATH} ...")
    cached = np.load(WINDOWS_CACHE_PATH, allow_pickle=True)
    features = cached["features"]
    window_end_date = cached["window_end_date"]
    labels = cached["eval_only_label_any"]

    window_split = build_chronological_split(window_end_date)
    X_train = features[window_split["train_mask"]]
    X_val = features[window_split["validation_mask"]]
    y_val = labels[window_split["validation_mask"]]

    print(f"Train windows: {len(X_train):,} | Validation windows: {len(X_val):,} "
          f"(threats: {int(y_val.sum())})")
    print(f"Search space: {len(DEFAULT_SEARCH_SPACE)} configurations\n")

    results = run_hyperparameter_sweep(
        X_train, X_val, y_val,
        search_space=DEFAULT_SEARCH_SPACE,
        fixed_kwargs=dict(
            encoder_size=32, latent_size=16,
            pretrain_epochs=3, fine_tune_epochs=3, batch_size=256,
        ),
    )

    print("\n--- Ranked by F1 (primary criterion, at F-beta-tuned threshold) ---")
    for rank, result in enumerate(results, start=1):
        print(f"{rank}. {result}")

    print("\n--- Same results, re-sorted by ROC-AUC (for cross-check) ---")
    by_auc = sorted(results, key=lambda r: r["roc_auc"], reverse=True)
    for rank, result in enumerate(by_auc, start=1):
        print(f"{rank}. {result}")

    best_by_f1 = results[0]
    best_by_auc = by_auc[0]
    print(f"\nBest by F1:      {best_by_f1}")
    print(f"Best by ROC-AUC: {best_by_auc}")
    if best_by_f1 == best_by_auc:
        print("-> Both criteria agree on the same configuration.")
    else:
        print("-> Criteria disagree — review both before deciding.")


if __name__ == "__main__":
    main()