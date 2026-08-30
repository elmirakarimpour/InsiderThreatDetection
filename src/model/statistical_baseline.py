"""
statistical_baseline.py

مسئولیت: ساده‌ترین baseline ممکن در کل پروژه — بدون هیچ شبکه‌ی عصبی.
فقط میانگین رفتار «عادی» (از train) را به‌عنوان یک بردار ثابت به خاطر
می‌سپارد، و فاصله‌ی هر پنجره‌ی جدید از این میانگین را به‌عنوان امتیاز
ناهنجاری برمی‌گرداند — دقیقاً همان ایده‌ی کلاسیک «انحراف از میانگین»
در آمار سنتی، پیش از ورود یادگیری عمیق به این حوزه.

از همان ابزار مشترک scoring.py استفاده می‌کند تا مقایسه کاملاً منصفانه
با بقیه‌ی مدل‌ها باشد.
"""

import numpy as np

from src.evaluation.scoring import compute_per_timestep_error, pool_temporal_errors
from src.model.base_model import AnomalyDetectionModel


class StatisticalBaselineModel(AnomalyDetectionModel):
    name = "statistical_baseline"

    def __init__(self, pooling_strategy: str = "max", top_k: int = 2):
        self.pooling_strategy = pooling_strategy
        self.top_k = top_k
        self._mean_profile = None  # شکل: (features,) — میانگین کلی هر ویژگی در کل train

    def fit(self, train_sequences: np.ndarray) -> "StatisticalBaselineModel":
        train_sequences = np.asarray(train_sequences, dtype=np.float32)
        # میانگین روی هر دو محور نمونه و زمان گرفته می‌شود تا یک «رفتار متوسط روزانه» به دست آید
        self._mean_profile = train_sequences.mean(axis=(0, 1))
        return self

    def score(self, sequences: np.ndarray) -> np.ndarray:
        if self._mean_profile is None:
            raise RuntimeError("قبل از score باید fit صدا زده شود.")

        sequences = np.asarray(sequences, dtype=np.float32)
        # «بازسازی» این baseline همیشه همان میانگین ثابت است، فارغ از ورودی —
        # یعنی امتیاز نهایی صرفاً میزان دوری هر پنجره از رفتار متوسط تاریخی است
        broadcasted_mean = np.broadcast_to(self._mean_profile, sequences.shape)
        per_timestep_error = compute_per_timestep_error(sequences, broadcasted_mean)
        return pool_temporal_errors(per_timestep_error, strategy=self.pooling_strategy, top_k=self.top_k)
