"""
isolation_forest_model.py

مسئولیت: baseline کلاسیک برای مقایسه با BiLSTM-DEC. برخلاف BiLSTM-DEC،
Isolation Forest اصلاً مفهوم «توالی زمانی» را نمی‌فهمد — فقط بردارهای
عددی ساده می‌گیرد. برای همین هر پنجره‌ی (۷ روز × N ویژگی) را قبل از
ورود به مدل، «صاف» (flatten) می‌کنیم و به یک بردار تک‌بعدی تبدیل می‌کنیم.

این baseline عمداً ساده و بدون آگاهی از ساختار زمانی نگه داشته شده —
دقیقاً همین سادگی است که در گزارش نهایی نشان می‌دهد چقدر از عملکرد
مدل‌های ما ناشی از «فهمیدن توالی زمانی» است (BiLSTM) در مقابل صرفاً
«فهمیدن اینکه این بردار غیرعادی است» (Isolation Forest).
"""

import numpy as np
from sklearn.ensemble import IsolationForest

from src.model.base_model import AnomalyDetectionModel


class IsolationForestModel(AnomalyDetectionModel):
    name = "isolation_forest"

    def __init__(self, n_estimators: int = 200, contamination="auto", random_state: int = 42):
        self.n_estimators = n_estimators
        self.contamination = contamination
        self.random_state = random_state
        self._model = None

    @staticmethod
    def _flatten(sequences: np.ndarray) -> np.ndarray:
        """(batch, time, features) -> (batch, time*features). هر پنجره یک بردار طولانی می‌شود."""
        return sequences.reshape(sequences.shape[0], -1)

    def fit(self, train_sequences: np.ndarray) -> "IsolationForestModel":
        self._model = IsolationForest(
            n_estimators=self.n_estimators,
            contamination=self.contamination,
            random_state=self.random_state,
        )
        self._model.fit(self._flatten(train_sequences))
        return self

    def score(self, sequences: np.ndarray) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("قبل از score باید fit صدا زده شود.")
        # score_samples در sklearn: عدد کمتر = غیرعادی‌تر. برای همینکه در کل
        # پروژه «عدد بزرگ‌تر = مشکوک‌تر» یک قرارداد ثابت بماند، علامتش را برعکس می‌کنیم.
        return -self._model.score_samples(self._flatten(sequences))
