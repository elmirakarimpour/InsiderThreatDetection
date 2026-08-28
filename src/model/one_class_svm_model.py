"""
one_class_svm_model.py

مسئولیت: دومین baseline کلاسیک. One-Class SVM یک مرز تصمیم‌گیری
اطراف «رفتار عادی» رسم می‌کند و هر نمونه‌ای که بیرون این مرز باشد را
ناهنجار در نظر می‌گیرد. مثل Isolation Forest، این مدل هم مفهوم توالی
زمانی را نمی‌فهمد، پس هر پنجره را قبل از ورود، صاف (flatten) می‌کنیم.
"""

import numpy as np
from sklearn.svm import OneClassSVM

from src.model.base_model import AnomalyDetectionModel


class OneClassSVMModel(AnomalyDetectionModel):
    name = "one_class_svm"

    def __init__(self, kernel: str = "rbf", nu: float = 0.01, gamma="scale"):
        # nu تقریباً همان «درصد داده‌ای که انتظار داریم ناهنجار باشد» است —
        # با توجه به شیوع پایین تهدید در این دیتاست (~۰.۴٪)، مقدار پیش‌فرض کوچک گذاشته شده.
        self.kernel = kernel
        self.nu = nu
        self.gamma = gamma
        self._model = None

    @staticmethod
    def _flatten(sequences: np.ndarray) -> np.ndarray:
        return sequences.reshape(sequences.shape[0], -1)

    def fit(self, train_sequences: np.ndarray) -> "OneClassSVMModel":
        self._model = OneClassSVM(kernel=self.kernel, nu=self.nu, gamma=self.gamma)
        self._model.fit(self._flatten(train_sequences))
        return self

    def score(self, sequences: np.ndarray) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("قبل از score باید fit صدا زده شود.")
        # decision_function: عدد بزرگ‌تر = عادی‌تر. علامتش را برعکس می‌کنیم
        # تا در کل پروژه «عدد بزرگ‌تر = مشکوک‌تر» باشد.
        return -self._model.decision_function(self._flatten(sequences))
