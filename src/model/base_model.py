"""
base_model.py

مسئولیت: تعریف یک «قرارداد مشترک» (interface) که همه‌ی مدل‌های تشخیص
ناهنجاری این پروژه — چه مدل اصلی (BiLSTM-DEC)، چه baselineها
(Isolation Forest، LSTM-AE، One-Class SVM و...) — باید از آن پیروی کنند.

این کار با «کلاس پایه‌ی انتزاعی» (Abstract Base Class) در پایتون انجام
می‌شود: یک کلاس که خودش مستقیماً قابل‌استفاده نیست (نمی‌شود از آن یک
شیء ساخت)، فقط مشخص می‌کند هر کلاس فرزندش چه متدهایی باید داشته باشد.
اگر فرزندی یکی از متدهای @abstractmethod را ننویسد، پایتون همان لحظه‌ی
ساختنش خطا می‌دهد — نه بعداً وسط اجرای آزمایش.
"""

from abc import ABC, abstractmethod

import numpy as np


class AnomalyDetectionModel(ABC):
    """
    قرارداد مشترک همه‌ی مدل‌های این پروژه.

    هر مدل جدید باید از این کلاس ارث‌بری کند و این دو متد را بنویسد:
        fit(train_sequences)  -> بدون هیچ لیبلی، فقط از روی توالی‌های
                                  رفتاری، الگوی عادی/کلی را یاد می‌گیرد.
        score(sequences)      -> برای هر توالی، یک عدد «میزان ناهنجاری»
                                  برمی‌گرداند (عدد بزرگ‌تر = مشکوک‌تر).
    """

    #: اسم مدل — در جدول‌های مقایسه‌ای و گزارش‌ها استفاده می‌شود
    name: str = "base_model"

    @abstractmethod
    def fit(self, train_sequences: np.ndarray) -> "AnomalyDetectionModel":
        """مدل را فقط روی train_sequences آموزش می‌دهد — هیچ لیبلی دریافت نمی‌کند."""
        raise NotImplementedError

    @abstractmethod
    def score(self, sequences: np.ndarray) -> np.ndarray:
        """به ازای هر توالی ورودی، یک عدد anomaly score برمی‌گرداند."""
        raise NotImplementedError

    def fit_score(self, train_sequences: np.ndarray, eval_sequences: np.ndarray) -> np.ndarray:
        """میان‌بر رایج: اول fit روی train، بعد score روی eval_sequences."""
        self.fit(train_sequences)
        return self.score(eval_sequences)
