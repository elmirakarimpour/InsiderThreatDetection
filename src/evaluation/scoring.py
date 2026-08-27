"""
scoring.py

مسئولیت: تبدیل خطای بازسازی (reconstruction error) هر روز از یک پنجره‌ی
۷روزه به یک عدد نهایی anomaly score برای کل پنجره.

پس‌زمینه‌ی مهم (چرا این فایل اصلاً وجود دارد):
در نسخه‌ی قبلی پروژه، خطای بازسازی با mean روی هم‌زمان محور «زمان» و
محور «فیچر» گرفته می‌شد. مشکل: چون یک پنجره فقط با ۱ روز مشکوک از ۷
روز هم «تهدید» برچسب می‌خورد (قانون any-day-positive)، وقتی خطای اون
۱ روز مشکوک را با میانگین ۶ روز کاملاً عادی دیگر ترکیب می‌کردیم، سیگنال
رقیق می‌شد و Recall مدل به‌شدت افت می‌کرد.

راه‌حل: خطای بازسازی را اول فقط روی محور فیچر میانگین می‌گیریم (تا یک
عدد به‌ازای هر روز داشته باشیم)، بعد روی محور زمان از یک استراتژی
قابل‌تنظیم (mean / max / topk) استفاده می‌کنیم.
"""

import numpy as np


def compute_per_timestep_error(sequences: np.ndarray, reconstructed: np.ndarray) -> np.ndarray:
    """
    خطای بازسازی را فقط روی محور فیچر (نه محور زمان) میانگین می‌گیرد.

    ورودی: sequences و reconstructed هر دو به شکل (batch, time, features)
    خروجی: آرایه‌ای به شکل (batch, time) — یک عدد خطا به‌ازای هر روز از هر پنجره
    """
    squared_error = (sequences - reconstructed) ** 2
    return squared_error.mean(axis=2)  # میانگین فقط روی محور فیچر (axis=2)


def pool_temporal_errors(per_timestep_error: np.ndarray, strategy: str = "max", top_k: int = 2) -> np.ndarray:
    """
    خطای هر روز از پنجره (شکل (batch, time)) را به یک عدد نهایی برای کل
    پنجره تبدیل می‌کند — طبق استراتژی انتخابی:

        "mean"  -> میانگین همه‌ی ۷ روز (رفتار قبلی، سیگنال تک‌روزه را رقیق می‌کند)
        "max"   -> فقط بدترین (مشکوک‌ترین) روز پنجره را در نظر می‌گیرد
        "topk"  -> میانگین k روز مشکوک‌ترین (پیش‌فرض k=2)، تعادلی بین mean و max

    چرا این سه گزینه؟ چون از قبل نمی‌دانیم کدام برای این دیتاست بهتر
    است — باید هر سه را روی validation امتحان کنیم (این دقیقاً هدف
    مرحله‌ی hyperparameter sweep بعدی است).
    """
    if strategy == "mean":
        return per_timestep_error.mean(axis=1)
    elif strategy == "max":
        return per_timestep_error.max(axis=1)
    elif strategy == "topk":
        sequence_length = per_timestep_error.shape[1]
        effective_k = min(top_k, sequence_length)
        sorted_descending = np.sort(per_timestep_error, axis=1)[:, ::-1]
        return sorted_descending[:, :effective_k].mean(axis=1)
    else:
        raise ValueError(f"استراتژی pooling نامعتبر: '{strategy}'. باید یکی از mean/max/topk باشد.")
