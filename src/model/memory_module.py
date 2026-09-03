"""
memory_module.py

مسئولیت: یک «حافظه‌ی محدود» از الگوهای رفتار عادی نگه می‌دارد.

ایده‌ی اصلی (بر پایه‌ی Gong et al., 2019 - Memory-Augmented Autoencoders):
یک autoencoder معمولی، اگر به‌اندازه‌ی کافی قدرتمند باشد (مثل BiLSTM
دوطرفه‌ی ما)، ممکن است رفتار غیرعادی را هم به‌خوبی بازسازی کند —
یعنی دقیقاً همان توانایی که برایش ارزشمند است، تبدیل به ضعفش می‌شود.

راه‌حل: به‌جای اینکه رمزگشا مستقیماً از خروجی رمزگذار بازسازی کند، اول
باید بردار رمزگذاری‌شده را با یک ترکیب وزن‌دار از چند ده «الگوی حافظه»
که فقط از روی رفتار عادی train یاد گرفته شده جایگزین کند. چون حافظه
محدود و فقط از رفتار عادی ساخته شده، رفتار غیرعادی را نمی‌تواند دقیق
بازسازی کند — و همین ضعف بازسازی، سیگنال ناهنجاری ماست.
"""

import torch
import torch.nn as nn
import torch.nn.functional as functional


class MemoryModule(nn.Module):
    """
    memory_size الگوی یادگرفتنی (هرکدام به طول latent_size) نگه می‌دارد.
    هر بردار ورودی را با نزدیک‌ترین ترکیب از این الگوها جایگزین می‌کند.
    """

    def __init__(self, memory_size: int, latent_size: int, sparsity_threshold: float = 1.0 / 100):
        super().__init__()
        self.memory_bank = nn.Parameter(torch.empty(memory_size, latent_size))
        nn.init.xavier_uniform_(self.memory_bank)
        # آستانه‌ی تنک‌سازی (hard shrinkage): وزن‌های خیلی کوچک را صفر می‌کند
        # تا هر نمونه فقط با تعداد کمی از الگوهای حافظه بازسازی شود، نه همه‌شان
        # (این جلوگیری می‌کند حافظه صرفاً مثل یک لایه‌ی خطی معمولی عمل کند)
        self.sparsity_threshold = sparsity_threshold

    def forward(self, latent_vectors: torch.Tensor):
        """
        latent_vectors: شکل (batch, latent_size)
        خروجی:
            retrieved_vectors: بازسازی latent از روی حافظه، همان شکل ورودی
            attention_weights: وزن هر الگوی حافظه برای هر نمونه، شکل (batch, memory_size)
        """
        similarity_scores = latent_vectors @ self.memory_bank.T  # (batch, memory_size)
        attention_weights = functional.softmax(similarity_scores, dim=1)

        # تنک‌سازی نرم: وزن‌های زیر آستانه را به صفر نزدیک می‌کند، بعد دوباره نرمال می‌کند
        shrunk_weights = functional.relu(attention_weights - self.sparsity_threshold) * attention_weights
        shrunk_weights = shrunk_weights / (shrunk_weights.sum(dim=1, keepdim=True) + 1e-12)

        retrieved_vectors = shrunk_weights @ self.memory_bank
        return retrieved_vectors, shrunk_weights
