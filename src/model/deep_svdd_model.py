"""
deep_svdd_model.py

مسئولیت: پیاده‌سازی Deep SVDD — برخلاف مدل‌های autoencoder-محور بالا
(که سعی می‌کنند ورودی را بازسازی کنند)، Deep SVDD یک شبکه دارد که
هر پنجره را به یک نقطه در فضای embedding نگاشت می‌کند، و یاد می‌گیرد
که نقاط رفتار عادی را تا حد امکان نزدیک به یک مرکز ثابت (c) جمع کند.
فاصله‌ی هر نمونه از این مرکز، امتیاز ناهنجاری آن است.

نکته‌ی مهم پیاده‌سازی: مرکز c را یک‌بار، از میانگین embeddingهای
مرحله‌ی اول (قبل از شروع آموزش واقعی)، محاسبه و سپس ثابت نگه می‌داریم
— این کار استاندارد Deep SVDD است تا شبکه به یک راه‌حل بی‌فایده (همه‌
چیز را به یک نقطه‌ی صفر نگاشت کردن) نرسد.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.model.base_model import AnomalyDetectionModel


class _SVDDEncoderNetwork(nn.Module):
    """یک رمزگذار LSTM ساده که هر پنجره را به یک بردار embedding تبدیل می‌کند (بدون رمزگشا)."""

    def __init__(self, n_features: int, encoder_size: int, embedding_size: int):
        super().__init__()
        # طبق توصیه‌ی استاندارد Deep SVDD، لایه‌های bias حذف می‌شوند تا
        # شبکه نتواند با یادگیری فقط بایاس، به‌راحتی همه‌چیز را به مرکز نگاشت کند
        self.encoder_lstm = nn.LSTM(n_features, encoder_size, batch_first=True, bias=False)
        self.to_embedding = nn.Linear(encoder_size, embedding_size, bias=False)

    def forward(self, sequences):
        _, (hidden_state, _) = self.encoder_lstm(sequences)
        return self.to_embedding(hidden_state[-1])


class DeepSVDDModel(AnomalyDetectionModel):
    name = "deep_svdd"

    def __init__(
        self,
        encoder_size: int = 64,
        embedding_size: int = 32,
        epochs: int = 20,
        batch_size: int = 512,
        learning_rate: float = 1e-3,
        device: str = None,
        seed: int = 42,
    ):
        self.encoder_size = encoder_size
        self.embedding_size = embedding_size
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.seed = seed
        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
        self._network = None
        self._center = None

    def _make_loader(self, sequences, shuffle):
        tensor = torch.from_numpy(np.asarray(sequences, dtype=np.float32))
        return DataLoader(TensorDataset(tensor), batch_size=self.batch_size, shuffle=shuffle, num_workers=0)

    def _compute_initial_center(self, train_loader, eps: float = 0.1):
        """مرکز اولیه را از میانگین embeddingهای شبکه‌ی تازه‌مقداردهی‌شده حساب می‌کند."""
        embeddings = []
        self._network.eval()
        with torch.no_grad():
            for (batch,) in train_loader:
                embeddings.append(self._network(batch.to(self.device)))
        center = torch.cat(embeddings, dim=0).mean(dim=0)
        # مقادیر خیلی نزدیک به صفر را کمی جابه‌جا می‌کنیم تا از فروپاشی به نقطه‌ی صفر جلوگیری شود
        center[(center.abs() < eps) & (center < 0)] = -eps
        center[(center.abs() < eps) & (center >= 0)] = eps
        return center

    def fit(self, train_sequences: np.ndarray) -> "DeepSVDDModel":
        train_sequences = np.asarray(train_sequences, dtype=np.float32)
        torch.manual_seed(self.seed)
        self._network = _SVDDEncoderNetwork(
            train_sequences.shape[2], self.encoder_size, self.embedding_size
        ).to(self.device)
        train_loader = self._make_loader(train_sequences, shuffle=True)

        self._center = self._compute_initial_center(train_loader).to(self.device)

        optimizer = torch.optim.Adam(self._network.parameters(), lr=self.learning_rate)
        self._network.train()
        for epoch in range(1, self.epochs + 1):
            epoch_loss = 0.0
            for (batch,) in train_loader:
                batch = batch.to(self.device)
                embeddings = self._network(batch)
                loss = ((embeddings - self._center) ** 2).sum(dim=1).mean()
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self._network.parameters(), max_norm=5.0)
                optimizer.step()
                epoch_loss += loss.item()
            print(f"[Deep-SVDD] Epoch {epoch}/{self.epochs} | Mean distance²: {epoch_loss / max(len(train_loader), 1):.6f}")

        return self

    def score(self, sequences: np.ndarray) -> np.ndarray:
        if self._network is None:
            raise RuntimeError("قبل از score باید fit صدا زده شود.")

        loader = self._make_loader(sequences, shuffle=False)
        distances = []
        self._network.eval()
        with torch.no_grad():
            for (batch,) in loader:
                embeddings = self._network(batch.to(self.device))
                batch_distances = ((embeddings - self._center) ** 2).sum(dim=1)
                distances.append(batch_distances.cpu().numpy())

        return np.concatenate(distances, axis=0)
