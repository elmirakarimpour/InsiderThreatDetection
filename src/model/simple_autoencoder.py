"""
simple_autoencoder.py

مسئولیت: ساده‌ترین baseline ممکن — یک autoencoder کاملاً MLP (بدون
هیچ لایه‌ی بازگشتی مثل LSTM/GRU). پنجره‌ی ۷روزه را کاملاً صاف
(flatten) می‌کند و مثل یک بردار عادی با آن رفتار می‌کند.

هدف این baseline اثبات این نکته است: چقدر از عملکرد مدل اصلی واقعاً
از «فهمیدن ترتیب زمانی» می‌آید؟ اگر این مدل ساده تقریباً به همان خوبی
BiLSTM-DEC عمل کند، یعنی ترتیب زمانی چندان مهم نبوده؛ اگر خیلی
ضعیف‌تر عمل کند، یعنی مدل‌سازی توالی واقعاً ارزش‌آفرین بوده است.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as functional
from torch.utils.data import DataLoader, TensorDataset

from src.evaluation.scoring import compute_per_timestep_error, pool_temporal_errors
from src.model.base_model import AnomalyDetectionModel


class _SimpleAutoencoderNetwork(nn.Module):
    """یک MLP رمزگذار-رمزگشا روی بردار صاف‌شده‌ی (time * features)."""

    def __init__(self, flattened_size: int, hidden_size: int, latent_size: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(flattened_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, latent_size),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, flattened_size),
        )

    def forward(self, flattened_sequences):
        latent = self.encoder(flattened_sequences)
        return self.decoder(latent)


class SimpleAutoencoderModel(AnomalyDetectionModel):
    name = "simple_autoencoder"

    def __init__(
        self,
        hidden_size: int = 128,
        latent_size: int = 32,
        pooling_strategy: str = "max",
        top_k: int = 2,
        epochs: int = 20,
        batch_size: int = 512,
        learning_rate: float = 1e-3,
        device: str = None,
        seed: int = 42,
    ):
        self.hidden_size = hidden_size
        self.latent_size = latent_size
        self.pooling_strategy = pooling_strategy
        self.top_k = top_k
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.seed = seed
        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
        self._network = None
        self._sequence_shape = None  # (time, features) برای بازسازی شکل اصلی بعد از flatten

    def _make_loader(self, sequences, shuffle):
        flattened = sequences.reshape(sequences.shape[0], -1)
        tensor = torch.from_numpy(np.asarray(flattened, dtype=np.float32))
        return DataLoader(TensorDataset(tensor), batch_size=self.batch_size, shuffle=shuffle, num_workers=0)

    def fit(self, train_sequences: np.ndarray) -> "SimpleAutoencoderModel":
        train_sequences = np.asarray(train_sequences, dtype=np.float32)
        self._sequence_shape = train_sequences.shape[1:]  # (time, features)
        flattened_size = int(np.prod(self._sequence_shape))

        torch.manual_seed(self.seed)
        self._network = _SimpleAutoencoderNetwork(flattened_size, self.hidden_size, self.latent_size).to(self.device)
        optimizer = torch.optim.Adam(self._network.parameters(), lr=self.learning_rate)
        train_loader = self._make_loader(train_sequences, shuffle=True)

        self._network.train()
        for epoch in range(1, self.epochs + 1):
            epoch_loss = 0.0
            for (batch,) in train_loader:
                batch = batch.to(self.device)
                reconstructed = self._network(batch)
                loss = functional.mse_loss(reconstructed, batch)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self._network.parameters(), max_norm=5.0)
                optimizer.step()
                epoch_loss += loss.item()
            print(f"[Simple-AE] Epoch {epoch}/{self.epochs} | MSE: {epoch_loss / max(len(train_loader), 1):.6f}")

        return self

    def score(self, sequences: np.ndarray) -> np.ndarray:
        if self._network is None:
            raise RuntimeError("قبل از score باید fit صدا زده شود.")

        sequences = np.asarray(sequences, dtype=np.float32)
        loader = self._make_loader(sequences, shuffle=False)
        reconstructed_batches = []

        self._network.eval()
        with torch.no_grad():
            for (batch,) in loader:
                reconstructed = self._network(batch.to(self.device))
                reconstructed_batches.append(reconstructed.cpu().numpy())

        # بردار صاف‌شده را به شکل اصلی (batch, time, features) برمی‌گردانیم
        # تا بشود دوباره از همان ابزار pooling مشترک بقیه‌ی مدل‌ها استفاده کرد
        reconstructed = np.concatenate(reconstructed_batches, axis=0).reshape(sequences.shape)
        per_timestep_error = compute_per_timestep_error(sequences, reconstructed)
        return pool_temporal_errors(per_timestep_error, strategy=self.pooling_strategy, top_k=self.top_k)
