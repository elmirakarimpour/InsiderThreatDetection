"""
lstm_autoencoder.py

مسئولیت: baseline ساده‌تر از مدل اصلی — یک رمزگذار-رمزگشای LSTM
یک‌طرفه (نه دوطرفه) و بدون لایه‌ی خوشه‌بندی DEC. این مدل دقیقاً همون
چیزیه که استادت خواسته: نشون بدیم مدل اصلی (BiLSTM + DEC) واقعاً
بهتر از یک LSTM ساده‌ی بدون این دو ویژگی عمل می‌کنه یا نه.

از همون pool_temporal_errors در scoring.py استفاده می‌کند تا مقایسه‌ی
منصفانه باشد — یعنی تفاوت نتیجه فقط از تفاوت معماری بیاید، نه از
تفاوت نحوه‌ی محاسبه‌ی امتیاز نهایی.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as functional
from torch.utils.data import DataLoader, TensorDataset

from src.evaluation.scoring import compute_per_timestep_error, pool_temporal_errors
from src.model.base_model import AnomalyDetectionModel


class _LSTMAutoencoderNetwork(nn.Module):
    """معماری خام: رمزگذار LSTM یک‌طرفه + رمزگشای LSTM یک‌طرفه. بدون خوشه‌بندی."""

    def __init__(self, n_features: int, encoder_size: int, latent_size: int):
        super().__init__()
        self.encoder_lstm = nn.LSTM(n_features, encoder_size, batch_first=True)
        self.to_latent = nn.Linear(encoder_size, latent_size)
        self.decoder_lstm = nn.LSTM(latent_size, encoder_size, batch_first=True)
        self.output_layer = nn.Linear(encoder_size, n_features)

    def forward(self, sequences):
        _, (hidden_state, _) = self.encoder_lstm(sequences)
        latent_vector = self.to_latent(hidden_state[-1])  # فقط جهت جلو، بدون بازگشتی/دوطرفه

        repeated_latent = latent_vector.unsqueeze(1).expand(-1, sequences.size(1), -1)
        decoded, _ = self.decoder_lstm(repeated_latent)
        reconstructed = self.output_layer(decoded)
        return reconstructed


class LSTMAutoencoderModel(AnomalyDetectionModel):
    name = "lstm_autoencoder"

    def __init__(
        self,
        encoder_size: int = 64,
        latent_size: int = 32,
        pooling_strategy: str = "max",
        top_k: int = 2,
        epochs: int = 20,
        batch_size: int = 512,
        learning_rate: float = 1e-3,
        device: str = None,
        seed: int = 42,
    ):
        self.encoder_size = encoder_size
        self.latent_size = latent_size
        self.pooling_strategy = pooling_strategy
        self.top_k = top_k
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.seed = seed
        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
        self._network = None

    def _make_loader(self, sequences, shuffle):
        tensor = torch.from_numpy(np.asarray(sequences, dtype=np.float32))
        return DataLoader(TensorDataset(tensor), batch_size=self.batch_size, shuffle=shuffle, num_workers=0)

    def fit(self, train_sequences: np.ndarray) -> "LSTMAutoencoderModel":
        train_sequences = np.asarray(train_sequences, dtype=np.float32)
        torch.manual_seed(self.seed)
        self._network = _LSTMAutoencoderNetwork(
            train_sequences.shape[2], self.encoder_size, self.latent_size
        ).to(self.device)
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
            print(f"[LSTM-AE] Epoch {epoch}/{self.epochs} | MSE: {epoch_loss / max(len(train_loader), 1):.6f}")

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

        reconstructed = np.concatenate(reconstructed_batches, axis=0)
        per_timestep_error = compute_per_timestep_error(sequences, reconstructed)
        return pool_temporal_errors(per_timestep_error, strategy=self.pooling_strategy, top_k=self.top_k)
