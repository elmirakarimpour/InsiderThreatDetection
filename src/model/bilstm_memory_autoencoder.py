"""
bilstm_memory_autoencoder.py

مسئولیت: مدل جایگزین DEC. همان رمزگذار/رمزگشای BiLSTM دوطرفه‌ی قبلی
را نگه می‌داریم (چون آن بخش، طبق تست DeLong، واقعاً مؤثر بود)، ولی
لایه‌ی خوشه‌بندی DEC را با MemoryModule جایگزین می‌کنیم.

جریان کار:
    ورودی -> رمزگذار BiLSTM -> بردار latent -> MemoryModule (بازسازی
    از روی الگوهای عادی) -> رمزگشای BiLSTM -> خروجی بازسازی‌شده

Loss = خطای بازسازی (MSE) + یک جریمه‌ی کوچک برای تشویق تنک‌بودن وزن‌های
حافظه (entropy regularization) — تا مدل مجبور شود واقعاً از تعداد کمی
الگو استفاده کند، نه اینکه یواشکی معادل یک لایه‌ی خطی معمولی عمل کند.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as functional
from torch.utils.data import DataLoader, TensorDataset

from src.evaluation.scoring import compute_per_timestep_error, pool_temporal_errors
from src.model.base_model import AnomalyDetectionModel
from src.model.memory_module import MemoryModule


class _BiLSTMMemoryNetwork(nn.Module):
    def __init__(self, n_features: int, encoder_size: int, latent_size: int, memory_size: int):
        super().__init__()
        if latent_size % 2 != 0:
            raise ValueError("latent_size باید زوج باشد چون رمزگذار دوطرفه است.")

        self.encoder_lstm_1 = nn.LSTM(n_features, encoder_size, batch_first=True, bidirectional=True)
        self.encoder_lstm_2 = nn.LSTM(encoder_size * 2, latent_size // 2, batch_first=True, bidirectional=True)
        self.memory = MemoryModule(memory_size=memory_size, latent_size=latent_size)
        self.decoder_lstm_1 = nn.LSTM(latent_size, latent_size // 2, batch_first=True, bidirectional=True)
        self.decoder_lstm_2 = nn.LSTM(latent_size, encoder_size, batch_first=True, bidirectional=True)
        self.output_layer = nn.Linear(encoder_size * 2, n_features)

    def forward(self, sequences):
        encoded, _ = self.encoder_lstm_1(sequences)
        _, (hidden_state, _) = self.encoder_lstm_2(encoded)
        latent_vectors = torch.cat((hidden_state[-2], hidden_state[-1]), dim=1)

        retrieved_vectors, attention_weights = self.memory(latent_vectors)

        repeated_latent = retrieved_vectors.unsqueeze(1).expand(-1, sequences.size(1), -1)
        decoded, _ = self.decoder_lstm_1(repeated_latent)
        decoded, _ = self.decoder_lstm_2(decoded)
        reconstructed = self.output_layer(decoded)

        return reconstructed, latent_vectors, attention_weights


def _entropy_regularization_loss(attention_weights: torch.Tensor) -> torch.Tensor:
    """
    آنتروپی وزن‌های توجه را جریمه می‌کند تا کم‌تر پخش (کم‌تنک) باشند —
    یعنی هر نمونه با تعداد کمی الگوی حافظه بازسازی شود، نه همه‌ی الگوها.
    """
    return -torch.sum(attention_weights * torch.log(attention_weights + 1e-12), dim=1).mean()


class BiLSTMMemoryAutoencoderModel(AnomalyDetectionModel):
    name = "bilstm_memory_autoencoder"

    def __init__(
        self,
        encoder_size: int = 64,
        latent_size: int = 64,
        memory_size: int = 50,
        entropy_weight: float = 0.0002,
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
        self.memory_size = memory_size
        self.entropy_weight = entropy_weight
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

    def fit(self, train_sequences: np.ndarray) -> "BiLSTMMemoryAutoencoderModel":
        train_sequences = np.asarray(train_sequences, dtype=np.float32)
        torch.manual_seed(self.seed)
        self._network = _BiLSTMMemoryNetwork(
            train_sequences.shape[2], self.encoder_size, self.latent_size, self.memory_size
        ).to(self.device)
        optimizer = torch.optim.Adam(self._network.parameters(), lr=self.learning_rate)
        train_loader = self._make_loader(train_sequences, shuffle=True)

        self._network.train()
        for epoch in range(1, self.epochs + 1):
            epoch_reconstruction_loss = 0.0
            epoch_entropy_loss = 0.0
            for (batch,) in train_loader:
                batch = batch.to(self.device)
                reconstructed, _, attention_weights = self._network(batch)

                reconstruction_loss = functional.mse_loss(reconstructed, batch)
                entropy_loss = _entropy_regularization_loss(attention_weights)
                loss = reconstruction_loss + self.entropy_weight * entropy_loss

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self._network.parameters(), max_norm=5.0)
                optimizer.step()

                epoch_reconstruction_loss += reconstruction_loss.item()
                epoch_entropy_loss += entropy_loss.item()

            n_batches = max(len(train_loader), 1)
            print(f"[BiLSTM-MemAE] Epoch {epoch}/{self.epochs} | "
                  f"MSE: {epoch_reconstruction_loss / n_batches:.6f} | "
                  f"Entropy: {epoch_entropy_loss / n_batches:.4f}")

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
                reconstructed, _, _ = self._network(batch.to(self.device))
                reconstructed_batches.append(reconstructed.cpu().numpy())

        reconstructed = np.concatenate(reconstructed_batches, axis=0)
        per_timestep_error = compute_per_timestep_error(sequences, reconstructed)
        return pool_temporal_errors(per_timestep_error, strategy=self.pooling_strategy, top_k=self.top_k)
