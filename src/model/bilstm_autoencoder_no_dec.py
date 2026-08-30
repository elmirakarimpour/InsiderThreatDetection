"""
bilstm_autoencoder_no_dec.py

مسئولیت: نسخه‌ی «نصفه»‌ی مدل اصلی — دقیقاً همان معماری رمزگذار-رمزگشای
BiLSTM دوطرفه، ولی بدون لایه‌ی خوشه‌بندی DEC. این مهم‌ترین baseline
برای اثبات ادعای اصلی پایان‌نامه است: آیا اضافه‌کردن خوشه‌بندی عمیق
واقعاً چیزی بهبود می‌دهد، یا فقط خود BiLSTM دوطرفه کافی بوده؟
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as functional
from torch.utils.data import DataLoader, TensorDataset

from src.evaluation.scoring import compute_per_timestep_error, pool_temporal_errors
from src.model.base_model import AnomalyDetectionModel


class _BiLSTMAutoencoderNoClusteringNetwork(nn.Module):
    """همان معماری رمزگذار/رمزگشای دوطرفه‌ی bilstm_dec.py، بدون ClusteringLayer."""

    def __init__(self, n_features: int, encoder_size: int, latent_size: int):
        super().__init__()
        if latent_size % 2 != 0:
            raise ValueError("latent_size باید زوج باشد چون رمزگذار دوطرفه است.")

        self.encoder_lstm_1 = nn.LSTM(n_features, encoder_size, batch_first=True, bidirectional=True)
        self.encoder_lstm_2 = nn.LSTM(encoder_size * 2, latent_size // 2, batch_first=True, bidirectional=True)
        self.decoder_lstm_1 = nn.LSTM(latent_size, latent_size // 2, batch_first=True, bidirectional=True)
        self.decoder_lstm_2 = nn.LSTM(latent_size, encoder_size, batch_first=True, bidirectional=True)
        self.output_layer = nn.Linear(encoder_size * 2, n_features)

    def forward(self, sequences):
        encoded, _ = self.encoder_lstm_1(sequences)
        _, (hidden_state, _) = self.encoder_lstm_2(encoded)
        latent_vectors = torch.cat((hidden_state[-2], hidden_state[-1]), dim=1)

        repeated_latent = latent_vectors.unsqueeze(1).expand(-1, sequences.size(1), -1)
        decoded, _ = self.decoder_lstm_1(repeated_latent)
        decoded, _ = self.decoder_lstm_2(decoded)
        return self.output_layer(decoded)


class BiLSTMAutoencoderNoDECModel(AnomalyDetectionModel):
    name = "bilstm_autoencoder_no_dec"

    def __init__(
        self,
        encoder_size: int = 64,
        latent_size: int = 64,
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

    def fit(self, train_sequences: np.ndarray) -> "BiLSTMAutoencoderNoDECModel":
        train_sequences = np.asarray(train_sequences, dtype=np.float32)
        torch.manual_seed(self.seed)
        self._network = _BiLSTMAutoencoderNoClusteringNetwork(
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
            print(f"[BiLSTM-AE-NoDEC] Epoch {epoch}/{self.epochs} | MSE: {epoch_loss / max(len(train_loader), 1):.6f}")

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
