"""
transformer_autoencoder.py
============================
Baseline دهم، تنها موردی که استاد راهنما خواسته و در repo موجود نبود.
یک رمزگذار-رمزگشای مبتنی بر Transformer (self-attention)، بدون هیچ
لایه‌ی بازگشتی (LSTM/GRU) و بدون خوشه‌بندی DEC — دقیقاً هم‌تراز با
baselineهای موجود (lstm_autoencoder.py، gru_autoencoder.py) از نظر
ساختار کد، تا مقایسه‌ی منصفانه باشد: تفاوت فقط از معماری بیاید، نه از
تفاوت نحوه‌ی محاسبه‌ی امتیاز.

معماری: encoder یک TransformerEncoder است که نمایش‌های زمینه‌مند
تولید می‌کند؛ میانگین‌گیری روی زمان یک بردار نهان (latent) با ابعاد
ثابت می‌سازد (همان ایده‌ی «گلوگاه» که در LSTM-AE با آخرین hidden state
پیاده‌سازی شده بود)؛ سپس این بردار به تعداد timestepها تکرار می‌شود،
embedding موقعیتی اضافه می‌شود، و یک TransformerEncoder دوم (نقش
رمزگشا) دنباله را بازسازی می‌کند.

باید در src/model/ (کنار بقیه‌ی baselineها) قرار بگیرد.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as functional
from torch.utils.data import DataLoader, TensorDataset

from src.evaluation.scoring import compute_per_timestep_error, pool_temporal_errors
from src.model.base_model import AnomalyDetectionModel


class _PositionalEmbedding(nn.Module):
    """embedding موقعیتی یادگرفته‌شده — چون طول دنباله (۷ روز) ثابت و کوتاه است، نیازی به sinusoidal نیست."""

    def __init__(self, sequence_length: int, d_model: int):
        super().__init__()
        self.position_embedding = nn.Parameter(torch.zeros(1, sequence_length, d_model))
        nn.init.trunc_normal_(self.position_embedding, std=0.02)

    def forward(self, x):
        return x + self.position_embedding


class _TransformerAutoencoderNetwork(nn.Module):
    """رمزگذار-رمزگشای مبتنی بر Transformer، بدون هیچ لایه‌ی بازگشتی."""

    def __init__(self, n_features: int, sequence_length: int, d_model: int, nhead: int, num_layers: int, latent_size: int):
        super().__init__()
        self.input_projection = nn.Linear(n_features, d_model)
        self.encoder_positional = _PositionalEmbedding(sequence_length, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 2, batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.to_latent = nn.Linear(d_model, latent_size)

        self.from_latent = nn.Linear(latent_size, d_model)
        self.decoder_positional = _PositionalEmbedding(sequence_length, d_model)
        decoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 2, batch_first=True,
        )
        self.decoder = nn.TransformerEncoder(decoder_layer, num_layers=num_layers)
        self.output_layer = nn.Linear(d_model, n_features)

    def forward(self, sequences):
        encoded = self.input_projection(sequences)
        encoded = self.encoder_positional(encoded)
        encoded = self.encoder(encoded)

        latent_vector = self.to_latent(encoded.mean(dim=1))  # میانگین‌گیری روی زمان = گلوگاه

        repeated_latent = self.from_latent(latent_vector).unsqueeze(1).expand(-1, sequences.size(1), -1)
        decoded = self.decoder_positional(repeated_latent)
        decoded = self.decoder(decoded)
        reconstructed = self.output_layer(decoded)
        return reconstructed


class TransformerAutoencoderModel(AnomalyDetectionModel):
    name = "transformer_autoencoder"

    def __init__(
        self,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        latent_size: int = 32,
        pooling_strategy: str = "max",
        top_k: int = 2,
        epochs: int = 20,
        batch_size: int = 512,
        learning_rate: float = 1e-3,
        device: str = None,
        seed: int = 42,
    ):
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
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

    def fit(self, train_sequences: np.ndarray) -> "TransformerAutoencoderModel":
        train_sequences = np.asarray(train_sequences, dtype=np.float32)
        torch.manual_seed(self.seed)
        self._network = _TransformerAutoencoderNetwork(
            n_features=train_sequences.shape[2],
            sequence_length=train_sequences.shape[1],
            d_model=self.d_model,
            nhead=self.nhead,
            num_layers=self.num_layers,
            latent_size=self.latent_size,
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
            print(f"[Transformer-AE] Epoch {epoch}/{self.epochs} | MSE: {epoch_loss / max(len(train_loader), 1):.6f}")

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