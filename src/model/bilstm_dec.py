"""
bilstm_dec.py

مسئولیت: مدل اصلی پیشنهادی پایان‌نامه — ترکیب رمزگذار-رمزگشای BiLSTM
با خوشه‌بندی عمیق (DEC). این فایل سه بخش دارد:

    ۱. ClusteringLayer   -> لایه‌ی خوشه‌بندی نرم (soft assignment) طبق DEC
    ۲. _BiLSTMAutoencoder -> معماری شبکه‌ی عصبی (nn.Module خام PyTorch)
    ۳. BiLSTMDECModel    -> کلاس نهایی که AnomalyDetectionModel را
                             پیاده‌سازی می‌کند (fit/score) و همه‌چیز
                             (آموزش دومرحله‌ای، pooling قابل‌تنظیم) را
                             به‌هم وصل می‌کند.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as functional
from sklearn.cluster import MiniBatchKMeans
from torch.utils.data import DataLoader, TensorDataset

from src.evaluation.scoring import compute_per_timestep_error, pool_temporal_errors
from src.model.base_model import AnomalyDetectionModel


class ClusteringLayer(nn.Module):
    """
    لایه‌ی خوشه‌بندی نرم طبق روش DEC (Deep Embedded Clustering).
    به هر بردار latent، احتمال تعلق به هر یک از n_clusters خوشه را
    می‌دهد — بر پایه‌ی فاصله تا مرکز هر خوشه (توزیع t-Student).
    """

    def __init__(self, n_clusters: int, latent_size: int, alpha: float = 1.0):
        super().__init__()
        self.alpha = alpha
        self.cluster_centers = nn.Parameter(torch.empty(n_clusters, latent_size))
        nn.init.xavier_uniform_(self.cluster_centers)

    def forward(self, latent_vectors):
        squared_distances = torch.sum(
            (latent_vectors.unsqueeze(1) - self.cluster_centers).pow(2), dim=2
        )
        soft_assignments = 1.0 / (1.0 + squared_distances / self.alpha)
        soft_assignments = soft_assignments.pow((self.alpha + 1.0) / 2.0)
        return soft_assignments / soft_assignments.sum(dim=1, keepdim=True)


def dec_target_distribution(soft_assignments: torch.Tensor) -> torch.Tensor:
    """
    توزیع هدف DEC را می‌سازد — خوشه‌هایی که مدل به آن‌ها مطمئن‌تر است
    را «تقویت» می‌کند، تا آموزش هر دور کمی قاطع‌تر از دور قبل شود.
    """
    cluster_frequency = soft_assignments.sum(dim=0, keepdim=True).clamp_min(1e-12)
    weights = soft_assignments.pow(2) / cluster_frequency
    return weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-12)


class _BiLSTMAutoencoder(nn.Module):
    """معماری خام: رمزگذار BiLSTM دولایه + رمزگشای BiLSTM دولایه + لایه‌ی خوشه‌بندی."""

    def __init__(self, n_features: int, n_clusters: int, encoder_size: int, latent_size: int):
        super().__init__()
        if latent_size % 2 != 0:
            raise ValueError("latent_size باید زوج باشد چون رمزگذار دوطرفه (bidirectional) است.")

        self.encoder_lstm_1 = nn.LSTM(n_features, encoder_size, batch_first=True, bidirectional=True)
        self.encoder_lstm_2 = nn.LSTM(encoder_size * 2, latent_size // 2, batch_first=True, bidirectional=True)
        self.decoder_lstm_1 = nn.LSTM(latent_size, latent_size // 2, batch_first=True, bidirectional=True)
        self.decoder_lstm_2 = nn.LSTM(latent_size, encoder_size, batch_first=True, bidirectional=True)
        self.output_layer = nn.Linear(encoder_size * 2, n_features)
        self.clustering_layer = ClusteringLayer(n_clusters, latent_size)

    def forward(self, sequences):
        encoded, _ = self.encoder_lstm_1(sequences)
        _, (hidden_state, _) = self.encoder_lstm_2(encoded)
        # hidden_state[-2] و hidden_state[-1] دو جهت آخرین لایه‌ی BiLSTM هستند (جلو و عقب)
        latent_vectors = torch.cat((hidden_state[-2], hidden_state[-1]), dim=1)
        soft_assignments = self.clustering_layer(latent_vectors)

        repeated_latent = latent_vectors.unsqueeze(1).expand(-1, sequences.size(1), -1)
        decoded, _ = self.decoder_lstm_1(repeated_latent)
        decoded, _ = self.decoder_lstm_2(decoded)
        reconstructed = self.output_layer(decoded)

        return reconstructed, latent_vectors, soft_assignments


class BiLSTMDECModel(AnomalyDetectionModel):
    """
    نسخه‌ی نهایی مدل که با بقیه‌ی پروژه (ablation_runner، evaluation و...)
    از طریق قرارداد AnomalyDetectionModel صحبت می‌کند.

    پارامترهای مهم برای آزمایش (hyperparameter sweep):
        gamma:            وزن clustering loss نسبت به reconstruction loss.
                           در نسخه‌ی قبلی ۰.۱۰ بود که خیلی کم به‌نظر می‌رسید.
        pooling_strategy: "mean" / "max" / "topk" — نحوه‌ی ترکیب خطای
                           ۷ روز پنجره به یک عدد نهایی (رجوع به scoring.py).
    """

    name = "bilstm_dec"

    def __init__(
        self,
        n_clusters: int = 5,
        encoder_size: int = 64,
        latent_size: int = 64,
        gamma: float = 0.5,
        pooling_strategy: str = "max",
        top_k: int = 2,
        pretrain_epochs: int = 20,
        fine_tune_epochs: int = 15,
        batch_size: int = 512,
        learning_rate: float = 1e-3,
        cluster_score_weight: float = 0.1,
        device: str = None,
        seed: int = 42,
    ):
        self.n_clusters = n_clusters
        self.encoder_size = encoder_size
        self.latent_size = latent_size
        self.gamma = gamma
        self.pooling_strategy = pooling_strategy
        self.top_k = top_k
        self.pretrain_epochs = pretrain_epochs
        self.fine_tune_epochs = fine_tune_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.cluster_score_weight = cluster_score_weight
        self.seed = seed
        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))

        self._network = None  # تا وقتی fit صدا زده نشده، معماری ساخته نمی‌شود (چون n_features را نمی‌دانیم)

    def _build_network(self, n_features: int):
        torch.manual_seed(self.seed)
        return _BiLSTMAutoencoder(n_features, self.n_clusters, self.encoder_size, self.latent_size).to(self.device)

    def _make_loader(self, sequences: np.ndarray, shuffle: bool):
        tensor = torch.from_numpy(np.asarray(sequences, dtype=np.float32))
        return DataLoader(TensorDataset(tensor), batch_size=self.batch_size, shuffle=shuffle, num_workers=0)

    def fit(self, train_sequences: np.ndarray) -> "BiLSTMDECModel":
        train_sequences = np.asarray(train_sequences, dtype=np.float32)
        n_features = train_sequences.shape[2]
        self._network = self._build_network(n_features)
        optimizer = torch.optim.Adam(self._network.parameters(), lr=self.learning_rate)
        train_loader = self._make_loader(train_sequences, shuffle=True)

        # ---- فاز ۱: پیش‌آموزش autoencoder (بدون خوشه‌بندی) ----
        self._network.train()
        for _ in range(self.pretrain_epochs):
            for (batch,) in train_loader:
                batch = batch.to(self.device)
                reconstructed, _, _ = self._network(batch)
                loss = functional.mse_loss(reconstructed, batch)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self._network.parameters(), max_norm=5.0)
                optimizer.step()

        # ---- مقداردهی اولیه‌ی مراکز خوشه با KMeans روی latent space ----
        latent_vectors = self._extract_latent_vectors(train_sequences)
        kmeans = MiniBatchKMeans(n_clusters=self.n_clusters, n_init=10, random_state=self.seed)
        kmeans.fit(latent_vectors)
        self._network.clustering_layer.cluster_centers.data.copy_(
            torch.as_tensor(kmeans.cluster_centers_, dtype=torch.float32, device=self.device)
        )

        # ---- فاز ۲: آموزش مشترک (reconstruction + gamma * clustering) ----
        train_tensor = torch.from_numpy(train_sequences)
        for _ in range(self.fine_tune_epochs):
            targets = self._compute_dec_targets(train_sequences)
            joint_loader = DataLoader(
                TensorDataset(train_tensor, targets), batch_size=self.batch_size, shuffle=True, num_workers=0
            )
            self._network.train()
            for batch, batch_targets in joint_loader:
                batch = batch.to(self.device)
                batch_targets = batch_targets.to(self.device)
                reconstructed, _, assignments = self._network(batch)

                reconstruction_loss = functional.mse_loss(reconstructed, batch)
                clustering_loss = functional.kl_div(
                    assignments.clamp_min(1e-12).log(), batch_targets, reduction="batchmean"
                )
                joint_loss = reconstruction_loss + self.gamma * clustering_loss

                optimizer.zero_grad(set_to_none=True)
                joint_loss.backward()
                torch.nn.utils.clip_grad_norm_(self._network.parameters(), max_norm=5.0)
                optimizer.step()

        return self

    def score(self, sequences: np.ndarray) -> np.ndarray:
        if self._network is None:
            raise RuntimeError("قبل از score باید fit صدا زده شود.")

        sequences = np.asarray(sequences, dtype=np.float32)
        loader = self._make_loader(sequences, shuffle=False)
        reconstructed_batches = []
        cluster_uncertainty_batches = []

        self._network.eval()
        with torch.no_grad():
            for (batch,) in loader:
                batch = batch.to(self.device)
                reconstructed, _, assignments = self._network(batch)
                reconstructed_batches.append(reconstructed.cpu().numpy())
                cluster_uncertainty_batches.append((1.0 - assignments.max(dim=1).values).cpu().numpy())

        reconstructed = np.concatenate(reconstructed_batches, axis=0)
        cluster_uncertainty = np.concatenate(cluster_uncertainty_batches, axis=0)

        # === اینجا دقیقاً همان اصلاح یافته‌شده اعمال می‌شود ===
        per_timestep_error = compute_per_timestep_error(sequences, reconstructed)
        reconstruction_score = pool_temporal_errors(
            per_timestep_error, strategy=self.pooling_strategy, top_k=self.top_k
        )

        return reconstruction_score + self.cluster_score_weight * cluster_uncertainty

    def _extract_latent_vectors(self, sequences: np.ndarray) -> np.ndarray:
        loader = self._make_loader(sequences, shuffle=False)
        latents = []
        self._network.eval()
        with torch.no_grad():
            for (batch,) in loader:
                _, batch_latents, _ = self._network(batch.to(self.device))
                latents.append(batch_latents.cpu().numpy())
        return np.concatenate(latents, axis=0)

    def _compute_dec_targets(self, sequences: np.ndarray) -> torch.Tensor:
        loader = self._make_loader(sequences, shuffle=False)
        assignments_all = []
        self._network.eval()
        with torch.no_grad():
            for (batch,) in loader:
                _, _, assignments = self._network(batch.to(self.device))
                assignments_all.append(assignments.cpu())
        return dec_target_distribution(torch.cat(assignments_all, dim=0))
