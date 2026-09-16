"""
src/model/bilstm_dec_contrastive.py
=====================================
معماری واحد: یک BiLSTM encoder مشترک با سه هدف آموزشی هم‌زمان:
  1) reconstruction loss   (autoencoder، همان سبک BiLSTM-DEC اصلی)
  2) clustering loss       (KL divergence نسبت به توزیع هدف DEC)
  3) contrastive loss      (InfoNCE بین context کاربر و action پنجره،
                             از طریق یک projection head جدا از فضای
                             بازسازی/خوشه‌بندی)

معماری پنج‌جزئی (BiLSTM + autoencoder + DEC + context-aware + contrastive)
طبق تصمیم پروژه دست‌نخورده می‌ماند. آنچه در score_sequences تغییر کرده،
فقط وزن هر جزء در امتیاز نهایی است، بر اساس ablation مستند:
  - Reconstruction only : AUC تست ≈ 0.8366
  - Clustering only     : AUC تست ≈ 0.4978 (≈ شانس تصادفی)
  - Clustering + بازآموزی دوره‌ای : AUC ≈ 0.4435 (باز هم ≈ شانس)
  - Contrastive only    : AUC تست ≈ 0.8749 (بهترین تکی)
  - Joint (وزن‌های قدیمی): AUC تست ≈ 0.8508
پس پیش‌فرض‌های جدید: cluster_weight=0.0, reconstruction_weight=0.0.
این وزن‌ها پارامتر تابع‌اند، پس هر وقت شواهد تازه اومد راحت قابل تغییرن.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as functional
from sklearn.cluster import MiniBatchKMeans
from torch.utils.data import DataLoader, TensorDataset


def _validate_sequences(sequences: np.ndarray) -> np.ndarray:
    sequences = np.asarray(sequences, dtype=np.float32)
    if sequences.ndim != 3:
        raise ValueError("Sequences must have shape (samples, timesteps, features).")
    if len(sequences) == 0:
        raise ValueError("At least one sequence is required.")
    return sequences


class ClusteringLayer(nn.Module):
    """DEC clustering head based on Student's t-distribution."""

    def __init__(self, n_clusters: int, latent_size: int, alpha: float = 1.0):
        super().__init__()
        self.alpha = alpha
        self.cluster_centers = nn.Parameter(torch.empty(n_clusters, latent_size))
        nn.init.xavier_uniform_(self.cluster_centers)

    def forward(self, latent_vectors: torch.Tensor) -> torch.Tensor:
        squared_distances = torch.sum(
            (latent_vectors.unsqueeze(1) - self.cluster_centers).pow(2), dim=2
        )
        soft_assignments = 1.0 / (1.0 + squared_distances / self.alpha)
        soft_assignments = soft_assignments.pow((self.alpha + 1.0) / 2.0)
        return soft_assignments / soft_assignments.sum(dim=1, keepdim=True)


def target_distribution(soft_assignments: torch.Tensor) -> torch.Tensor:
    """Builds the DEC target distribution from global soft assignments."""
    cluster_frequency = soft_assignments.sum(dim=0, keepdim=True).clamp_min(1e-12)
    weights = soft_assignments.pow(2) / cluster_frequency
    return weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-12)


class BiLSTMDECContrastive(nn.Module):
    """
    یک encoder مشترک با سه سر خروجی:
      - decoder            -> بازسازی (reconstruction)
      - clustering_layer   -> خوشه‌بندی عمیق (DEC)
      - projection_head + user_embedding -> contrastive learning (InfoNCE)
    """

    def __init__(
        self,
        n_features: int,
        n_users_vocab: int,
        n_clusters: int = 3,
        encoder_size: int = 64,
        latent_size: int = 64,
        contrastive_dim: int = 64,
    ):
        super().__init__()
        if latent_size % 2 != 0:
            raise ValueError("latent_size must be even for a bidirectional encoder.")

        self.n_features = n_features
        self.n_users_vocab = n_users_vocab
        self.n_clusters = n_clusters
        self.encoder_size = encoder_size
        self.latent_size = latent_size
        self.contrastive_dim = contrastive_dim

        self.encoder_lstm_1 = nn.LSTM(
            input_size=n_features, hidden_size=encoder_size,
            batch_first=True, bidirectional=True,
        )
        self.encoder_lstm_2 = nn.LSTM(
            input_size=encoder_size * 2, hidden_size=latent_size // 2,
            batch_first=True, bidirectional=True,
        )

        self.decoder_lstm_1 = nn.LSTM(
            input_size=latent_size, hidden_size=latent_size // 2,
            batch_first=True, bidirectional=True,
        )
        self.decoder_lstm_2 = nn.LSTM(
            input_size=latent_size, hidden_size=encoder_size,
            batch_first=True, bidirectional=True,
        )
        self.output_layer = nn.Linear(encoder_size * 2, n_features)

        self.clustering_layer = ClusteringLayer(n_clusters, latent_size)

        self.projection_head = nn.Sequential(
            nn.Linear(latent_size, latent_size),
            nn.ReLU(),
            nn.Linear(latent_size, contrastive_dim),
        )
        self.user_embedding = nn.Embedding(n_users_vocab, contrastive_dim)

    def encode(self, sequences: torch.Tensor) -> torch.Tensor:
        encoded_sequences, _ = self.encoder_lstm_1(sequences)
        _, (hidden_state, _) = self.encoder_lstm_2(encoded_sequences)
        return torch.cat((hidden_state[-2], hidden_state[-1]), dim=1)

    def decode(self, latent_vectors: torch.Tensor, sequence_length: int) -> torch.Tensor:
        repeated_latent = latent_vectors.unsqueeze(1).expand(-1, sequence_length, -1)
        decoded_sequences, _ = self.decoder_lstm_1(repeated_latent)
        decoded_sequences, _ = self.decoder_lstm_2(decoded_sequences)
        return self.output_layer(decoded_sequences)

    def forward(self, sequences: torch.Tensor, user_indices: torch.Tensor) -> dict:
        latent_vectors = self.encode(sequences)
        reconstructed_sequences = self.decode(latent_vectors, sequences.size(1))
        soft_assignments = self.clustering_layer(latent_vectors)

        action_embedding = functional.normalize(self.projection_head(latent_vectors), dim=1)
        context_embedding = functional.normalize(self.user_embedding(user_indices), dim=1)

        return {
            "reconstructed": reconstructed_sequences,
            "latent": latent_vectors,
            "soft_assignments": soft_assignments,
            "action_embedding": action_embedding,
            "context_embedding": context_embedding,
        }


def _create_loader(sequences, user_indices, batch_size, shuffle, device):
    # torch.device("cuda") روی خودش idempotent است، پس چه caller رشته
    # بدهد و چه شیء device، اینجا همیشه درست کار می‌کند.
    device = torch.device(device)
    sequence_tensor = torch.from_numpy(_validate_sequences(sequences))
    user_tensor = torch.from_numpy(np.asarray(user_indices, dtype=np.int64))
    return DataLoader(
        TensorDataset(sequence_tensor, user_tensor),
        batch_size=batch_size,
        shuffle=shuffle,
        pin_memory=device.type == "cuda",
        num_workers=0,
    )


def build_user_vocabulary(train_user_ids: np.ndarray) -> dict:
    unique_users = sorted(set(train_user_ids.tolist()))
    return {user: index for index, user in enumerate(unique_users)}


def map_user_indices(user_ids: np.ndarray, user_to_index: dict, fallback_index: int = 0) -> np.ndarray:
    return np.array([user_to_index.get(user, fallback_index) for user in user_ids], dtype=np.int64)


def contrastive_infonce_loss(
    action_embedding: torch.Tensor, context_embedding: torch.Tensor, temperature: float
) -> torch.Tensor:
    """InfoNCE متقارن: هر context باید action متناظر خودش را از میان همه‌ی action های batch تشخیص دهد و برعکس."""
    similarity = context_embedding @ action_embedding.T / temperature
    labels = torch.arange(len(action_embedding), device=action_embedding.device)
    return functional.cross_entropy(similarity, labels) + functional.cross_entropy(similarity.T, labels)


def _extract_latent_vectors(model, sequences, user_indices, device, batch_size):
    data_loader = _create_loader(sequences, user_indices, batch_size, shuffle=False, device=device)
    collected_latents = []
    model.eval()
    with torch.no_grad():
        for batch_sequences, batch_users in data_loader:
            outputs = model(
                batch_sequences.to(device, non_blocking=True),
                batch_users.to(device, non_blocking=True),
            )
            collected_latents.append(outputs["latent"].cpu().numpy())
    return np.concatenate(collected_latents, axis=0)


def initialize_cluster_centers(model, train_sequences, train_user_indices, device, batch_size=512, seed=42):
    """Initializes DEC centers using MiniBatch K-Means after shared-encoder pretraining."""
    latent_vectors = _extract_latent_vectors(model, train_sequences, train_user_indices, device, batch_size)
    if len(latent_vectors) < model.n_clusters:
        raise ValueError("The number of training sequences must be at least n_clusters.")

    kmeans = MiniBatchKMeans(
        n_clusters=model.n_clusters,
        batch_size=min(4096, len(latent_vectors)),
        n_init=10,
        random_state=seed,
    )
    kmeans.fit(latent_vectors)
    cluster_centers = torch.as_tensor(kmeans.cluster_centers_, dtype=torch.float32, device=device)
    model.clustering_layer.cluster_centers.data.copy_(cluster_centers)


def _compute_global_targets(model, train_sequences, train_user_indices, device, batch_size):
    data_loader = _create_loader(train_sequences, train_user_indices, batch_size, shuffle=False, device=device)
    collected_assignments = []
    model.eval()
    with torch.no_grad():
        for batch_sequences, batch_users in data_loader:
            outputs = model(
                batch_sequences.to(device, non_blocking=True),
                batch_users.to(device, non_blocking=True),
            )
            collected_assignments.append(outputs["soft_assignments"].cpu())
    return target_distribution(torch.cat(collected_assignments, dim=0))


def train_bilstm_dec_contrastive(
    model: BiLSTMDECContrastive,
    train_sequences: np.ndarray,
    train_user_ids: np.ndarray,
    device: torch.device,
    pretrain_epochs: int = 20,
    joint_epochs: int = 15,
    batch_size: int = 512,
    learning_rate: float = 1e-3,
    gamma: float = 0.1,
    delta: float = 1.0,
    temperature: float = 0.1,
    seed: int = 42,
):
    """
    آموزش دو فازی:
      فاز ۱ - Pretraining: فقط reconstruction loss.
      فاز ۲ - Joint fine-tuning: reconstruction + gamma * clustering (KL)
               + delta * contrastive (InfoNCE) روی encoder مشترک.
    """
    torch.manual_seed(seed)
    train_sequences = _validate_sequences(train_sequences)
    if len(train_sequences) < model.n_clusters:
        raise ValueError("The training split must contain at least n_clusters sequences.")

    user_to_index = build_user_vocabulary(train_user_ids)
    train_user_indices = map_user_indices(train_user_ids, user_to_index)

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    train_loader = _create_loader(train_sequences, train_user_indices, batch_size, shuffle=True, device=device)

    history = {
        "pretrain_reconstruction": [],
        "joint_loss": [],
        "reconstruction_loss": [],
        "clustering_loss": [],
        "contrastive_loss": [],
    }

    print("\n--- Phase 1: Shared-Encoder Pretraining (reconstruction only) ---")
    for epoch in range(1, pretrain_epochs + 1):
        model.train()
        total_loss = 0.0
        for batch_sequences, batch_users in train_loader:
            batch_sequences = batch_sequences.to(device, non_blocking=True)
            batch_users = batch_users.to(device, non_blocking=True)
            outputs = model(batch_sequences, batch_users)
            reconstruction_loss = functional.mse_loss(outputs["reconstructed"], batch_sequences)

            optimizer.zero_grad(set_to_none=True)
            reconstruction_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            total_loss += reconstruction_loss.item()

        mean_loss = total_loss / max(len(train_loader), 1)
        history["pretrain_reconstruction"].append(mean_loss)
        print(f"Pretrain epoch {epoch:02d}/{pretrain_epochs} | MSE: {mean_loss:.6f}")

    initialize_cluster_centers(model, train_sequences, train_user_indices, device, batch_size, seed)

    print("\n--- Phase 2: Joint Fine-tuning (reconstruction + clustering + contrastive) ---")
    train_sequence_tensor = torch.from_numpy(train_sequences)
    train_user_tensor = torch.from_numpy(train_user_indices)
    for epoch in range(1, joint_epochs + 1):
        dec_targets = _compute_global_targets(model, train_sequences, train_user_indices, device, batch_size)
        joint_loader = DataLoader(
            TensorDataset(train_sequence_tensor, train_user_tensor, dec_targets),
            batch_size=batch_size,
            shuffle=True,
            pin_memory=device.type == "cuda",
            num_workers=0,
        )

        model.train()
        total_joint = total_reconstruction = total_clustering = total_contrastive = 0.0
        for batch_sequences, batch_users, batch_targets in joint_loader:
            batch_sequences = batch_sequences.to(device, non_blocking=True)
            batch_users = batch_users.to(device, non_blocking=True)
            batch_targets = batch_targets.to(device, non_blocking=True)
            outputs = model(batch_sequences, batch_users)

            reconstruction_loss = functional.mse_loss(outputs["reconstructed"], batch_sequences)
            clustering_loss = functional.kl_div(
                outputs["soft_assignments"].clamp_min(1e-12).log(),
                batch_targets,
                reduction="batchmean",
            )
            contrastive_loss = contrastive_infonce_loss(
                outputs["action_embedding"], outputs["context_embedding"], temperature
            )
            joint_loss = reconstruction_loss + gamma * clustering_loss + delta * contrastive_loss

            optimizer.zero_grad(set_to_none=True)
            joint_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            total_joint += joint_loss.item()
            total_reconstruction += reconstruction_loss.item()
            total_clustering += clustering_loss.item()
            total_contrastive += contrastive_loss.item()

        n_batches = max(len(joint_loader), 1)
        history["joint_loss"].append(total_joint / n_batches)
        history["reconstruction_loss"].append(total_reconstruction / n_batches)
        history["clustering_loss"].append(total_clustering / n_batches)
        history["contrastive_loss"].append(total_contrastive / n_batches)
        print(
            f"Joint epoch {epoch:02d}/{joint_epochs} | "
            f"Joint: {history['joint_loss'][-1]:.6f} | "
            f"MSE: {history['reconstruction_loss'][-1]:.6f} | "
            f"KL: {history['clustering_loss'][-1]:.6f} | "
            f"InfoNCE: {history['contrastive_loss'][-1]:.6f}"
        )

    return model, user_to_index, history


def score_sequences(
    model: BiLSTMDECContrastive,
    sequences: np.ndarray,
    user_ids: np.ndarray,
    user_to_index: dict,
    device: torch.device,
    batch_size: int = 512,
    cluster_weight: float = 0.0,          # ← تغییر: قبلاً 0.1 بود؛ طبق ablation مستند فاز ۱، اکنون 0.0
    reconstruction_weight: float = 0.0,   # ← جدید: قبلاً همیشه با وزن ثابت 1.0 جمع می‌شد
    contrastive_weight: float = 1.0,
) -> dict:
    """
    امتیاز آنومالی نهایی = reconstruction_weight * reconstruction_error
                          + cluster_weight * cluster_uncertainty
                          + contrastive_weight * contrastive_mismatch
    پیش‌فرض‌های فعلی معادل «فقط contrastive» است، بر اساس ablation مستند
    (فایل‌های contrastive_dec_score_ablation_report.json و
    periodic_recluster_report.json). این پارامترها را دستی صفر نکن؛
    اگر شواهد جدیدی به دست آمد، اینجا مستند و تغییرشان بده.
    """
    user_indices = map_user_indices(user_ids, user_to_index)
    data_loader = _create_loader(sequences, user_indices, batch_size, shuffle=False, device=device)

    reconstruction_scores, clustering_scores, contrastive_scores = [], [], []
    model.eval()
    with torch.no_grad():
        for batch_sequences, batch_users in data_loader:
            batch_sequences = batch_sequences.to(device, non_blocking=True)
            batch_users = batch_users.to(device, non_blocking=True)
            outputs = model(batch_sequences, batch_users)

            reconstruction_error = (batch_sequences - outputs["reconstructed"]).pow(2).mean(dim=(1, 2))
            cluster_uncertainty = 1.0 - outputs["soft_assignments"].max(dim=1).values
            contrastive_mismatch = 1.0 - (outputs["action_embedding"] * outputs["context_embedding"]).sum(dim=1)

            reconstruction_scores.append(reconstruction_error.cpu().numpy())
            clustering_scores.append(cluster_uncertainty.cpu().numpy())
            contrastive_scores.append(contrastive_mismatch.cpu().numpy())

    reconstruction_scores = np.concatenate(reconstruction_scores)
    clustering_scores = np.concatenate(clustering_scores)
    contrastive_scores = np.concatenate(contrastive_scores)
    joint_scores = (
        reconstruction_weight * reconstruction_scores
        + cluster_weight * clustering_scores
        + contrastive_weight * contrastive_scores
    )

    return {
        "reconstruction_score": reconstruction_scores,
        "cluster_score": clustering_scores,
        "contrastive_score": contrastive_scores,
        "joint_score": joint_scores,
    }


def save_model_checkpoint(model, user_to_index, feature_names, training_history, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_configuration": {
                "n_features": model.n_features,
                "n_users_vocab": model.n_users_vocab,
                "n_clusters": model.n_clusters,
                "encoder_size": model.encoder_size,
                "latent_size": model.latent_size,
                "contrastive_dim": model.contrastive_dim,
            },
            "user_to_index": user_to_index,
            "feature_names": feature_names,
            "training_history": training_history,
        },
        output_path,
    )


if __name__ == "__main__":
    print("This module defines the unified BiLSTM-DEC + Contrastive architecture.")
    print("Run 05_train_contrastive_dec.py in Colab to train and cache results.")