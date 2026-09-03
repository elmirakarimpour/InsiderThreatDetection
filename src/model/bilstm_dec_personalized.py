"""
bilstm_dec_personalized.py

مسئولیت: نسخه‌ی شخصی‌سازی‌شده‌ی مدل اصلی. دقیقاً همان معماری
BiLSTM-DEC قبلی، با یک تفاوت: قبل از رمزگشایی و خوشه‌بندی، یک بردار
«هویت کاربر» (که فقط از روی شناسه‌ی کاربر یاد گرفته می‌شود، نه از روی
هیچ لیبل تهدیدی) به بردار latent اضافه می‌شود.

چرا این نقض بدون‌نظارت‌بودن نیست؟
چون فقط از «این پنجره مال کدام کاربر است» استفاده می‌کنیم (که یک واقعیت
ساختاری داده است، نه یک لیبل تهدید)، دقیقاً هم‌خانواده‌ی استفاده از
department/role که از قبل در پروژه داشتیم.

هشدار مهم درباره‌ی رابط (interface):
برخلاف بقیه‌ی مدل‌های پروژه، این مدل به AnomalyDetectionModel استاندارد
(fit(sequences) / score(sequences)) پایبند نیست — چون به شناسه‌ی کاربر
هر پنجره هم نیاز دارد. متدهایش fit(sequences, user_ids) و
score(sequences, user_ids) هستند. این یعنی نمی‌توان این مدل را مستقیم
و بدون تغییر در run_full_model_comparison.py استفاده کرد.

مشکل شناخته‌شده (cold start): اگر کاربری فقط در validation/test دیده
شود و هیچ روزی در train نداشته باشد، مدل embedding مخصوص او را یاد
نگرفته است. برای این کاربرها از یک بردار «ناشناس» (UNKNOWN) مشترک
استفاده می‌شود.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as functional
from sklearn.cluster import MiniBatchKMeans
from torch.utils.data import DataLoader, TensorDataset

from src.evaluation.scoring import compute_per_timestep_error, pool_temporal_errors
from src.model.bilstm_dec import ClusteringLayer, dec_target_distribution

UNKNOWN_USER_INDEX = 0  # اندیس ۰ همیشه برای کاربرهای دیده‌نشده در train رزرو است


class _PersonalizedBiLSTMNetwork(nn.Module):
    def __init__(self, n_features: int, n_users_in_vocab: int, n_clusters: int, encoder_size: int, latent_size: int):
        super().__init__()
        if latent_size % 2 != 0:
            raise ValueError("latent_size باید زوج باشد چون رمزگذار دوطرفه است.")

        self.encoder_lstm_1 = nn.LSTM(n_features, encoder_size, batch_first=True, bidirectional=True)
        self.encoder_lstm_2 = nn.LSTM(encoder_size * 2, latent_size // 2, batch_first=True, bidirectional=True)

        # اندیس ۰ = UNKNOWN، بقیه = کاربرهای دیده‌شده در train
        self.user_embedding = nn.Embedding(num_embeddings=n_users_in_vocab, embedding_dim=latent_size)
        nn.init.zeros_(self.user_embedding.weight)  # شروع از صفر: اول مثل مدل غیرشخصی‌سازی‌شده رفتار می‌کند

        self.clustering_layer = ClusteringLayer(n_clusters, latent_size)
        self.decoder_lstm_1 = nn.LSTM(latent_size, latent_size // 2, batch_first=True, bidirectional=True)
        self.decoder_lstm_2 = nn.LSTM(latent_size, encoder_size, batch_first=True, bidirectional=True)
        self.output_layer = nn.Linear(encoder_size * 2, n_features)

    def forward(self, sequences, user_indices):
        encoded, _ = self.encoder_lstm_1(sequences)
        _, (hidden_state, _) = self.encoder_lstm_2(encoded)
        raw_latent = torch.cat((hidden_state[-2], hidden_state[-1]), dim=1)

        user_context = self.user_embedding(user_indices)  # (batch, latent_size)
        personalized_latent = raw_latent + user_context

        soft_assignments = self.clustering_layer(personalized_latent)

        repeated_latent = personalized_latent.unsqueeze(1).expand(-1, sequences.size(1), -1)
        decoded, _ = self.decoder_lstm_1(repeated_latent)
        decoded, _ = self.decoder_lstm_2(decoded)
        reconstructed = self.output_layer(decoded)

        return reconstructed, personalized_latent, soft_assignments


class BiLSTMDECPersonalizedModel:
    """
    نسخه‌ی شخصی‌سازی‌شده. توجه: از AnomalyDetectionModel ارث‌بری
    نمی‌کند چون امضای fit/score متفاوتی دارد (نیاز به user_ids).
    """

    name = "bilstm_dec_personalized"

    def __init__(
        self,
        n_clusters: int = 3,
        encoder_size: int = 64,
        latent_size: int = 64,
        gamma: float = 0.5,
        pooling_strategy: str = "max",
        top_k: int = 2,
        pretrain_epochs: int = 20,
        fine_tune_epochs: int = 15,
        batch_size: int = 512,
        learning_rate: float = 1e-3,
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
        self.seed = seed
        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
        self._network = None
        self._user_to_index = {}

    def _encode_user_ids(self, user_ids) -> np.ndarray:
        """شناسه‌های متنی کاربر را به اندیس عددی تبدیل می‌کند (کاربرهای ناشناس -> UNKNOWN_USER_INDEX)."""
        return np.array(
            [self._user_to_index.get(user_id, UNKNOWN_USER_INDEX) for user_id in user_ids],
            dtype=np.int64,
        )

    def _make_loader(self, sequences, user_indices, shuffle):
        sequence_tensor = torch.from_numpy(np.asarray(sequences, dtype=np.float32))
        user_tensor = torch.from_numpy(np.asarray(user_indices, dtype=np.int64))
        return DataLoader(
            TensorDataset(sequence_tensor, user_tensor), batch_size=self.batch_size, shuffle=shuffle, num_workers=0
        )

    def fit(self, train_sequences: np.ndarray, train_user_ids) -> "BiLSTMDECPersonalizedModel":
        train_sequences = np.asarray(train_sequences, dtype=np.float32)
        if len(train_user_ids) != len(train_sequences):
            raise ValueError("train_user_ids باید هم‌طول train_sequences باشد.")

        # ساخت واژه‌نامه‌ی کاربر فقط از روی train (اندیس ۰ رزرو UNKNOWN است)
        unique_users = sorted(set(train_user_ids))
        self._user_to_index = {user_id: index + 1 for index, user_id in enumerate(unique_users)}
        vocab_size = len(unique_users) + 1

        torch.manual_seed(self.seed)
        self._network = _PersonalizedBiLSTMNetwork(
            train_sequences.shape[2], vocab_size, self.n_clusters, self.encoder_size, self.latent_size
        ).to(self.device)
        optimizer = torch.optim.Adam(self._network.parameters(), lr=self.learning_rate)

        train_user_indices = self._encode_user_ids(train_user_ids)
        train_loader = self._make_loader(train_sequences, train_user_indices, shuffle=True)

        # ---- فاز ۱: پیش‌آموزش autoencoder ----
        self._network.train()
        for epoch in range(1, self.pretrain_epochs + 1):
            epoch_loss = 0.0
            for batch, user_batch in train_loader:
                batch, user_batch = batch.to(self.device), user_batch.to(self.device)
                reconstructed, _, _ = self._network(batch, user_batch)
                loss = functional.mse_loss(reconstructed, batch)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self._network.parameters(), max_norm=5.0)
                optimizer.step()
                epoch_loss += loss.item()
            print(f"[BiLSTM-DEC-Personalized] Pretrain epoch {epoch}/{self.pretrain_epochs} "
                  f"| MSE: {epoch_loss / max(len(train_loader), 1):.6f}")

        # ---- مقداردهی اولیه‌ی خوشه‌ها با KMeans روی latent شخصی‌سازی‌شده ----
        latent_vectors = self._extract_latent_vectors(train_sequences, train_user_indices)
        kmeans = MiniBatchKMeans(n_clusters=self.n_clusters, n_init=10, random_state=self.seed)
        kmeans.fit(latent_vectors)
        self._network.clustering_layer.cluster_centers.data.copy_(
            torch.as_tensor(kmeans.cluster_centers_, dtype=torch.float32, device=self.device)
        )

        # ---- فاز ۲: آموزش مشترک ----
        sequence_tensor = torch.from_numpy(train_sequences)
        user_index_tensor = torch.from_numpy(train_user_indices)
        for epoch in range(1, self.fine_tune_epochs + 1):
            targets = self._compute_dec_targets(train_sequences, train_user_indices)
            joint_loader = DataLoader(
                TensorDataset(sequence_tensor, user_index_tensor, targets),
                batch_size=self.batch_size, shuffle=True, num_workers=0,
            )
            self._network.train()
            epoch_joint_loss = 0.0
            for batch, user_batch, batch_targets in joint_loader:
                batch = batch.to(self.device)
                user_batch = user_batch.to(self.device)
                batch_targets = batch_targets.to(self.device)
                reconstructed, _, assignments = self._network(batch, user_batch)

                reconstruction_loss = functional.mse_loss(reconstructed, batch)
                clustering_loss = functional.kl_div(
                    assignments.clamp_min(1e-12).log(), batch_targets, reduction="batchmean"
                )
                joint_loss = reconstruction_loss + self.gamma * clustering_loss

                optimizer.zero_grad(set_to_none=True)
                joint_loss.backward()
                torch.nn.utils.clip_grad_norm_(self._network.parameters(), max_norm=5.0)
                optimizer.step()
                epoch_joint_loss += joint_loss.item()

            print(f"[BiLSTM-DEC-Personalized] Fine-tune epoch {epoch}/{self.fine_tune_epochs} "
                  f"| Joint loss: {epoch_joint_loss / max(len(joint_loader), 1):.6f}")

        return self

    def score(self, sequences: np.ndarray, user_ids) -> np.ndarray:
        if self._network is None:
            raise RuntimeError("قبل از score باید fit صدا زده شود.")

        sequences = np.asarray(sequences, dtype=np.float32)
        user_indices = self._encode_user_ids(user_ids)
        loader = self._make_loader(sequences, user_indices, shuffle=False)
        reconstructed_batches = []

        self._network.eval()
        with torch.no_grad():
            for batch, user_batch in loader:
                reconstructed, _, _ = self._network(batch.to(self.device), user_batch.to(self.device))
                reconstructed_batches.append(reconstructed.cpu().numpy())

        reconstructed = np.concatenate(reconstructed_batches, axis=0)
        per_timestep_error = compute_per_timestep_error(sequences, reconstructed)
        return pool_temporal_errors(per_timestep_error, strategy=self.pooling_strategy, top_k=self.top_k)

    def fit_score(self, train_sequences, train_user_ids, eval_sequences, eval_user_ids) -> np.ndarray:
        self.fit(train_sequences, train_user_ids)
        return self.score(eval_sequences, eval_user_ids)

    def _extract_latent_vectors(self, sequences, user_indices) -> np.ndarray:
        loader = self._make_loader(sequences, user_indices, shuffle=False)
        latents = []
        self._network.eval()
        with torch.no_grad():
            for batch, user_batch in loader:
                _, batch_latents, _ = self._network(batch.to(self.device), user_batch.to(self.device))
                latents.append(batch_latents.cpu().numpy())
        return np.concatenate(latents, axis=0)

    def _compute_dec_targets(self, sequences, user_indices) -> torch.Tensor:
        loader = self._make_loader(sequences, user_indices, shuffle=False)
        assignments_all = []
        self._network.eval()
        with torch.no_grad():
            for batch, user_batch in loader:
                _, _, assignments = self._network(batch.to(self.device), user_batch.to(self.device))
                assignments_all.append(assignments.cpu())
        return dec_target_distribution(torch.cat(assignments_all, dim=0))
