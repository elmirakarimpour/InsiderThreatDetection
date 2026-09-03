"""
test_bilstm_dec_personalized.py
"""

import numpy as np

from src.model.bilstm_dec_personalized import BiLSTMDECPersonalizedModel


def test_fit_score_pipeline_runs_without_error():
    rng = np.random.default_rng(42)
    train_sequences = rng.random((30, 7, 4)).astype(np.float32)
    train_user_ids = ["alice"] * 10 + ["bob"] * 10 + ["carol"] * 10

    eval_sequences = rng.random((6, 7, 4)).astype(np.float32)
    eval_user_ids = ["alice", "bob", "carol", "alice", "bob", "carol"]

    model = BiLSTMDECPersonalizedModel(
        n_clusters=2, encoder_size=8, latent_size=4,
        pretrain_epochs=1, fine_tune_epochs=1, batch_size=8,
    )
    scores = model.fit_score(train_sequences, train_user_ids, eval_sequences, eval_user_ids)

    assert scores.shape == (6,)
    assert np.isfinite(scores).all()


def test_unknown_user_at_eval_time_does_not_crash_cold_start():
    """کاربری که فقط در eval است و اصلاً در train نبوده، نباید باعث خطا شود."""
    rng = np.random.default_rng(0)
    train_sequences = rng.random((20, 7, 4)).astype(np.float32)
    train_user_ids = ["alice"] * 10 + ["bob"] * 10

    eval_sequences = rng.random((2, 7, 4)).astype(np.float32)
    eval_user_ids = ["dave", "eve"]  # هیچ‌کدام در train نبودند

    model = BiLSTMDECPersonalizedModel(
        n_clusters=2, encoder_size=8, latent_size=4,
        pretrain_epochs=1, fine_tune_epochs=1, batch_size=8,
    )
    model.fit(train_sequences, train_user_ids)
    scores = model.score(eval_sequences, eval_user_ids)

    assert scores.shape == (2,)
    assert np.isfinite(scores).all()


def test_score_before_fit_raises_error():
    model = BiLSTMDECPersonalizedModel(pretrain_epochs=1, fine_tune_epochs=1)
    try:
        model.score(np.zeros((2, 7, 3), dtype=np.float32), ["alice", "bob"])
        assert False, "انتظار می‌رفت RuntimeError بدهد"
    except RuntimeError:
        pass


def test_mismatched_user_ids_length_raises_error():
    rng = np.random.default_rng(0)
    train_sequences = rng.random((10, 7, 3)).astype(np.float32)
    model = BiLSTMDECPersonalizedModel(pretrain_epochs=1, fine_tune_epochs=1)
    try:
        model.fit(train_sequences, ["alice"] * 5)  # طول اشتباه عمدی
        assert False, "انتظار می‌رفت ValueError بدهد"
    except ValueError:
        pass
