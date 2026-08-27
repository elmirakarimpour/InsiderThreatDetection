"""
test_bilstm_dec.py

این تست یک آموزش واقعی (ولی خیلی کوچک و سریع) اجرا می‌کند تا مطمئن
شویم کل لوله‌کشی (pretrain -> kmeans init -> fine-tune -> score) بدون
خطا کار می‌کند. هدف این تست دقت مدل نیست (با ۲ epoch و داده‌ی ساختگی
دقتی وجود ندارد)، فقط سالم بودن مسیر اجراست.
"""

import numpy as np

from src.model.bilstm_dec import BiLSTMDECModel


def test_fit_score_pipeline_runs_without_error_on_tiny_data():
    rng = np.random.default_rng(42)
    train_sequences = rng.random((20, 7, 4)).astype(np.float32)
    eval_sequences = rng.random((5, 7, 4)).astype(np.float32)

    model = BiLSTMDECModel(
        n_clusters=2,
        encoder_size=8,
        latent_size=4,
        pretrain_epochs=1,
        fine_tune_epochs=1,
        batch_size=8,
        pooling_strategy="max",
    )
    scores = model.fit_score(train_sequences, eval_sequences)

    assert scores.shape == (5,)
    assert np.isfinite(scores).all()


def test_score_before_fit_raises_error():
    model = BiLSTMDECModel(pretrain_epochs=1, fine_tune_epochs=1)
    dummy_sequences = np.zeros((2, 7, 3), dtype=np.float32)
    try:
        model.score(dummy_sequences)
        assert False, "انتظار می‌رفت RuntimeError بدهد"
    except RuntimeError:
        pass


def test_different_pooling_strategies_produce_different_scores():
    """اثبات عملی: تغییر pooling_strategy واقعاً روی خروجی نهایی مدل اثر می‌گذارد."""
    rng = np.random.default_rng(7)
    train_sequences = rng.random((20, 7, 4)).astype(np.float32)
    eval_sequences = rng.random((5, 7, 4)).astype(np.float32)

    shared_kwargs = dict(
        n_clusters=2, encoder_size=8, latent_size=4,
        pretrain_epochs=1, fine_tune_epochs=1, batch_size=8, seed=42,
    )

    mean_model = BiLSTMDECModel(pooling_strategy="mean", **shared_kwargs)
    max_model = BiLSTMDECModel(pooling_strategy="max", **shared_kwargs)

    mean_scores = mean_model.fit_score(train_sequences, eval_sequences)
    max_scores = max_model.fit_score(train_sequences, eval_sequences)

    # چون seed یکسان است، معماری اولیه یکسان می‌شود؛ فقط pooling فرق دارد
    assert not np.allclose(mean_scores, max_scores)
