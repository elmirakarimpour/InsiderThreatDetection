"""
test_simple_autoencoder.py
"""

import numpy as np

from src.model.simple_autoencoder import SimpleAutoencoderModel


def test_fit_score_pipeline_runs_without_error_on_tiny_data():
    rng = np.random.default_rng(42)
    train_sequences = rng.random((20, 7, 4)).astype(np.float32)
    eval_sequences = rng.random((5, 7, 4)).astype(np.float32)

    model = SimpleAutoencoderModel(hidden_size=16, latent_size=4, epochs=1, batch_size=8)
    scores = model.fit_score(train_sequences, eval_sequences)

    assert scores.shape == (5,)
    assert np.isfinite(scores).all()


def test_score_before_fit_raises_error():
    model = SimpleAutoencoderModel(epochs=1)
    try:
        model.score(np.zeros((2, 7, 3), dtype=np.float32))
        assert False, "انتظار می‌رفت RuntimeError بدهد"
    except RuntimeError:
        pass
