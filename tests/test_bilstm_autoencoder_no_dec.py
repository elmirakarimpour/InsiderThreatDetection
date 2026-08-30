"""
test_bilstm_autoencoder_no_dec.py
"""

import numpy as np

from src.model.bilstm_autoencoder_no_dec import BiLSTMAutoencoderNoDECModel


def test_fit_score_pipeline_runs_without_error_on_tiny_data():
    rng = np.random.default_rng(42)
    train_sequences = rng.random((20, 7, 4)).astype(np.float32)
    eval_sequences = rng.random((5, 7, 4)).astype(np.float32)

    model = BiLSTMAutoencoderNoDECModel(encoder_size=8, latent_size=4, epochs=1, batch_size=8)
    scores = model.fit_score(train_sequences, eval_sequences)

    assert scores.shape == (5,)
    assert np.isfinite(scores).all()


def test_score_before_fit_raises_error():
    model = BiLSTMAutoencoderNoDECModel(epochs=1)
    try:
        model.score(np.zeros((2, 7, 3), dtype=np.float32))
        assert False, "انتظار می‌رفت RuntimeError بدهد"
    except RuntimeError:
        pass


def test_odd_latent_size_raises_error():
    rng = np.random.default_rng(0)
    train_sequences = rng.random((10, 7, 3)).astype(np.float32)
    model = BiLSTMAutoencoderNoDECModel(latent_size=5, epochs=1)  # فرد است، نباید مجاز باشد
    try:
        model.fit(train_sequences)
        assert False, "انتظار می‌رفت ValueError بدهد"
    except ValueError:
        pass
