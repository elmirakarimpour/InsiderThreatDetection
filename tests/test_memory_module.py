"""
test_memory_module.py
"""

import torch

from src.model.memory_module import MemoryModule


def test_output_shapes_match_input():
    memory = MemoryModule(memory_size=10, latent_size=8)
    latent_vectors = torch.rand(5, 8)  # (batch=5, latent_size=8)

    retrieved, attention_weights = memory(latent_vectors)

    assert retrieved.shape == (5, 8)
    assert attention_weights.shape == (5, 10)


def test_attention_weights_sum_to_approximately_one():
    memory = MemoryModule(memory_size=10, latent_size=8)
    latent_vectors = torch.rand(5, 8)

    _, attention_weights = memory(latent_vectors)
    row_sums = attention_weights.sum(dim=1)

    assert torch.allclose(row_sums, torch.ones(5), atol=1e-4)


def test_retrieved_vectors_are_combinations_of_memory_bank():
    """بردار بازیابی‌شده باید همیشه ترکیبی از سطرهای memory_bank باشد، نه چیز دیگری."""
    memory = MemoryModule(memory_size=4, latent_size=3)
    latent_vectors = torch.rand(2, 3)

    retrieved, attention_weights = memory(latent_vectors)
    manual_reconstruction = attention_weights @ memory.memory_bank

    assert torch.allclose(retrieved, manual_reconstruction, atol=1e-5)
