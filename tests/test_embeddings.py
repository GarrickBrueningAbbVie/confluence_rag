"""Unit tests for EmbeddingManager class."""

import pytest
import numpy as np
from src.rag.embeddings import EmbeddingManager


@pytest.fixture
def embedding_manager() -> EmbeddingManager:
    """Create an EmbeddingManager instance for testing."""
    return EmbeddingManager(model_name="all-MiniLM-L6-v2")


def test_embedding_manager_initialization(embedding_manager: EmbeddingManager) -> None:
    """Test EmbeddingManager initialization."""
    assert embedding_manager.model_name == "all-MiniLM-L6-v2"
    assert embedding_manager.embedding_dimension > 0


def test_generate_embedding(embedding_manager: EmbeddingManager) -> None:
    """Test generating a single embedding."""
    text = "This is a test sentence for embedding generation"
    embedding = embedding_manager.generate_embedding(text)

    assert isinstance(embedding, np.ndarray)
    assert len(embedding) == embedding_manager.embedding_dimension
    assert embedding.dtype == np.float32 or embedding.dtype == np.float64


def test_generate_embeddings_batch(embedding_manager: EmbeddingManager) -> None:
    """Test generating multiple embeddings."""
    texts = [
        "First test sentence",
        "Second test sentence",
        "Third test sentence",
    ]
    embeddings = embedding_manager.generate_embeddings(texts, show_progress=False)

    assert isinstance(embeddings, np.ndarray)
    assert len(embeddings) == len(texts)
    assert embeddings.shape[1] == embedding_manager.embedding_dimension


def test_compute_similarity(embedding_manager: EmbeddingManager) -> None:
    """Test computing similarity between embeddings."""
    text1 = "Machine learning is fascinating"
    text2 = "Artificial intelligence is interesting"
    text3 = "The weather is nice today"

    emb1 = embedding_manager.generate_embedding(text1)
    emb2 = embedding_manager.generate_embedding(text2)
    emb3 = embedding_manager.generate_embedding(text3)

    sim_12 = embedding_manager.compute_similarity(emb1, emb2)
    sim_13 = embedding_manager.compute_similarity(emb1, emb3)

    # Similar texts should have higher similarity
    assert sim_12 > sim_13
    assert -1 <= sim_12 <= 1
    assert -1 <= sim_13 <= 1


def test_empty_text_embedding(embedding_manager: EmbeddingManager) -> None:
    """Test handling of empty text."""
    embedding = embedding_manager.generate_embedding("")
    assert isinstance(embedding, np.ndarray)
    assert len(embedding) == embedding_manager.embedding_dimension


def test_get_model_info(embedding_manager: EmbeddingManager) -> None:
    """Test getting model information."""
    info = embedding_manager.get_model_info()
    assert "model_name" in info
    assert "embedding_dimension" in info
    assert info["model_name"] == "all-MiniLM-L6-v2"
