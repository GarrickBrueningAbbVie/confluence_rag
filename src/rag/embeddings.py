"""Embedding generation and management for the RAG pipeline."""

from typing import List, Optional
from sentence_transformers import SentenceTransformer
import numpy as np
from loguru import logger

from rag.similarity import cosine_similarity


class EmbeddingManager:
    """
    Manager for generating and handling text embeddings.

    This class uses sentence transformers to generate embeddings
    for text chunks used in the RAG pipeline.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        """
        Initialize the embedding manager.

        Args:
            model_name: Name of the sentence transformer model to use.
                       Defaults to 'all-MiniLM-L6-v2'.
        """
        self.model_name = model_name
        logger.info(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.embedding_dimension = self.model.get_sentence_embedding_dimension()
        logger.info(f"Embedding model loaded. Dimension: {self.embedding_dimension}")

    def generate_embedding(self, text: str) -> np.ndarray:
        """
        Generate embedding for a single text.

        Args:
            text: Input text to embed.

        Returns:
            Numpy array containing the embedding vector.
        """
        if not text or not text.strip():
            logger.warning("Empty text provided for embedding")
            return np.zeros(self.embedding_dimension)

        embedding = self.model.encode(text, convert_to_numpy=True)
        logger.debug(f"Generated embedding for text of length {len(text)}")
        return embedding

    def generate_embeddings(
        self, texts: List[str], batch_size: int = 32, show_progress: bool = True
    ) -> np.ndarray:
        """
        Generate embeddings for multiple texts efficiently.

        Args:
            texts: List of texts to embed.
            batch_size: Number of texts to process at once. Defaults to 32.
            show_progress: Whether to show progress bar. Defaults to True.

        Returns:
            Numpy array of shape (n_texts, embedding_dim) containing all embeddings.
        """
        if not texts:
            logger.warning("Empty text list provided for embeddings")
            return np.array([])

        logger.info(f"Generating embeddings for {len(texts)} texts")

        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
        )

        logger.info(f"Generated {len(embeddings)} embeddings")
        return embeddings

    def compute_similarity(
        self, embedding1: np.ndarray, embedding2: np.ndarray
    ) -> float:
        """
        Compute cosine similarity between two embeddings.

        Args:
            embedding1: First embedding vector.
            embedding2: Second embedding vector.

        Returns:
            Cosine similarity score between -1 and 1.
        """
        return cosine_similarity(embedding1, embedding2)

    def get_model_info(self) -> dict:
        """
        Get information about the embedding model.

        Returns:
            Dictionary with model name and embedding dimension.
        """
        return {
            "model_name": self.model_name,
            "embedding_dimension": self.embedding_dimension,
        }
