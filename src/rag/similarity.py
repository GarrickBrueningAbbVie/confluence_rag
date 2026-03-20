"""
Shared similarity computation utilities for the RAG module.

This module provides the canonical implementation of cosine similarity
computation used across the RAG pipeline. All similarity computations
should use these functions to ensure consistency and reduce duplication.

Example:
    >>> from rag.similarity import cosine_similarity, batch_cosine_similarity
    >>> similarity = cosine_similarity(query_embedding, doc_embedding)
    >>> similarities = batch_cosine_similarity(query_embedding, doc_embeddings)
"""

import numpy as np
from typing import List, Tuple, Union

# Type alias for embeddings
Embedding = Union[List[float], np.ndarray]


def cosine_similarity(embedding1: Embedding, embedding2: Embedding) -> float:
    """
    Compute cosine similarity between two embeddings.

    Args:
        embedding1: First embedding vector
        embedding2: Second embedding vector

    Returns:
        Cosine similarity score between -1 and 1

    Example:
        >>> emb1 = [1.0, 0.0, 0.0]
        >>> emb2 = [1.0, 0.0, 0.0]
        >>> cosine_similarity(emb1, emb2)
        1.0
    """
    vec1 = np.array(embedding1)
    vec2 = np.array(embedding2)

    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return float(np.dot(vec1, vec2) / (norm1 * norm2))


def batch_cosine_similarity(
    query_embedding: Embedding,
    embeddings: Union[List[Embedding], np.ndarray],
) -> np.ndarray:
    """
    Compute cosine similarity between a query and multiple embeddings.

    Optimized for batch computation using vectorized numpy operations.

    Args:
        query_embedding: Query embedding vector
        embeddings: Array of embedding vectors to compare against

    Returns:
        Array of similarity scores

    Example:
        >>> query = [1.0, 0.0]
        >>> docs = [[1.0, 0.0], [0.0, 1.0], [0.707, 0.707]]
        >>> batch_cosine_similarity(query, docs)
        array([1.0, 0.0, 0.707...])
    """
    query = np.array(query_embedding)
    docs = np.array(embeddings)

    if docs.ndim == 1:
        docs = docs.reshape(1, -1)

    # Normalize query
    query_norm = np.linalg.norm(query)
    if query_norm == 0:
        return np.zeros(len(docs))
    query_normalized = query / query_norm

    # Normalize documents (row-wise)
    doc_norms = np.linalg.norm(docs, axis=1, keepdims=True)
    doc_norms = np.where(doc_norms == 0, 1, doc_norms)  # Avoid division by zero
    docs_normalized = docs / doc_norms

    # Compute similarities via dot product
    similarities = np.dot(docs_normalized, query_normalized)

    return similarities


def find_top_k_similar(
    query_embedding: Embedding,
    embeddings: Union[List[Embedding], np.ndarray],
    top_k: int = 5,
) -> List[Tuple[int, float]]:
    """
    Find the top-k most similar embeddings to a query.

    Args:
        query_embedding: Query embedding vector
        embeddings: Array of embedding vectors to search
        top_k: Number of results to return

    Returns:
        List of (index, similarity) tuples, sorted by similarity descending

    Example:
        >>> query = [1.0, 0.0]
        >>> docs = [[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]]
        >>> find_top_k_similar(query, docs, top_k=2)
        [(0, 1.0), (2, 0.707...)]
    """
    similarities = batch_cosine_similarity(query_embedding, embeddings)

    # Get top-k indices
    if len(similarities) <= top_k:
        top_indices = np.argsort(similarities)[::-1]
    else:
        # Use argpartition for efficiency with large arrays
        partition_idx = np.argpartition(similarities, -top_k)[-top_k:]
        top_indices = partition_idx[np.argsort(similarities[partition_idx])[::-1]]

    return [(int(idx), float(similarities[idx])) for idx in top_indices]


def cosine_distance(embedding1: Embedding, embedding2: Embedding) -> float:
    """
    Compute cosine distance between two embeddings.

    Distance is defined as 1 - similarity, ranging from 0 to 2.

    Args:
        embedding1: First embedding vector
        embedding2: Second embedding vector

    Returns:
        Cosine distance (0 = identical, 2 = opposite)

    Example:
        >>> emb1 = [1.0, 0.0]
        >>> emb2 = [1.0, 0.0]
        >>> cosine_distance(emb1, emb2)
        0.0
    """
    return 1.0 - cosine_similarity(embedding1, embedding2)
