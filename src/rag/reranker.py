"""Re-ranking module for improved document retrieval.

This module provides composite scoring functionality that combines multiple
signals to re-rank retrieved documents for better relevance.
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
import numpy as np
from loguru import logger

from rag.query_processor import ProcessedQuery


@dataclass
class RankingWeights:
    """
    Configurable weights for composite scoring.

    All weights should sum to 1.0 for normalized scoring.
    """

    content_similarity: float = 0.35  # Vector similarity to page content
    title_similarity: float = 0.25  # Vector similarity to page title
    keyword_in_title: float = 0.20  # Keywords found in page title
    page_depth: float = 0.10  # Page hierarchy depth (shallower = higher score)
    has_children: float = 0.05  # Pages with children may be more important
    keyword_in_content: float = 0.05  # Keywords found in page content

    def __post_init__(self):
        """Validate weights sum to approximately 1.0."""
        total = (
            self.content_similarity
            + self.title_similarity
            + self.keyword_in_title
            + self.page_depth
            + self.has_children
            + self.keyword_in_content
        )
        if abs(total - 1.0) > 0.01:
            logger.warning(
                f"Ranking weights sum to {total:.3f}, not 1.0. "
                "Scores will not be normalized."
            )

    def to_dict(self) -> Dict[str, float]:
        """Convert weights to dictionary."""
        return {
            "content_similarity": self.content_similarity,
            "title_similarity": self.title_similarity,
            "keyword_in_title": self.keyword_in_title,
            "page_depth": self.page_depth,
            "has_children": self.has_children,
            "keyword_in_content": self.keyword_in_content,
        }


@dataclass
class ScoredDocument:
    """Container for a document with its component scores."""

    document: str
    metadata: Dict[str, Any]
    doc_id: str
    scores: Dict[str, float] = field(default_factory=dict)
    composite_score: float = 0.0

    def __repr__(self) -> str:
        title = self.metadata.get("title", "Unknown")[:30]
        return f"ScoredDocument(title='{title}...', score={self.composite_score:.4f})"


class DocumentReranker:
    """
    Re-ranks retrieved documents using composite scoring.

    This class combines multiple ranking signals:
    1. Content similarity score (from vector search)
    2. Title similarity score (embedding comparison)
    3. Keyword matches in title
    4. Page depth in hierarchy
    5. Whether page has children (indicates importance)
    6. Keyword matches in content
    """

    def __init__(
        self,
        weights: Optional[RankingWeights] = None,
        embedding_manager=None,
        max_depth: int = 10,
    ) -> None:
        """
        Initialize the document re-ranker.

        Args:
            weights: Custom ranking weights. Uses defaults if None.
            embedding_manager: EmbeddingManager for title similarity.
            max_depth: Maximum page depth for normalization.
        """
        self.weights = weights or RankingWeights()
        self.embedding_manager = embedding_manager
        self.max_depth = max_depth

        logger.info(f"DocumentReranker initialized with weights: {self.weights.to_dict()}")

    def rerank(
        self,
        documents: List[str],
        metadatas: List[Dict[str, Any]],
        ids: List[str],
        distances: List[float],
        processed_query: ProcessedQuery,
        query_embedding: Optional[np.ndarray] = None,
    ) -> List[ScoredDocument]:
        """
        Re-rank documents using composite scoring.

        Args:
            documents: List of document texts.
            metadatas: List of metadata dictionaries.
            ids: List of document IDs.
            distances: Original similarity distances from vector search.
            processed_query: Processed query with keywords.
            query_embedding: Query embedding for title similarity.

        Returns:
            List of ScoredDocuments sorted by composite score (descending).
        """
        logger.info(f"Re-ranking {len(documents)} documents")

        scored_docs = []
        keywords = set(processed_query.lemmatized_keywords)
        project_names = set(processed_query.potential_project_names)
        all_search_terms = keywords | project_names

        logger.debug(f"Search terms for re-ranking: {all_search_terms}")

        for i, (doc, meta, doc_id, distance) in enumerate(
            zip(documents, metadatas, ids, distances)
        ):
            scored_doc = ScoredDocument(
                document=doc,
                metadata=meta,
                doc_id=doc_id,
            )

            # Score 1: Content similarity (convert distance to similarity)
            content_sim = 1.0 - distance
            scored_doc.scores["content_similarity"] = content_sim

            # Score 2: Title similarity
            title = meta.get("title", "").lower()
            title_sim = self._compute_title_similarity(
                query_embedding, title
            ) if query_embedding is not None and self.embedding_manager else content_sim
            scored_doc.scores["title_similarity"] = title_sim

            # Score 3: Keywords in title
            keyword_title_score = self._compute_keyword_title_score(
                title, all_search_terms
            )
            scored_doc.scores["keyword_in_title"] = keyword_title_score

            # Score 4: Page depth score (shallower = higher score)
            depth = meta.get("depth", 1)
            depth_score = self._compute_depth_score(depth)
            scored_doc.scores["page_depth"] = depth_score

            # Score 5: Has children score
            children = meta.get("children", [])
            children_score = 1.0 if children else 0.5
            scored_doc.scores["has_children"] = children_score

            # Score 6: Keywords in content
            keyword_content_score = self._compute_keyword_content_score(
                doc.lower(), all_search_terms
            )
            scored_doc.scores["keyword_in_content"] = keyword_content_score

            # Compute composite score
            scored_doc.composite_score = self._compute_composite_score(scored_doc.scores)

            scored_docs.append(scored_doc)

            logger.debug(
                f"Document {i+1} '{title[:30]}...' scores: "
                f"content={content_sim:.3f}, title={title_sim:.3f}, "
                f"kw_title={keyword_title_score:.3f}, depth={depth_score:.3f}, "
                f"children={children_score:.3f}, kw_content={keyword_content_score:.3f} "
                f"-> composite={scored_doc.composite_score:.3f}"
            )

        # Sort by composite score (descending)
        scored_docs.sort(key=lambda x: x.composite_score, reverse=True)

        logger.info(
            f"Re-ranking complete. Top score: {scored_docs[0].composite_score:.3f}, "
            f"Bottom score: {scored_docs[-1].composite_score:.3f}"
        )

        return scored_docs

    def _compute_title_similarity(
        self,
        query_embedding: np.ndarray,
        title: str,
    ) -> float:
        """
        Compute similarity between query and page title.

        Args:
            query_embedding: Query embedding vector.
            title: Page title text.

        Returns:
            Similarity score between 0 and 1.
        """
        if not self.embedding_manager or not title:
            return 0.5

        try:
            title_embedding = self.embedding_manager.generate_embedding(title)
            similarity = self.embedding_manager.compute_similarity(
                query_embedding, title_embedding
            )
            # Normalize to 0-1 range (cosine similarity can be negative)
            return (similarity + 1) / 2
        except Exception as e:
            logger.warning(f"Error computing title similarity: {e}")
            return 0.5

    def _compute_keyword_title_score(
        self,
        title: str,
        keywords: set,
    ) -> float:
        """
        Compute score based on keyword presence in title.

        Args:
            title: Page title (lowercase).
            keywords: Set of keywords to search for.

        Returns:
            Score between 0 and 1.
        """
        if not keywords or not title:
            return 0.0

        matches = sum(1 for kw in keywords if kw in title)
        # Scale by number of keywords, cap at 1.0
        score = min(matches / max(len(keywords), 1), 1.0)

        if matches > 0:
            logger.debug(f"Found {matches} keywords in title: '{title[:50]}...'")

        return score

    def _compute_keyword_content_score(
        self,
        content: str,
        keywords: set,
    ) -> float:
        """
        Compute score based on keyword presence in content.

        Args:
            content: Document content (lowercase).
            keywords: Set of keywords to search for.

        Returns:
            Score between 0 and 1.
        """
        if not keywords or not content:
            return 0.0

        matches = sum(1 for kw in keywords if kw in content)
        # Scale by number of keywords, cap at 1.0
        return min(matches / max(len(keywords), 1), 1.0)

    def _compute_depth_score(self, depth: int) -> float:
        """
        Compute score based on page depth (shallower = higher score).

        Args:
            depth: Page depth in hierarchy (1 = top-level).

        Returns:
            Score between 0 and 1.
        """
        # Linear decay: depth 1 = 1.0, max_depth = 0.0
        normalized = max(0, min(depth, self.max_depth))
        score = 1.0 - (normalized - 1) / max(self.max_depth - 1, 1)
        return max(0.0, score)

    def _compute_composite_score(self, scores: Dict[str, float]) -> float:
        """
        Compute weighted composite score.

        Args:
            scores: Dictionary of individual scores.

        Returns:
            Weighted composite score.
        """
        composite = (
            scores.get("content_similarity", 0) * self.weights.content_similarity
            + scores.get("title_similarity", 0) * self.weights.title_similarity
            + scores.get("keyword_in_title", 0) * self.weights.keyword_in_title
            + scores.get("page_depth", 0) * self.weights.page_depth
            + scores.get("has_children", 0) * self.weights.has_children
            + scores.get("keyword_in_content", 0) * self.weights.keyword_in_content
        )
        return composite

    def update_weights(self, **kwargs) -> None:
        """
        Update ranking weights.

        Args:
            **kwargs: Weight names and values to update.
        """
        for key, value in kwargs.items():
            if hasattr(self.weights, key):
                setattr(self.weights, key, value)
                logger.info(f"Updated weight '{key}' to {value}")
            else:
                logger.warning(f"Unknown weight parameter: {key}")

    def extract_reranked_results(
        self,
        scored_docs: List[ScoredDocument],
        n_results: int,
    ) -> Dict[str, Any]:
        """
        Extract top results in the standard format.

        Args:
            scored_docs: List of scored documents.
            n_results: Number of results to return.

        Returns:
            Dictionary with documents, metadatas, distances, ids.
        """
        top_docs = scored_docs[:n_results]

        return {
            "documents": [d.document for d in top_docs],
            "metadatas": [d.metadata for d in top_docs],
            "distances": [1.0 - d.composite_score for d in top_docs],  # Convert back to distance
            "ids": [d.doc_id for d in top_docs],
            "composite_scores": [d.composite_score for d in top_docs],
            "score_breakdown": [d.scores for d in top_docs],
        }
