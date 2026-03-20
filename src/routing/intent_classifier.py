"""
Intent classification for query routing.

This module classifies user queries to determine which pipeline(s)
should handle them: RAG (semantic), Database (structured), or both.

Note: This is the RULE-BASED classifier, used as a FALLBACK when
LLM-based classification (UnifiedQueryAnalyzer) is unavailable.
For smart routing, prefer UnifiedQueryAnalyzer instead.

Example:
    >>> from routing.intent_classifier import IntentClassifier
    >>> classifier = IntentClassifier()
    >>> intent = classifier.classify("How many pages use Python?")
    >>> print(intent)  # QueryIntent.DATABASE
"""

from dataclasses import dataclass
from typing import List, Optional

from loguru import logger

# Import from shared modules
from .types import QueryIntent
from .patterns import (
    DATABASE_INDICATORS,
    RAG_INDICATORS,
    CHART_INDICATORS,
    HYBRID_INDICATORS,
)

# Import types for type hints
try:
    from iliad.client import IliadClient
except ImportError:
    pass


@dataclass
class ClassificationResult:
    """Result of intent classification."""

    intent: QueryIntent
    confidence: float
    reasoning: str
    sub_queries: Optional[List[str]] = None  # For hybrid queries


class IntentClassifier:
    """
    Classify query intent for routing.

    Uses rule-based classification with optional LLM fallback
    for ambiguous cases.

    Attributes:
        iliad_client: Optional Iliad client for LLM fallback
        use_llm_fallback: Whether to use LLM for ambiguous cases

    Example:
        >>> classifier = IntentClassifier()
        >>> result = classifier.classify("How many pages use Airflow?")
        >>> print(result.intent)  # QueryIntent.DATABASE
    """

    def __init__(
        self,
        iliad_client: Optional["IliadClient"] = None,
        use_llm_fallback: bool = False,
        model: str = "gpt-5-mini-global",
    ) -> None:
        """Initialize classifier.

        Args:
            iliad_client: Optional Iliad client for LLM fallback
            use_llm_fallback: Whether to use LLM for ambiguous cases
            model: Model to use for LLM classification
        """
        self.iliad_client = iliad_client
        self.use_llm_fallback = use_llm_fallback and iliad_client is not None
        self.model = model

        logger.info(
            f"Initialized IntentClassifier (LLM fallback: {self.use_llm_fallback})"
        )

    def classify(self, query: str) -> ClassificationResult:
        """
        Classify a query's intent.

        Args:
            query: User query string

        Returns:
            ClassificationResult with intent and confidence

        Example:
            >>> result = classifier.classify("What is the RAG pipeline?")
            >>> print(result.intent)  # QueryIntent.RAG
        """
        query_lower = query.lower()

        # Check for chart requests first
        chart_score = self._calculate_score(query_lower, CHART_INDICATORS)
        if chart_score > 0.3:
            return ClassificationResult(
                intent=QueryIntent.CHART,
                confidence=min(0.9, chart_score + 0.3),
                reasoning="Query contains visualization keywords",
            )

        # Calculate scores for each intent
        db_score = self._calculate_score(query_lower, DATABASE_INDICATORS)
        rag_score = self._calculate_score(query_lower, RAG_INDICATORS)
        hybrid_score = self._calculate_score(query_lower, HYBRID_INDICATORS)

        logger.debug(
            f"Scores - DB: {db_score:.2f}, RAG: {rag_score:.2f}, "
            f"Hybrid: {hybrid_score:.2f}, Chart: {chart_score:.2f}"
        )

        # Determine intent based on scores
        if hybrid_score > 0.2 or (db_score > 0.3 and rag_score > 0.3):
            return ClassificationResult(
                intent=QueryIntent.HYBRID,
                confidence=min(0.85, (db_score + rag_score) / 2 + hybrid_score),
                reasoning="Query requires both structured and semantic search",
            )

        if db_score > rag_score and db_score > 0.2:
            return ClassificationResult(
                intent=QueryIntent.DATABASE,
                confidence=min(0.9, db_score + 0.2),
                reasoning="Query matches database/structured patterns",
            )

        if rag_score > 0.2:
            return ClassificationResult(
                intent=QueryIntent.RAG,
                confidence=min(0.9, rag_score + 0.2),
                reasoning="Query matches semantic search patterns",
            )

        # Ambiguous case - use LLM fallback or default to RAG
        if self.use_llm_fallback:
            return self._llm_classify(query)

        # Default to RAG for ambiguous queries
        return ClassificationResult(
            intent=QueryIntent.RAG,
            confidence=0.5,
            reasoning="Defaulting to semantic search for ambiguous query",
        )

    def _calculate_score(self, query: str, indicators: List[str]) -> float:
        """Calculate match score against indicator list.

        Args:
            query: Lowercase query string
            indicators: List of indicator patterns

        Returns:
            Score from 0.0 to 1.0
        """
        matches = sum(1 for ind in indicators if ind in query)

        if matches == 0:
            return 0.0

        # Normalize by number of indicators
        # Higher weight for multiple matches
        base_score = min(1.0, matches / 3)

        return base_score

    def _llm_classify(self, query: str) -> ClassificationResult:
        """Use LLM to classify ambiguous query.

        Args:
            query: Query string

        Returns:
            ClassificationResult from LLM
        """
        prompt = f"""Classify this query into one of these categories:
            - RAG: Questions about concepts, explanations, documentation (e.g., "What is X?", "Explain Y")
            - DATABASE: Questions about counts, lists, filters, aggregations (e.g., "How many?", "List all")
            - HYBRID: Questions needing both semantic search and structured data
            - CHART: Requests for visualizations

            Query: "{query}"

            Respond with ONLY one word: RAG, DATABASE, HYBRID, or CHART
        """

        try:
            messages = [{"role": "user", "content": prompt}]
            response = self.iliad_client.chat(messages=messages, model=self.model)
            content = self.iliad_client.extract_content(response)

            intent_str = content.strip().upper()

            intent_map = {
                "RAG": QueryIntent.RAG,
                "DATABASE": QueryIntent.DATABASE,
                "HYBRID": QueryIntent.HYBRID,
                "CHART": QueryIntent.CHART,
            }

            intent = intent_map.get(intent_str, QueryIntent.RAG)

            return ClassificationResult(
                intent=intent,
                confidence=0.75,
                reasoning=f"LLM classified as {intent_str}",
            )

        except Exception as e:
            logger.warning(f"LLM classification failed: {e}")
            return ClassificationResult(
                intent=QueryIntent.RAG,
                confidence=0.5,
                reasoning="LLM fallback failed, defaulting to RAG",
            )

    def get_intent_description(self, intent: QueryIntent) -> str:
        """Get human-readable description of intent.

        Args:
            intent: QueryIntent enum value

        Returns:
            Description string
        """
        descriptions = {
            QueryIntent.RAG: "Semantic search for conceptual information",
            QueryIntent.DATABASE: "Structured query for data aggregation/filtering",
            QueryIntent.HYBRID: "Combined semantic and structured search",
            QueryIntent.CHART: "Data visualization request",
        }

        return descriptions.get(intent, "Unknown intent")
