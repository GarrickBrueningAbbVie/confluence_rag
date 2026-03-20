"""
Shared types for the query routing system.

This module provides the canonical definitions for query intent
classification and sub-query representation used across the routing
and agents modules.

Example:
    >>> from routing.types import QueryIntent, SubQuery
    >>> intent = QueryIntent.RAG
    >>> sub_query = SubQuery(text="What is ALFA?", intent=QueryIntent.RAG)
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class QueryIntent(Enum):
    """Classification of query intent for routing.

    Determines which pipeline(s) should handle the query:
    - RAG: Semantic search for conceptual/explanatory questions
    - DATABASE: Structured queries for counts, lists, aggregations
    - HYBRID: Requires both RAG and database pipelines
    - CHART: Visualization/charting requests
    - TABLE: Tabular data display requests
    """

    RAG = "rag"
    DATABASE = "database"
    HYBRID = "hybrid"
    CHART = "chart"
    TABLE = "table"


@dataclass
class SubQuery:
    """A decomposed sub-query with its classification.

    Represents an atomic query unit that can be executed independently
    or as part of a larger query decomposition.

    Attributes:
        text: The sub-query text to execute
        intent: Classified intent (RAG/DATABASE/HYBRID/CHART/TABLE)
        depends_on: Index of sub-query this depends on (for sequential execution)
        priority: Execution priority (lower = higher priority, 0 is highest)
        context_from: List of sub-query indices to inject results from
        store_as: Optional key to store result in context for later use

    Example:
        >>> sq = SubQuery(
        ...     text="Describe the ALFA project",
        ...     intent=QueryIntent.RAG,
        ...     priority=0
        ... )
    """

    text: str
    intent: QueryIntent
    depends_on: Optional[int] = None
    priority: int = 0
    context_from: List[int] = field(default_factory=list)
    store_as: Optional[str] = None


@dataclass
class ClassificationResult:
    """Result of intent classification.

    Attributes:
        intent: The classified query intent
        confidence: Confidence score (0.0 to 1.0)
        reasoning: Explanation for the classification
        sub_queries: Optional list of sub-query strings for hybrid queries
    """

    intent: QueryIntent
    confidence: float
    reasoning: str
    sub_queries: Optional[List[str]] = None
