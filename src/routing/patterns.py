"""
Pattern matching utilities for query routing.

This module provides keyword patterns and helper functions for
rule-based query classification. These are used as FALLBACK when
LLM-based analysis is unavailable.

Note: The primary routing mechanism is LLM-based analysis via
UnifiedQueryAnalyzer. These patterns are for fallback scenarios only.

Example:
    >>> from routing.patterns import is_database_query, is_chart_query
    >>> is_database_query("How many pages use Python?")
    True
    >>> is_chart_query("Show a bar chart of pages")
    True
"""

from typing import List, Tuple

from .types import QueryIntent


# =============================================================================
# Intent Indicator Patterns
# =============================================================================

DATABASE_INDICATORS: List[str] = [
    # Count/aggregation
    "how many",
    "count",
    "total number",
    "sum of",
    "average",
    "mean",
    # Listing
    "list all",
    "list the",
    "show all",
    "show me all",
    "get all",
    # Filtering
    "filter",
    "where",
    "with score",
    "above",
    "below",
    "greater than",
    "less than",
    "between",
    # Grouping
    "by project",
    "by author",
    "by technology",
    "per project",
    "per author",
    "group by",
    # Specific lookups
    "who created",
    "who has",
    "which projects",
    "which pages",
    "what projects",
]

RAG_INDICATORS: List[str] = [
    # Explanatory
    "what is",
    "what are",
    "explain",
    "describe",
    "tell me about",
    "how does",
    "how do",
    "why",
    # Information seeking
    "documentation for",
    "information about",
    "details about",
    "overview of",
    # Understanding
    "understand",
    "meaning of",
    "purpose of",
    "definition of",
]

CHART_INDICATORS: List[str] = [
    "chart",
    "graph",
    "plot",
    "visualize",
    "visualization",
    "bar chart",
    "pie chart",
    "line chart",
    "histogram",
    "show me a chart",
    "create a graph",
]

TABLE_INDICATORS: List[str] = [
    "table",
    "tabular",
    "in a table",
    "as a table",
    "table format",
    "table showing",
    "table of",
]

HYBRID_INDICATORS: List[str] = [
    # Database-like + summary
    "summarize",
    "summary of",
    "and explain",
    "and describe",
    "with details",
    # Comparative
    "compare",
    "difference between",
    "similar to",
    "vs",
    "versus",
]

# =============================================================================
# List + Describe Patterns (for iterative agent routing)
# =============================================================================

LIST_DESCRIBE_PATTERNS: List[str] = [
    "list all",
    "list the",
    "what projects",
    "which projects",
    "show all",
    "get all",
    "find all",
]

DESCRIBE_PATTERNS: List[str] = [
    "describe all",
    "describe each",
    "describe these",
    "describe them",
    "explain all",
    "explain each",
    "and describe",
    "and explain",
    "then describe",
    "then explain",
]

# =============================================================================
# Comparison Patterns
# =============================================================================

COMPARISON_PATTERNS: List[str] = [
    "compare",
    "vs",
    "versus",
    "difference between",
    "differences between",
    "similar to",
    "similarities between",
    "how does .* compare",
    ".* vs .*",
]

# =============================================================================
# Chart Type Detection
# =============================================================================

CHART_TYPE_KEYWORDS = {
    # Explicit chart type keywords (highest priority)
    "bar chart": "bar",
    "bar graph": "bar",
    "column chart": "bar",
    "pie chart": "pie",
    "pie graph": "pie",
    "donut chart": "pie",
    "line chart": "line",
    "line graph": "line",
    "scatter chart": "scatter",
    "scatter plot": "scatter",
    "histogram": "histogram",
}

CHART_TYPE_IMPLICIT = {
    "bar": "bar",
    "column": "bar",
    "pie": "pie",
    "donut": "pie",
    "line": "line",
    "trend": "line",
    "time series": "line",
    "over time": "line",
    "timeline": "line",
    "scatter": "scatter",
    "distribution": "histogram",
}

TEMPORAL_INDICATORS: List[str] = [
    "when",
    "timeline",
    "created",
    "modified",
    "date",
    "time",
    "history",
    "progression",
    "growth",
    "trend",
    "monthly",
    "weekly",
    "daily",
    "yearly",
    "annual",
]

COMPARISON_CHART_INDICATORS: List[str] = [
    "most",
    "top",
    "highest",
    "lowest",
    "ranking",
    "compare",
    "comparison",
    "by user",
    "by author",
    "per project",
    "per team",
    "by project",
    "by team",
]


# =============================================================================
# Helper Functions
# =============================================================================


def _calculate_score(query: str, indicators: List[str]) -> float:
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
    return min(1.0, matches / 3)


def is_database_query(query: str) -> bool:
    """Check if query matches database/structured patterns.

    Args:
        query: User query string

    Returns:
        True if likely a database query
    """
    query_lower = query.lower()
    return _calculate_score(query_lower, DATABASE_INDICATORS) > 0.2


def is_rag_query(query: str) -> bool:
    """Check if query matches RAG/semantic patterns.

    Args:
        query: User query string

    Returns:
        True if likely a RAG query
    """
    query_lower = query.lower()
    return _calculate_score(query_lower, RAG_INDICATORS) > 0.2


def is_chart_query(query: str) -> bool:
    """Check if query requests a chart/visualization.

    Args:
        query: User query string

    Returns:
        True if likely a chart request
    """
    query_lower = query.lower()
    return _calculate_score(query_lower, CHART_INDICATORS) > 0.2


def is_table_query(query: str) -> bool:
    """Check if query requests tabular output.

    Args:
        query: User query string

    Returns:
        True if likely a table request
    """
    query_lower = query.lower()
    return any(ind in query_lower for ind in TABLE_INDICATORS)


def is_hybrid_query(query: str) -> bool:
    """Check if query requires both RAG and database.

    Args:
        query: User query string

    Returns:
        True if likely a hybrid query
    """
    query_lower = query.lower()
    hybrid_score = _calculate_score(query_lower, HYBRID_INDICATORS)
    db_score = _calculate_score(query_lower, DATABASE_INDICATORS)
    rag_score = _calculate_score(query_lower, RAG_INDICATORS)

    return hybrid_score > 0.2 or (db_score > 0.3 and rag_score > 0.3)


def is_comparison_query(query: str) -> bool:
    """Check if query is comparing entities.

    Args:
        query: User query string

    Returns:
        True if likely a comparison query
    """
    query_lower = query.lower()
    return any(pattern in query_lower for pattern in ["compare", "vs", "versus", "difference"])


def is_list_describe_query(query: str) -> bool:
    """Check if query matches list+describe pattern.

    These queries need the IterativeDescribeAgent to first get a list
    from the database, then describe each item via RAG.

    Args:
        query: User query string

    Returns:
        True if this is a list+describe pattern
    """
    query_lower = query.lower()
    has_list = any(pattern in query_lower for pattern in LIST_DESCRIBE_PATTERNS)
    has_describe = any(pattern in query_lower for pattern in DESCRIBE_PATTERNS)
    return has_list and has_describe


def detect_chart_type(query: str) -> str:
    """Detect the appropriate chart type from query.

    Args:
        query: Chart request query

    Returns:
        Chart type string ('bar', 'pie', 'line', 'scatter', 'histogram', or 'auto')
    """
    query_lower = query.lower()

    # Check explicit chart type keywords first
    for keyword, chart_type in CHART_TYPE_KEYWORDS.items():
        if keyword in query_lower:
            return chart_type

    # Check implicit keywords
    for keyword, chart_type in CHART_TYPE_IMPLICIT.items():
        if keyword in query_lower:
            return chart_type

    # Temporal indicators suggest line chart
    if any(ind in query_lower for ind in TEMPORAL_INDICATORS):
        return "line"

    # Comparison/ranking indicators suggest bar chart
    if any(ind in query_lower for ind in COMPARISON_CHART_INDICATORS):
        return "bar"

    return "auto"


def classify_intent_fallback(query: str) -> Tuple[QueryIntent, float, str]:
    """Classify query intent using pattern matching (fallback).

    This is the fallback classification when LLM is unavailable.
    Returns the most likely intent based on keyword patterns.

    Args:
        query: User query string

    Returns:
        Tuple of (QueryIntent, confidence, reasoning)
    """
    query_lower = query.lower()

    # Check for chart first (most specific)
    if is_chart_query(query):
        return (
            QueryIntent.CHART,
            min(0.9, _calculate_score(query_lower, CHART_INDICATORS) + 0.3),
            "Query contains visualization keywords",
        )

    # Check for table
    if is_table_query(query):
        return (
            QueryIntent.TABLE,
            0.8,
            "Query requests tabular output",
        )

    # Calculate scores
    db_score = _calculate_score(query_lower, DATABASE_INDICATORS)
    rag_score = _calculate_score(query_lower, RAG_INDICATORS)
    hybrid_score = _calculate_score(query_lower, HYBRID_INDICATORS)

    # Determine intent
    if hybrid_score > 0.2 or (db_score > 0.3 and rag_score > 0.3):
        confidence = min(0.85, (db_score + rag_score) / 2 + hybrid_score)
        return (QueryIntent.HYBRID, confidence, "Query requires both structured and semantic search")

    if db_score > rag_score and db_score > 0.2:
        return (
            QueryIntent.DATABASE,
            min(0.9, db_score + 0.2),
            "Query matches database/structured patterns",
        )

    if rag_score > 0.2:
        return (
            QueryIntent.RAG,
            min(0.9, rag_score + 0.2),
            "Query matches semantic search patterns",
        )

    # Default to RAG for ambiguous queries
    return (QueryIntent.RAG, 0.5, "Defaulting to semantic search for ambiguous query")
