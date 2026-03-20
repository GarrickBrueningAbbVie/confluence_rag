"""
Query routing for hybrid RAG + Database pipeline.

This package provides intelligent routing of user queries to the
appropriate pipeline(s) based on query intent.

Recommended approach (unified analysis):
    SmartQueryRouter with UnifiedQueryAnalyzer - performs entity extraction,
    intent classification, and query decomposition in a SINGLE LLM call.

Example (smart routing with unified analysis):
    >>> from routing import SmartQueryRouter
    >>> router = SmartQueryRouter(rag_pipeline, db_pipeline, iliad_client)
    >>> result = router.route("Describe ALFA and how many projects reference it")

Example (simple routing):
    >>> from routing import QueryRouter
    >>> router = QueryRouter(rag_pipeline, db_pipeline, iliad_client)
    >>> result = router.route("How many pages use Python?")

For lower-level access to unified analysis:
    >>> from routing import UnifiedQueryAnalyzer
    >>> analyzer = UnifiedQueryAnalyzer(iliad_client)
    >>> analysis = analyzer.analyze("Compare ALFA and CloverX")
    >>> print(analysis.sub_queries)  # 3 sub-queries for comparison
"""

# Core types (canonical source - use these, not query_analyzer.py)
from .types import QueryIntent, SubQuery

# Classification result (kept in intent_classifier for backwards compatibility)
from .intent_classifier import ClassificationResult

# Unified analyzer (recommended - single LLM call)
from .unified_analyzer import UnifiedQueryAnalyzer, UnifiedAnalysisResult, EntityExtractionResult

# Pattern matching utilities (fallback when LLM unavailable)
from .patterns import (
    is_list_describe_query,
    is_chart_query,
    is_comparison_query,
    detect_chart_type,
    classify_intent_fallback,
)

# Formatters
from .formatters import (
    format_db_answer,
    format_single_answer,
    format_list_result,
    format_sources,
)

# Routers
from .query_router import QueryRouter
from .smart_router import SmartQueryRouter, SmartRouteResult

# Execution and aggregation
from .parallel_executor import ParallelQueryExecutor, SubQueryResult, ExecutionPlan
from .result_aggregator import ResultAggregator, AggregatedResult

# Legacy components (kept for backwards compatibility)
from .intent_classifier import IntentClassifier
from .response_combiner import ResponseCombiner

__all__ = [
    # Core types
    "QueryIntent",
    "SubQuery",
    "ClassificationResult",
    # Unified analyzer (recommended)
    "UnifiedQueryAnalyzer",
    "UnifiedAnalysisResult",
    "EntityExtractionResult",
    # Pattern utilities
    "is_list_describe_query",
    "is_chart_query",
    "is_comparison_query",
    "detect_chart_type",
    "classify_intent_fallback",
    # Formatters
    "format_db_answer",
    "format_single_answer",
    "format_list_result",
    "format_sources",
    # Routers
    "QueryRouter",
    "SmartQueryRouter",
    "SmartRouteResult",
    # Execution and aggregation
    "ParallelQueryExecutor",
    "SubQueryResult",
    "ExecutionPlan",
    "ResultAggregator",
    "AggregatedResult",
    # Legacy
    "IntentClassifier",
    "ResponseCombiner",
]
