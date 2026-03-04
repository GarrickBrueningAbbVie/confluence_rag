"""
Query routing for hybrid RAG + Database pipeline.

This package provides intelligent routing of user queries to the
appropriate pipeline(s) based on query intent.

Two routing approaches are available:
1. QueryRouter: Rule-based keyword matching (fast, simple)
2. SmartQueryRouter: LLM-based decomposition with parallel execution (powerful, flexible)

Example (simple routing):
    >>> from routing import QueryRouter
    >>> router = QueryRouter(rag_pipeline, db_pipeline, iliad_client)
    >>> result = router.route("How many pages use Python?")

Example (smart routing with query decomposition):
    >>> from routing import SmartQueryRouter
    >>> router = SmartQueryRouter(rag_pipeline, db_pipeline, iliad_client)
    >>> result = router.route("Describe ALFA and how many projects reference it")
"""

from .intent_classifier import IntentClassifier, QueryIntent, ClassificationResult
from .query_router import QueryRouter
from .response_combiner import ResponseCombiner
from .query_analyzer import (
    LLMQueryAnalyzer,
    QueryAnalysisResult,
    SubQuery,
    SubQueryIntent,
)
from .parallel_executor import ParallelQueryExecutor, SubQueryResult, ExecutionPlan
from .result_aggregator import ResultAggregator, AggregatedResult
from .smart_router import SmartQueryRouter, SmartRouteResult

__all__ = [
    # Legacy rule-based routing
    "IntentClassifier",
    "QueryIntent",
    "ClassificationResult",
    "QueryRouter",
    "ResponseCombiner",
    # New LLM-based smart routing
    "LLMQueryAnalyzer",
    "QueryAnalysisResult",
    "SubQuery",
    "SubQueryIntent",
    "ParallelQueryExecutor",
    "SubQueryResult",
    "ExecutionPlan",
    "ResultAggregator",
    "AggregatedResult",
    "SmartQueryRouter",
    "SmartRouteResult",
]
