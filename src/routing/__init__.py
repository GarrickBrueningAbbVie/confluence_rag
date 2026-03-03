"""
Query routing for hybrid RAG + Database pipeline.

This package provides intelligent routing of user queries to the
appropriate pipeline(s) based on query intent.

Example:
    >>> from routing import QueryRouter
    >>> router = QueryRouter(rag_pipeline, db_pipeline, iliad_client)
    >>> result = router.route("How many pages use Python?")
"""

from .intent_classifier import IntentClassifier, QueryIntent
from .query_router import QueryRouter
from .response_combiner import ResponseCombiner

__all__ = [
    "IntentClassifier",
    "QueryIntent",
    "QueryRouter",
    "ResponseCombiner",
]
