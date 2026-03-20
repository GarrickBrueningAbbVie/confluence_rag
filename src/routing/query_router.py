"""
Query router for hybrid RAG + Database pipeline.

This module routes queries to the appropriate pipeline(s) based on
intent classification and combines results.

Routing modes:
1. Smart mode (default): LLM-based unified analysis with query decomposition
2. Rule-based (fallback): Fast keyword matching for simple queries

The smart mode uses UnifiedQueryAnalyzer which performs entity extraction,
intent classification, and query decomposition in a SINGLE LLM call.

Example:
    >>> from routing.query_router import QueryRouter
    >>> router = QueryRouter(rag_pipeline, db_pipeline, iliad_client)
    >>> result = router.route("How many pages use Python?")

    >>> # Smart mode handles complex queries automatically
    >>> result = router.route("Compare ALFA and CloverX projects")
"""

from typing import Any, Dict, List, Optional

from loguru import logger

from .types import QueryIntent, ClassificationResult
from .intent_classifier import IntentClassifier
from .response_combiner import ResponseCombiner
from .formatters import format_db_answer

# Import types for type hints
try:
    from iliad.client import IliadClient
    from database.pipeline import DatabasePipeline
except ImportError:
    pass


class QueryRouter:
    """
    Route queries to appropriate pipelines.

    Classifies query intent and routes to RAG, Database, or both
    pipelines as needed.

    Attributes:
        rag_pipeline: RAG pipeline for semantic search
        db_pipeline: Database pipeline for structured queries
        classifier: Intent classifier
        combiner: Response combiner for hybrid queries

    Example:
        >>> router = QueryRouter(rag_pipeline, db_pipeline, iliad_client)
        >>> result = router.route("What technologies does Project X use?")
    """

    def __init__(
        self,
        rag_pipeline: Any,  # Type hint as Any to avoid circular import
        db_pipeline: Optional["DatabasePipeline"] = None,
        iliad_client: Optional["IliadClient"] = None,
        use_llm_fallback: bool = False,
        use_smart_routing: bool = True,
    ) -> None:
        """Initialize query router.

        Args:
            rag_pipeline: RAG pipeline instance
            db_pipeline: Optional database pipeline instance
            iliad_client: Optional Iliad client for LLM operations
            use_llm_fallback: Whether to use LLM for ambiguous classification (rule-based fallback)
            use_smart_routing: Use LLM-based smart routing with unified analysis (default: True)
        """
        self.rag_pipeline = rag_pipeline
        self.db_pipeline = db_pipeline
        self.iliad_client = iliad_client
        self.use_smart_routing = use_smart_routing

        # Intent classifier for rule-based fallback
        self.classifier = IntentClassifier(
            iliad_client=iliad_client,
            use_llm_fallback=use_llm_fallback,
        )

        # Response combiner for hybrid queries
        self.combiner = ResponseCombiner(iliad_client=iliad_client)

        # Initialize smart router with unified analyzer if enabled
        if use_smart_routing and iliad_client:
            from .smart_router import SmartQueryRouter

            self.smart_router = SmartQueryRouter(
                rag_pipeline=rag_pipeline,
                db_pipeline=db_pipeline,
                iliad_client=iliad_client,
            )
        else:
            self.smart_router = None

        logger.info(
            f"Initialized QueryRouter "
            f"(DB: {db_pipeline is not None}, Smart: {use_smart_routing})"
        )

    def route(
        self,
        query: str,
        force_intent: Optional[QueryIntent] = None,
        return_metadata: bool = True,
        use_smart: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """
        Route a query to appropriate pipeline(s).

        Routing logic:
        1. If smart routing enabled: Use UnifiedQueryAnalyzer for single-LLM-call analysis
        2. Otherwise: Fall back to rule-based keyword classification

        Args:
            query: User query string
            force_intent: Optional intent override (disables smart routing)
            return_metadata: Whether to include routing metadata
            use_smart: Override smart routing setting for this query

        Returns:
            Dict with:
            - success: Whether query succeeded
            - answer: Combined response
            - sources: Source information (RAG)
            - query: Generated query (Database)
            - intent: Classified intent
            - metadata: Routing metadata (if return_metadata=True)

        Example:
            >>> result = router.route("How many pages use Airflow?")
            >>> print(result['answer'])
        """
        # Determine if we should use smart routing
        should_use_smart = use_smart if use_smart is not None else self.use_smart_routing

        logger.info(f"Routing query: '{query[:100]}{'...' if len(query) > 100 else ''}'")

        # Use smart router if enabled and no forced intent
        if should_use_smart and self.smart_router and not force_intent:
            logger.info("Route selected: SMART (unified LLM analysis)")
            return self._route_smart(query, return_metadata)

        # Fall back to rule-based routing
        logger.info("Route selected: RULE-BASED (keyword matching)")
        result = {
            "success": False,
            "answer": None,
            "sources": [],
            "query": None,
            "intent": None,
            "metadata": {},
        }

        # Classify intent using rule-based classifier
        if force_intent:
            classification = ClassificationResult(
                intent=force_intent,
                confidence=1.0,
                reasoning="Intent forced by user",
            )
        else:
            classification = self.classifier.classify(query)

        final_intent = classification.intent
        result["intent"] = final_intent.value

        if return_metadata:
            result["metadata"] = {
                "intent": final_intent.value,
                "confidence": classification.confidence,
                "reasoning": classification.reasoning,
                "routing_mode": "rule_based",
            }

        logger.info(
            f"Query classified as {final_intent.value.upper()} "
            f"(confidence: {classification.confidence:.2f})"
        )

        # Route based on final intent
        try:
            if final_intent == QueryIntent.DATABASE:
                logger.info("Executing pipeline: DATABASE (structured query)")
                result = self._route_database(query, result)

            elif final_intent == QueryIntent.RAG:
                logger.info("Executing pipeline: RAG (semantic search)")
                result = self._route_rag(query, result)

            elif final_intent == QueryIntent.HYBRID:
                logger.info("Executing pipeline: HYBRID (RAG + Database)")
                result = self._route_hybrid(query, result)

            elif final_intent == QueryIntent.CHART:
                logger.info("Executing pipeline: CHART (visualization)")
                result = self._route_chart(query, result)

            elif final_intent == QueryIntent.TABLE:
                logger.info("Executing pipeline: TABLE (tabular data)")
                result = self._route_database(query, result)

        except Exception as e:
            logger.error(f"Routing failed: {e}")
            result["error"] = str(e)

        return result

    def _route_smart(
        self,
        query: str,
        return_metadata: bool = True,
    ) -> Dict[str, Any]:
        """Route using smart LLM-based router.

        Args:
            query: User query
            return_metadata: Whether to include metadata

        Returns:
            Result dict in same format as rule-based routing
        """
        logger.info("Using smart LLM-based routing")

        smart_result = self.smart_router.route(query)

        # Log extracted sub-queries
        if smart_result.sub_results:
            logger.info(f"Smart routing decomposed into {len(smart_result.sub_results)} sub-queries:")
            for i, sr in enumerate(smart_result.sub_results, 1):
                status = "✓" if sr.success else "✗"
                logger.info(f"  [{i}] {status} [{sr.sub_query.intent.value.upper()}] {sr.sub_query.text}")
        else:
            logger.info("Smart routing: No sub-queries extracted (treated as single query)")

        # Convert to standard result format
        result = {
            "success": smart_result.success,
            "answer": smart_result.answer,
            "sources": smart_result.sources,
            "query": smart_result.queries[0] if smart_result.queries else None,
            "intent": "smart",
        }

        # Extract figures and tables from CHART/TABLE sub_results
        figures = []
        tables = []
        for sr in smart_result.sub_results:
            if sr.success and sr.metadata:
                # Extract chart figures
                if sr.metadata.get("figure"):
                    figures.append({
                        "figure": sr.metadata["figure"],
                        "html": sr.metadata.get("html"),
                        "code": sr.metadata.get("code"),
                        "chart_type": sr.metadata.get("chart_type", "auto"),
                        "query": sr.sub_query.text,
                    })
                # Extract tables
                if sr.metadata.get("html_table"):
                    tables.append({
                        "html": sr.metadata["html_table"],
                        "markdown": sr.answer,
                        "row_count": sr.metadata.get("row_count", 0),
                        "query": sr.sub_query.text,
                    })

        if figures:
            result["figures"] = figures
            logger.info(f"Extracted {len(figures)} chart figure(s) from sub-results")

        if tables:
            result["tables"] = tables
            logger.info(f"Extracted {len(tables)} table(s) from sub-results")

        if return_metadata:
            result["metadata"] = {
                "routing_mode": "smart",
                "is_complex": smart_result.metadata.get("is_complex", False),
                "num_sub_queries": len(smart_result.sub_results),
                "execution_time": smart_result.execution_time,
                "has_figures": len(figures) > 0,
                "has_tables": len(tables) > 0,
                "sub_queries": [
                    {
                        "text": sr.sub_query.text,
                        "intent": sr.sub_query.intent.value,
                        "success": sr.success,
                    }
                    for sr in smart_result.sub_results
                ],
                "all_queries": smart_result.queries,
            }

        return result

    def _route_database(
        self,
        query: str,
        result: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Route to database pipeline.

        Args:
            query: User query
            result: Result dict to update

        Returns:
            Updated result dict
        """
        if not self.db_pipeline:
            logger.warning("Database pipeline not available, falling back to RAG")
            return self._route_rag(query, result)

        logger.info("Routing to database pipeline")

        db_result = self.db_pipeline.query(query)

        if db_result["success"]:
            result["success"] = True
            result["answer"] = self._format_db_answer(db_result["answer"])
            result["query"] = db_result.get("query")
        else:
            result["error"] = db_result.get("error")

        return result

    def _route_rag(
        self,
        query: str,
        result: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Route to RAG pipeline.

        Args:
            query: User query
            result: Result dict to update

        Returns:
            Updated result dict
        """
        logger.info("Routing to RAG pipeline")

        rag_result = self.rag_pipeline.query(query)

        if rag_result.get("success", True):  # RAG may not have explicit success flag
            result["success"] = True
            result["answer"] = rag_result.get("response", rag_result.get("answer", ""))
            result["sources"] = rag_result.get("sources", [])
        else:
            result["error"] = rag_result.get("error", "RAG query failed")

        return result

    def _route_hybrid(
        self,
        query: str,
        result: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Route to both pipelines and combine results.

        Args:
            query: User query
            result: Result dict to update

        Returns:
            Updated result dict
        """
        logger.info("Routing to hybrid (RAG + Database)")

        rag_result = None
        db_result = None

        # Execute RAG
        try:
            rag_response = self.rag_pipeline.query(query)
            rag_result = {
                "answer": rag_response.get("response", rag_response.get("answer", "")),
                "sources": rag_response.get("sources", []),
            }
        except Exception as e:
            logger.warning(f"RAG query failed in hybrid: {e}")

        # Execute Database
        if self.db_pipeline:
            try:
                db_response = self.db_pipeline.query(query)
                if db_response["success"]:
                    db_result = {
                        "answer": self._format_db_answer(db_response["answer"]),
                        "query": db_response.get("query"),
                    }
            except Exception as e:
                logger.warning(f"Database query failed in hybrid: {e}")

        # Combine results
        combined = self.combiner.combine(
            query=query,
            rag_result=rag_result,
            db_result=db_result,
        )

        result["success"] = True
        result["answer"] = combined["answer"]
        result["sources"] = rag_result.get("sources", []) if rag_result else []

        if db_result:
            result["query"] = db_result.get("query")

        return result

    def _route_chart(
        self,
        query: str,
        result: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Route to chart generation.

        Args:
            query: User query
            result: Result dict to update

        Returns:
            Updated result dict
        """
        logger.info("Routing to chart generation")

        # First, get data from database pipeline
        if not self.db_pipeline:
            result["error"] = "Database pipeline required for chart generation"
            return result

        # Try to get data for the chart
        db_result = self.db_pipeline.query(query)

        if not db_result["success"]:
            result["error"] = f"Failed to get data for chart: {db_result.get('error')}"
            return result

        result["success"] = True
        result["answer"] = db_result["answer"]
        result["query"] = db_result.get("query")
        result["metadata"]["chart_data"] = db_result["answer"]
        result["metadata"]["requires_visualization"] = True

        return result

    def _format_db_answer(self, answer: Any) -> str:
        """Format database result as readable text.

        Args:
            answer: Raw database result

        Returns:
            Formatted string
        """
        return format_db_answer(answer)

    def get_available_modes(self) -> List[str]:
        """Get list of available routing modes.

        Returns:
            List of mode names
        """
        modes = ["auto", "rag"]

        if self.db_pipeline:
            modes.extend(["database", "hybrid", "chart"])

        return modes
