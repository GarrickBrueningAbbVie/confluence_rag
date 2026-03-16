"""
Query router for hybrid RAG + Database pipeline.

This module routes queries to the appropriate pipeline(s) based on
intent classification, entity extraction, and combines results.

Routing logic:
1. Extract entities (projects, people) from query via LLM
2. If entities found → RAG with entity-based filtering
3. If no entities AND aggregation/listing query → Database pipeline
4. If no entities AND informational query → RAG with similarity fallback

Two routing modes available:
1. Rule-based (default): Fast keyword matching for simple queries
2. Smart mode: LLM-based query decomposition for complex queries

Example:
    >>> from routing.query_router import QueryRouter
    >>> router = QueryRouter(rag_pipeline, db_pipeline, iliad_client)
    >>> result = router.route("How many pages use Python?")

    >>> # Enable smart mode for complex queries
    >>> router = QueryRouter(rag_pipeline, db_pipeline, iliad_client, use_smart_routing=True)
    >>> result = router.route("Describe ALFA and count how many projects reference it")
"""

from typing import Any, Dict, List, Optional

from loguru import logger

from .intent_classifier import IntentClassifier, QueryIntent, ClassificationResult
from .response_combiner import ResponseCombiner

# Import types for type hints
try:
    from iliad.client import IliadClient
    from database.pipeline import DatabasePipeline
    from rag.query_processor import QueryProcessor, ProcessedQuery
except ImportError:
    QueryProcessor = None
    ProcessedQuery = None


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
        use_smart_routing: bool = False,
        use_entity_routing: bool = True,
    ) -> None:
        """Initialize query router.

        Args:
            rag_pipeline: RAG pipeline instance
            db_pipeline: Optional database pipeline instance
            iliad_client: Optional Iliad client for LLM operations
            use_llm_fallback: Whether to use LLM for ambiguous classification
            use_smart_routing: Use LLM-based smart routing with query decomposition
            use_entity_routing: Use entity extraction to inform routing decisions
        """
        self.rag_pipeline = rag_pipeline
        self.db_pipeline = db_pipeline
        self.iliad_client = iliad_client
        self.use_smart_routing = use_smart_routing
        self.use_entity_routing = use_entity_routing

        self.classifier = IntentClassifier(
            iliad_client=iliad_client,
            use_llm_fallback=use_llm_fallback,
        )

        self.combiner = ResponseCombiner(iliad_client=iliad_client)

        # Initialize query processor for entity extraction
        if use_entity_routing and QueryProcessor is not None:
            self.query_processor = QueryProcessor(
                iliad_client=iliad_client,
                use_llm=iliad_client is not None,
            )
        else:
            self.query_processor = None

        # Initialize smart router if enabled
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
            f"(DB: {db_pipeline is not None}, Smart: {use_smart_routing}, "
            f"EntityRouting: {use_entity_routing and self.query_processor is not None})"
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

        Routing logic with entity extraction:
        1. Extract entities (projects, people) from query via LLM
        2. Classify query intent (RAG, DATABASE, HYBRID, CHART)
        3. If no entities extracted AND intent suggests database query → Database
        4. If entities extracted OR informational intent → RAG (with entity filtering)

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

        # Use smart router if enabled and no forced intent
        if should_use_smart and self.smart_router and not force_intent:
            return self._route_smart(query, return_metadata)

        # Fall back to rule-based routing
        result = {
            "success": False,
            "answer": None,
            "sources": [],
            "query": None,
            "intent": None,
            "metadata": {},
        }

        # Extract entities from query (if entity routing enabled)
        processed_query = None
        has_entities = False
        if self.use_entity_routing and self.query_processor:
            processed_query = self.query_processor.process_query(query)
            has_entities = (
                len(processed_query.potential_project_names) > 0 or
                len(processed_query.potential_person_names) > 0
            )
            logger.info(
                f"Entity extraction - Projects: {processed_query.potential_project_names}, "
                f"People: {processed_query.potential_person_names}, "
                f"Has entities: {has_entities}"
            )

        # Classify intent
        if force_intent:
            classification = ClassificationResult(
                intent=force_intent,
                confidence=1.0,
                reasoning="Intent forced by user",
            )
        else:
            classification = self.classifier.classify(query)

        # Apply entity-based routing override
        # If no entities and query is aggregation/listing → prefer Database
        # If entities found → prefer RAG (entity filtering handles it)
        final_intent = classification.intent
        routing_override = None

        if self.use_entity_routing and not force_intent:
            if not has_entities and classification.intent == QueryIntent.RAG:
                # Check if this looks like a database query based on processed_query intent
                if processed_query and processed_query.query_intent in ["aggregation", "listing"]:
                    final_intent = QueryIntent.DATABASE
                    routing_override = "No entities + aggregation/listing → Database"
                    logger.info(f"Routing override: {routing_override}")
            elif has_entities and classification.intent == QueryIntent.DATABASE:
                # If entities found but classified as database, consider hybrid
                # since we have specific entities to search for
                if processed_query and processed_query.query_intent not in ["aggregation"]:
                    final_intent = QueryIntent.HYBRID
                    routing_override = "Entities found + database intent → Hybrid"
                    logger.info(f"Routing override: {routing_override}")

        result["intent"] = final_intent.value

        if return_metadata:
            result["metadata"] = {
                "intent": final_intent.value,
                "original_intent": classification.intent.value,
                "confidence": classification.confidence,
                "reasoning": classification.reasoning,
                "routing_mode": "entity_aware" if self.use_entity_routing else "rule_based",
                "routing_override": routing_override,
                "has_entities": has_entities,
            }
            if processed_query:
                result["metadata"]["extracted_entities"] = {
                    "projects": processed_query.potential_project_names,
                    "people": processed_query.potential_person_names,
                    "query_intent": processed_query.query_intent,
                }

        logger.info(
            f"Query classified as {final_intent.value} "
            f"(original: {classification.intent.value}, confidence: {classification.confidence:.2f})"
        )

        # Route based on final intent
        try:
            if final_intent == QueryIntent.DATABASE:
                result = self._route_database(query, result)

            elif final_intent == QueryIntent.RAG:
                result = self._route_rag(query, result)

            elif final_intent == QueryIntent.HYBRID:
                result = self._route_hybrid(query, result)

            elif final_intent == QueryIntent.CHART:
                result = self._route_chart(query, result)

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

        # Convert to standard result format
        result = {
            "success": smart_result.success,
            "answer": smart_result.answer,
            "sources": smart_result.sources,
            "query": smart_result.queries[0] if smart_result.queries else None,
            "intent": "smart",
        }

        if return_metadata:
            result["metadata"] = {
                "routing_mode": "smart",
                "is_complex": smart_result.metadata.get("is_complex", False),
                "num_sub_queries": len(smart_result.sub_results),
                "execution_time": smart_result.execution_time,
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
        if answer is None:
            return "No results found."

        if isinstance(answer, (int, float)):
            return str(answer)

        if isinstance(answer, str):
            return answer

        if isinstance(answer, list):
            if len(answer) == 0:
                return "No results found."

            if len(answer) <= 10:
                # Format as list
                if isinstance(answer[0], dict):
                    lines = []
                    for item in answer:
                        line = ", ".join(f"{k}: {v}" for k, v in item.items())
                        lines.append(f"- {line}")
                    return "\n".join(lines)
                return "\n".join(f"- {item}" for item in answer)

            # Truncate long lists
            preview = answer[:10]
            if isinstance(preview[0], dict):
                lines = []
                for item in preview:
                    line = ", ".join(f"{k}: {v}" for k, v in item.items())
                    lines.append(f"- {line}")
                lines.append(f"... and {len(answer) - 10} more items")
                return "\n".join(lines)

            return "\n".join(f"- {item}" for item in preview) + f"\n... and {len(answer) - 10} more items"

        if isinstance(answer, dict):
            lines = [f"- {k}: {v}" for k, v in answer.items()]
            return "\n".join(lines)

        return str(answer)

    def get_available_modes(self) -> List[str]:
        """Get list of available routing modes.

        Returns:
            List of mode names
        """
        modes = ["auto", "rag"]

        if self.db_pipeline:
            modes.extend(["database", "hybrid", "chart"])

        return modes
