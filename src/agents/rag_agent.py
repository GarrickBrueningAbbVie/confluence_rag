"""
RAG Agent for semantic search and document retrieval.

This agent wraps the existing RAGPipeline to provide semantic search
capabilities within the agent framework. It handles conceptual questions,
explanations, and document-based queries.

Example:
    >>> from agents.rag_agent import RAGAgent
    >>> from rag.pipeline import RAGPipeline
    >>>
    >>> rag_agent = RAGAgent(rag_pipeline, iliad_client)
    >>> context = AgentContext(original_query="What is ALFA?")
    >>> result = rag_agent.execute("Describe the ALFA project", context)
    >>> print(result.data["answer"])
"""

from typing import Any, Dict, List, Optional, Tuple

from loguru import logger

from agents.base import BaseAgent, AgentContext, AgentResult, AgentStatus
from routing.patterns import RAG_INDICATORS as ROUTING_RAG_INDICATORS
from agents.query_utils import enhance_query_with_context

# Type hints for optional imports
try:
    from rag.pipeline import RAGPipeline
    from iliad.client import IliadClient
except ImportError:
    pass


class RAGAgent(BaseAgent):
    """Agent for RAG-based semantic search and document retrieval.

    Wraps the RAGPipeline to provide semantic search capabilities.
    Handles conceptual questions, explanations, and document lookup.

    The agent can enhance queries with context from previous steps,
    evaluate result quality, and trigger refinement when needed.

    Attributes:
        rag_pipeline: Configured RAGPipeline instance
        iliad_client: Optional Iliad client for result evaluation
        min_answer_length: Minimum answer length to consider valid
        max_distance_threshold: Maximum distance to consider relevant

    Example:
        >>> rag_agent = RAGAgent(rag_pipeline, iliad_client)
        >>> result = rag_agent.execute("Explain the RAG architecture", context)
        >>> if result.success:
        ...     print(result.data["answer"])
    """

    def __init__(
        self,
        rag_pipeline: "RAGPipeline",
        iliad_client: Optional["IliadClient"] = None,
        min_answer_length: int = 50,
        max_distance_threshold: float = 0.7,
    ) -> None:
        """Initialize RAG agent.

        Args:
            rag_pipeline: Configured RAGPipeline instance
            iliad_client: Optional Iliad client for result evaluation
            min_answer_length: Minimum answer length to consider valid
            max_distance_threshold: Maximum distance to consider relevant
        """
        super().__init__(
            name="rag_agent",
            description="Semantic search for conceptual questions, explanations, and documentation",
            iliad_client=iliad_client,
        )
        self.rag_pipeline = rag_pipeline
        self.min_answer_length = min_answer_length
        self.max_distance_threshold = max_distance_threshold

        logger.info("Initialized RAGAgent with semantic search capabilities")

    def execute(
        self,
        query: str,
        context: AgentContext,
    ) -> AgentResult:
        """Execute RAG query with optional context injection.

        If context contains intermediate results (e.g., from a previous
        step), they can be used to enhance the query for better retrieval.

        Args:
            query: The query to process
            context: Shared execution context

        Returns:
            AgentResult with answer, sources, and distances

        Example:
            >>> # Simple query
            >>> result = agent.execute("What is ALFA?", context)
            >>>
            >>> # Query with context injection
            >>> context.intermediate_results["project_summary"] = "ALFA is..."
            >>> result = agent.execute(
            ...     "Find similar projects to: {project_summary}",
            ...     context
            ... )
        """
        logger.info(f"RAGAgent executing: {query[:80]}...")

        try:
            # Step 1: Enhance query with context if available
            enhanced_query = self._enhance_query_with_context(query, context)

            if enhanced_query != query:
                logger.debug(f"Enhanced query: {enhanced_query[:100]}...")

            # Step 2: Execute RAG pipeline
            result = self.rag_pipeline.query(enhanced_query)

            # Step 3: Extract and structure results
            answer = result.get("answer", "")
            sources = result.get("sources", [])
            distances = result.get("distances", [])

            # Step 4: Evaluate result quality
            needs_followup, followup_query = self._evaluate_result(
                result, query, context
            )

            # Step 5: Calculate confidence
            confidence = self._calculate_confidence(result)

            # Record execution in context
            context.record_execution(self.name, query)

            logger.info(
                f"RAGAgent completed: {len(sources)} sources, "
                f"confidence={confidence:.2f}, needs_followup={needs_followup}"
            )

            return AgentResult(
                status=AgentStatus.SUCCESS,
                data={
                    "answer": answer,
                    "sources": sources,
                    "distances": distances,
                    "retrieved_documents": result.get("retrieved_documents", []),
                    "query_analysis": result.get("query_analysis", {}),
                },
                needs_followup=needs_followup,
                followup_query=followup_query,
                confidence=confidence,
                reasoning=f"Retrieved {len(sources)} relevant documents",
                metadata={
                    "enhanced_query": enhanced_query,
                    "original_query": query,
                },
            )

        except Exception as e:
            logger.error(f"RAGAgent execution failed: {e}")
            return AgentResult(
                status=AgentStatus.FAILED,
                reasoning=str(e),
                metadata={"error": str(e)},
            )

    def can_handle(self, query: str, context: AgentContext) -> float:
        """Determine if this agent can handle the query.

        Evaluates the query against known RAG indicators and returns
        a confidence score.

        Args:
            query: The query to evaluate
            context: Execution context

        Returns:
            Confidence score (0.0 - 1.0)

        Example:
            >>> score = agent.can_handle("What is ALFA?", context)
            >>> print(f"Confidence: {score:.2f}")  # ~0.6
        """
        query_lower = query.lower()

        # Base score
        score = 0.3

        # Check for RAG indicators
        for indicator in ROUTING_RAG_INDICATORS:
            if indicator in query_lower:
                score += 0.12

        # Penalize if query looks like database query
        db_indicators = ["how many", "count", "list all", "list the", "total", "average"]
        for indicator in db_indicators:
            if indicator in query_lower:
                score -= 0.15

        # Bonus if context suggests semantic search
        if context.metadata.get("prefer_semantic", False):
            score += 0.2

        return max(0.0, min(score, 1.0))

    def _enhance_query_with_context(
        self,
        query: str,
        context: AgentContext,
    ) -> str:
        """Inject intermediate results into query.

        Supports template syntax with {placeholder} for result injection.
        Also automatically adds project_summary context if available.

        Args:
            query: Original query (may contain {placeholders})
            context: Context with intermediate results

        Returns:
            Enhanced query with placeholders filled

        Example:
            >>> context.intermediate_results["summary"] = "ML project"
            >>> enhanced = agent._enhance_query_with_context(
            ...     "Find similar to: {summary}", context
            ... )
            >>> # Returns: "Find similar to: ML project"
        """
        # Use shared utility for placeholder replacement
        enhanced = enhance_query_with_context(
            query, context.intermediate_results, max_value_length=500
        )

        # RAG-specific: Auto-inject project_summary for similarity queries
        if (
            enhanced == query
            and "project_summary" in context.intermediate_results
            and "similar" in query.lower()
        ):
            summary = context.intermediate_results["project_summary"]
            if len(str(summary)) > 500:
                summary = str(summary)[:500] + "..."
            enhanced = f"{query}\n\nContext about the project:\n{summary}"

        return enhanced

    def _evaluate_result(
        self,
        result: Dict[str, Any],
        original_query: str,
        context: AgentContext,
    ) -> Tuple[bool, Optional[str]]:
        """Evaluate if result needs follow-up query.

        Checks for common issues like empty answers, short responses,
        or low-confidence retrieval.

        Args:
            result: RAG pipeline result
            original_query: The original query
            context: Execution context

        Returns:
            Tuple of (needs_followup, followup_query)
        """
        answer = result.get("answer", "")
        distances = result.get("distances", [])

        # Check if answer is too short
        if len(answer) < self.min_answer_length:
            if context.can_iterate():
                return True, f"Provide more detailed information about: {original_query}"

        # Check if retrieval was low-confidence (high distance)
        if distances and all(d > self.max_distance_threshold for d in distances[:3]):
            if context.can_iterate():
                return True, f"Search for alternative terms related to: {original_query}"

        # Check for "not found" type responses
        not_found_indicators = [
            "no information",
            "not found",
            "cannot find",
            "no relevant",
            "unable to find",
        ]
        answer_lower = answer.lower()
        if any(ind in answer_lower for ind in not_found_indicators):
            if context.can_iterate():
                # Try to reformulate query
                return True, f"Rephrase and search for: {original_query}"

        return False, None

    def _calculate_confidence(self, result: Dict[str, Any]) -> float:
        """Calculate confidence based on retrieval quality.

        Uses distance scores and answer characteristics to estimate
        result confidence.

        Args:
            result: RAG pipeline result

        Returns:
            Confidence score (0.0 - 1.0)
        """
        distances = result.get("distances", [])
        answer = result.get("answer", "")

        if not distances:
            return 0.5

        # Calculate average similarity (1 - distance) for top results
        top_distances = distances[:3]
        avg_similarity = sum(1 - d for d in top_distances) / len(top_distances)

        # Adjust based on answer length
        if len(answer) < self.min_answer_length:
            avg_similarity *= 0.7
        elif len(answer) > 200:
            avg_similarity *= 1.1

        return max(0.0, min(avg_similarity, 1.0))
