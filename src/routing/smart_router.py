"""
LLM-based smart query router with unified analysis and parallel execution.

This module provides an intelligent query routing system that:
1. Analyzes queries using a SINGLE LLM call (UnifiedQueryAnalyzer)
2. Extracts entities, classifies intent, and decomposes queries all at once
3. Executes sub-queries in parallel across appropriate pipelines
4. Aggregates results into a coherent response
5. Supports multi-step queries with feedback loops via AgentOrchestrator

Example:
    >>> from routing.smart_router import SmartQueryRouter
    >>> router = SmartQueryRouter(rag_pipeline, db_pipeline, iliad_client)
    >>> result = router.route("Describe ALFA and how many projects reference it")
    >>>
    >>> # Multi-step query with agent orchestration
    >>> result = router.route_multistep("What projects are similar to ALFA?")
"""

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from loguru import logger

from .types import QueryIntent, SubQuery
from .unified_analyzer import UnifiedQueryAnalyzer, UnifiedAnalysisResult
from .parallel_executor import ParallelQueryExecutor, SubQueryResult
from .result_aggregator import ResultAggregator, AggregatedResult
from .patterns import is_list_describe_query

# Import types for type hints
try:
    from iliad.client import IliadClient
    from database.pipeline import DatabasePipeline
    from rag.pipeline import RAGPipeline
except ImportError:
    pass


@dataclass
class SmartRouteResult:
    """Result from smart query routing.

    Attributes:
        success: Overall success status
        answer: Final synthesized answer
        original_query: The original user query
        analysis: Unified analysis result (entities, intent, sub-queries)
        sub_results: Individual sub-query results
        sources: Combined sources
        queries: Generated database queries
        execution_time: Total execution time
        metadata: Additional metadata
    """

    success: bool
    answer: str
    original_query: str
    analysis: Optional[UnifiedAnalysisResult] = None
    sub_results: List[SubQueryResult] = field(default_factory=list)
    sources: List[Dict[str, Any]] = field(default_factory=list)
    queries: List[str] = field(default_factory=list)
    execution_time: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


class SmartQueryRouter:
    """
    Intelligent query router with LLM-based decomposition.

    Uses LLM to analyze queries, decompose complex ones into
    sub-queries, execute them in parallel, and synthesize results.

    Attributes:
        rag_pipeline: RAG pipeline for semantic queries
        db_pipeline: Database pipeline for structured queries
        iliad_client: Iliad client for LLM operations
        analyzer: Query analyzer
        executor: Parallel query executor
        aggregator: Result aggregator

    Example:
        >>> router = SmartQueryRouter(rag_pipeline, db_pipeline, iliad_client)
        >>> result = router.route("What is ALFA and who works on it?")
        >>> print(result.answer)
    """

    def __init__(
        self,
        rag_pipeline: "RAGPipeline",
        db_pipeline: Optional["DatabasePipeline"] = None,
        iliad_client: Optional["IliadClient"] = None,
        analyzer_model: str = "gpt-5-mini-global",
        synthesis_model: str = "gpt-4o-mini-global",
        max_workers: int = 4,
        query_timeout: float = 30.0,
    ) -> None:
        """Initialize smart query router.

        Args:
            rag_pipeline: RAG pipeline instance
            db_pipeline: Optional database pipeline
            iliad_client: Iliad client for LLM operations
            analyzer_model: Model for query analysis
            synthesis_model: Model for result synthesis
            max_workers: Max parallel workers
            query_timeout: Timeout per query in seconds
        """
        self.rag_pipeline = rag_pipeline
        self.db_pipeline = db_pipeline
        self.iliad_client = iliad_client

        # Initialize unified analyzer (single LLM call for all analysis)
        if iliad_client:
            self.analyzer = UnifiedQueryAnalyzer(
                iliad_client=iliad_client,
                model=analyzer_model,
            )
            self.aggregator = ResultAggregator(
                iliad_client=iliad_client,
                model=synthesis_model,
            )
        else:
            self.analyzer = None
            self.aggregator = ResultAggregator(use_llm=False)

        self.executor = ParallelQueryExecutor(
            rag_pipeline=rag_pipeline,
            db_pipeline=db_pipeline,
            iliad_client=iliad_client,
            max_workers=max_workers,
            timeout=query_timeout,
        )

        # Initialize agent orchestrator for multi-step queries
        self.orchestrator = None
        self._init_orchestrator(
            rag_pipeline=rag_pipeline,
            db_pipeline=db_pipeline,
            iliad_client=iliad_client,
            max_workers=max_workers,
        )

        logger.info(
            f"Initialized SmartQueryRouter "
            f"(LLM: {iliad_client is not None}, DB: {db_pipeline is not None}, "
            f"Orchestrator: {self.orchestrator is not None})"
        )

    def _init_orchestrator(
        self,
        rag_pipeline: "RAGPipeline",
        db_pipeline: Optional["DatabasePipeline"],
        iliad_client: Optional["IliadClient"],
        max_workers: int,
    ) -> None:
        """Initialize agent orchestrator with available agents.

        Args:
            rag_pipeline: RAG pipeline instance
            db_pipeline: Optional database pipeline
            iliad_client: Iliad client for LLM operations
            max_workers: Max parallel workers
        """
        try:
            from agents.orchestrator import AgentOrchestrator
            from agents.rag_agent import RAGAgent
            from agents.database_agent import DatabaseAgent

            agents = []

            # Initialize RAG agent
            rag_agent = RAGAgent(
                rag_pipeline=rag_pipeline,
                iliad_client=iliad_client,
            )
            agents.append(rag_agent)

            # Initialize Database agent if pipeline available
            if db_pipeline:
                db_agent = DatabaseAgent(
                    db_pipeline=db_pipeline,
                    iliad_client=iliad_client,
                )
                agents.append(db_agent)

            # Create orchestrator
            self.orchestrator = AgentOrchestrator(
                agents=agents,
                iliad_client=iliad_client,
                max_workers=max_workers,
                max_iterations=3,
            )

            logger.info(f"Initialized orchestrator with {len(agents)} agents")

        except ImportError as e:
            logger.warning(f"Could not initialize orchestrator: {e}")
            self.orchestrator = None

    def route(
        self,
        query: str,
        force_simple: bool = False,
    ) -> SmartRouteResult:
        """
        Route a query through the smart pipeline.

        Steps:
        1. Analyze and decompose query using LLM
        2. Execute sub-queries in parallel
        3. Aggregate results into final answer

        Args:
            query: User's natural language query
            force_simple: Skip decomposition, treat as single query

        Returns:
            SmartRouteResult with answer and metadata

        Example:
            >>> result = router.route("What is ALFA and how many pages mention it?")
            >>> print(result.answer)
            >>> print(f"Executed {len(result.sub_results)} sub-queries")
        """
        start_time = time.time()

        logger.info(f"Smart routing query: {query[:100]}...")

        # Check for list+describe pattern - delegate to orchestrator
        if not force_simple and self._is_list_describe_query(query):
            logger.info("Detected list+describe pattern, using orchestrator")
            return self.route_multistep(query)

        try:
            # Step 1: Analyze and decompose query
            if self.analyzer and not force_simple:
                analysis = self.analyzer.analyze(query)
            else:
                # Fallback: simple single-query analysis using types from types.py
                from .unified_analyzer import UnifiedAnalysisResult, EntityExtractionResult

                analysis = UnifiedAnalysisResult(
                    original_query=query,
                    entities=EntityExtractionResult(),
                    primary_intent=QueryIntent.RAG,
                    confidence=0.5,
                    sub_queries=[
                        SubQuery(
                            text=query,
                            intent=QueryIntent.RAG,
                        )
                    ],
                    is_complex=False,
                    is_comparative=False,
                    comparative_entities=[],
                    analysis_reasoning="Fallback: no LLM analyzer available",
                )

            logger.info(
                f"Analysis complete: {len(analysis.sub_queries)} sub-queries, "
                f"complex={analysis.is_complex}"
            )

            # Log sub-queries for debugging
            for i, sq in enumerate(analysis.sub_queries):
                logger.debug(f"  Sub-query {i + 1}: [{sq.intent.value}] {sq.text[:50]}...")

            # Step 2: Execute sub-queries in parallel
            sub_results = self.executor.execute(analysis.sub_queries)

            # Step 3: Aggregate results
            aggregated = self.aggregator.aggregate(query, sub_results)

            execution_time = time.time() - start_time

            logger.info(
                f"Smart routing complete in {execution_time:.2f}s "
                f"(success: {aggregated.success})"
            )

            return SmartRouteResult(
                success=aggregated.success,
                answer=aggregated.answer,
                original_query=query,
                analysis=analysis,
                sub_results=sub_results,
                sources=aggregated.sources,
                queries=aggregated.queries,
                execution_time=execution_time,
                metadata={
                    "is_complex": analysis.is_complex,
                    "num_sub_queries": len(analysis.sub_queries),
                    **aggregated.metadata,
                },
            )

        except Exception as e:
            logger.error(f"Smart routing failed: {e}")
            execution_time = time.time() - start_time

            return SmartRouteResult(
                success=False,
                answer=f"An error occurred while processing your query: {str(e)}",
                original_query=query,
                execution_time=execution_time,
                metadata={"error": str(e)},
            )

    def route_batch(
        self,
        queries: List[str],
        force_simple: bool = False,
    ) -> List[SmartRouteResult]:
        """
        Route multiple queries.

        Args:
            queries: List of queries to route
            force_simple: Skip decomposition for all queries

        Returns:
            List of SmartRouteResult objects
        """
        results = []
        for i, query in enumerate(queries, 1):
            logger.info(f"Processing query {i}/{len(queries)}")
            results.append(self.route(query, force_simple=force_simple))
        return results

    def get_route_summary(self, result: SmartRouteResult) -> Dict[str, Any]:
        """
        Get a summary of the routing decision.

        Args:
            result: SmartRouteResult to summarize

        Returns:
            Dict with routing summary
        """
        intent_counts = {}
        for sr in result.sub_results:
            intent = sr.sub_query.intent.value
            intent_counts[intent] = intent_counts.get(intent, 0) + 1

        return {
            "original_query": result.original_query,
            "is_complex": result.metadata.get("is_complex", False),
            "num_sub_queries": len(result.sub_results),
            "intent_breakdown": intent_counts,
            "success_rate": (
                sum(1 for sr in result.sub_results if sr.success) / len(result.sub_results)
                if result.sub_results
                else 0
            ),
            "execution_time": result.execution_time,
            "sources_count": len(result.sources),
            "queries_generated": result.queries,
        }

    def format_result(
        self,
        result: SmartRouteResult,
        include_sources: bool = True,
        include_debug: bool = False,
    ) -> str:
        """
        Format result for display.

        Args:
            result: SmartRouteResult to format
            include_sources: Include source citations
            include_debug: Include debug information

        Returns:
            Formatted string
        """
        output = result.answer

        if include_sources and result.sources:
            output += "\n\n**Sources:**\n"
            for i, source in enumerate(result.sources[:5], 1):
                title = source.get("title", "Unknown")
                url = source.get("url", "")
                if url:
                    output += f"{i}. [{title}]({url})\n"
                else:
                    output += f"{i}. {title}\n"

        if include_debug:
            output += f"\n\n---\n**Debug Info:**\n"
            output += f"- Execution time: {result.execution_time:.2f}s\n"
            output += f"- Sub-queries: {len(result.sub_results)}\n"
            output += f"- Complex query: {result.metadata.get('is_complex', False)}\n"

            if result.analysis:
                output += f"- Analysis: {result.analysis.analysis_reasoning}\n"

            for i, sr in enumerate(result.sub_results, 1):
                status = "✓" if sr.success else "✗"
                output += f"  {i}. [{sr.sub_query.intent.value}] {status} {sr.sub_query.text[:40]}...\n"

            if result.queries:
                output += f"- Generated queries:\n"
                for q in result.queries:
                    output += f"  `{q[:80]}...`\n"

        return output

    def route_multistep(
        self,
        query: str,
        force_agents: Optional[List[str]] = None,
    ) -> SmartRouteResult:
        """
        Route query using agent orchestrator for multi-step execution.

        Uses the AgentOrchestrator to handle complex queries that require
        multiple steps with context passing between steps. This is ideal
        for queries like "What projects are similar to ALFA?" which require:
        1. First getting information about ALFA
        2. Using that information to find similar projects

        Args:
            query: User's natural language query
            force_agents: Optional list of agent names to use

        Returns:
            SmartRouteResult with answer and metadata

        Example:
            >>> result = router.route_multistep("What projects are similar to ALFA?")
            >>> print(result.answer)
            >>> print(f"Executed in {len(result.metadata.get('agents_used', []))} steps")
        """
        if not self.orchestrator:
            logger.warning("Orchestrator not available, falling back to standard routing")
            return self.route(query)

        logger.info(f"Multi-step routing query: {query[:100]}...")

        try:
            # Execute via orchestrator
            orch_result = self.orchestrator.execute(query, force_agents=force_agents)

            # Convert OrchestrationResult to SmartRouteResult
            sources = []
            sub_results = []

            # Extract sources and sub_results from orchestration steps
            for step in orch_result.steps_executed:
                step_result = step.get("result")
                if step_result and step_result.data:
                    # Extract sources if present
                    if isinstance(step_result.data, dict):
                        step_sources = step_result.data.get("sources", [])
                        sources.extend(step_sources)

            return SmartRouteResult(
                success=orch_result.success,
                answer=orch_result.final_answer,
                original_query=query,
                analysis=None,  # Orchestrator doesn't use QueryAnalysisResult
                sub_results=sub_results,
                sources=sources,
                queries=[],  # Could extract from steps if needed
                execution_time=orch_result.execution_time,
                metadata={
                    "routing_mode": "multi_step",
                    "agents_used": orch_result.metadata.get("agents_used", []),
                    "num_steps": orch_result.metadata.get("num_steps", 0),
                    "iterations": orch_result.metadata.get("iterations", 0),
                    "steps": [
                        {
                            "agent": s.get("agent"),
                            "query": s.get("query", "")[:100],
                            "success": s.get("result").success if s.get("result") else False,
                        }
                        for s in orch_result.steps_executed
                    ],
                },
            )

        except Exception as e:
            logger.error(f"Multi-step routing failed: {e}")
            return SmartRouteResult(
                success=False,
                answer=f"Error in multi-step processing: {str(e)}",
                original_query=query,
                metadata={"error": str(e), "routing_mode": "multi_step"},
            )

    def _is_list_describe_query(self, query: str) -> bool:
        """Check if query matches list+describe pattern.

        These queries need the IterativeDescribeAgent to first get a list
        from the database, then describe each item via RAG.

        Args:
            query: User query to check

        Returns:
            True if this is a list+describe pattern
        """
        # Use the shared function from patterns.py
        result = is_list_describe_query(query)
        if result:
            logger.debug(f"Matched list+describe pattern for query: {query[:50]}...")
        return result

    def supports_multistep(self) -> bool:
        """Check if multi-step routing is available.

        Returns:
            True if orchestrator is initialized
        """
        return self.orchestrator is not None
