"""
Parallel execution of sub-queries across multiple pipelines.

This module executes decomposed sub-queries in parallel using
ThreadPoolExecutor for efficient multi-pipeline query handling.

Example:
    >>> from routing.parallel_executor import ParallelQueryExecutor
    >>> executor = ParallelQueryExecutor(rag_pipeline, db_pipeline)
    >>> results = executor.execute(sub_queries)
"""

import time
from concurrent.futures import ThreadPoolExecutor, as_completed, Future
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from loguru import logger

from .query_analyzer import SubQuery, SubQueryIntent

# Import types for type hints
try:
    from database.pipeline import DatabasePipeline
    from rag.pipeline import RAGPipeline
except ImportError:
    pass


@dataclass
class SubQueryResult:
    """Result of executing a single sub-query.

    Attributes:
        sub_query: The original sub-query
        success: Whether execution succeeded
        answer: The answer/result
        sources: Source documents (for RAG queries)
        query: Generated query (for database queries)
        execution_time: Time taken in seconds
        error: Error message if failed
        metadata: Additional metadata
    """

    sub_query: SubQuery
    success: bool
    answer: Any = None
    sources: List[Dict[str, Any]] = field(default_factory=list)
    query: Optional[str] = None
    execution_time: float = 0.0
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExecutionPlan:
    """Execution plan for sub-queries.

    Groups sub-queries by execution order (parallel vs sequential).

    Attributes:
        parallel_groups: List of groups that can run in parallel
        total_queries: Total number of queries
    """

    parallel_groups: List[List[SubQuery]]
    total_queries: int


class ParallelQueryExecutor:
    """
    Execute sub-queries in parallel across pipelines.

    Uses ThreadPoolExecutor to run independent sub-queries
    concurrently while respecting dependencies.

    Attributes:
        rag_pipeline: RAG pipeline for semantic queries
        db_pipeline: Database pipeline for structured queries
        max_workers: Maximum parallel workers
        timeout: Timeout per query in seconds

    Example:
        >>> executor = ParallelQueryExecutor(rag_pipeline, db_pipeline)
        >>> results = executor.execute(sub_queries)
        >>> for r in results:
        ...     print(f"{r.sub_query.intent}: {r.answer}")
    """

    def __init__(
        self,
        rag_pipeline: "RAGPipeline",
        db_pipeline: Optional["DatabasePipeline"] = None,
        max_workers: int = 4,
        timeout: float = 30.0,
    ) -> None:
        """Initialize parallel executor.

        Args:
            rag_pipeline: RAG pipeline instance
            db_pipeline: Optional database pipeline
            max_workers: Maximum concurrent workers
            timeout: Timeout per query in seconds
        """
        self.rag_pipeline = rag_pipeline
        self.db_pipeline = db_pipeline
        self.max_workers = max_workers
        self.timeout = timeout

        logger.info(
            f"Initialized ParallelQueryExecutor "
            f"(workers: {max_workers}, db_pipeline: {db_pipeline is not None})"
        )

    def create_execution_plan(self, sub_queries: List[SubQuery]) -> ExecutionPlan:
        """
        Create an execution plan from sub-queries.

        Groups queries into parallel batches based on dependencies.

        Args:
            sub_queries: List of sub-queries to plan

        Returns:
            ExecutionPlan with grouped queries
        """
        if not sub_queries:
            return ExecutionPlan(parallel_groups=[], total_queries=0)

        # Sort by priority
        sorted_queries = sorted(sub_queries, key=lambda sq: sq.priority)

        # Group by dependency level
        # Queries with no dependencies can run in parallel
        # Queries with dependencies must wait
        groups: List[List[SubQuery]] = []
        executed_indices: set = set()

        while len(executed_indices) < len(sorted_queries):
            current_group = []

            for i, sq in enumerate(sorted_queries):
                if i in executed_indices:
                    continue

                # Check if dependencies are satisfied
                if sq.depends_on is None or sq.depends_on in executed_indices:
                    current_group.append(sq)
                    executed_indices.add(i)

            if current_group:
                groups.append(current_group)
            else:
                # Circular dependency or error - force add remaining
                remaining = [
                    sq for i, sq in enumerate(sorted_queries) if i not in executed_indices
                ]
                if remaining:
                    groups.append(remaining)
                break

        return ExecutionPlan(parallel_groups=groups, total_queries=len(sub_queries))

    def execute(
        self,
        sub_queries: List[SubQuery],
        previous_results: Optional[List[SubQueryResult]] = None,
    ) -> List[SubQueryResult]:
        """
        Execute sub-queries with parallel processing.

        Args:
            sub_queries: List of sub-queries to execute
            previous_results: Results from previous batches (for context)

        Returns:
            List of SubQueryResult objects

        Example:
            >>> results = executor.execute(sub_queries)
            >>> successful = [r for r in results if r.success]
        """
        if not sub_queries:
            return []

        start_time = time.time()
        logger.info(f"Executing {len(sub_queries)} sub-queries")

        # Create execution plan
        plan = self.create_execution_plan(sub_queries)
        logger.info(f"Execution plan: {len(plan.parallel_groups)} parallel groups")

        all_results: List[SubQueryResult] = previous_results or []

        # Execute each group
        for group_idx, group in enumerate(plan.parallel_groups):
            logger.info(
                f"Executing group {group_idx + 1}/{len(plan.parallel_groups)} "
                f"({len(group)} queries)"
            )

            group_results = self._execute_parallel_group(group, all_results)
            all_results.extend(group_results)

        total_time = time.time() - start_time
        success_count = sum(1 for r in all_results if r.success)

        logger.info(
            f"Execution complete: {success_count}/{len(all_results)} successful "
            f"in {total_time:.2f}s"
        )

        return all_results

    def _execute_parallel_group(
        self,
        group: List[SubQuery],
        previous_results: List[SubQueryResult],
    ) -> List[SubQueryResult]:
        """Execute a group of queries in parallel.

        Args:
            group: Sub-queries to execute in parallel
            previous_results: Previous results for context

        Returns:
            List of results for this group
        """
        results: List[SubQueryResult] = []

        if len(group) == 1:
            # Single query - no need for thread pool
            result = self._execute_single(group[0], previous_results)
            return [result]

        # Multiple queries - use thread pool
        with ThreadPoolExecutor(max_workers=min(len(group), self.max_workers)) as executor:
            # Submit all queries
            future_to_query: Dict[Future, SubQuery] = {}

            for sq in group:
                future = executor.submit(self._execute_single, sq, previous_results)
                future_to_query[future] = sq

            # Collect results as they complete
            for future in as_completed(future_to_query, timeout=self.timeout * len(group)):
                sq = future_to_query[future]
                try:
                    result = future.result(timeout=self.timeout)
                    results.append(result)
                except Exception as e:
                    logger.error(f"Query execution failed: {e}")
                    results.append(
                        SubQueryResult(
                            sub_query=sq,
                            success=False,
                            error=str(e),
                        )
                    )

        return results

    def _execute_single(
        self,
        sub_query: SubQuery,
        previous_results: List[SubQueryResult],
    ) -> SubQueryResult:
        """Execute a single sub-query.

        Args:
            sub_query: The sub-query to execute
            previous_results: Previous results for context

        Returns:
            SubQueryResult
        """
        start_time = time.time()

        logger.debug(f"Executing {sub_query.intent.value}: {sub_query.text[:50]}...")

        try:
            if sub_query.intent == SubQueryIntent.RAG:
                result = self._execute_rag(sub_query)
            elif sub_query.intent == SubQueryIntent.DATABASE:
                result = self._execute_database(sub_query)
            elif sub_query.intent == SubQueryIntent.HYBRID:
                result = self._execute_hybrid(sub_query, previous_results)
            else:
                result = self._execute_rag(sub_query)  # Default fallback

            result.execution_time = time.time() - start_time
            return result

        except Exception as e:
            logger.error(f"Error executing sub-query: {e}")
            return SubQueryResult(
                sub_query=sub_query,
                success=False,
                error=str(e),
                execution_time=time.time() - start_time,
            )

    def _execute_rag(self, sub_query: SubQuery) -> SubQueryResult:
        """Execute a RAG query.

        Args:
            sub_query: The RAG sub-query

        Returns:
            SubQueryResult
        """
        try:
            rag_result = self.rag_pipeline.query(sub_query.text)

            return SubQueryResult(
                sub_query=sub_query,
                success=True,
                answer=rag_result.get("answer", ""),
                sources=rag_result.get("sources", []),
                metadata={
                    "distances": rag_result.get("distances", []),
                    "query_analysis": rag_result.get("query_analysis", {}),
                },
            )

        except Exception as e:
            logger.error(f"RAG execution failed: {e}")
            return SubQueryResult(
                sub_query=sub_query,
                success=False,
                error=f"RAG error: {str(e)}",
            )

    def _execute_database(self, sub_query: SubQuery) -> SubQueryResult:
        """Execute a database query.

        Args:
            sub_query: The database sub-query

        Returns:
            SubQueryResult
        """
        if not self.db_pipeline:
            logger.warning("Database pipeline not available, falling back to RAG")
            return self._execute_rag(sub_query)

        try:
            db_result = self.db_pipeline.query(sub_query.text)

            if db_result["success"]:
                return SubQueryResult(
                    sub_query=sub_query,
                    success=True,
                    answer=db_result["answer"],
                    query=db_result.get("query"),
                    metadata={"raw_result": db_result.get("answer")},
                )
            else:
                return SubQueryResult(
                    sub_query=sub_query,
                    success=False,
                    error=db_result.get("error", "Database query failed"),
                    query=db_result.get("query"),
                )

        except Exception as e:
            logger.error(f"Database execution failed: {e}")
            return SubQueryResult(
                sub_query=sub_query,
                success=False,
                error=f"Database error: {str(e)}",
            )

    def _execute_hybrid(
        self,
        sub_query: SubQuery,
        previous_results: List[SubQueryResult],
    ) -> SubQueryResult:
        """Execute a hybrid query (both RAG and database).

        Args:
            sub_query: The hybrid sub-query
            previous_results: Previous results for context

        Returns:
            SubQueryResult with combined information
        """
        # Execute both pipelines
        rag_result = self._execute_rag(sub_query)
        db_result = self._execute_database(sub_query)

        # Combine results
        combined_answer = ""
        sources = []

        if rag_result.success:
            combined_answer += f"From documentation:\n{rag_result.answer}\n\n"
            sources = rag_result.sources

        if db_result.success:
            combined_answer += f"From data:\n{self._format_db_answer(db_result.answer)}"

        return SubQueryResult(
            sub_query=sub_query,
            success=rag_result.success or db_result.success,
            answer=combined_answer.strip() if combined_answer else "No results found",
            sources=sources,
            query=db_result.query,
            metadata={
                "rag_success": rag_result.success,
                "db_success": db_result.success,
            },
        )

    def _format_db_answer(self, answer: Any) -> str:
        """Format database answer for display.

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

            return (
                "\n".join(f"- {item}" for item in preview)
                + f"\n... and {len(answer) - 10} more items"
            )

        if isinstance(answer, dict):
            return "\n".join(f"- {k}: {v}" for k, v in answer.items())

        return str(answer)
