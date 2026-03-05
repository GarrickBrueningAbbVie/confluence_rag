"""
Database Agent for structured queries.

This agent wraps the existing DatabasePipeline to provide structured
query capabilities within the agent framework. It handles counting,
listing, filtering, and aggregation queries.

Example:
    >>> from agents.database_agent import DatabaseAgent
    >>> from database.pipeline import DatabasePipeline
    >>>
    >>> db_agent = DatabaseAgent(db_pipeline, iliad_client)
    >>> context = AgentContext(original_query="How many Python projects?")
    >>> result = db_agent.execute("Count projects using Python", context)
    >>> print(result.data["answer"])
"""

from typing import Any, Dict, List, Optional

from loguru import logger

from agents.base import BaseAgent, AgentContext, AgentResult, AgentStatus

# Type hints for optional imports
try:
    from database.pipeline import DatabasePipeline
    from iliad.client import IliadClient
except ImportError:
    pass


class DatabaseAgent(BaseAgent):
    """Agent for structured database queries.

    Wraps the DatabasePipeline to provide structured query capabilities.
    Handles queries requiring counting, listing, filtering, or aggregation.

    The agent generates pandas queries from natural language and executes
    them safely against the Confluence data.

    Attributes:
        db_pipeline: Configured DatabasePipeline instance
        iliad_client: Optional Iliad client for query enhancement

    Example:
        >>> db_agent = DatabaseAgent(db_pipeline, iliad_client)
        >>> result = db_agent.execute("List all projects using Airflow", context)
        >>> if result.success:
        ...     for project in result.data["answer"]:
        ...         print(project)
    """

    # Keywords indicating database query is appropriate
    DB_INDICATORS: List[str] = [
        "how many",
        "count",
        "list all",
        "list the",
        "show all",
        "show me",
        "who created",
        "who works",
        "filter",
        "average",
        "total",
        "most common",
        "least common",
        "maximum",
        "minimum",
        "sum",
        "group by",
        "sort by",
        "top",
        "bottom",
        "all projects",
        "all pages",
        "references",
        "mentions",
    ]

    def __init__(
        self,
        db_pipeline: "DatabasePipeline",
        iliad_client: Optional["IliadClient"] = None,
    ) -> None:
        """Initialize Database agent.

        Args:
            db_pipeline: Configured DatabasePipeline instance
            iliad_client: Optional Iliad client for query enhancement
        """
        super().__init__(
            name="database_agent",
            description="Structured queries for counts, lists, filtering, and aggregations",
            iliad_client=iliad_client,
        )
        self.db_pipeline = db_pipeline

        logger.info("Initialized DatabaseAgent with structured query capabilities")

    def execute(
        self,
        query: str,
        context: AgentContext,
    ) -> AgentResult:
        """Execute structured database query.

        Converts natural language query to pandas code and executes
        it against the Confluence data.

        Args:
            query: The query to process
            context: Shared execution context

        Returns:
            AgentResult with query results

        Example:
            >>> result = agent.execute("How many pages use Python?", context)
            >>> print(result.data["answer"])  # 42
            >>> print(result.data["query"])   # df[df['technologies'].str.contains...
        """
        logger.info(f"DatabaseAgent executing: {query[:80]}...")

        try:
            # Step 1: Enhance query with context if available
            enhanced_query = self._enhance_query_with_context(query, context)

            # Step 2: Execute database pipeline
            result = self.db_pipeline.query(enhanced_query)

            # Record execution
            context.record_execution(self.name, query)

            if result["success"]:
                answer = result["answer"]
                generated_query = result.get("query", "")

                # Format answer for storage
                formatted_answer = self._format_answer(answer)

                logger.info(
                    f"DatabaseAgent completed: query='{generated_query[:50]}...'"
                )

                return AgentResult(
                    status=AgentStatus.SUCCESS,
                    data={
                        "answer": answer,
                        "formatted_answer": formatted_answer,
                        "query": generated_query,
                        "row_count": self._count_results(answer),
                    },
                    confidence=0.9,  # Database queries are generally reliable
                    reasoning=f"Generated pandas query: {generated_query[:60]}...",
                    metadata={
                        "enhanced_query": enhanced_query,
                        "original_query": query,
                    },
                )
            else:
                error_msg = result.get("error", "Database query failed")
                logger.warning(f"DatabaseAgent query failed: {error_msg}")

                return AgentResult(
                    status=AgentStatus.FAILED,
                    reasoning=error_msg,
                    metadata={
                        "error": error_msg,
                        "original_query": query,
                    },
                )

        except Exception as e:
            logger.error(f"DatabaseAgent execution failed: {e}")
            return AgentResult(
                status=AgentStatus.FAILED,
                reasoning=str(e),
                metadata={"error": str(e)},
            )

    def can_handle(self, query: str, context: AgentContext) -> float:
        """Determine if this agent can handle the query.

        Evaluates the query against known database indicators and returns
        a confidence score.

        Args:
            query: The query to evaluate
            context: Execution context

        Returns:
            Confidence score (0.0 - 1.0)

        Example:
            >>> score = agent.can_handle("How many pages use Python?", context)
            >>> print(f"Confidence: {score:.2f}")  # ~0.8
        """
        query_lower = query.lower()

        # Base score
        score = 0.2

        # Check for database indicators
        for indicator in self.DB_INDICATORS:
            if indicator in query_lower:
                score += 0.15

        # Penalize if query looks like semantic search
        semantic_indicators = ["explain", "describe", "what is", "how does", "why"]
        for indicator in semantic_indicators:
            if indicator in query_lower:
                score -= 0.1

        # Bonus if context suggests structured query
        if context.metadata.get("prefer_structured", False):
            score += 0.2

        # Bonus if previous result is chartable (follow-up aggregation)
        if context.has_result("chart_data"):
            score += 0.1

        return max(0.0, min(score, 1.0))

    def _enhance_query_with_context(
        self,
        query: str,
        context: AgentContext,
    ) -> str:
        """Inject intermediate results into query.

        Supports template syntax with {placeholder} for result injection.

        Args:
            query: Original query (may contain {placeholders})
            context: Context with intermediate results

        Returns:
            Enhanced query with placeholders filled
        """
        enhanced = query

        # Replace template placeholders
        for key, value in context.intermediate_results.items():
            placeholder = "{" + key + "}"
            if placeholder in enhanced:
                # Convert lists to comma-separated strings for queries
                if isinstance(value, list):
                    value_str = ", ".join(str(v) for v in value[:20])
                else:
                    value_str = str(value)
                enhanced = enhanced.replace(placeholder, value_str)

        return enhanced

    def _format_answer(self, answer: Any) -> str:
        """Format answer as human-readable string.

        Args:
            answer: Raw database result

        Returns:
            Formatted string representation
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

    def _count_results(self, answer: Any) -> int:
        """Count number of results.

        Args:
            answer: Database result

        Returns:
            Number of result items
        """
        if isinstance(answer, list):
            return len(answer)
        if isinstance(answer, dict):
            return len(answer)
        if isinstance(answer, (int, float)):
            return 1
        return 1


class DatabaseStatsAgent(BaseAgent):
    """Agent for retrieving database statistics.

    Provides schema information, sample questions, and statistics
    about the underlying data. Useful for meta-queries.

    Attributes:
        db_pipeline: Configured DatabasePipeline instance
    """

    def __init__(
        self,
        db_pipeline: "DatabasePipeline",
    ) -> None:
        """Initialize Database stats agent.

        Args:
            db_pipeline: Configured DatabasePipeline instance
        """
        super().__init__(
            name="database_stats_agent",
            description="Provides schema info and statistics about the data",
        )
        self.db_pipeline = db_pipeline

    def execute(
        self,
        query: str,
        context: AgentContext,
    ) -> AgentResult:
        """Get database statistics.

        Args:
            query: The query to process
            context: Shared execution context

        Returns:
            AgentResult with statistics
        """
        logger.info(f"DatabaseStatsAgent executing: {query[:80]}...")

        try:
            stats = self.db_pipeline.get_stats()
            sample_questions = self.db_pipeline.get_sample_questions()

            context.record_execution(self.name, query)

            return AgentResult(
                status=AgentStatus.SUCCESS,
                data={
                    "stats": stats,
                    "sample_questions": sample_questions,
                },
                confidence=1.0,
                reasoning="Retrieved database statistics",
            )

        except Exception as e:
            logger.error(f"DatabaseStatsAgent failed: {e}")
            return AgentResult(
                status=AgentStatus.FAILED,
                reasoning=str(e),
            )

    def can_handle(self, query: str, context: AgentContext) -> float:
        """Check if query is asking for stats/schema info.

        Args:
            query: The query to evaluate
            context: Execution context

        Returns:
            Confidence score (0.0 - 1.0)
        """
        query_lower = query.lower()
        stats_indicators = [
            "schema",
            "columns",
            "fields",
            "what data",
            "what can i ask",
            "sample questions",
            "statistics",
        ]

        for indicator in stats_indicators:
            if indicator in query_lower:
                return 0.9

        return 0.1
