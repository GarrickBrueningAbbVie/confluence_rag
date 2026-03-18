"""
Plotting Agent for data visualization generation.

This agent wraps the existing ChartGenerator to provide visualization
capabilities within the agent framework. It generates charts and graphs
from data using Plotly.

Example:
    >>> from agents.plotting_agent import PlottingAgent
    >>> from visualization.chart_generator import ChartGenerator
    >>>
    >>> plotting_agent = PlottingAgent(chart_generator, iliad_client)
    >>> context = AgentContext(original_query="Show pages by author")
    >>> context.intermediate_results["chart_data"] = {"Alice": 10, "Bob": 5}
    >>> result = plotting_agent.execute("Bar chart of pages by author", context)
"""

from typing import Any, Dict, List, Optional

from loguru import logger

from agents.base import BaseAgent, AgentContext, AgentResult, AgentStatus

# Type hints for optional imports
try:
    from visualization.chart_generator import ChartGenerator
    from iliad.client import IliadClient
except ImportError:
    pass


class PlottingAgent(BaseAgent):
    """Agent for generating visualizations from data.

    Wraps the ChartGenerator to provide visualization capabilities.
    Generates charts from data stored in context (typically from a
    previous database query).

    The agent supports both LLM-based chart generation for complex
    requests and quick chart generation for common patterns.

    Attributes:
        chart_generator: Configured ChartGenerator instance
        iliad_client: Optional Iliad client for chart generation
        default_chart_type: Default chart type when not specified

    Example:
        >>> plotting_agent = PlottingAgent(chart_generator, iliad_client)
        >>> context.intermediate_results["chart_data"] = data
        >>> result = plotting_agent.execute("Show me a pie chart", context)
        >>> if result.success:
        ...     result.data["figure"].show()
    """

    # Keywords indicating visualization is needed
    VIZ_INDICATORS: List[str] = [
        "chart",
        "plot",
        "graph",
        "visualize",
        "visualization",
        "show me",
        "display",
        "bar chart",
        "pie chart",
        "line chart",
        "histogram",
        "scatter",
        "heatmap",
        "treemap",
    ]

    # Chart type keywords
    CHART_TYPE_KEYWORDS: Dict[str, str] = {
        "bar chart": "bar",
        "bar graph": "bar",
        "bar": "bar",
        "column": "bar",
        "pie chart": "pie",
        "pie": "pie",
        "donut": "pie",
        "line chart": "line",
        "line graph": "line",
        "line": "line",
        "trend": "line",
        "time series": "line",
        "over time": "line",
        "timeline": "line",
        "scatter": "scatter",
        "histogram": "histogram",
        "distribution": "histogram",
        "heatmap": "heatmap",
        "heat map": "heatmap",
        "tree": "treemap",
        "treemap": "treemap",
    }

    # Temporal indicators that suggest line chart
    TEMPORAL_INDICATORS: List[str] = [
        "when", "timeline", "created", "modified", "date",
        "time", "history", "progression", "growth",
        "monthly", "weekly", "daily", "yearly", "annual",
    ]

    # Comparison indicators that suggest bar chart
    COMPARISON_INDICATORS: List[str] = [
        "most", "top", "highest", "lowest", "ranking",
        "compare", "comparison", "by user", "by author",
        "per project", "per team", "by project", "by team",
    ]

    def __init__(
        self,
        chart_generator: "ChartGenerator",
        iliad_client: Optional["IliadClient"] = None,
        default_chart_type: str = "bar",
    ) -> None:
        """Initialize Plotting agent.

        Args:
            chart_generator: Configured ChartGenerator instance
            iliad_client: Optional Iliad client for complex chart generation
            default_chart_type: Default chart type when not specified
        """
        super().__init__(
            name="plotting_agent",
            description="Generate charts and visualizations from data",
            iliad_client=iliad_client,
        )
        self.chart_generator = chart_generator
        self.default_chart_type = default_chart_type

        logger.info("Initialized PlottingAgent with visualization capabilities")

    def execute(
        self,
        query: str,
        context: AgentContext,
    ) -> AgentResult:
        """Generate visualization from context data.

        Looks for data in context.intermediate_results (typically from
        a previous database query) and generates a chart.

        Args:
            query: The visualization request
            context: Shared execution context with data

        Returns:
            AgentResult with figure, HTML, and code

        Example:
            >>> context.intermediate_results["chart_data"] = {"A": 10, "B": 5}
            >>> result = agent.execute("Create a bar chart", context)
            >>> if result.success:
            ...     st.plotly_chart(result.data["figure"])
        """
        logger.info(f"PlottingAgent executing: {query[:80]}...")

        try:
            # Step 1: Get data from context
            data = self._get_chart_data(context)

            if data is None:
                logger.warning("No chart data available in context")
                return AgentResult(
                    status=AgentStatus.FAILED,
                    reasoning="No data available for visualization. Run a database query first.",
                    metadata={"available_keys": list(context.intermediate_results.keys())},
                )

            # Step 2: Determine chart type
            chart_type = self._detect_chart_type(query)

            logger.info(f"Generating {chart_type} chart with {self._count_data_points(data)} points")

            # Step 3: Generate chart
            # Use LLM generation for complex requests, quick chart for simple ones
            if self._is_complex_request(query):
                result = self.chart_generator.generate(
                    request=query,
                    data=data,
                    chart_type=chart_type,
                )
            else:
                result = self.chart_generator.generate_quick_chart(
                    data=data,
                    chart_type=chart_type,
                    title=self._generate_title(query, context),
                )

            # Record execution
            context.record_execution(self.name, query)

            if result["success"]:
                logger.info("PlottingAgent completed: chart generated successfully")

                return AgentResult(
                    status=AgentStatus.SUCCESS,
                    data={
                        "figure": result["figure"],
                        "html": result.get("html"),
                        "code": result.get("code"),
                        "chart_type": chart_type,
                    },
                    confidence=0.85,
                    reasoning=f"Generated {chart_type} chart successfully",
                    metadata={
                        "chart_type": chart_type,
                        "data_points": self._count_data_points(data),
                    },
                )
            else:
                error_msg = result.get("error", "Chart generation failed")
                logger.warning(f"PlottingAgent failed: {error_msg}")

                return AgentResult(
                    status=AgentStatus.FAILED,
                    reasoning=error_msg,
                    metadata={
                        "error": error_msg,
                        "chart_type": chart_type,
                    },
                )

        except Exception as e:
            logger.error(f"PlottingAgent execution failed: {e}")
            return AgentResult(
                status=AgentStatus.FAILED,
                reasoning=str(e),
                metadata={"error": str(e)},
            )

    def can_handle(self, query: str, context: AgentContext) -> float:
        """Determine if this agent can handle the query.

        Evaluates the query for visualization keywords and checks
        if chart data is available in context.

        Args:
            query: The query to evaluate
            context: Execution context

        Returns:
            Confidence score (0.0 - 1.0)

        Example:
            >>> context.intermediate_results["chart_data"] = data
            >>> score = agent.can_handle("Show me a bar chart", context)
            >>> print(f"Confidence: {score:.2f}")  # ~0.9
        """
        query_lower = query.lower()

        # Base score
        score = 0.1

        # Check for visualization indicators
        for indicator in self.VIZ_INDICATORS:
            if indicator in query_lower:
                score += 0.2

        # Check for specific chart type mentions
        for keyword in self.CHART_TYPE_KEYWORDS:
            if keyword in query_lower:
                score += 0.1

        # Critical: Must have chart data available
        if self._get_chart_data(context) is not None:
            score += 0.3
        else:
            # Without data, significantly reduce score
            score *= 0.3

        return max(0.0, min(score, 1.0))

    def _get_chart_data(self, context: AgentContext) -> Optional[Any]:
        """Get chart data from context.

        Looks for data under various common keys.

        Args:
            context: Execution context

        Returns:
            Data for charting or None
        """
        # Priority order for data keys
        data_keys = [
            "chart_data",
            "db_result",
            "database_result",
            "query_result",
            "data",
            "answer",
        ]

        for key in data_keys:
            if context.has_result(key):
                data = context.get_result(key)
                # Validate data is chartable
                if self._is_chartable(data):
                    return data

        # Also check for any dict/list result that could be charted
        for key, value in context.intermediate_results.items():
            if self._is_chartable(value):
                return value

        return None

    def _is_chartable(self, data: Any) -> bool:
        """Check if data can be charted.

        Args:
            data: Data to check

        Returns:
            True if data can be visualized
        """
        if data is None:
            return False

        if isinstance(data, dict):
            # Need at least one value
            return len(data) > 0

        if isinstance(data, list):
            if len(data) == 0:
                return False
            # List of dicts or primitives
            return True

        return False

    def _detect_chart_type(self, query: str) -> str:
        """Detect chart type from query.

        Args:
            query: User's visualization request

        Returns:
            Chart type string
        """
        query_lower = query.lower()

        # Check explicit chart type keywords first
        for keyword, chart_type in self.CHART_TYPE_KEYWORDS.items():
            if keyword in query_lower:
                return chart_type

        # Check temporal indicators (suggests line chart)
        if any(ind in query_lower for ind in self.TEMPORAL_INDICATORS):
            return "line"

        # Check comparison indicators (suggests bar chart)
        if any(ind in query_lower for ind in self.COMPARISON_INDICATORS):
            return "bar"

        return self.default_chart_type

    def _is_complex_request(self, query: str) -> bool:
        """Check if request requires LLM generation.

        Args:
            query: Visualization request

        Returns:
            True if LLM generation needed
        """
        complex_indicators = [
            "with labels",
            "color by",
            "grouped",
            "stacked",
            "sorted by",
            "top",
            "bottom",
            "filter",
            "exclude",
            "compared",
            "correlation",
            "trend",
            "custom",
        ]

        query_lower = query.lower()
        return any(ind in query_lower for ind in complex_indicators)

    def _generate_title(self, query: str, context: AgentContext) -> str:
        """Generate chart title from query and context.

        Args:
            query: Visualization request
            context: Execution context

        Returns:
            Chart title
        """
        # Try to extract title from query
        title_prefixes = ["show", "chart of", "graph of", "plot of", "visualize"]
        query_lower = query.lower()

        for prefix in title_prefixes:
            if prefix in query_lower:
                idx = query_lower.index(prefix)
                title = query[idx + len(prefix):].strip()
                if title:
                    return title.capitalize()

        # Use original query as title
        return context.original_query[:50]

    def _count_data_points(self, data: Any) -> int:
        """Count data points.

        Args:
            data: Chart data

        Returns:
            Number of data points
        """
        if isinstance(data, dict):
            return len(data)
        if isinstance(data, list):
            return len(data)
        return 1
