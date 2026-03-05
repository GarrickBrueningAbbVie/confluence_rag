"""
Abstract base agent and core data structures for the agent framework.

This module defines the foundational classes that all agents inherit from,
as well as the shared context and result structures used throughout the
agent execution pipeline.

Classes:
    AgentStatus: Enum representing agent execution states
    AgentContext: Shared context passed between agents during execution
    AgentResult: Standardized result from agent execution
    BaseAgent: Abstract base class for all agents

Example:
    >>> from agents.base import BaseAgent, AgentContext, AgentResult, AgentStatus
    >>>
    >>> class CustomAgent(BaseAgent):
    ...     def execute(self, query: str, context: AgentContext) -> AgentResult:
    ...         # Custom implementation
    ...         return AgentResult(
    ...             status=AgentStatus.SUCCESS,
    ...             data={"answer": "Custom response"},
    ...         )
    ...
    ...     def can_handle(self, query: str, context: AgentContext) -> float:
    ...         return 0.5  # 50% confidence
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from loguru import logger


class AgentStatus(Enum):
    """Status of agent execution.

    Attributes:
        PENDING: Agent has not started execution
        RUNNING: Agent is currently executing
        SUCCESS: Agent completed successfully
        NEEDS_REFINEMENT: Agent completed but result needs follow-up
        FAILED: Agent execution failed
    """

    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    NEEDS_REFINEMENT = "needs_refinement"
    FAILED = "failed"


@dataclass
class AgentContext:
    """Shared context passed between agents during execution.

    This context allows agents to share intermediate results and maintain
    state across a multi-step query execution. The orchestrator manages
    this context and passes it to each agent.

    Attributes:
        original_query: The user's original query string
        intermediate_results: Dict storing results from previous agent executions,
            keyed by result name (e.g., "project_summary", "chart_data")
        metadata: Additional context information (routing mode, user preferences)
        iteration: Current iteration count for feedback loops
        max_iterations: Maximum allowed iterations before stopping refinement
        execution_history: List of (agent_name, query) tuples showing execution path

    Example:
        >>> context = AgentContext(
        ...     original_query="What projects are similar to ALFA?",
        ...     max_iterations=3,
        ... )
        >>> # After first agent executes:
        >>> context.intermediate_results["project_summary"] = "ALFA is a ML project..."
        >>> context.iteration += 1
    """

    original_query: str
    intermediate_results: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    iteration: int = 0
    max_iterations: int = 3
    execution_history: List[tuple] = field(default_factory=list)

    def store_result(self, key: str, value: Any) -> None:
        """Store a result in intermediate_results.

        Args:
            key: Key to store the result under
            value: Result value to store
        """
        self.intermediate_results[key] = value
        logger.debug(f"Stored result '{key}' in context")

    def get_result(self, key: str, default: Any = None) -> Any:
        """Retrieve a result from intermediate_results.

        Args:
            key: Key to retrieve
            default: Default value if key not found

        Returns:
            The stored result or default value
        """
        return self.intermediate_results.get(key, default)

    def has_result(self, key: str) -> bool:
        """Check if a result exists in intermediate_results.

        Args:
            key: Key to check

        Returns:
            True if key exists in intermediate_results
        """
        return key in self.intermediate_results

    def can_iterate(self) -> bool:
        """Check if more iterations are allowed.

        Returns:
            True if iteration < max_iterations
        """
        return self.iteration < self.max_iterations

    def record_execution(self, agent_name: str, query: str) -> None:
        """Record an agent execution in history.

        Args:
            agent_name: Name of the agent that executed
            query: Query that was executed
        """
        self.execution_history.append((agent_name, query))


@dataclass
class AgentResult:
    """Result from agent execution.

    Standardized result structure that all agents return. Includes
    the execution status, data, and optional follow-up information
    for feedback loops.

    Attributes:
        status: Execution status (SUCCESS, FAILED, NEEDS_REFINEMENT, etc.)
        data: Result data - structure varies by agent type:
            - RAGAgent: {"answer": str, "sources": list, "distances": list}
            - DatabaseAgent: {"answer": Any, "query": str}
            - PlottingAgent: {"figure": PlotlyFigure, "html": str}
        needs_followup: Whether a follow-up query is needed
        followup_query: The follow-up query if needs_followup is True
        confidence: Confidence score (0.0 - 1.0) in the result
        reasoning: Human-readable explanation of the result
        metadata: Additional metadata (execution time, debug info)

    Example:
        >>> result = AgentResult(
        ...     status=AgentStatus.SUCCESS,
        ...     data={"answer": "ALFA is a machine learning project..."},
        ...     confidence=0.85,
        ...     reasoning="Retrieved 5 relevant documents",
        ... )
    """

    status: AgentStatus
    data: Any = None
    needs_followup: bool = False
    followup_query: Optional[str] = None
    confidence: float = 1.0
    reasoning: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def success(self) -> bool:
        """Check if execution was successful.

        Returns:
            True if status is SUCCESS
        """
        return self.status == AgentStatus.SUCCESS

    @property
    def failed(self) -> bool:
        """Check if execution failed.

        Returns:
            True if status is FAILED
        """
        return self.status == AgentStatus.FAILED

    def get_answer(self, default: str = "") -> str:
        """Extract answer from data if available.

        Args:
            default: Default value if no answer found

        Returns:
            Answer string or default
        """
        if isinstance(self.data, dict):
            return self.data.get("answer", default)
        if isinstance(self.data, str):
            return self.data
        return default


class BaseAgent(ABC):
    """Abstract base class for all agents in the framework.

    Each agent implements a specific capability (RAG, Database, Plotting, etc.)
    and can be orchestrated by the AgentOrchestrator for complex queries.

    Agents should:
    - Implement execute() to perform their core function
    - Implement can_handle() to indicate confidence in handling a query
    - Optionally override validate_result() for custom validation

    Attributes:
        name: Unique identifier for the agent (e.g., "rag_agent")
        description: Human-readable description of agent capabilities
        iliad_client: Optional Iliad client for LLM operations

    Example:
        >>> class MyAgent(BaseAgent):
        ...     def __init__(self):
        ...         super().__init__(
        ...             name="my_agent",
        ...             description="Does something specific",
        ...         )
        ...
        ...     def execute(self, query, context):
        ...         return AgentResult(status=AgentStatus.SUCCESS, data="result")
        ...
        ...     def can_handle(self, query, context):
        ...         return 0.5 if "specific" in query else 0.1
    """

    def __init__(
        self,
        name: str,
        description: str,
        iliad_client: Optional[Any] = None,
    ) -> None:
        """Initialize base agent.

        Args:
            name: Unique identifier for the agent
            description: Human-readable description of capabilities
            iliad_client: Optional Iliad client for LLM operations
        """
        self.name = name
        self.description = description
        self.iliad_client = iliad_client

        logger.info(f"Initialized agent: {name}")

    @abstractmethod
    def execute(
        self,
        query: str,
        context: AgentContext,
    ) -> AgentResult:
        """Execute the agent's primary function.

        This method should perform the agent's core task and return
        a standardized AgentResult. The context provides access to
        intermediate results from previous executions.

        Args:
            query: The query to process
            context: Shared execution context with intermediate results

        Returns:
            AgentResult with execution outcome

        Raises:
            NotImplementedError: If not implemented by subclass
        """
        pass

    @abstractmethod
    def can_handle(self, query: str, context: AgentContext) -> float:
        """Determine if this agent can handle the query.

        Returns a confidence score indicating how well-suited this
        agent is for the given query. The orchestrator uses these
        scores to select agents.

        Args:
            query: The query to evaluate
            context: Execution context

        Returns:
            Confidence score between 0.0 (cannot handle) and 1.0 (ideal match)

        Raises:
            NotImplementedError: If not implemented by subclass
        """
        pass

    def validate_result(
        self,
        result: AgentResult,
        context: AgentContext,
    ) -> bool:
        """Validate if result is satisfactory.

        Override in subclasses for custom validation logic. The default
        implementation considers a result valid if it succeeded and
        doesn't need follow-up.

        Args:
            result: The execution result to validate
            context: Execution context

        Returns:
            True if result is valid and complete, False if refinement needed
        """
        return result.success and not result.needs_followup

    def __repr__(self) -> str:
        """String representation of the agent.

        Returns:
            String with agent name and description
        """
        return f"{self.__class__.__name__}(name='{self.name}')"
