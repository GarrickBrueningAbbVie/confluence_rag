"""
Agent framework for the Confluence RAG system.

This package provides an extensible agent-based architecture for executing
complex, multi-step queries with feedback loops and parallel execution.

Core Components:
    - BaseAgent: Abstract base class for all agents
    - AgentContext: Shared context passed between agents
    - AgentResult: Standardized result from agent execution
    - AgentOrchestrator: Coordinates multi-agent execution
    - FeedbackController: Manages feedback loops and refinement

Available Agents:
    - RAGAgent: Semantic search and document retrieval
    - DatabaseAgent: Structured queries (counts, lists, aggregations)
    - PlottingAgent: Data visualization generation

Example:
    >>> from agents import AgentOrchestrator, RAGAgent, DatabaseAgent
    >>> from agents.base import AgentContext
    >>>
    >>> # Initialize agents
    >>> rag_agent = RAGAgent(rag_pipeline, iliad_client)
    >>> db_agent = DatabaseAgent(db_pipeline, iliad_client)
    >>>
    >>> # Create orchestrator
    >>> orchestrator = AgentOrchestrator(
    ...     agents=[rag_agent, db_agent],
    ...     iliad_client=iliad_client,
    ... )
    >>>
    >>> # Execute multi-step query
    >>> result = orchestrator.execute("What projects are similar to ALFA?")
    >>> print(result.final_answer)
"""

from agents.base import (
    AgentStatus,
    AgentContext,
    AgentResult,
    BaseAgent,
)
from agents.rag_agent import RAGAgent
from agents.database_agent import DatabaseAgent
from agents.plotting_agent import PlottingAgent
from agents.feedback_controller import FeedbackController, RefinementTrigger
from agents.orchestrator import (
    AgentOrchestrator,
    ExecutionStep,
    OrchestrationResult,
)

__all__ = [
    # Base classes
    "AgentStatus",
    "AgentContext",
    "AgentResult",
    "BaseAgent",
    # Agents
    "RAGAgent",
    "DatabaseAgent",
    "PlottingAgent",
    # Orchestration
    "AgentOrchestrator",
    "ExecutionStep",
    "OrchestrationResult",
    # Feedback
    "FeedbackController",
    "RefinementTrigger",
]
