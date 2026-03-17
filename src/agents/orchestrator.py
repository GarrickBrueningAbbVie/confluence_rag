"""
Agent orchestrator for coordinating multi-agent execution.

This module provides the AgentOrchestrator class which manages agent selection,
execution ordering, dependency handling, and result synthesis for complex
multi-step queries.

Example:
    >>> from agents.orchestrator import AgentOrchestrator
    >>> from agents import RAGAgent, DatabaseAgent
    >>>
    >>> orchestrator = AgentOrchestrator(
    ...     agents=[rag_agent, db_agent],
    ...     iliad_client=iliad_client,
    ... )
    >>> result = orchestrator.execute("What projects are similar to ALFA?")
    >>> print(result.final_answer)
"""

import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

from loguru import logger

from agents.base import BaseAgent, AgentContext, AgentResult, AgentStatus
from agents.feedback_controller import FeedbackController

# Type hints for optional imports
try:
    from iliad.client import IliadClient
    from agents.iterative_agent import IterativeDescribeAgent
except ImportError:
    pass


# Patterns for detecting list+describe queries
LIST_DESCRIBE_PATTERNS = [
    "list all",
    "list the",
    "what projects",
    "which projects",
    "show all",
    "get all",
    "find all",
]

DESCRIBE_PATTERNS = [
    "describe all",
    "describe each",
    "describe these",
    "describe them",
    "explain all",
    "explain each",
    "and describe",
    "and explain",
    "then describe",
    "then explain",
]


@dataclass
class ExecutionStep:
    """A step in the execution plan.

    Represents a single agent execution with its query, dependencies,
    and result storage configuration.

    Attributes:
        agent: Agent to execute
        query: Query for the agent (may contain {placeholders})
        depends_on: List of step indices this depends on
        store_as: Key to store result in context
        priority: Execution priority (lower = higher priority)

    Example:
        >>> step = ExecutionStep(
        ...     agent=rag_agent,
        ...     query="Summarize ALFA project",
        ...     depends_on=[],
        ...     store_as="project_summary",
        ... )
    """

    agent: BaseAgent
    query: str
    depends_on: List[int] = field(default_factory=list)
    store_as: Optional[str] = None
    priority: int = 0


@dataclass
class OrchestrationResult:
    """Result from orchestrated multi-agent execution.

    Contains the final synthesized answer along with metadata about
    the execution process.

    Attributes:
        success: Overall success status
        final_answer: Synthesized final answer from all agents
        steps_executed: List of execution details for each step
        context: Final execution context with all intermediate results
        execution_time: Total execution time in seconds
        metadata: Additional metadata (agent counts, error info)

    Example:
        >>> result = orchestrator.execute(query)
        >>> if result.success:
        ...     print(result.final_answer)
        ...     for step in result.steps_executed:
        ...         print(f"{step['agent']}: {step['result'].status}")
    """

    success: bool
    final_answer: str
    steps_executed: List[Dict[str, Any]] = field(default_factory=list)
    context: Optional[AgentContext] = None
    execution_time: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


class AgentOrchestrator:
    """Orchestrates multi-agent execution with feedback loops.

    Coordinates agent selection, creates execution plans with dependencies,
    handles parallel/sequential execution, manages feedback loops, and
    synthesizes results from multiple agents.

    The orchestrator supports two planning modes:
    1. LLM-based planning: Uses Iliad to decompose queries and select agents
    2. Score-based planning: Uses agent can_handle() scores

    Attributes:
        agents: Dict of available agents keyed by name
        iliad_client: Optional Iliad client for planning and synthesis
        max_workers: Maximum parallel workers
        feedback_controller: Controller for feedback loops

    Example:
        >>> orchestrator = AgentOrchestrator(
        ...     agents=[rag_agent, db_agent, plotting_agent],
        ...     iliad_client=iliad_client,
        ...     max_iterations=3,
        ... )
        >>> result = orchestrator.execute("What projects are similar to ALFA?")
        >>> print(result.final_answer)

    Multi-Step Query Flow:
        1. Query "What projects are similar to ALFA?"
        2. Orchestrator creates plan:
           - Step 1: RAGAgent("Summarize ALFA") -> store as "project_summary"
           - Step 2: RAGAgent("Find similar to {project_summary}") -> depends on step 1
        3. Executes steps respecting dependencies
        4. Synthesizes final answer
    """

    # Prompt for LLM-based execution planning
    PLANNING_PROMPT = """Create an execution plan for this query using available agents.

Query: {query}

Available Agents:
{agent_descriptions}

For multi-step queries that require information from one step to inform the next:
- Use "depends_on" to specify which steps must complete first
- Use "store_as" to save results for later steps
- Use {{placeholder}} syntax in queries to reference stored results

Examples of multi-step patterns:
1. "What projects are similar to ALFA?"
   - Step 1: Get info about ALFA (store as "project_summary")
   - Step 2: Find similar projects using {{project_summary}}

2. "Who works on Python projects and how many pages have they created?"
   - Step 1: List Python project authors (store as "authors")
   - Step 2: Count pages by authors

Return ONLY a JSON array of steps:
[
    {{
        "agent": "agent_name",
        "query": "specific query (may include {{placeholder}})",
        "depends_on": [],
        "store_as": "optional_key"
    }}
]

No explanations, only JSON."""

    def __init__(
        self,
        agents: List[BaseAgent],
        iliad_client: Optional["IliadClient"] = None,
        max_workers: int = 4,
        max_iterations: int = 3,
        planning_model: str = "gpt-4o-mini-global",
        synthesis_model: str = "gpt-4o-mini-global",
    ) -> None:
        """Initialize orchestrator.

        Args:
            agents: List of available agents
            iliad_client: Iliad client for planning and synthesis
            max_workers: Max parallel workers for execution
            max_iterations: Max feedback iterations per agent
            planning_model: Model to use for execution planning
            synthesis_model: Model to use for result synthesis
        """
        self.agents = {agent.name: agent for agent in agents}
        self.iliad_client = iliad_client
        self.max_workers = max_workers
        self.planning_model = planning_model
        self.synthesis_model = synthesis_model

        self.feedback_controller = FeedbackController(
            iliad_client=iliad_client,
            max_iterations=max_iterations,
        )

        logger.info(
            f"Initialized AgentOrchestrator with {len(agents)} agents: "
            f"{list(self.agents.keys())}"
        )

    def execute(
        self,
        query: str,
        force_agents: Optional[List[str]] = None,
        force_plan: Optional[List[ExecutionStep]] = None,
    ) -> OrchestrationResult:
        """Execute query using appropriate agents.

        Main entry point for the orchestrator. Analyzes the query,
        creates an execution plan, executes with feedback loops,
        and synthesizes the final answer.

        Args:
            query: User's natural language query
            force_agents: Optional list of agent names to use
            force_plan: Optional predefined execution plan

        Returns:
            OrchestrationResult with final answer and metadata

        Example:
            >>> result = orchestrator.execute("What is ALFA and how many pages mention it?")
            >>> print(result.final_answer)
            >>> print(f"Executed {len(result.steps_executed)} steps")
        """
        start_time = time.time()

        logger.info(f"Orchestrating query: {query[:100]}...")

        # Reset feedback controller for new query
        self.feedback_controller.reset()

        # Initialize context
        context = AgentContext(
            original_query=query,
            max_iterations=self.feedback_controller.max_iterations,
        )

        try:
            # Check for list+describe pattern first (highest priority)
            if not force_plan and self._is_list_describe_query(query):
                logger.info("Detected list+describe pattern, using IterativeDescribeAgent")
                return self._execute_iterative_describe(query, context, start_time)

            # Step 1: Create execution plan
            if force_plan:
                plan = force_plan
            else:
                plan = self._create_execution_plan(query, context, force_agents)

            if not plan:
                return OrchestrationResult(
                    success=False,
                    final_answer="Unable to create execution plan for this query.",
                    execution_time=time.time() - start_time,
                )

            logger.info(f"Created execution plan with {len(plan)} steps")
            for i, step in enumerate(plan):
                logger.debug(f"  Step {i}: {step.agent.name} - {step.query[:50]}...")

            # Step 2: Execute plan with feedback loops
            results = self._execute_plan(plan, context)

            # Step 3: Synthesize final answer
            final_answer = self._synthesize_results(query, results, context)

            execution_time = time.time() - start_time

            logger.info(f"Orchestration completed in {execution_time:.2f}s")

            return OrchestrationResult(
                success=True,
                final_answer=final_answer,
                steps_executed=results,
                context=context,
                execution_time=execution_time,
                metadata={
                    "num_steps": len(plan),
                    "agents_used": [s.agent.name for s in plan],
                    "iterations": context.iteration,
                },
            )

        except Exception as e:
            logger.error(f"Orchestration failed: {e}")
            execution_time = time.time() - start_time

            return OrchestrationResult(
                success=False,
                final_answer=f"Error processing query: {str(e)}",
                execution_time=execution_time,
                metadata={"error": str(e)},
            )

    def _create_execution_plan(
        self,
        query: str,
        context: AgentContext,
        force_agents: Optional[List[str]] = None,
    ) -> List[ExecutionStep]:
        """Create execution plan for query.

        Uses LLM to decompose complex queries into steps, or falls
        back to agent scoring for simple queries.

        Args:
            query: User query
            context: Execution context
            force_agents: Optional agent names to force

        Returns:
            List of ExecutionStep objects
        """
        if force_agents:
            # Create simple plan with forced agents
            return [
                ExecutionStep(
                    agent=self.agents[name],
                    query=query,
                    store_as=f"{name}_result",
                )
                for name in force_agents
                if name in self.agents
            ]

        # Try LLM-based planning for complex queries
        if self.iliad_client and self._is_complex_query(query):
            plan = self._llm_create_plan(query, context)
            if plan:
                return plan

        # Fallback to score-based planning
        return self._score_based_plan(query, context)

    def _is_complex_query(self, query: str) -> bool:
        """Check if query likely needs multi-step processing.

        Args:
            query: User query

        Returns:
            True if query appears complex
        """
        complex_indicators = [
            " and ",
            " then ",
            " also ",
            " similar to ",
            " like ",
            " compare ",
            " correlation ",
            " relationship ",
            "as well as",
            "followed by",
        ]

        query_lower = query.lower()
        return any(ind in query_lower for ind in complex_indicators)

    def _llm_create_plan(
        self,
        query: str,
        context: AgentContext,
    ) -> Optional[List[ExecutionStep]]:
        """Use LLM to create execution plan.

        Args:
            query: User query
            context: Execution context

        Returns:
            List of ExecutionStep or None if planning fails
        """
        agent_descriptions = "\n".join(
            f"- {name}: {agent.description}" for name, agent in self.agents.items()
        )

        prompt = self.PLANNING_PROMPT.format(
            query=query,
            agent_descriptions=agent_descriptions,
        )

        try:
            messages = [{"role": "user", "content": prompt}]
            response = self.iliad_client.chat(
                messages=messages, model=self.planning_model
            )
            content = self.iliad_client.extract_content(response)

            # Parse JSON response
            content = content.strip()
            if content.startswith("```"):
                content = content.split("\n", 1)[1].rsplit("```", 1)[0]

            steps_data = json.loads(content)

            # Convert to ExecutionStep objects
            plan = []
            for i, step_data in enumerate(steps_data):
                agent_name = step_data.get("agent")
                if agent_name not in self.agents:
                    logger.warning(f"Unknown agent '{agent_name}' in plan, skipping")
                    continue

                plan.append(
                    ExecutionStep(
                        agent=self.agents[agent_name],
                        query=step_data.get("query", query),
                        depends_on=step_data.get("depends_on", []),
                        store_as=step_data.get("store_as"),
                        priority=i,
                    )
                )

            if plan:
                logger.info(f"LLM created {len(plan)}-step execution plan")
                return plan

        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse LLM plan JSON: {e}")
        except Exception as e:
            logger.warning(f"LLM planning failed: {e}")

        return None

    def _score_based_plan(
        self,
        query: str,
        context: AgentContext,
    ) -> List[ExecutionStep]:
        """Create plan based on agent scoring.

        Args:
            query: User query
            context: Execution context

        Returns:
            List of ExecutionStep objects
        """
        # Score each agent
        scores = [
            (name, agent.can_handle(query, context))
            for name, agent in self.agents.items()
        ]

        # Sort by score descending
        scores.sort(key=lambda x: x[1], reverse=True)

        # Use agents above threshold
        plan = []
        for name, score in scores:
            if score > 0.4:
                plan.append(
                    ExecutionStep(
                        agent=self.agents[name],
                        query=query,
                        store_as=f"{name}_result",
                    )
                )

        # Default to RAG if no good match
        if not plan and "rag_agent" in self.agents:
            plan.append(
                ExecutionStep(
                    agent=self.agents["rag_agent"],
                    query=query,
                    store_as="rag_result",
                )
            )

        logger.info(f"Score-based plan: {[s.agent.name for s in plan]}")
        return plan

    def _execute_plan(
        self,
        plan: List[ExecutionStep],
        context: AgentContext,
    ) -> List[Dict[str, Any]]:
        """Execute plan with dependency handling and feedback loops.

        Executes steps in dependency order, with independent steps
        running in parallel.

        Args:
            plan: List of ExecutionStep objects
            context: Execution context

        Returns:
            List of execution result dicts
        """
        results = []
        completed: Set[int] = set()

        while len(completed) < len(plan):
            # Find steps ready to execute (dependencies satisfied)
            ready = [
                (i, step)
                for i, step in enumerate(plan)
                if i not in completed
                and all(d in completed for d in step.depends_on)
            ]

            if not ready:
                logger.error("Circular dependency detected in execution plan")
                break

            logger.debug(f"Executing {len(ready)} ready step(s)")

            # Execute ready steps
            if len(ready) == 1:
                # Single step - execute directly
                idx, step = ready[0]
                result = self._execute_step_with_feedback(step, context)
                results.append(result)
                completed.add(idx)
            else:
                # Multiple steps - execute in parallel
                with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                    futures = {
                        executor.submit(
                            self._execute_step_with_feedback, step, context
                        ): idx
                        for idx, step in ready
                    }

                    for future in as_completed(futures):
                        idx = futures[future]
                        try:
                            result = future.result()
                            results.append(result)
                        except Exception as e:
                            logger.error(f"Step {idx} failed: {e}")
                            results.append(
                                {
                                    "step_index": idx,
                                    "agent": plan[idx].agent.name,
                                    "success": False,
                                    "error": str(e),
                                }
                            )
                        completed.add(idx)

        return results

    def _execute_step_with_feedback(
        self,
        step: ExecutionStep,
        context: AgentContext,
    ) -> Dict[str, Any]:
        """Execute single step with feedback loop.

        Injects context into query, executes agent, evaluates result,
        and optionally triggers refinement.

        Args:
            step: ExecutionStep to execute
            context: Execution context

        Returns:
            Dict with execution details
        """
        logger.info(f"Executing step: {step.agent.name}")

        # Inject context into query
        query = self.feedback_controller.inject_context(
            step.query, {"intermediate_results": context.intermediate_results}
        )

        if query != step.query:
            logger.debug(f"Injected context: {query[:80]}...")

        # Execute agent
        result = step.agent.execute(query, context)

        # Store result if specified
        if step.store_as and result.data:
            answer = result.get_answer() or result.data
            context.store_result(step.store_as, answer)

        # Check for feedback loop
        if result.needs_followup and context.can_iterate():
            context.iteration += 1
            logger.info(f"Refinement triggered, iteration {context.iteration}")

            # Execute follow-up
            followup_query = result.followup_query or query
            followup_result = step.agent.execute(followup_query, context)

            # Use better result
            if followup_result.success:
                result = followup_result
                # Update stored result
                if step.store_as and result.data:
                    answer = result.get_answer() or result.data
                    context.store_result(step.store_as, answer)

        return {
            "step_index": step.priority,
            "agent": step.agent.name,
            "query": query,
            "result": result,
            "stored_as": step.store_as,
        }

    def _synthesize_results(
        self,
        original_query: str,
        results: List[Dict[str, Any]],
        context: AgentContext,
    ) -> str:
        """Synthesize final answer from all results.

        Args:
            original_query: Original user query
            results: List of execution result dicts
            context: Execution context

        Returns:
            Synthesized answer string
        """
        # Filter successful results
        successful = [r for r in results if r.get("result") and r["result"].success]

        if not successful:
            return "Unable to find relevant information for your query."

        # Single result - return directly
        if len(successful) == 1:
            result = successful[0]["result"]
            return result.get_answer() or str(result.data)

        # Multiple results - use LLM synthesis
        if self.iliad_client:
            return self._llm_synthesize(original_query, successful, context)

        # Fallback: concatenate results
        return self._simple_synthesize(successful)

    def _llm_synthesize(
        self,
        original_query: str,
        results: List[Dict[str, Any]],
        context: AgentContext,
    ) -> str:
        """Use LLM to synthesize multiple results.

        Args:
            original_query: Original query
            results: Successful results
            context: Execution context

        Returns:
            Synthesized answer
        """
        results_text = []
        for i, r in enumerate(results, 1):
            result = r.get("result")
            agent_name = r.get("agent", "unknown")
            query_used = r.get("query", "")

            answer = result.get_answer() if result else "No answer"

            results_text.append(
                f"### Result {i} ({agent_name})\n"
                f"Query: {query_used[:100]}...\n"
                f"Answer: {answer[:500]}..."
            )

        prompt = f"""Synthesize these results into a coherent answer.

Original Question: {original_query}

Results:
{chr(10).join(results_text)}

Create a unified, well-structured answer that:
1. Directly addresses the original question
2. Integrates information from all results
3. Is clear and concise
4. Notes any contradictions or gaps

Your synthesized answer:"""

        try:
            messages = [{"role": "user", "content": prompt}]
            response = self.iliad_client.chat(
                messages=messages, model=self.synthesis_model
            )
            return self.iliad_client.extract_content(response)
        except Exception as e:
            logger.error(f"LLM synthesis failed: {e}")
            return self._simple_synthesize(results)

    def _simple_synthesize(self, results: List[Dict[str, Any]]) -> str:
        """Simple concatenation of results.

        Args:
            results: Successful results

        Returns:
            Concatenated answer
        """
        parts = []
        for r in results:
            result = r.get("result")
            if result and result.success:
                answer = result.get_answer()
                if answer:
                    parts.append(answer)

        return "\n\n".join(parts) if parts else "No results available."

    def register_agent(self, agent: BaseAgent) -> None:
        """Register a new agent.

        Args:
            agent: Agent to register

        Example:
            >>> orchestrator.register_agent(JiraAgent(jira_client))
        """
        self.agents[agent.name] = agent
        logger.info(f"Registered agent: {agent.name}")

    def get_available_agents(self) -> List[str]:
        """Get list of available agent names.

        Returns:
            List of agent names
        """
        return list(self.agents.keys())

    def _is_list_describe_query(self, query: str) -> bool:
        """Check if query matches list+describe pattern.

        These queries need the IterativeDescribeAgent to first get a list
        from the database, then describe each item via RAG.

        Args:
            query: User query to check

        Returns:
            True if this is a list+describe pattern
        """
        query_lower = query.lower()

        # Check for both list AND describe keywords
        has_list = any(pattern in query_lower for pattern in LIST_DESCRIBE_PATTERNS)
        has_describe = any(pattern in query_lower for pattern in DESCRIBE_PATTERNS)

        if has_list and has_describe:
            logger.debug(f"Matched list+describe pattern: list={has_list}, describe={has_describe}")
            return True

        return False

    def _execute_iterative_describe(
        self,
        query: str,
        context: AgentContext,
        start_time: float,
    ) -> OrchestrationResult:
        """Execute query using IterativeDescribeAgent.

        Creates an IterativeDescribeAgent dynamically with the RAG and
        Database agents, then executes the list+describe query.

        Args:
            query: User query
            context: Execution context
            start_time: Start time for timing

        Returns:
            OrchestrationResult with combined descriptions
        """
        # Get RAG and Database agents
        rag_agent = self.agents.get("rag_agent")
        db_agent = self.agents.get("database_agent")

        if not rag_agent or not db_agent:
            logger.warning("Missing rag_agent or database_agent for iterative describe")
            # Fallback to normal execution
            plan = self._score_based_plan(query, context)
            results = self._execute_plan(plan, context)
            final_answer = self._synthesize_results(query, results, context)
            return OrchestrationResult(
                success=True,
                final_answer=final_answer,
                steps_executed=results,
                context=context,
                execution_time=time.time() - start_time,
            )

        # Create IterativeDescribeAgent
        iterative_agent = IterativeDescribeAgent(
            rag_agent=rag_agent,
            db_agent=db_agent,
            iliad_client=self.iliad_client,
            max_items_to_describe=10,
        )

        # Execute iterative query
        result = iterative_agent.execute(query, context)

        execution_time = time.time() - start_time

        if result.success:
            logger.info(
                f"IterativeDescribeAgent completed: "
                f"{result.data.get('items_found', 0)} items found, "
                f"{result.data.get('items_described', 0)} described"
            )

            return OrchestrationResult(
                success=True,
                final_answer=result.data.get("answer", str(result.data)),
                steps_executed=[
                    {
                        "step_index": 0,
                        "agent": "iterative_describe_agent",
                        "query": query,
                        "result": result,
                        "stored_as": "iterative_result",
                    }
                ],
                context=context,
                execution_time=execution_time,
                metadata={
                    "pattern": "list_describe",
                    "items_found": result.data.get("items_found", 0),
                    "items_described": result.data.get("items_described", 0),
                    "sources": result.data.get("sources", []),
                },
            )
        else:
            logger.warning(f"IterativeDescribeAgent failed: {result.reasoning}")
            # Fallback to normal execution
            plan = self._score_based_plan(query, context)
            results = self._execute_plan(plan, context)
            final_answer = self._synthesize_results(query, results, context)

            return OrchestrationResult(
                success=bool(final_answer),
                final_answer=final_answer or "Unable to process query.",
                steps_executed=results,
                context=context,
                execution_time=time.time() - start_time,
                metadata={"fallback": True, "original_error": result.reasoning},
            )
