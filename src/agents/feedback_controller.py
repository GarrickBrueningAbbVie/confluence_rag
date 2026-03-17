"""
Feedback loop controller for multi-step query refinement.

This module manages iterative query refinement based on intermediate results.
It evaluates execution results, determines if refinement is needed, and
generates follow-up queries when necessary.

Example:
    >>> from agents.feedback_controller import FeedbackController
    >>>
    >>> controller = FeedbackController(iliad_client, max_iterations=3)
    >>> needs_refinement, followup, reasoning = controller.evaluate(
    ...     result={"answer": "Short answer"},
    ...     original_query="What is ALFA?",
    ...     context={"intermediate_results": {}},
    ... )
    >>> if needs_refinement:
    ...     print(f"Follow-up needed: {followup}")
"""

import json
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from loguru import logger

# Type hints for optional imports
try:
    from iliad.client import IliadClient
except ImportError:
    pass


@dataclass
class RefinementTrigger:
    """Defines when and how to trigger query refinement.

    A trigger consists of a condition function that evaluates results,
    and a function to generate the follow-up query if the condition is met.

    Attributes:
        name: Human-readable name for the trigger
        condition: Function(result, query) -> bool that returns True if refinement needed
        generate_followup: Function(result, query) -> str that generates follow-up query
        max_triggers: Maximum times this trigger can fire per query
        triggered_count: Internal counter for trigger activations
        priority: Trigger priority (lower = higher priority)

    Example:
        >>> trigger = RefinementTrigger(
        ...     name="empty_result",
        ...     condition=lambda r, q: not r.get("answer"),
        ...     generate_followup=lambda r, q: f"Provide details about: {q}",
        ...     max_triggers=2,
        ... )
    """

    name: str
    condition: Callable[[Dict[str, Any], str], bool]
    generate_followup: Callable[[Dict[str, Any], str], str]
    max_triggers: int = 2
    triggered_count: int = field(default=0, repr=False)
    priority: int = 0

    def can_trigger(self) -> bool:
        """Check if trigger can still fire.

        Returns:
            True if triggered_count < max_triggers
        """
        return self.triggered_count < self.max_triggers

    def reset(self) -> None:
        """Reset trigger counter."""
        self.triggered_count = 0


class FeedbackController:
    """Controls feedback loops for multi-step queries.

    Evaluates intermediate results and determines if refinement
    or follow-up queries are needed. Supports both rule-based
    triggers and optional LLM-based evaluation.

    Attributes:
        iliad_client: Optional Iliad client for LLM-based evaluation
        max_iterations: Maximum feedback iterations allowed
        triggers: List of registered refinement triggers
        use_llm_evaluation: Whether to use LLM for sophisticated evaluation

    Example:
        >>> controller = FeedbackController(iliad_client, max_iterations=3)
        >>>
        >>> # Register custom trigger
        >>> controller.register_trigger(RefinementTrigger(
        ...     name="custom",
        ...     condition=lambda r, q: "error" in str(r).lower(),
        ...     generate_followup=lambda r, q: f"Retry: {q}",
        ... ))
        >>>
        >>> # Evaluate result
        >>> needs_refinement, followup, reason = controller.evaluate(
        ...     result, query, context
        ... )
    """

    def __init__(
        self,
        iliad_client: Optional["IliadClient"] = None,
        max_iterations: int = 3,
        use_llm_evaluation: bool = True,
    ) -> None:
        """Initialize feedback controller.

        Args:
            iliad_client: Optional Iliad client for LLM-based evaluation
            max_iterations: Maximum feedback iterations
            use_llm_evaluation: Whether to use LLM for evaluation
        """
        self.iliad_client = iliad_client
        self.max_iterations = max_iterations
        self.use_llm_evaluation = use_llm_evaluation and iliad_client is not None
        self.triggers: List[RefinementTrigger] = []
        self._iteration = 0

        # Register default triggers
        self._register_default_triggers()

        logger.info(
            f"Initialized FeedbackController (max_iter: {max_iterations}, "
            f"llm_eval: {self.use_llm_evaluation})"
        )

    def _register_default_triggers(self) -> None:
        """Register built-in refinement triggers."""

        # Trigger 1: Empty or very short results
        self.triggers.append(
            RefinementTrigger(
                name="empty_result",
                condition=lambda result, query: (
                    not result.get("answer")
                    or len(str(result.get("answer", ""))) < 30
                ),
                generate_followup=lambda result, query: (
                    f"Provide more detailed information about: {query}"
                ),
                priority=0,
            )
        )

        # Trigger 2: Low confidence retrieval (high distance scores)
        self.triggers.append(
            RefinementTrigger(
                name="low_confidence",
                condition=lambda result, query: (
                    result.get("distances")
                    and len(result.get("distances", [])) > 0
                    and all(d > 0.6 for d in result.get("distances", [])[:3])
                ),
                generate_followup=lambda result, query: (
                    f"Search for alternative terms or synonyms related to: {query}"
                ),
                priority=1,
            )
        )

        # Trigger 3: "Not found" type responses
        self.triggers.append(
            RefinementTrigger(
                name="not_found",
                condition=lambda result, query: (
                    any(
                        ind in str(result.get("answer", "")).lower()
                        for ind in [
                            "no information",
                            "not found",
                            "cannot find",
                            "no relevant",
                            "unable to find",
                            "no results",
                            "i don't have",
                        ]
                    )
                ),
                generate_followup=lambda result, query: (
                    f"Try a broader search for: {query}"
                ),
                priority=2,
            )
        )

        # Trigger 4: Database query returned empty
        self.triggers.append(
            RefinementTrigger(
                name="empty_db_result",
                condition=lambda result, query: (
                    isinstance(result.get("answer"), list)
                    and len(result.get("answer", [])) == 0
                ),
                generate_followup=lambda result, query: (
                    f"Relax filters and retry: {query}"
                ),
                priority=3,
            )
        )

    def register_trigger(self, trigger: RefinementTrigger) -> None:
        """Register a new refinement trigger.

        Args:
            trigger: RefinementTrigger to register

        Example:
            >>> controller.register_trigger(RefinementTrigger(
            ...     name="custom",
            ...     condition=my_condition_func,
            ...     generate_followup=my_followup_func,
            ... ))
        """
        self.triggers.append(trigger)
        # Sort by priority
        self.triggers.sort(key=lambda t: t.priority)
        logger.debug(f"Registered refinement trigger: {trigger.name}")

    def evaluate(
        self,
        result: Dict[str, Any],
        original_query: str,
        context: Dict[str, Any],
    ) -> Tuple[bool, Optional[str], str]:
        """Evaluate result and determine if refinement needed.

        Checks registered triggers in priority order, then optionally
        uses LLM for more sophisticated evaluation.

        Args:
            result: Current execution result (dict with 'answer', 'sources', etc.)
            original_query: The original query being refined
            context: Execution context with intermediate results

        Returns:
            Tuple of (needs_refinement, followup_query, reasoning)

        Example:
            >>> needs_ref, followup, reason = controller.evaluate(
            ...     {"answer": "Short"},
            ...     "What is ALFA?",
            ...     {"intermediate_results": {}},
            ... )
            >>> if needs_ref:
            ...     print(f"Follow-up: {followup}")
        """
        self._iteration += 1

        # Check iteration limit
        if self._iteration >= self.max_iterations:
            logger.info(f"Max iterations ({self.max_iterations}) reached")
            return False, None, "Maximum iterations reached"

        logger.debug(f"Evaluating result (iteration {self._iteration}/{self.max_iterations})")

        # Check registered triggers in priority order
        for trigger in self.triggers:
            if not trigger.can_trigger():
                continue

            try:
                if trigger.condition(result, original_query):
                    trigger.triggered_count += 1
                    followup = trigger.generate_followup(result, original_query)

                    logger.info(
                        f"Trigger '{trigger.name}' fired, generating follow-up: {followup[:50]}..."
                    )
                    return True, followup, f"Trigger '{trigger.name}' activated"
            except Exception as e:
                logger.warning(f"Trigger '{trigger.name}' evaluation failed: {e}")

        # Use LLM for sophisticated evaluation if enabled
        if self.use_llm_evaluation:
            return self._llm_evaluate(result, original_query, context)

        return False, None, "Result satisfactory"

    def _llm_evaluate(
        self,
        result: Dict[str, Any],
        original_query: str,
        context: Dict[str, Any],
    ) -> Tuple[bool, Optional[str], str]:
        """Use LLM to evaluate if refinement is needed.

        Args:
            result: Execution result
            original_query: Original query
            context: Execution context

        Returns:
            Tuple of (needs_refinement, followup_query, reasoning)
        """
        answer = str(result.get("answer", ""))[:1000]
        intermediate = str(context.get("intermediate_results", {}))[:500]

        prompt = f"""Evaluate if this answer adequately addresses the user's query.

Query: {original_query}

Answer: {answer}

Previous context: {intermediate}

Analyze:
1. Does the answer directly address what was asked?
2. Is the answer complete and informative?
3. Are there gaps that require follow-up?

Respond with JSON only:
{{
    "is_sufficient": true or false,
    "reasoning": "brief explanation",
    "followup_query": "optional follow-up query if needed, or null"
}}"""

        try:
            messages = [{"role": "user", "content": prompt}]
            response = self.iliad_client.chat(messages=messages)
            content = self.iliad_client.extract_content(response)

            # Parse JSON response
            content = content.strip()
            if content.startswith("```"):
                content = content.split("\n", 1)[1].rsplit("```", 1)[0]

            evaluation = json.loads(content)

            if not evaluation.get("is_sufficient", True):
                followup = evaluation.get("followup_query")
                reasoning = evaluation.get("reasoning", "LLM evaluation")
                logger.info(f"LLM evaluation: refinement needed - {reasoning}")
                return True, followup, f"LLM evaluation: {reasoning}"

            return False, None, "LLM evaluation: result sufficient"

        except Exception as e:
            logger.warning(f"LLM evaluation failed: {e}")
            return False, None, "LLM evaluation failed, accepting result"

    def inject_context(
        self,
        query_template: str,
        context: Dict[str, Any],
    ) -> str:
        """Inject intermediate results into query template.

        Replaces {placeholders} in the template with values from
        the context's intermediate_results.

        Args:
            query_template: Template with {placeholders}
            context: Dict with 'intermediate_results' key

        Returns:
            Query with placeholders filled

        Example:
            >>> template = "Find projects similar to: {project_summary}"
            >>> context = {"intermediate_results": {"project_summary": "ALFA is an ML project"}}
            >>> query = controller.inject_context(template, context)
            >>> print(query)
            "Find projects similar to: ALFA is an ML project"
        """
        result = query_template
        intermediate = context.get("intermediate_results", {})

        for key, value in intermediate.items():
            placeholder = "{" + key + "}"
            if placeholder in result:
                # Truncate long values
                value_str = str(value)
                if len(value_str) > 500:
                    value_str = value_str[:500] + "..."
                result = result.replace(placeholder, value_str)

        return result

    def reset(self) -> None:
        """Reset controller state for new query.

        Resets iteration counter and all trigger counters.
        """
        self._iteration = 0
        for trigger in self.triggers:
            trigger.reset()
        logger.debug("FeedbackController reset for new query")

    @property
    def iteration(self) -> int:
        """Current iteration count.

        Returns:
            Current iteration number
        """
        return self._iteration

    @property
    def can_iterate(self) -> bool:
        """Check if more iterations are allowed.

        Returns:
            True if iteration < max_iterations
        """
        return self._iteration < self.max_iterations


def create_default_controller(
    iliad_client: Optional["IliadClient"] = None,
    max_iterations: int = 3,
) -> FeedbackController:
    """Factory function to create a FeedbackController with default settings.

    Args:
        iliad_client: Optional Iliad client
        max_iterations: Maximum iterations

    Returns:
        Configured FeedbackController

    Example:
        >>> controller = create_default_controller(iliad_client)
    """
    return FeedbackController(
        iliad_client=iliad_client,
        max_iterations=max_iterations,
        use_llm_evaluation=True,
    )
