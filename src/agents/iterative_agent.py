"""
Iterative Agent for list-then-describe query patterns.

This agent handles queries that require:
1. First getting a list of items (via Database)
2. Then describing EACH item in that list (via RAG)
3. Combining all descriptions into a comprehensive answer

This solves the problem where "List X and describe all X" queries
would miss items because RAG only retrieves a few documents.

Example:
    >>> from agents.iterative_agent import IterativeDescribeAgent
    >>> agent = IterativeDescribeAgent(rag_agent, db_agent, iliad_client)
    >>> result = agent.execute(
    ...     "List all projects that use XGBoost and describe all these projects",
    ...     context
    ... )
"""

import re
from typing import Any, Dict, List, Optional, Tuple

from loguru import logger

from agents.base import BaseAgent, AgentContext, AgentResult, AgentStatus

# Import shared patterns for list+describe detection
from routing.patterns import LIST_DESCRIBE_PATTERNS as LIST_KEYWORDS_SHARED
from routing.patterns import DESCRIBE_PATTERNS as DESCRIBE_KEYWORDS_SHARED

# Type hints for optional imports
try:
    from agents.rag_agent import RAGAgent
    from agents.database_agent import DatabaseAgent
    from iliad.client import IliadClient
except ImportError:
    pass


class IterativeDescribeAgent(BaseAgent):
    """Agent for list-then-describe query patterns.

    Handles queries like:
    - "List all projects that use X and describe all these projects"
    - "What projects use Y and explain each one"
    - "Show all pages about Z and summarize them"

    The agent:
    1. Extracts the list query component
    2. Executes database query to get all items
    3. Iterates over each item and executes RAG query
    4. Combines all results into comprehensive answer

    Attributes:
        rag_agent: RAGAgent for descriptions
        db_agent: DatabaseAgent for listing
        iliad_client: Iliad client for synthesis
        max_items_to_describe: Maximum items to describe (prevents runaway)

    Example:
        >>> agent = IterativeDescribeAgent(rag_agent, db_agent, iliad_client)
        >>> result = agent.execute(
        ...     "List all projects using Python and describe each",
        ...     context
        ... )
    """

    # Patterns that indicate list+describe queries
    LIST_DESCRIBE_PATTERNS = [
        r"list\s+(?:all\s+)?(.+?)\s+(?:and|then)\s+(?:describe|explain|summarize|tell me about)\s+(?:all\s+)?(?:these|them|each|every)",
        r"(?:what|which)\s+(.+?)\s+(?:and|then)\s+(?:describe|explain|summarize)\s+(?:all\s+)?(?:these|them|each|every)?",
        r"show\s+(?:me\s+)?(?:all\s+)?(.+?)\s+(?:and|then)\s+(?:describe|explain|summarize)",
        r"(?:get|find)\s+(?:all\s+)?(.+?)\s+(?:and|then)\s+(?:describe|explain|summarize)",
        r"list\s+(?:all\s+)?(.+?)\s+(?:and|,)\s+(?:describe|explain)\s+(?:all\s+)?(?:of\s+)?them",
        r"(?:describe|explain)\s+all\s+(.+?)\s+that\s+(?:use|have|contain)",
    ]

    # Use shared patterns from routing.patterns (imported at module level)
    # Additional keywords specific to this agent's detection
    DESCRIBE_KEYWORDS = DESCRIBE_KEYWORDS_SHARED + [
        "summarize all",
        "summarize each",
        "tell me about all",
        "tell me about each",
    ]

    LIST_KEYWORDS = LIST_KEYWORDS_SHARED + [
        "what pages",
        "which pages",
        "show me all",
    ]

    def __init__(
        self,
        rag_agent: "RAGAgent",
        db_agent: "DatabaseAgent",
        iliad_client: Optional["IliadClient"] = None,
        max_items_to_describe: int = 5,
    ) -> None:
        """Initialize iterative describe agent.

        Args:
            rag_agent: RAGAgent for descriptions
            db_agent: DatabaseAgent for listing
            iliad_client: Iliad client for synthesis
            max_items_to_describe: Maximum items to describe
        """
        super().__init__(
            name="iterative_describe_agent",
            description="Handles list-then-describe queries by iterating over items",
            iliad_client=iliad_client,
        )
        self.rag_agent = rag_agent
        self.db_agent = db_agent
        self.max_items_to_describe = max_items_to_describe

        logger.info(
            f"Initialized IterativeDescribeAgent (max_items: {max_items_to_describe})"
        )

    def execute(
        self,
        query: str,
        context: AgentContext,
    ) -> AgentResult:
        """Execute list-then-describe query.

        Steps:
        1. Parse query to extract list and describe components
        2. Execute database query to get all items
        3. For each item, execute RAG query to get description
        4. Synthesize all descriptions into final answer

        Args:
            query: The query to process
            context: Shared execution context

        Returns:
            AgentResult with combined descriptions
        """
        logger.info(f"IterativeDescribeAgent executing: {query[:80]}...")

        try:
            # Step 1: Parse query to extract components
            list_query, describe_template = self._parse_query(query)

            if not list_query:
                logger.warning("Could not parse list+describe pattern, falling back")
                return AgentResult(
                    status=AgentStatus.FAILED,
                    reasoning="Could not parse list+describe pattern from query",
                )

            logger.info(f"List query: {list_query}")
            logger.info(f"Describe template: {describe_template}")

            # Step 2: Execute database query to get all items
            db_result = self.db_agent.execute(list_query, context)

            if not db_result.success:
                return AgentResult(
                    status=AgentStatus.FAILED,
                    reasoning=f"Database query failed: {db_result.reasoning}",
                )

            # Extract items with full metadata for ranking
            all_items = self._extract_items_with_metadata(db_result.data)

            if not all_items:
                return AgentResult(
                    status=AgentStatus.SUCCESS,
                    data={
                        "answer": "No items found matching the criteria.",
                        "items_found": 0,
                        "items_described": 0,
                    },
                    reasoning="Database query returned no results",
                )

            logger.info(f"Found {len(all_items)} items total")

            # Rank items by importance metrics and select top N
            ranked_items = self._rank_items(all_items)
            items_to_describe = ranked_items[: self.max_items_to_describe]

            if len(all_items) > self.max_items_to_describe:
                logger.info(
                    f"Selected top {self.max_items_to_describe} items by importance "
                    f"(from {len(all_items)} total)"
                )

            # Step 3: Describe each selected item
            descriptions = []
            sources = []

            for i, item in enumerate(items_to_describe):
                item_name = self._get_item_name(item)
                describe_query = describe_template.format(item=item_name)

                logger.debug(f"Describing item {i + 1}/{len(items_to_describe)}: {item_name}")

                rag_result = self.rag_agent.execute(describe_query, context)

                if rag_result.success and rag_result.data:
                    answer = rag_result.data.get("answer", "")
                    item_sources = rag_result.data.get("sources", [])

                    descriptions.append({
                        "item": item_name,
                        "description": answer,
                        "found": bool(answer and len(answer) > 50),
                    })
                    sources.extend(item_sources)
                else:
                    descriptions.append({
                        "item": item_name,
                        "description": f"No detailed information found for {item_name}.",
                        "found": False,
                    })

            # Step 4: Synthesize final answer
            final_answer = self._synthesize_answer(
                query, all_items, descriptions, context
            )

            # Record execution
            context.record_execution(self.name, query)

            # Calculate success metrics
            total_items_found = len(all_items)
            items_selected = len(items_to_describe)
            items_described = sum(1 for d in descriptions if d.get("found", False))

            logger.info(
                f"IterativeDescribeAgent completed: {total_items_found} found, "
                f"{items_selected} selected, {items_described} described successfully"
            )

            return AgentResult(
                status=AgentStatus.SUCCESS,
                data={
                    "answer": final_answer,
                    "all_items": [self._get_item_name(i) for i in all_items],
                    "items_selected": [self._get_item_name(i) for i in items_to_describe],
                    "descriptions": descriptions,
                    "sources": sources,
                    "items_found": total_items_found,
                    "items_selected_count": items_selected,
                    "items_described": items_described,
                },
                confidence=items_described / max(items_selected, 1),
                reasoning=f"Found {total_items_found} items, selected top {items_selected}, described {items_described}",
                metadata={
                    "list_query": list_query,
                    "describe_template": describe_template,
                    "selection_limited": total_items_found > self.max_items_to_describe,
                    "ranking_applied": total_items_found > self.max_items_to_describe,
                },
            )

        except Exception as e:
            logger.error(f"IterativeDescribeAgent execution failed: {e}")
            return AgentResult(
                status=AgentStatus.FAILED,
                reasoning=str(e),
                metadata={"error": str(e)},
            )

    def can_handle(self, query: str, context: AgentContext) -> float:
        """Determine if this agent can handle the query.

        Looks for list+describe patterns in the query.

        Args:
            query: The query to evaluate
            context: Execution context

        Returns:
            Confidence score (0.0 - 1.0)
        """
        query_lower = query.lower()

        # Check for combined list+describe keywords
        has_list = any(kw in query_lower for kw in self.LIST_KEYWORDS)
        has_describe = any(kw in query_lower for kw in self.DESCRIBE_KEYWORDS)

        if has_list and has_describe:
            return 0.95  # Very high confidence for explicit patterns

        # Check regex patterns
        for pattern in self.LIST_DESCRIBE_PATTERNS:
            if re.search(pattern, query_lower):
                return 0.9

        # Check for implicit patterns
        # "List X and describe" without explicit "all/each/these"
        if has_list and ("describe" in query_lower or "explain" in query_lower):
            return 0.7

        # Check for "describe all X that use/have Y"
        if re.search(r"describe\s+all\s+\w+\s+that", query_lower):
            return 0.85

        return 0.1

    def _parse_query(self, query: str) -> Tuple[Optional[str], str]:
        """Parse query to extract list query and describe template.

        Preserves original case in the list_query to ensure proper matching
        (e.g., 'XGBoost' stays as 'XGBoost', not 'xgboost').

        Args:
            query: Original query

        Returns:
            Tuple of (list_query, describe_template)
        """
        # Use IGNORECASE for pattern matching but extract from original query
        # to preserve case (e.g., 'XGBoost' not 'xgboost')

        # Try to split on common conjunctions
        split_patterns = [
            r"(.+?)\s+and\s+(?:then\s+)?(?:describe|explain|summarize)\s+(?:all\s+)?(?:these|them|each|every)?(.*)$",
            r"(.+?)\s+,\s*(?:and\s+)?(?:describe|explain|summarize)\s+(?:all\s+)?(?:these|them|each)?(.*)$",
        ]

        for pattern in split_patterns:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                list_part = match.group(1).strip()
                # Clean up list part - preserve original case
                list_query = list_part

                # Build describe template
                describe_template = "Describe {item} in detail, including its purpose, features, and key information."
                return list_query, describe_template

        # Fallback: look for "describe all X that use Y" pattern
        match = re.search(
            r"describe\s+all\s+(\w+)\s+that\s+(.+)",
            query,
            re.IGNORECASE
        )
        if match:
            entity = match.group(1)  # e.g., "projects"
            condition = match.group(2)  # e.g., "use XGBoost" - preserves case
            list_query = f"List all {entity} that {condition}"
            describe_template = f"Describe {{item}} in detail."
            return list_query, describe_template

        # Another fallback: split on "and describe" (case-insensitive)
        and_describe_match = re.search(r"\s+and\s+describe", query, re.IGNORECASE)
        if and_describe_match:
            list_query = query[:and_describe_match.start()].strip()
            describe_template = "Describe {item} in detail, including its purpose and key features."
            return list_query, describe_template

        # Try to extract list query from the beginning
        list_match = re.search(
            r"^(list\s+(?:all\s+)?[^,]+)",
            query,
            re.IGNORECASE
        )
        if list_match:
            list_query = list_match.group(1)
            describe_template = "Describe {item} in detail."
            return list_query, describe_template

        return None, ""

    def _extract_items_with_metadata(self, db_data: Any) -> List[Dict[str, Any]]:
        """Extract items with full metadata from database result.

        Preserves metadata fields like page_size, children_count, depth
        for ranking purposes.

        Args:
            db_data: Database result data

        Returns:
            List of item dicts with metadata
        """
        if db_data is None:
            return []

        answer = db_data.get("answer") if isinstance(db_data, dict) else db_data

        if isinstance(answer, list):
            items = []
            for item in answer:
                if isinstance(item, dict):
                    # Keep full dict with metadata
                    items.append(item)
                else:
                    # Wrap simple values
                    items.append({"name": str(item)})
            return items

        if isinstance(answer, str):
            # Try to parse as newline-separated list
            lines = [l.strip().lstrip("- •").strip() for l in answer.split("\n")]
            return [{"name": l} for l in lines if l and not l.startswith("#")]

        return []

    def _rank_items(self, items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Rank items by importance metrics.

        Ranks by (in order of priority):
        1. page_size / content_length - larger content = more to describe
        2. children_count - more children = more comprehensive project
        3. depth - shallower = more top-level/important

        Args:
            items: List of item dicts with metadata

        Returns:
            Items sorted by importance (most important first)
        """
        def get_score(item: Dict[str, Any]) -> float:
            score = 0.0

            # Page size / content length (normalize, higher = better)
            page_size = item.get("page_size", item.get("content_length", 0))
            if isinstance(page_size, (int, float)):
                score += min(page_size / 10000, 10)  # Cap at 10 points

            # Children count (more children = more comprehensive)
            children = item.get("children_count", item.get("num_children", 0))
            if isinstance(children, (int, float)):
                score += min(children, 10)  # Cap at 10 points

            # Depth (shallower = more important, invert so lower depth = higher score)
            depth = item.get("depth", item.get("page_depth", 5))
            if isinstance(depth, (int, float)):
                score += max(0, 5 - depth)  # Depth 0-1 = 4-5 points, depth 5+ = 0

            return score

        # Sort by score descending (highest score = most important)
        ranked = sorted(items, key=get_score, reverse=True)

        if ranked:
            logger.debug(
                f"Ranked {len(ranked)} items. Top item score: {get_score(ranked[0]):.1f}"
            )

        return ranked

    def _get_item_name(self, item: Any) -> str:
        """Get display name for an item.

        Args:
            item: Item from database

        Returns:
            Item name string
        """
        if isinstance(item, dict):
            for key in ["title", "name", "project", "page"]:
                if key in item:
                    return str(item[key])
            return str(list(item.values())[0]) if item else "Unknown"

        return str(item)

    def _synthesize_answer(
        self,
        original_query: str,
        items: List[Any],
        descriptions: List[Dict[str, Any]],
        context: AgentContext,
    ) -> str:
        """Synthesize final answer from all descriptions.

        Args:
            original_query: Original user query
            items: All items found
            descriptions: Descriptions for each item
            context: Execution context

        Returns:
            Synthesized answer string
        """
        # If we have Iliad client, use LLM synthesis
        if self.iliad_client and len(descriptions) > 1:
            return self._llm_synthesize(original_query, items, descriptions)

        # Otherwise, format manually
        total_found = len(items)
        num_described = len(descriptions)

        if total_found > num_described:
            parts = [f"Found {total_found} items. Describing the top {num_described} by importance:\n"]
        else:
            parts = [f"Found {total_found} items:\n"]

        for i, desc in enumerate(descriptions, 1):
            item_name = desc.get("item", f"Item {i}")
            description = desc.get("description", "No description available.")

            parts.append(f"\n## {i}. {item_name}\n")
            parts.append(description)

        if total_found > num_described:
            remaining = total_found - num_described
            parts.append(
                f"\n\n*Note: {remaining} additional items were found but not described. "
                f"Items were selected based on content size, depth, and comprehensiveness.*"
            )

        return "\n".join(parts)

    def _llm_synthesize(
        self,
        original_query: str,
        items: List[Any],
        descriptions: List[Dict[str, Any]],
    ) -> str:
        """Use LLM to synthesize descriptions.

        Args:
            original_query: Original query
            items: Items found
            descriptions: Item descriptions

        Returns:
            Synthesized answer
        """
        desc_text = "\n\n".join(
            f"### {d.get('item', 'Item')}\n{d.get('description', 'No description.')}"
            for d in descriptions
        )

        total_found = len(items)
        num_described = len(descriptions)

        selection_note = ""
        if total_found > num_described:
            selection_note = f" (selected top {num_described} by importance from {total_found} total)"

        prompt = f"""Synthesize these project descriptions into a comprehensive answer.

Original Question: {original_query}

Found {total_found} items{selection_note}. Here are the descriptions:

{desc_text}

Create a well-organized response that:
1. Provides a clear summary of each project described
2. Gives the essential information about each one
3. Is well-structured and easy to read
4. Mentions if there are additional items not described

Your synthesized answer:"""

        try:
            messages = [{"role": "user", "content": prompt}]
            response = self.iliad_client.chat(messages=messages)
            return self.iliad_client.extract_content(response)
        except Exception as e:
            logger.error(f"LLM synthesis failed: {e}")
            # Fallback to simple formatting
            parts = [f"Found {len(items)} items:\n"]
            for d in descriptions:
                parts.append(f"\n**{d.get('item', 'Item')}**: {d.get('description', 'No description.')[:200]}...")
            return "\n".join(parts)

    def validate_coverage(
        self,
        result: AgentResult,
        min_coverage: float = 0.5,
    ) -> Tuple[bool, str]:
        """Validate that a sufficient number of items were described.

        Checks that the coverage ratio (items_described / items_found)
        meets the minimum threshold.

        Args:
            result: AgentResult from execute()
            min_coverage: Minimum acceptable coverage ratio (0.0 - 1.0)

        Returns:
            Tuple of (is_valid, message)

        Example:
            >>> result = agent.execute(query, context)
            >>> is_valid, msg = agent.validate_coverage(result, min_coverage=0.7)
            >>> if not is_valid:
            ...     logger.warning(f"Low coverage: {msg}")
        """
        if not result.success or not result.data:
            return False, "Result was not successful"

        # Use items_selected_count (items we attempted to describe) for coverage
        items_selected = result.data.get("items_selected_count", result.data.get("items_found", 0))
        items_described = result.data.get("items_described", 0)

        if items_selected == 0:
            return True, "No items to describe"

        coverage = items_described / items_selected

        if coverage < min_coverage:
            missing = items_selected - items_described
            return (
                False,
                f"Low coverage: {items_described}/{items_selected} selected items described "
                f"({coverage:.1%}), {missing} items missing descriptions"
            )

        return True, f"Good coverage: {items_described}/{items_selected} ({coverage:.1%})"
