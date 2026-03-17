"""
LLM-based query analysis and decomposition.

This module uses an LLM to analyze complex user queries, decompose them
into atomic sub-queries, and classify each sub-query for routing.

Example:
    >>> from routing.query_analyzer import LLMQueryAnalyzer
    >>> analyzer = LLMQueryAnalyzer(iliad_client)
    >>> result = analyzer.analyze("Describe ALFA and how many projects reference it")
    >>> print(result.sub_queries)
    [SubQuery(text="Describe the ALFA project", intent=RAG),
     SubQuery(text="Count projects that reference ALFA", intent=DATABASE)]
"""

import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from loguru import logger

# Import types for type hints
try:
    from iliad.client import IliadClient, IliadModel
except ImportError:
    pass


class SubQueryIntent(Enum):
    """Intent classification for sub-queries."""

    RAG = "rag"  # Semantic search (explain, describe, what is)
    DATABASE = "database"  # Structured query (count, list, filter, find references)
    HYBRID = "hybrid"  # Needs both pipelines combined


@dataclass
class SubQuery:
    """A decomposed sub-query with its classification.

    Attributes:
        text: The sub-query text
        intent: Classified intent (RAG/DATABASE/HYBRID)
        depends_on: Index of sub-query this depends on (for sequential execution)
        priority: Execution priority (lower = higher priority)
        context_from: List of sub-query indices to inject results from
        query_template: Template with {placeholders} for result injection
        validation_criteria: Criteria for result validation
        max_refinements: Maximum refinement attempts for this sub-query
        store_as: Key to store result in context for later use
    """

    text: str
    intent: SubQueryIntent
    depends_on: Optional[int] = None
    priority: int = 0
    context_from: List[int] = field(default_factory=list)
    query_template: Optional[str] = None
    validation_criteria: Optional[str] = None
    max_refinements: int = 2
    store_as: Optional[str] = None


@dataclass
class QueryAnalysisResult:
    """Result of query analysis and decomposition.

    Attributes:
        original_query: The original user query
        sub_queries: List of decomposed sub-queries
    """

    original_query: str
    sub_queries: List[SubQuery]

    @property
    def is_complex(self) -> bool:
        """Whether the query was decomposed into multiple sub-queries."""
        return len(self.sub_queries) > 1


# Prompt template for query analysis
QUERY_ANALYSIS_PROMPT = """You are a query analyzer for a knowledge base system. Your job is to:
1. Analyze the user's query
2. Split into atomic sub-queries if needed
3. Classify each sub-query by intent

## Intent Types

**RAG (Semantic Search)** - Use for:
- Explanations: "What is X?", "Explain Y", "Describe Z"
- Conceptual questions: "How does X work?", "Why is Y important?"
- Documentation lookup: "Tell me about X", "Overview of Y"
- Qualitative information about projects, technologies, methodologies

**DATABASE (Structured Query)** - Use for:
- Counting: "How many X?", "Count of Y"
- Listing: "List all X", "Show all Y", "What projects use Z?"
- Filtering: "Projects with score above X", "Pages by author Y"
- Finding references: "What references X?", "Where is Y mentioned?"
- Aggregations: "Average score", "Most common technology"
- Metadata lookups: "Who created X?", "When was Y modified?"

**HYBRID** - Use when a SINGLE atomic question genuinely needs both semantic understanding AND structured data together (rare).

## Rules

1. ALWAYS decompose multi-part questions into separate sub-queries
2. Each sub-query should be atomic (one question, one intent)
3. Preserve the user's intent and context in each sub-query
4. Questions connected by "and", "also", "as well as" are usually separate queries
5. If a query can be answered by ONE pipeline, use that - don't force HYBRID
6. Order sub-queries logically (context-building queries first)

## Output Format

Return ONLY a JSON object:
```json
{
  "sub_queries": [
    {
      "text": "The rewritten sub-query",
      "intent": "rag" | "database" | "hybrid",
      "priority": 0,
      "depends_on": null
    }
  ]
}
```

## Examples

Query: "What is the ALFA project and how many pages mention it?"
```json
{
  "sub_queries": [
    {"text": "Describe the ALFA project, its purpose, and key details", "intent": "rag", "priority": 0, "depends_on": null},
    {"text": "Count how many pages mention or reference ALFA", "intent": "database", "priority": 1, "depends_on": null}
  ]
}
```

Query: "List all projects using Python"
```json
{
  "sub_queries": [
    {"text": "List all projects that use Python", "intent": "database", "priority": 0, "depends_on": null}
  ]
}
```

Query: "Explain the RAG pipeline architecture"
```json
{
  "sub_queries": [
    {"text": "Explain the RAG pipeline architecture", "intent": "rag", "priority": 0, "depends_on": null}
  ]
}
```

Query: "What projects does John Smith work on and what technologies do they use?"
```json
{
  "sub_queries": [
    {"text": "List all projects that John Smith works on or has created", "intent": "database", "priority": 0, "depends_on": null},
    {"text": "What technologies are used by John Smith's projects?", "intent": "database", "priority": 1, "depends_on": 0}
  ]
}
```

## User Query to Analyze

{query}
"""


class LLMQueryAnalyzer:
    """
    Analyze and decompose queries using LLM.

    Uses Iliad API to intelligently split complex queries into
    atomic sub-queries with appropriate intent classification.

    Attributes:
        iliad_client: Iliad API client
        model: Model to use for analysis (default: gpt-4o-mini)
        max_sub_queries: Maximum number of sub-queries allowed

    Example:
        >>> analyzer = LLMQueryAnalyzer(iliad_client)
        >>> result = analyzer.analyze("Describe X and count Y")
        >>> for sq in result.sub_queries:
        ...     print(f"{sq.intent}: {sq.text}")
    """

    def __init__(
        self,
        iliad_client: "IliadClient",
        model: str = "gpt-4o-mini-global",
        max_sub_queries: int = 5,
    ) -> None:
        """Initialize the query analyzer.

        Args:
            iliad_client: Configured Iliad client
            model: Model to use for analysis
            max_sub_queries: Maximum sub-queries to generate
        """
        self.iliad_client = iliad_client
        self.model = model
        self.max_sub_queries = max_sub_queries

        logger.info(f"Initialized LLMQueryAnalyzer with model: {model}")

    def analyze(self, query: str) -> QueryAnalysisResult:
        """
        Analyze a query and decompose into sub-queries.

        Args:
            query: User's natural language query

        Returns:
            QueryAnalysisResult with decomposed sub-queries

        Example:
            >>> result = analyzer.analyze("What is ALFA and who created it?")
            >>> print(len(result.sub_queries))
            2
        """
        logger.info(f"Analyzing query: {query[:100]}...")

        try:
            # Build prompt
            prompt = QUERY_ANALYSIS_PROMPT.format(query=query)

            # Call LLM
            messages = [{"role": "user", "content": prompt}]
            response = self.iliad_client.chat(messages=messages, model=self.model)
            content = self.iliad_client.extract_content(response)

            # Parse response
            result = self._parse_response(content, query)

            logger.info(
                f"Query decomposed into {len(result.sub_queries)} sub-queries "
                f"(complex: {result.is_complex})"
            )

            return result

        except Exception as e:
            logger.error(f"Query analysis failed: {e}")
            # Fallback: treat as single RAG query
            return self._fallback_analysis(query, str(e))

    def _parse_response(self, response: str, original_query: str) -> QueryAnalysisResult:
        """Parse LLM response into QueryAnalysisResult.

        Args:
            response: Raw LLM response
            original_query: Original user query

        Returns:
            Parsed QueryAnalysisResult
        """
        # Clean up response
        response = response.strip()

        # Remove markdown code blocks if present
        if response.startswith("```"):
            lines = response.split("\n")
            # Find start and end of JSON
            start_idx = 1 if lines[0].startswith("```") else 0
            end_idx = len(lines)
            for i in range(len(lines) - 1, -1, -1):
                if lines[i].strip() == "```":
                    end_idx = i
                    break
            response = "\n".join(lines[start_idx:end_idx])

        try:
            data = json.loads(response)
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse JSON response: {e}")
            return self._fallback_analysis(original_query, f"JSON parse error: {e}")

        # Extract sub-queries
        sub_queries = []
        raw_sub_queries = data.get("sub_queries", [])

        for i, sq in enumerate(raw_sub_queries[: self.max_sub_queries]):
            intent_str = sq.get("intent", "rag").lower()
            intent_map = {
                "rag": SubQueryIntent.RAG,
                "database": SubQueryIntent.DATABASE,
                "hybrid": SubQueryIntent.HYBRID,
            }
            intent = intent_map.get(intent_str, SubQueryIntent.RAG)

            sub_queries.append(
                SubQuery(
                    text=sq.get("text", original_query),
                    intent=intent,
                    depends_on=sq.get("depends_on"),
                    priority=sq.get("priority", i),
                )
            )

        # Ensure at least one sub-query
        if not sub_queries:
            sub_queries.append(
                SubQuery(
                    text=original_query,
                    intent=SubQueryIntent.RAG,
                )
            )

        return QueryAnalysisResult(
            original_query=original_query,
            sub_queries=sub_queries,
        )

    def _fallback_analysis(self, query: str, error: str) -> QueryAnalysisResult:
        """Create fallback analysis when LLM fails.

        Args:
            query: Original query
            error: Error message

        Returns:
            Simple single-query result
        """
        logger.warning(f"Using fallback analysis due to: {error}")

        # Simple heuristic fallback
        query_lower = query.lower()

        # Check for database indicators
        db_indicators = ["how many", "count", "list all", "list the", "show all", "who created"]
        is_database = any(ind in query_lower for ind in db_indicators)

        intent = SubQueryIntent.DATABASE if is_database else SubQueryIntent.RAG

        return QueryAnalysisResult(
            original_query=query,
            sub_queries=[
                SubQuery(
                    text=query,
                    intent=intent,
                )
            ],
        )

    def analyze_batch(self, queries: List[str]) -> List[QueryAnalysisResult]:
        """
        Analyze multiple queries.

        Args:
            queries: List of queries to analyze

        Returns:
            List of QueryAnalysisResult objects
        """
        results = []
        for query in queries:
            results.append(self.analyze(query))
        return results
