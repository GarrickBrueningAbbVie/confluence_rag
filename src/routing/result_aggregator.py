"""
Result aggregation for multi-pipeline query responses.

This module combines results from multiple sub-queries executed
across different pipelines into a unified, coherent response.

Example:
    >>> from routing.result_aggregator import ResultAggregator
    >>> aggregator = ResultAggregator(iliad_client)
    >>> final = aggregator.aggregate(original_query, sub_query_results)
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from loguru import logger

from .types import QueryIntent
from .parallel_executor import SubQueryResult
from .formatters import format_db_answer

# Import types for type hints
try:
    from iliad.client import IliadClient
except ImportError:
    pass


@dataclass
class AggregatedResult:
    """Final aggregated result from all sub-queries.

    Attributes:
        original_query: The original user query
        answer: Combined answer text
        sub_results: Individual sub-query results
        sources: Combined unique sources
        queries: Generated database queries
        success: Overall success status
        metadata: Additional metadata
    """

    original_query: str
    answer: str
    sub_results: List[SubQueryResult]
    sources: List[Dict[str, Any]] = field(default_factory=list)
    queries: List[str] = field(default_factory=list)
    success: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)


# Prompt template for result synthesis
SYNTHESIS_PROMPT = """You are synthesizing answers from multiple information sources to create a comprehensive response.

## Original Question
{original_query}

## Sub-Query Results

{sub_results_text}

## Instructions

1. Create a unified, coherent answer that addresses the original question
2. Integrate information from all successful sub-queries
3. Preserve specific facts, numbers, and data from database results
4. Include relevant context and explanations from documentation
5. If some sub-queries failed, work with what's available
6. Don't repeat information unnecessarily
7. Format the response clearly with sections if appropriate
8. If results contradict, note the discrepancy

## Response Format

Provide a well-structured response that:
- Directly answers the user's question
- Integrates all relevant information
- Is clear and concise
- References sources where appropriate

Your synthesized answer:"""


class ResultAggregator:
    """
    Aggregate results from multiple sub-queries.

    Uses LLM to synthesize coherent responses from multiple
    pipeline results.

    Attributes:
        iliad_client: Optional Iliad client for LLM synthesis
        model: Model to use for synthesis
        use_llm: Whether to use LLM for synthesis

    Example:
        >>> aggregator = ResultAggregator(iliad_client)
        >>> result = aggregator.aggregate(query, sub_results)
        >>> print(result.answer)
    """

    def __init__(
        self,
        iliad_client: Optional["IliadClient"] = None,
        model: str = "gpt-4o-mini-global",
        use_llm: bool = True,
    ) -> None:
        """Initialize result aggregator.

        Args:
            iliad_client: Optional Iliad client for synthesis
            model: Model to use for synthesis
            use_llm: Whether to use LLM (falls back to simple if False)
        """
        self.iliad_client = iliad_client
        self.model = model
        self.use_llm = use_llm and iliad_client is not None

        logger.info(f"Initialized ResultAggregator (LLM synthesis: {self.use_llm})")

    def aggregate(
        self,
        original_query: str,
        sub_results: List[SubQueryResult],
    ) -> AggregatedResult:
        """
        Aggregate sub-query results into a unified response.

        Args:
            original_query: The original user query
            sub_results: List of sub-query results

        Returns:
            AggregatedResult with combined answer

        Example:
            >>> result = aggregator.aggregate("What is X and how many Y?", results)
            >>> print(result.answer)
        """
        logger.info(f"Aggregating {len(sub_results)} sub-query results")

        # Handle edge cases
        if not sub_results:
            return AggregatedResult(
                original_query=original_query,
                answer="No results available to aggregate.",
                sub_results=[],
                success=False,
            )

        # Check success rate
        successful = [r for r in sub_results if r.success]
        success_rate = len(successful) / len(sub_results)

        logger.info(f"Success rate: {len(successful)}/{len(sub_results)} ({success_rate:.0%})")

        # Single successful result - no synthesis needed
        if len(successful) == 1 and len(sub_results) == 1:
            result = successful[0]
            return AggregatedResult(
                original_query=original_query,
                answer=self._format_single_answer(result),
                sub_results=sub_results,
                sources=result.sources,
                queries=[result.query] if result.query else [],
                success=True,
                metadata={"synthesis_method": "single_result"},
            )

        # Multiple results - synthesize
        if self.use_llm and len(successful) > 0:
            answer = self._synthesize_with_llm(original_query, sub_results)
            method = "llm_synthesis"
        else:
            answer = self._synthesize_simple(sub_results)
            method = "simple_concatenation"

        # Collect sources and queries
        all_sources = self._collect_sources(sub_results)
        all_queries = [r.query for r in sub_results if r.query]

        return AggregatedResult(
            original_query=original_query,
            answer=answer,
            sub_results=sub_results,
            sources=all_sources,
            queries=all_queries,
            success=len(successful) > 0,
            metadata={
                "synthesis_method": method,
                "success_rate": success_rate,
                "total_sub_queries": len(sub_results),
                "successful_sub_queries": len(successful),
            },
        )

    def _format_single_answer(self, result: SubQueryResult) -> str:
        """Format a single result's answer.

        Args:
            result: Single sub-query result

        Returns:
            Formatted answer string
        """
        return format_db_answer(result.answer, max_items=20)

    def _synthesize_with_llm(
        self,
        original_query: str,
        sub_results: List[SubQueryResult],
    ) -> str:
        """Synthesize results using LLM.

        Args:
            original_query: Original user query
            sub_results: All sub-query results

        Returns:
            Synthesized answer string
        """
        # Build sub-results text
        sub_results_parts = []

        for i, result in enumerate(sub_results, 1):
            status = "SUCCESS" if result.success else "FAILED"
            intent = result.sub_query.intent.value.upper()

            part = f"### Sub-Query {i} [{intent}] - {status}\n"
            part += f"**Question:** {result.sub_query.text}\n"

            if result.success:
                answer = self._format_single_answer(result)
                part += f"**Answer:**\n{answer}\n"
                if result.query:
                    part += f"**Generated Query:** `{result.query}`\n"
            else:
                part += f"**Error:** {result.error}\n"

            sub_results_parts.append(part)

        sub_results_text = "\n".join(sub_results_parts)

        # Build prompt
        prompt = SYNTHESIS_PROMPT.format(
            original_query=original_query,
            sub_results_text=sub_results_text,
        )

        try:
            messages = [{"role": "user", "content": prompt}]
            response = self.iliad_client.chat(messages=messages, model=self.model)
            answer = self.iliad_client.extract_content(response)

            return answer.strip()

        except Exception as e:
            logger.error(f"LLM synthesis failed: {e}")
            return self._synthesize_simple(sub_results)

    def _synthesize_simple(self, sub_results: List[SubQueryResult]) -> str:
        """Simple concatenation of results (fallback).

        Args:
            sub_results: All sub-query results

        Returns:
            Concatenated answer string
        """
        parts = []

        for result in sub_results:
            if not result.success:
                continue

            intent_label = {
                QueryIntent.RAG: "Documentation",
                QueryIntent.DATABASE: "Data Query",
                QueryIntent.HYBRID: "Combined",
                QueryIntent.CHART: "Visualization",
                QueryIntent.TABLE: "Table",
            }.get(result.sub_query.intent, "Result")

            answer = self._format_single_answer(result)

            if len(sub_results) > 1:
                parts.append(f"**{intent_label}:** {result.sub_query.text}")
                parts.append(answer)
                parts.append("")  # Blank line
            else:
                parts.append(answer)

        if not parts:
            return "Unable to find relevant information for your query."

        return "\n".join(parts).strip()

    def _collect_sources(self, sub_results: List[SubQueryResult]) -> List[Dict[str, Any]]:
        """Collect sources from all results, preserving document indices.

        Sources are collected in order with their document_index preserved
        to match LLM document references. Re-indexes across sub-queries.

        Args:
            sub_results: All sub-query results

        Returns:
            List of sources with updated document_index
        """
        sources = []
        doc_index = 1

        for result in sub_results:
            for source in result.sources:
                # Create a copy with updated document_index for combined results
                source_copy = dict(source)
                source_copy["document_index"] = doc_index
                sources.append(source_copy)
                doc_index += 1

        return sources

    def format_with_sources(self, result: AggregatedResult) -> str:
        """Format result with source citations.

        Args:
            result: Aggregated result

        Returns:
            Formatted answer with sources
        """
        output = result.answer

        if result.sources:
            output += "\n\n**Sources:**\n"
            for i, source in enumerate(result.sources[:5], 1):
                title = source.get("title", "Unknown")
                url = source.get("url", "")
                if url:
                    output += f"{i}. [{title}]({url})\n"
                else:
                    output += f"{i}. {title}\n"

        return output
