"""
Response combiner for hybrid queries.

This module combines responses from RAG and Database pipelines
into a coherent unified response.

Example:
    >>> from routing.response_combiner import ResponseCombiner
    >>> combiner = ResponseCombiner(iliad_client)
    >>> combined = combiner.combine(query, rag_result, db_result)
"""

from typing import Any, Dict, Optional

from loguru import logger

# Import types for type hints
try:
    from iliad.client import IliadClient
except ImportError:
    pass


class ResponseCombiner:
    """
    Combine responses from multiple pipelines.

    Uses LLM to synthesize coherent responses from RAG semantic
    search and Database structured query results.

    Attributes:
        iliad_client: Optional Iliad client for LLM synthesis
        model: Model to use for synthesis

    Example:
        >>> combiner = ResponseCombiner(iliad_client)
        >>> result = combiner.combine(query, rag_result, db_result)
    """

    def __init__(
        self,
        iliad_client: Optional["IliadClient"] = None,
        model: str = "gpt-5-mini-global",
    ) -> None:
        """Initialize response combiner.

        Args:
            iliad_client: Optional Iliad client for LLM synthesis
            model: Model to use for synthesis
        """
        self.iliad_client = iliad_client
        self.model = model

        logger.info("Initialized ResponseCombiner")

    def combine(
        self,
        query: str,
        rag_result: Optional[Dict[str, Any]] = None,
        db_result: Optional[Dict[str, Any]] = None,
        use_llm: bool = True,
    ) -> Dict[str, Any]:
        """
        Combine results from RAG and Database pipelines.

        Args:
            query: Original user query
            rag_result: Result from RAG pipeline (optional)
            db_result: Result from Database pipeline (optional)
            use_llm: Whether to use LLM for synthesis

        Returns:
            Dict with:
            - answer: Combined response
            - sources: Combined sources
            - method: Combination method used

        Example:
            >>> combined = combiner.combine(
            ...     "What is Project X and how many pages does it have?",
            ...     rag_result={"answer": "Project X is...", "sources": [...]},
            ...     db_result={"answer": "42 pages"}
            ... )
        """
        result = {
            "answer": "",
            "sources": [],
            "method": "none",
        }

        # Handle cases where only one result is available
        if not rag_result and not db_result:
            result["answer"] = "Unable to find relevant information."
            return result

        if not rag_result:
            result["answer"] = self._format_db_only(db_result)
            result["method"] = "database_only"
            return result

        if not db_result:
            result["answer"] = rag_result.get("answer", "")
            result["sources"] = rag_result.get("sources", [])
            result["method"] = "rag_only"
            return result

        # Both results available - combine them
        if use_llm and self.iliad_client:
            result = self._combine_with_llm(query, rag_result, db_result)
        else:
            result = self._combine_simple(rag_result, db_result)

        return result

    def _format_db_only(self, db_result: Dict[str, Any]) -> str:
        """Format database-only result.

        Args:
            db_result: Database pipeline result

        Returns:
            Formatted answer string
        """
        answer = db_result.get("answer", "")

        if isinstance(answer, str):
            return answer

        if isinstance(answer, (int, float)):
            return str(answer)

        if isinstance(answer, list):
            if len(answer) == 0:
                return "No results found."

            if isinstance(answer[0], dict):
                lines = []
                for item in answer[:20]:  # Limit to 20 items
                    line = ", ".join(f"{k}: {v}" for k, v in item.items())
                    lines.append(f"- {line}")
                if len(answer) > 20:
                    lines.append(f"... and {len(answer) - 20} more items")
                return "\n".join(lines)

            return "\n".join(f"- {item}" for item in answer[:20])

        if isinstance(answer, dict):
            return "\n".join(f"- {k}: {v}" for k, v in answer.items())

        return str(answer)

    def _combine_simple(
        self,
        rag_result: Dict[str, Any],
        db_result: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Simple combination without LLM.

        Args:
            rag_result: RAG pipeline result
            db_result: Database pipeline result

        Returns:
            Combined result dict
        """
        rag_answer = rag_result.get("answer", "")
        db_answer = self._format_db_only(db_result)

        combined = f"""Based on the documentation:
{rag_answer}

Data summary:
{db_answer}"""

        return {
            "answer": combined,
            "sources": rag_result.get("sources", []),
            "method": "simple_concatenation",
        }

    def _combine_with_llm(
        self,
        query: str,
        rag_result: Dict[str, Any],
        db_result: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Combine results using LLM synthesis.

        Args:
            query: Original user query
            rag_result: RAG pipeline result
            db_result: Database pipeline result

        Returns:
            Combined result dict
        """
        rag_answer = rag_result.get("answer", "")
        db_answer = self._format_db_only(db_result)

        prompt = f"""You are combining information from two sources to answer a user's question.

User Question: {query}

Documentation Search Result:
{rag_answer}

Database Query Result:
{db_answer}

Please synthesize these results into a coherent, comprehensive answer.
- Integrate the factual data with the contextual information
- Don't repeat information unnecessarily
- If the results contradict, note the discrepancy
- Keep the response concise and focused on the question

Answer:"""

        try:
            messages = [{"role": "user", "content": prompt}]
            response = self.iliad_client.chat(messages=messages, model=self.model)
            content = self.iliad_client.extract_content(response)

            return {
                "answer": content.strip(),
                "sources": rag_result.get("sources", []),
                "method": "llm_synthesis",
            }

        except Exception as e:
            logger.warning(f"LLM synthesis failed: {e}, falling back to simple")
            return self._combine_simple(rag_result, db_result)

    def format_sources(self, sources: list) -> str:
        """Format source list for display.

        Args:
            sources: List of source documents

        Returns:
            Formatted sources string
        """
        if not sources:
            return ""

        lines = ["\n\nSources:"]
        for i, source in enumerate(sources[:5], 1):
            title = source.get("title", "Unknown")
            url = source.get("url", "")

            if url:
                lines.append(f"{i}. [{title}]({url})")
            else:
                lines.append(f"{i}. {title}")

        return "\n".join(lines)
