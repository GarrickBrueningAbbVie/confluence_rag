"""
Database query pipeline orchestrator.

This module orchestrates the database query pipeline, combining
DataFrame loading, query generation, and execution.

Example:
    >>> from database.pipeline import DatabasePipeline
    >>> pipeline = DatabasePipeline("Data_Storage/confluence_pages.json")
    >>> result = pipeline.query("How many pages use Python?")
"""

from pathlib import Path
from typing import Any, Dict, Optional, Union

from loguru import logger

from .dataframe_loader import DataFrameLoader
from .query_generator import QueryGenerator
from .query_executor import QueryExecutor
from .fuzzy_matcher import (
    FuzzyMatcher,
    extract_search_terms_from_query,
    detect_searchable_column,
)

# Import types for type hints
try:
    from iliad.client import IliadClient
except ImportError:
    pass


class DatabasePipeline:
    """
    Orchestrate database queries on Confluence data.

    Combines DataFrame loading, LLM-based query generation, and
    safe query execution.

    Attributes:
        loader: DataFrameLoader instance
        generator: QueryGenerator instance
        executor: QueryExecutor instance

    Example:
        >>> pipeline = DatabasePipeline(json_path, iliad_client)
        >>> result = pipeline.query("Count pages by author")
        >>> print(result['answer'])
    """

    def __init__(
        self,
        json_path: Union[str, Path],
        iliad_client: "IliadClient",
        model: str = "gpt-5-mini-global",
    ) -> None:
        """Initialize the database pipeline.

        Args:
            json_path: Path to Confluence JSON data
            iliad_client: Configured Iliad client
            model: Model to use for query generation
        """
        self.json_path = Path(json_path)
        self.iliad_client = iliad_client
        self.model = model

        # Initialize components
        self.loader = DataFrameLoader(json_path)
        self.df = self.loader.load()

        # Get schema info for query generator
        schema_info = self.loader.get_column_info()
        self.generator = QueryGenerator(iliad_client, schema_info, model)

        self.executor = QueryExecutor(self.df)

        # Fuzzy matcher for fallback (lazy initialized)
        self._fuzzy_matcher: Optional[FuzzyMatcher] = None

        logger.info(f"Initialized DatabasePipeline with {len(self.df)} pages")

    def query(
        self,
        question: str,
        return_query: bool = True,
        enable_fuzzy_fallback: bool = True,
    ) -> Dict[str, Any]:
        """
        Execute a natural language query.

        Generates a pandas query from the question, executes it,
        and returns the results. If the query returns empty results
        and fuzzy fallback is enabled, attempts to find similar terms
        using embedding similarity.

        Args:
            question: Natural language question
            return_query: Whether to include generated query in response
            enable_fuzzy_fallback: Whether to try fuzzy matching on empty results

        Returns:
            Dict with:
            - success: Whether query succeeded
            - answer: Query result
            - query: Generated pandas query (if return_query=True)
            - fuzzy_corrections: Dict of term corrections (if fuzzy fallback used)
            - error: Error message (if failed)

        Example:
            >>> result = pipeline.query("How many pages use Airflow?")
            >>> print(result['answer'])
        """
        result = {
            "success": False,
            "answer": None,
            "query": None,
            "fuzzy_corrections": None,
            "error": None,
        }

        logger.info(f"Processing database query: {question[:100]}")

        # Generate query
        gen_result = self.generator.generate(question)

        if not gen_result["success"]:
            result["error"] = f"Query generation failed: {gen_result['error']}"
            logger.error(f"Query generation failed: {gen_result['error']}")
            return result

        if return_query:
            result["query"] = gen_result["query"]

        # Execute query
        exec_result = self.executor.execute(gen_result["query"])

        if not exec_result["success"]:
            result["error"] = f"Query execution failed: {exec_result['error']}"
            result["query"] = gen_result["query"]  # Include query for debugging
            logger.error(f"Query execution failed: {exec_result['error']}")
            return result

        # Check if result is empty and fuzzy fallback is enabled
        if enable_fuzzy_fallback and self._is_empty_result(exec_result["result"]):
            logger.info("Query returned empty result, attempting fuzzy fallback...")

            fuzzy_result = self._try_fuzzy_fallback(
                gen_result["query"],
                question,
            )

            if fuzzy_result:
                logger.info("Fuzzy fallback succeeded")
                if return_query:
                    fuzzy_result["query"] = fuzzy_result.get("query")
                return fuzzy_result

            logger.info("Fuzzy fallback did not find matches")

        result["success"] = True
        result["answer"] = exec_result["result"]

        logger.info(f"Query executed successfully, answer: {repr(exec_result['result'])}")

        return result

    def execute_raw_query(
        self,
        query: str,
    ) -> Dict[str, Any]:
        """
        Execute a raw pandas query directly.

        Bypasses query generation for direct query execution.

        Args:
            query: Pandas query string

        Returns:
            Execution result dictionary

        Example:
            >>> result = pipeline.execute_raw_query("df.shape[0]")
            >>> print(result['result'])
        """
        return self.executor.execute(query)

    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the loaded data.

        Returns:
            Dict with data statistics
        """
        stats = {
            "total_pages": len(self.df),
            "unique_projects": self.df["parent_project"].nunique(),
            "unique_authors": self.df["created_by"].nunique(),
            "pages_with_technologies": self.df["technologies"].apply(
                lambda x: len(x) > 0 if isinstance(x, list) else False
            ).sum(),
            "pages_with_completeness": self.df["completeness_score"].notna().sum(),
            "avg_completeness": self.df["completeness_score"].mean(),
        }

        return stats

    def get_sample_questions(self) -> list:
        """
        Get sample questions that can be answered.

        Returns:
            List of example questions
        """
        return [
            "How many pages are there in total?",
            "Who has created the most pages?",
            "What technologies are most commonly used?",
            "List all projects with completeness score above 50",
            "How many pages are in each project?",
            "What pages use Python?",
            "Show the average completeness score by project",
            "Which projects have the lowest completeness scores?",
            "How many pages were created in the last 30 days?",
            "What is the depth distribution of pages?",
        ]

    def reload_data(self) -> None:
        """Reload data from JSON file."""
        self.df = self.loader.load()
        self.executor = QueryExecutor(self.df)
        # Clear fuzzy matcher cache on reload
        if self._fuzzy_matcher:
            self._fuzzy_matcher.clear_cache()
        logger.info(f"Reloaded data: {len(self.df)} pages")

    @property
    def fuzzy_matcher(self) -> FuzzyMatcher:
        """Lazy initialization of fuzzy matcher."""
        if self._fuzzy_matcher is None:
            self._fuzzy_matcher = FuzzyMatcher(self.df)
            logger.info("Initialized FuzzyMatcher for fallback queries")
        return self._fuzzy_matcher

    def _is_empty_result(self, result: Any) -> bool:
        """Check if a query result is empty or trivially zero.

        Args:
            result: The query execution result

        Returns:
            True if result is considered empty
        """
        if result is None:
            return True
        if isinstance(result, (list, dict)) and len(result) == 0:
            return True
        if isinstance(result, (int, float)) and result == 0:
            return True
        # Check for empty DataFrame result (converted to empty list/dict)
        if isinstance(result, str) and result in ("[]", "{}", ""):
            return True
        return False

    def _try_fuzzy_fallback(
        self,
        original_query: str,
        question: str,
    ) -> Optional[Dict[str, Any]]:
        """Attempt fuzzy matching fallback when exact query returns empty.

        Extracts search terms from the failed query, finds similar terms
        in the data, and retries with corrected terms.

        Args:
            original_query: The pandas query that returned empty
            question: The original natural language question

        Returns:
            New query result dict if fuzzy match succeeds, None otherwise
        """
        # Extract search terms from the query
        search_terms = extract_search_terms_from_query(original_query)
        if not search_terms:
            logger.debug("No search terms found in query for fuzzy fallback")
            return None

        # Detect which column was being searched
        column = detect_searchable_column(original_query)

        logger.info(
            f"Attempting fuzzy fallback for terms: {search_terms}, "
            f"column: {column or 'auto-detect'}"
        )

        # Try to find similar terms
        corrections = {}
        for term in search_terms:
            if column:
                # Search in detected column
                match = self.fuzzy_matcher.find_best_match(term, column)
                if match:
                    corrections[term] = match
            else:
                # Search across common columns
                suggestions = self.fuzzy_matcher.suggest_corrections(term)
                if suggestions:
                    # Take the best match across all columns
                    best_match = None
                    best_score = 0.0
                    for col, matches in suggestions.items():
                        if matches and matches[0][1] > best_score:
                            best_match = (col, matches[0])
                            best_score = matches[0][1]
                    if best_match:
                        corrections[term] = best_match[1]

        if not corrections:
            logger.info("No fuzzy matches found for search terms")
            return None

        # Log corrections found
        for original, (corrected, score) in corrections.items():
            logger.info(
                f"Fuzzy correction: '{original}' -> '{corrected}' "
                f"(similarity: {score:.3f})"
            )

        # Build corrected query by replacing terms
        corrected_query = original_query
        for original, (corrected, _) in corrections.items():
            # Replace in query (case-insensitive for _lower columns)
            corrected_query = corrected_query.replace(
                original.lower(), corrected.lower()
            )
            # Also try original case
            corrected_query = corrected_query.replace(original, corrected)

        if corrected_query == original_query:
            logger.debug("Corrected query same as original, skipping retry")
            return None

        logger.info(f"Retrying with corrected query: {corrected_query[:100]}...")

        # Execute corrected query
        exec_result = self.executor.execute(corrected_query)

        if exec_result["success"] and not self._is_empty_result(exec_result["result"]):
            return {
                "success": True,
                "answer": exec_result["result"],
                "query": corrected_query,
                "fuzzy_corrections": {
                    orig: corr for orig, (corr, _) in corrections.items()
                },
                "error": None,
            }

        return None
