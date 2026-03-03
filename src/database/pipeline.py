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

        logger.info(f"Initialized DatabasePipeline with {len(self.df)} pages")

    def query(
        self,
        question: str,
        return_query: bool = True,
    ) -> Dict[str, Any]:
        """
        Execute a natural language query.

        Generates a pandas query from the question, executes it,
        and returns the results.

        Args:
            question: Natural language question
            return_query: Whether to include generated query in response

        Returns:
            Dict with:
            - success: Whether query succeeded
            - answer: Query result
            - query: Generated pandas query (if return_query=True)
            - error: Error message (if failed)

        Example:
            >>> result = pipeline.query("How many pages use Airflow?")
            >>> print(result['answer'])
        """
        result = {
            "success": False,
            "answer": None,
            "query": None,
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
        logger.info(f"Reloaded data: {len(self.df)} pages")
