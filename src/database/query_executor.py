"""
Safe pandas query executor.

This module provides sandboxed execution of generated pandas queries
with security restrictions and error handling.

Example:
    >>> from database.query_executor import QueryExecutor
    >>> executor = QueryExecutor(df)
    >>> result = executor.execute("df['created_by'].value_counts().head(5)")
"""

import pandas as pd
from typing import Any, Dict, Optional

from loguru import logger


class QueryExecutor:
    """
    Safely execute pandas queries.

    Provides a sandboxed environment for executing generated pandas
    queries with restricted globals and error handling.

    Attributes:
        df: DataFrame to query against
        max_result_rows: Maximum rows to return

    Example:
        >>> executor = QueryExecutor(df)
        >>> result = executor.execute("df.shape[0]")
        >>> print(result['result'])
    """

    # Allowed modules/functions in execution context
    ALLOWED_BUILTINS = {
        "len": len,
        "str": str,
        "int": int,
        "float": float,
        "list": list,
        "dict": dict,
        "set": set,
        "sorted": sorted,
        "sum": sum,
        "min": min,
        "max": max,
        "abs": abs,
        "round": round,
        "enumerate": enumerate,
        "zip": zip,
        "isinstance": isinstance,
        "True": True,
        "False": False,
        "None": None,
    }

    def __init__(
        self,
        df: pd.DataFrame,
        max_result_rows: int = 1000,
    ) -> None:
        """Initialize executor.

        Args:
            df: DataFrame to query against
            max_result_rows: Maximum rows to return in results
        """
        self.df = df.copy()  # Work with a copy for safety
        self.max_result_rows = max_result_rows

        logger.info(f"Initialized QueryExecutor with {len(df)} rows")

    def execute(
        self,
        query: str,
        timeout_seconds: float = 30.0,
    ) -> Dict[str, Any]:
        """
        Execute a pandas query safely.

        Args:
            query: Pandas query string to execute
            timeout_seconds: Maximum execution time (not enforced, for logging)

        Returns:
            Dict with:
            - success: Whether execution succeeded
            - result: Query result (if successful)
            - result_type: Type of result (scalar, list, dict, dataframe)
            - error: Error message (if failed)
            - row_count: Number of result rows (for DataFrame results)

        Example:
            >>> result = executor.execute("df['title'].head(5).tolist()")
            >>> if result['success']:
            ...     print(result['result'])
        """
        result = {
            "success": False,
            "result": None,
            "result_type": None,
            "error": None,
            "row_count": None,
        }

        # Validate query first
        validation = self._validate_query(query)
        if not validation["valid"]:
            result["error"] = f"Query validation failed: {', '.join(validation['issues'])}"
            logger.warning(f"Query validation failed: {validation['issues']}")
            return result

        try:
            # Create restricted execution context
            exec_globals = self._create_execution_context()

            # Execute query
            logger.debug(f"Executing query: {query[:100]}...")
            query_result = eval(query, exec_globals, {})

            # Process result
            result["result"], result["result_type"] = self._process_result(query_result)
            result["success"] = True

            if result["result_type"] == "dataframe":
                result["row_count"] = len(query_result)

            logger.info(f"Query executed successfully, result type: {result['result_type']}")

        except SyntaxError as e:
            result["error"] = f"Syntax error in query: {e}"
            logger.error(f"Syntax error: {e}")

        except NameError as e:
            result["error"] = f"Unknown variable or function: {e}"
            logger.error(f"Name error: {e}")

        except KeyError as e:
            result["error"] = f"Column not found: {e}"
            logger.error(f"Key error: {e}")

        except Exception as e:
            result["error"] = f"Query execution failed: {type(e).__name__}: {e}"
            logger.error(f"Execution error: {e}")

        return result

    def _validate_query(self, query: str) -> Dict[str, Any]:
        """Validate query for safety before execution.

        Args:
            query: Query string to validate

        Returns:
            Dict with valid (bool) and issues (list)
        """
        issues = []

        # Check for dangerous patterns
        dangerous_patterns = [
            ("import ", "Import statements not allowed"),
            ("__", "Dunder methods not allowed"),
            ("exec(", "exec() not allowed"),
            ("eval(", "Nested eval() not allowed"),
            ("open(", "File operations not allowed"),
            ("os.", "OS module not allowed"),
            ("sys.", "sys module not allowed"),
            ("subprocess", "subprocess not allowed"),
            ("globals(", "globals() not allowed"),
            ("locals(", "locals() not allowed"),
            ("compile(", "compile() not allowed"),
            ("getattr(", "getattr() not allowed"),
            ("setattr(", "setattr() not allowed"),
            ("delattr(", "delattr() not allowed"),
        ]

        for pattern, message in dangerous_patterns:
            if pattern in query:
                issues.append(message)

        # Check query references df
        if "df" not in query:
            issues.append("Query must reference 'df'")

        # Check query length
        if len(query) > 2000:
            issues.append("Query too long (max 2000 chars)")

        return {
            "valid": len(issues) == 0,
            "issues": issues,
        }

    def _create_execution_context(self) -> Dict[str, Any]:
        """Create restricted globals for query execution.

        Returns:
            Dictionary of allowed names for eval()
        """
        context = {
            "__builtins__": self.ALLOWED_BUILTINS,
            "df": self.df,
            "pd": pd,
        }

        return context

    def _process_result(self, query_result: Any) -> tuple:
        """Process query result into serializable format.

        Args:
            query_result: Raw result from query execution

        Returns:
            Tuple of (processed_result, result_type)
        """
        # Handle None
        if query_result is None:
            return None, "none"

        # Handle scalar values
        if isinstance(query_result, (int, float, str, bool)):
            return query_result, "scalar"

        # Handle pandas Series
        if isinstance(query_result, pd.Series):
            # Convert to list or dict depending on index
            if query_result.index.is_numeric():
                result = query_result.head(self.max_result_rows).tolist()
            else:
                result = query_result.head(self.max_result_rows).to_dict()
            return result, "series"

        # Handle pandas DataFrame
        if isinstance(query_result, pd.DataFrame):
            result = query_result.head(self.max_result_rows).to_dict("records")
            return result, "dataframe"

        # Handle lists
        if isinstance(query_result, list):
            if len(query_result) > self.max_result_rows:
                query_result = query_result[: self.max_result_rows]
            return query_result, "list"

        # Handle dicts
        if isinstance(query_result, dict):
            return query_result, "dict"

        # Handle sets
        if isinstance(query_result, set):
            return list(query_result), "set"

        # Fallback: convert to string
        return str(query_result), "string"

    def get_available_columns(self) -> list:
        """Get list of available DataFrame columns.

        Returns:
            List of column names
        """
        return self.df.columns.tolist()

    def get_sample_data(self, n: int = 5) -> list:
        """Get sample rows from DataFrame.

        Args:
            n: Number of rows to return

        Returns:
            List of row dictionaries
        """
        return self.df.head(n).to_dict("records")
