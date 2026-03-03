"""
Safe code executor for chart generation.

This module provides sandboxed execution of generated Plotly code
with security restrictions.

Example:
    >>> from visualization.code_executor import CodeExecutor
    >>> executor = CodeExecutor()
    >>> result = executor.execute(code, data)
"""

from typing import Any, Dict, List, Optional, Union

from loguru import logger


class CodeExecutor:
    """
    Safely execute generated Plotly code.

    Provides a sandboxed environment for executing chart generation
    code with restricted globals.

    Attributes:
        max_execution_time: Maximum execution time in seconds

    Example:
        >>> executor = CodeExecutor()
        >>> result = executor.execute(code, data)
        >>> if result['success']:
        ...     result['figure'].show()
    """

    # Blocked patterns in code
    BLOCKED_PATTERNS = [
        "import os",
        "import sys",
        "import subprocess",
        "import shutil",
        "__import__",
        "exec(",
        "eval(",
        "open(",
        "file(",
        "compile(",
        "globals(",
        "locals(",
        "getattr(",
        "setattr(",
        "delattr(",
        "breakpoint(",
        "input(",
        "help(",
    ]

    def __init__(
        self,
        max_execution_time: float = 30.0,
    ) -> None:
        """Initialize code executor.

        Args:
            max_execution_time: Maximum execution time (not enforced, for logging)
        """
        self.max_execution_time = max_execution_time

        logger.info("Initialized CodeExecutor")

    def execute(
        self,
        code: str,
        data: Union[Dict, List, Any],
    ) -> Dict[str, Any]:
        """
        Execute generated Plotly code.

        Args:
            code: Python code to execute
            data: Data to make available to the code

        Returns:
            Dict with:
            - success: Whether execution succeeded
            - figure: Plotly figure (if successful)
            - html: HTML representation
            - error: Error message (if failed)

        Example:
            >>> result = executor.execute("fig = px.bar(data)", data)
        """
        result = {
            "success": False,
            "figure": None,
            "html": None,
            "error": None,
        }

        # Validate code
        validation = self._validate_code(code)
        if not validation["valid"]:
            result["error"] = f"Code validation failed: {validation['reason']}"
            logger.warning(f"Code validation failed: {validation['reason']}")
            return result

        try:
            # Create execution context
            exec_globals = self._create_execution_context(data)
            exec_locals = {}

            # Execute code
            logger.debug("Executing generated code...")
            exec(code, exec_globals, exec_locals)

            # Extract figure
            fig = exec_locals.get("fig")

            if fig is None:
                # Try to find figure in globals
                fig = exec_globals.get("fig")

            if fig is None:
                result["error"] = "No figure (fig) was created by the code"
                return result

            # Validate it's a Plotly figure
            import plotly.graph_objects as go
            if not isinstance(fig, go.Figure):
                result["error"] = f"Result is not a Plotly figure: {type(fig)}"
                return result

            result["success"] = True
            result["figure"] = fig
            result["html"] = fig.to_html(full_html=False, include_plotlyjs="cdn")

            logger.info("Code execution successful")

        except SyntaxError as e:
            result["error"] = f"Syntax error: {e}"
            logger.error(f"Syntax error in generated code: {e}")

        except NameError as e:
            result["error"] = f"Name error: {e}"
            logger.error(f"Name error: {e}")

        except ImportError as e:
            result["error"] = f"Import error: {e}"
            logger.error(f"Import error: {e}")

        except Exception as e:
            result["error"] = f"Execution error: {type(e).__name__}: {e}"
            logger.error(f"Execution error: {e}")

        return result

    def _validate_code(self, code: str) -> Dict[str, Any]:
        """Validate code for safety.

        Args:
            code: Code to validate

        Returns:
            Dict with valid (bool) and reason (str)
        """
        # Check for blocked patterns
        code_lower = code.lower()

        for pattern in self.BLOCKED_PATTERNS:
            if pattern.lower() in code_lower:
                return {
                    "valid": False,
                    "reason": f"Blocked pattern: {pattern}",
                }

        # Check for dunder methods
        if "__" in code and "__name__" not in code:
            return {
                "valid": False,
                "reason": "Dunder methods not allowed",
            }

        # Check code length
        if len(code) > 10000:
            return {
                "valid": False,
                "reason": "Code too long (max 10000 chars)",
            }

        # Check it actually uses Plotly
        if "plotly" not in code.lower() and "px." not in code and "go." not in code:
            return {
                "valid": False,
                "reason": "Code must use Plotly library",
            }

        return {"valid": True, "reason": ""}

    def _create_execution_context(
        self,
        data: Union[Dict, List, Any],
    ) -> Dict[str, Any]:
        """Create restricted execution context.

        Args:
            data: Data to include in context

        Returns:
            Dictionary of allowed names
        """
        import pandas as pd
        import numpy as np
        import plotly.express as px
        import plotly.graph_objects as go
        import json

        # Restricted builtins
        safe_builtins = {
            "len": len,
            "str": str,
            "int": int,
            "float": float,
            "list": list,
            "dict": dict,
            "set": set,
            "tuple": tuple,
            "sorted": sorted,
            "sum": sum,
            "min": min,
            "max": max,
            "abs": abs,
            "round": round,
            "enumerate": enumerate,
            "zip": zip,
            "range": range,
            "isinstance": isinstance,
            "True": True,
            "False": False,
            "None": None,
            "print": lambda *args, **kwargs: None,  # Suppress output
        }

        context = {
            "__builtins__": safe_builtins,
            "data": data,
            "pd": pd,
            "np": np,
            "px": px,
            "go": go,
            "json": json,
        }

        return context

    def test_plotly_available(self) -> bool:
        """Test if Plotly is available.

        Returns:
            True if Plotly is installed
        """
        try:
            import plotly.express as px
            import plotly.graph_objects as go
            return True
        except ImportError:
            return False
