"""
Visualization generation for data queries.

This package provides chart generation capabilities using
Iliad API for code generation and Plotly for rendering.

Example:
    >>> from visualization import ChartGenerator
    >>> generator = ChartGenerator(iliad_client)
    >>> fig = generator.generate("Show pages by author", data)
"""

from .chart_generator import ChartGenerator
from .code_executor import CodeExecutor

__all__ = [
    "ChartGenerator",
    "CodeExecutor",
]
