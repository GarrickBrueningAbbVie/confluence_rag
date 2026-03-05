"""
Database pipeline for structured queries on Confluence data.

This package provides pandas-based querying capabilities for
structured data extracted from Confluence pages.

Example:
    >>> from database import DatabasePipeline
    >>> pipeline = DatabasePipeline("Data_Storage/confluence_pages.json")
    >>> result = pipeline.query("How many pages use Python?")
"""

from .dataframe_loader import DataFrameLoader
from .query_generator import QueryGenerator
from .query_executor import QueryExecutor
from .pipeline import DatabasePipeline

__all__ = [
    "DataFrameLoader",
    "QueryGenerator",
    "QueryExecutor",
    "DatabasePipeline",
]
