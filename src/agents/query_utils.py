"""
Shared utilities for agent query processing.

This module provides common utilities used across multiple agents,
reducing code duplication and ensuring consistent behavior.

Example:
    >>> from agents.query_utils import enhance_query_with_context
    >>> context = {"project": "ALFA", "items": ["a", "b", "c"]}
    >>> query = "Find info about {project}"
    >>> enhance_query_with_context(query, context)
    'Find info about ALFA'
"""

from typing import Any, Dict, Optional


def enhance_query_with_context(
    query: str,
    intermediate_results: Dict[str, Any],
    max_value_length: Optional[int] = None,
    list_limit: int = 20,
) -> str:
    """Inject intermediate results into query using {placeholder} syntax.

    Replaces placeholders in the query string with values from
    intermediate_results. Supports truncation and list limiting.

    Args:
        query: Original query (may contain {placeholders})
        intermediate_results: Dict of key-value pairs for substitution
        max_value_length: Optional max length for values (truncates with ...)
        list_limit: Max items for list values (default 20)

    Returns:
        Enhanced query with placeholders filled

    Example:
        >>> results = {"project": "ALFA", "items": ["a", "b", "c"]}
        >>> enhance_query_with_context("About {project}", results)
        'About ALFA'
        >>> enhance_query_with_context("Items: {items}", results)
        'Items: a, b, c'
    """
    enhanced = query

    for key, value in intermediate_results.items():
        placeholder = "{" + key + "}"
        if placeholder in enhanced:
            # Convert lists to comma-separated strings
            if isinstance(value, list):
                value_str = ", ".join(str(v) for v in value[:list_limit])
            else:
                value_str = str(value)

            # Apply truncation if specified
            if max_value_length and len(value_str) > max_value_length:
                value_str = value_str[:max_value_length] + "..."

            enhanced = enhanced.replace(placeholder, value_str)

    return enhanced
