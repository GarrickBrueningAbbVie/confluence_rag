"""Custom template filters for RAG display.

This module provides template filters for formatting:
- Markdown content
- Source documents
- Relevance scores
"""

import json
from typing import Any, Dict, List, Union

from django import template
from django.utils.safestring import mark_safe

register = template.Library()


@register.filter
def to_json(value: Any) -> str:
    """Convert a Python object to JSON string.

    Args:
        value: The value to convert.

    Returns:
        str: JSON string representation.
    """
    return json.dumps(value)


@register.filter
def format_score(value: float) -> str:
    """Format a relevance score as a percentage.

    Args:
        value: The relevance score (0-1).

    Returns:
        str: Formatted percentage string.
    """
    if value is None:
        return "N/A"
    return f"{value * 100:.1f}%"


@register.filter
def truncate_text(value: str, length: int = 100) -> str:
    """Truncate text to a maximum length.

    Args:
        value: The text to truncate.
        length: Maximum length (default 100).

    Returns:
        str: Truncated text with ellipsis if needed.
    """
    if not value:
        return ""
    if len(value) <= length:
        return value
    return value[:length] + "..."


@register.filter
def intent_badge_class(intent: str) -> str:
    """Get the CSS class for an intent badge.

    Args:
        intent: The query intent (rag, database, hybrid, smart, chart).

    Returns:
        str: CSS class name for the intent badge.
    """
    intent_classes = {
        "rag": "intent-rag",
        "database": "intent-database",
        "hybrid": "intent-hybrid",
        "smart": "intent-smart",
        "chart": "intent-chart",
    }
    return intent_classes.get(intent.lower(), "intent-default")


@register.simple_tag
def get_source_count(sources: List[Dict]) -> int:
    """Count unique sources by URL.

    Args:
        sources: List of source dictionaries.

    Returns:
        int: Number of unique sources.
    """
    if not sources:
        return 0
    unique_urls = set(s.get("url", "") for s in sources)
    return len(unique_urls)
