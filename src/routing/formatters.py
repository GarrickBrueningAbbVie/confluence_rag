"""
Formatting utilities for query results.

This module provides shared formatting functions for displaying
database results, lists, and other structured data in a human-readable format.

Example:
    >>> from routing.formatters import format_db_answer
    >>> result = [{"name": "Project A"}, {"name": "Project B"}]
    >>> print(format_db_answer(result))
    - name: Project A
    - name: Project B
"""

from typing import Any, Dict, List


def format_db_answer(answer: Any, max_items: int = 10) -> str:
    """Format database result as readable text.

    Handles various result types: numbers, strings, lists, and dicts.
    Truncates long lists for readability.

    Args:
        answer: Raw database result (can be int, float, str, list, dict)
        max_items: Maximum items to show before truncating

    Returns:
        Formatted string suitable for display

    Example:
        >>> format_db_answer(42)
        '42'
        >>> format_db_answer([{"name": "A"}, {"name": "B"}])
        '- name: A\\n- name: B'
    """
    if answer is None:
        return "No results found."

    if isinstance(answer, (int, float)):
        return str(answer)

    if isinstance(answer, str):
        return answer

    if isinstance(answer, list):
        return format_list_result(answer, max_items)

    if isinstance(answer, dict):
        return format_dict_result(answer)

    return str(answer)


def format_list_result(items: List[Any], max_items: int = 10) -> str:
    """Format a list of items for display.

    Args:
        items: List of items (can be dicts, strings, or other types)
        max_items: Maximum items to show

    Returns:
        Formatted string with bullet points

    Example:
        >>> format_list_result(["item1", "item2", "item3"], max_items=2)
        '- item1\\n- item2\\n... and 1 more items'
    """
    if not items:
        return "No results found."

    if len(items) <= max_items:
        return _format_items(items)

    # Truncate long lists
    preview = items[:max_items]
    formatted = _format_items(preview)
    formatted += f"\n... and {len(items) - max_items} more items"
    return formatted


def _format_items(items: List[Any]) -> str:
    """Format a list of items as bullet points.

    Args:
        items: List of items

    Returns:
        Formatted string
    """
    if not items:
        return ""

    if isinstance(items[0], dict):
        lines = []
        for item in items:
            line = ", ".join(f"{k}: {v}" for k, v in item.items())
            lines.append(f"- {line}")
        return "\n".join(lines)

    return "\n".join(f"- {item}" for item in items)


def format_dict_result(data: Dict[str, Any]) -> str:
    """Format a dictionary for display.

    Args:
        data: Dictionary to format

    Returns:
        Formatted string with key-value pairs

    Example:
        >>> format_dict_result({"count": 5, "status": "ok"})
        '- count: 5\\n- status: ok'
    """
    if not data:
        return "No results found."

    lines = [f"- {k}: {v}" for k, v in data.items()]
    return "\n".join(lines)


def format_single_answer(answer: Any, max_list_items: int = 20) -> str:
    """Format a single result answer for aggregation.

    Similar to format_db_answer but with higher default limits
    for aggregation contexts.

    Args:
        answer: Result answer (can be various types)
        max_list_items: Maximum list items to show

    Returns:
        Formatted string
    """
    if isinstance(answer, str):
        return answer

    if isinstance(answer, (int, float)):
        return str(answer)

    if isinstance(answer, list):
        if not answer:
            return "No results found."

        if isinstance(answer[0], dict):
            lines = []
            for item in answer[:max_list_items]:
                line = ", ".join(f"{k}: {v}" for k, v in item.items())
                lines.append(f"- {line}")
            if len(answer) > max_list_items:
                lines.append(f"... and {len(answer) - max_list_items} more")
            return "\n".join(lines)

        return "\n".join(f"- {item}" for item in answer[:max_list_items])

    if isinstance(answer, dict):
        return "\n".join(f"- {k}: {v}" for k, v in answer.items())

    return str(answer)


def format_sources(sources: List[Dict[str, Any]], max_sources: int = 5) -> str:
    """Format source citations for display.

    Args:
        sources: List of source dictionaries with 'title' and 'url' keys
        max_sources: Maximum sources to show

    Returns:
        Formatted markdown string with links

    Example:
        >>> sources = [{"title": "Doc 1", "url": "http://example.com"}]
        >>> print(format_sources(sources))
        **Sources:**
        1. [Doc 1](http://example.com)
    """
    if not sources:
        return ""

    output = "**Sources:**\n"
    for i, source in enumerate(sources[:max_sources], 1):
        title = source.get("title", "Unknown")
        url = source.get("url", "")
        if url:
            output += f"{i}. [{title}]({url})\n"
        else:
            output += f"{i}. {title}\n"

    return output


def truncate_text(text: str, max_length: int = 100, suffix: str = "...") -> str:
    """Truncate text to a maximum length.

    Args:
        text: Text to truncate
        max_length: Maximum length before truncation
        suffix: Suffix to add if truncated

    Returns:
        Truncated text with suffix if needed
    """
    if len(text) <= max_length:
        return text
    return text[: max_length - len(suffix)] + suffix
