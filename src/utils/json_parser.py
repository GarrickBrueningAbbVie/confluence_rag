"""
JSON parsing utilities for LLM responses.

This module provides robust JSON parsing that handles common LLM response
formats including markdown code blocks and malformed JSON.

Example:
    >>> from utils.json_parser import parse_llm_json_response
    >>> response = '''```json
    ... {"key": "value"}
    ... ```'''
    >>> result = parse_llm_json_response(response)
    >>> print(result)  # {'key': 'value'}
"""

import json
import re
from typing import Optional

from loguru import logger


def parse_llm_json_response(content: str) -> Optional[dict]:
    """
    Parse JSON from an LLM response, handling common formats.

    Handles:
    - Raw JSON strings
    - Markdown code blocks (```json ... ```)
    - JSON embedded in other text

    Args:
        content: Raw LLM response string

    Returns:
        Parsed dictionary or None if parsing fails

    Example:
        >>> parse_llm_json_response('{"intent": "rag"}')
        {'intent': 'rag'}
        >>> parse_llm_json_response('```json\\n{"intent": "rag"}\\n```')
        {'intent': 'rag'}
    """
    if not content:
        return None

    content = content.strip()

    # Handle markdown code blocks
    if content.startswith("```"):
        content = _strip_markdown_code_block(content)

    # Try direct JSON parsing
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        pass

    # Try to extract JSON object from text
    extracted = extract_json_from_text(content)
    if extracted is not None:
        return extracted

    logger.warning(f"Could not parse LLM response as JSON: {content[:200]}...")
    return None


def extract_json_from_text(text: str) -> Optional[dict]:
    """
    Extract a JSON object from text that may contain other content.

    Uses regex to find JSON-like patterns and attempts to parse them.

    Args:
        text: Text potentially containing JSON

    Returns:
        Parsed dictionary or None if no valid JSON found

    Example:
        >>> extract_json_from_text('The result is: {"count": 5}')
        {'count': 5}
    """
    if not text:
        return None

    # Try to find JSON object pattern
    # Match from first { to last }
    json_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'

    matches = re.findall(json_pattern, text, re.DOTALL)

    for match in matches:
        try:
            return json.loads(match)
        except json.JSONDecodeError:
            continue

    # Try a more permissive pattern for nested objects
    start_idx = text.find('{')
    if start_idx == -1:
        return None

    # Find matching closing brace
    brace_count = 0
    end_idx = start_idx

    for i, char in enumerate(text[start_idx:], start=start_idx):
        if char == '{':
            brace_count += 1
        elif char == '}':
            brace_count -= 1
            if brace_count == 0:
                end_idx = i + 1
                break

    if end_idx > start_idx:
        candidate = text[start_idx:end_idx]
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass

    return None


def _strip_markdown_code_block(content: str) -> str:
    """
    Remove markdown code block markers from content.

    Args:
        content: Content potentially wrapped in code blocks

    Returns:
        Content with code block markers removed
    """
    lines = content.split("\n")

    # Find start and end of actual content
    start_idx = 0
    end_idx = len(lines)

    # Skip opening ``` line
    if lines and lines[0].strip().startswith("```"):
        start_idx = 1

    # Skip closing ``` line
    for i in range(len(lines) - 1, -1, -1):
        if lines[i].strip() == "```":
            end_idx = i
            break

    return "\n".join(lines[start_idx:end_idx]).strip()


def safe_json_loads(content: str, default: Optional[dict] = None) -> dict:
    """
    Safely parse JSON with a default fallback.

    Args:
        content: JSON string to parse
        default: Default value if parsing fails (defaults to empty dict)

    Returns:
        Parsed dictionary or default value
    """
    result = parse_llm_json_response(content)
    if result is None:
        return default if default is not None else {}
    return result
