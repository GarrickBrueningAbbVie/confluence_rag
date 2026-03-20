"""
Security utilities for the database module.

This module provides centralized security validation for generated
pandas queries. All query validation should use these utilities to
ensure consistent security checks across the codebase.

IMPORTANT: Any changes to security patterns must be made HERE ONLY.
This is the single source of truth for query validation.

Example:
    >>> from database.security import validate_query_security, is_safe_query
    >>> result = validate_query_security("df.head()")
    >>> if not result["valid"]:
    ...     print(f"Security issue: {result['issues']}")
"""

import re
from typing import Dict, List, Tuple

# =============================================================================
# Dangerous Patterns - SINGLE SOURCE OF TRUTH
# =============================================================================
# These patterns detect potentially malicious code in generated queries.
# Add new patterns here and they will apply across the entire codebase.

DANGEROUS_PATTERNS: List[Tuple[str, str]] = [
    # Code execution
    ("exec(", "Code execution via exec() is not allowed"),
    ("eval(", "Code execution via eval() is not allowed"),
    ("compile(", "Code compilation is not allowed"),

    # File operations
    ("open(", "File operations are not allowed"),
    ("read(", "File read operations are not allowed"),
    ("write(", "File write operations are not allowed"),

    # System access
    ("os.", "Operating system access is not allowed"),
    ("sys.", "System module access is not allowed"),
    ("subprocess", "Subprocess execution is not allowed"),
    ("shutil", "Shell utilities are not allowed"),

    # Import statements
    ("import ", "Import statements are not allowed"),
    ("__import__", "Dynamic imports are not allowed"),

    # Dunder access (potential for arbitrary code execution)
    ("__", "Dunder attribute access is not allowed"),

    # Network operations
    ("requests.", "Network requests are not allowed"),
    ("urllib", "URL operations are not allowed"),
    ("socket", "Socket operations are not allowed"),

    # Pickle (arbitrary code execution risk)
    ("pickle", "Pickle operations are not allowed"),

    # Lambda with dangerous patterns
    ("lambda x: x(", "Lambda calling functions is not allowed"),
]

# Patterns that should trigger warnings but not outright rejection
WARNING_PATTERNS: List[Tuple[str, str]] = [
    ("drop(", "Warning: drop() may modify data unexpectedly"),
    ("del ", "Warning: del statements may cause issues"),
    ("inplace=True", "Warning: inplace operations modify the original DataFrame"),
]

# =============================================================================
# Validation Functions
# =============================================================================


def validate_query_security(query: str) -> Dict[str, any]:
    """
    Validate a query for security issues.

    This is the primary validation function that should be called
    before executing any generated pandas query.

    Args:
        query: The pandas query string to validate

    Returns:
        Dict with:
        - valid (bool): Whether the query passed security checks
        - issues (List[str]): List of security issues found
        - warnings (List[str]): List of warnings (non-blocking)

    Example:
        >>> result = validate_query_security("df.head()")
        >>> print(result)
        {'valid': True, 'issues': [], 'warnings': []}

        >>> result = validate_query_security("import os; os.system('rm -rf /')")
        >>> print(result['valid'])
        False
    """
    issues: List[str] = []
    warnings: List[str] = []

    # Check dangerous patterns
    for pattern, message in DANGEROUS_PATTERNS:
        if pattern in query:
            issues.append(message)

    # Check warning patterns
    for pattern, message in WARNING_PATTERNS:
        if pattern in query:
            warnings.append(message)

    # Additional regex-based checks
    regex_issues = _check_regex_patterns(query)
    issues.extend(regex_issues)

    return {
        "valid": len(issues) == 0,
        "issues": issues,
        "warnings": warnings,
    }


def _check_regex_patterns(query: str) -> List[str]:
    """
    Check for patterns that require regex matching.

    Args:
        query: Query string to check

    Returns:
        List of issue messages
    """
    issues = []

    # Check for potential shell injection via string formatting
    shell_patterns = [
        (r'["\'].*\$\{.*\}.*["\']', "Potential shell variable injection"),
        (r'["\'].*`.*`.*["\']', "Potential command substitution"),
        (r';\s*\w+\s*\(', "Potential command chaining"),
    ]

    for pattern, message in shell_patterns:
        if re.search(pattern, query):
            issues.append(message)

    return issues


def is_safe_query(query: str) -> bool:
    """
    Quick check if a query is safe to execute.

    This is a convenience function for simple boolean checks.
    For detailed information, use validate_query_security().

    Args:
        query: The pandas query string to validate

    Returns:
        True if query is safe, False otherwise

    Example:
        >>> is_safe_query("df.head()")
        True
        >>> is_safe_query("import os")
        False
    """
    result = validate_query_security(query)
    return result["valid"]


def sanitize_query(query: str) -> str:
    """
    Attempt to sanitize a query by removing dangerous content.

    WARNING: This is a best-effort sanitization and should NOT be
    relied upon for security. Always validate after sanitization.

    Args:
        query: The query to sanitize

    Returns:
        Sanitized query string

    Example:
        >>> sanitize_query("import os; df.head()")
        'df.head()'
    """
    # Remove import statements
    query = re.sub(r'^import\s+\w+.*$', '', query, flags=re.MULTILINE)
    query = re.sub(r'^from\s+\w+.*$', '', query, flags=re.MULTILINE)

    # Remove semicolon-separated commands (keep only first valid)
    if ';' in query:
        parts = query.split(';')
        safe_parts = []
        for part in parts:
            part = part.strip()
            if part and is_safe_query(part):
                safe_parts.append(part)
        query = safe_parts[0] if safe_parts else ""

    return query.strip()


def must_reference_dataframe(query: str, df_name: str = "df") -> bool:
    """
    Check if a query references the expected DataFrame variable.

    Ensures the query operates on the expected data source.

    Args:
        query: Query string to check
        df_name: Expected DataFrame variable name

    Returns:
        True if query references the DataFrame

    Example:
        >>> must_reference_dataframe("df.head()", "df")
        True
        >>> must_reference_dataframe("other_df.head()", "df")
        False
    """
    # Check for df reference at start of expression or after operators
    pattern = rf'(?:^|[=\(\[\s]){df_name}[.\[\s]'
    return bool(re.search(pattern, query))


# =============================================================================
# Query Extraction Helpers
# =============================================================================


def extract_query_from_response(response: str, df_name: str = "df") -> str:
    """
    Extract a pandas query from an LLM response.

    Handles common LLM response formats including markdown code blocks.

    Args:
        response: Raw LLM response text
        df_name: Expected DataFrame variable name

    Returns:
        Extracted query string, or empty string if not found

    Example:
        >>> response = "```python\\ndf.head()\\n```"
        >>> extract_query_from_response(response)
        'df.head()'
    """
    query = response.strip()

    # Remove markdown code blocks
    if query.startswith("```python"):
        query = query[9:]
    elif query.startswith("```"):
        query = query[3:]

    if query.endswith("```"):
        query = query[:-3]

    query = query.strip()

    # Handle variable assignments (e.g., "result = df.head()")
    if "=" in query:
        match = re.search(rf'=\s*({df_name}[.\[].*)', query)
        if match:
            query = match.group(1).strip()

    # Handle multiple statements - take last one referencing df
    if ";" in query:
        parts = [p.strip() for p in query.split(";") if p.strip()]
        for part in reversed(parts):
            if must_reference_dataframe(part, df_name):
                query = part
                break

    return query
