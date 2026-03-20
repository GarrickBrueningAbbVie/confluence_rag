"""
Fuzzy string matching for database query fallback.

This module provides fuzzy matching capabilities to handle typos and
misspellings in database queries. When an exact match returns no results,
fuzzy matching can find similar terms in the data.

Uses embedding similarity for semantic matching, with caching for performance.

Example:
    >>> from database.fuzzy_matcher import FuzzyMatcher
    >>> matcher = FuzzyMatcher(df)
    >>> similar = matcher.find_similar_terms('xgbost', 'technologies')
    >>> print(similar)  # [('xgboost', 0.95)]
"""

import re
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
from loguru import logger


class FuzzyMatcher:
    """
    Fuzzy string matcher for database query fallback.

    Uses embedding similarity to find terms that are semantically similar
    to a search term, helping to handle typos and misspellings.

    Caches unique values and their embeddings per column for performance.

    Attributes:
        df: The DataFrame to search in
        embedding_manager: EmbeddingManager for generating embeddings
        _cache: Cache of unique values and embeddings per column

    Example:
        >>> matcher = FuzzyMatcher(df)
        >>> matches = matcher.find_similar_terms('pythn', 'technologies')
        >>> print(matches)  # [('python', 0.92), ('pytorch', 0.85)]
    """

    def __init__(
        self,
        df: Any,
        embedding_manager: Optional[Any] = None,
        similarity_threshold: float = 0.7,
    ) -> None:
        """Initialize fuzzy matcher.

        Args:
            df: DataFrame containing the data
            embedding_manager: Optional EmbeddingManager instance.
                              If not provided, will create one lazily.
            similarity_threshold: Minimum similarity score (0-1) for matches
        """
        self.df = df
        self._embedding_manager = embedding_manager
        self.similarity_threshold = similarity_threshold

        # Cache: {column_name: {'values': [...], 'embeddings': np.array}}
        self._cache: Dict[str, Dict[str, Any]] = {}

        logger.info(
            f"Initialized FuzzyMatcher (threshold={similarity_threshold})"
        )

    @property
    def embedding_manager(self) -> Any:
        """Lazy initialization of embedding manager."""
        if self._embedding_manager is None:
            from rag.embeddings import EmbeddingManager
            self._embedding_manager = EmbeddingManager()
            logger.info("Created EmbeddingManager for FuzzyMatcher")
        return self._embedding_manager

    def _get_unique_values(self, column: str) -> List[str]:
        """Get unique values from a column, handling list columns.

        Args:
            column: Column name to extract values from

        Returns:
            List of unique string values
        """
        if column not in self.df.columns:
            logger.warning(f"Column '{column}' not found in DataFrame")
            return []

        values: Set[str] = set()

        for val in self.df[column].dropna():
            if isinstance(val, list):
                # Handle list columns like 'technologies'
                for item in val:
                    if isinstance(item, str) and item.strip():
                        values.add(item.strip())
            elif isinstance(val, str) and val.strip():
                values.add(val.strip())

        return sorted(values)

    def _ensure_column_cached(self, column: str) -> None:
        """Ensure column values and embeddings are cached.

        Args:
            column: Column name to cache
        """
        if column in self._cache:
            return

        logger.info(f"Caching embeddings for column '{column}'")

        values = self._get_unique_values(column)

        if not values:
            self._cache[column] = {'values': [], 'embeddings': np.array([])}
            return

        # Generate embeddings for all unique values
        embeddings = self.embedding_manager.generate_embeddings(
            values, show_progress=False
        )

        self._cache[column] = {
            'values': values,
            'embeddings': embeddings,
        }

        logger.info(
            f"Cached {len(values)} unique values for column '{column}'"
        )

    def find_similar_terms(
        self,
        search_term: str,
        column: str,
        top_k: int = 5,
        threshold: Optional[float] = None,
    ) -> List[Tuple[str, float]]:
        """Find terms similar to the search term in a column.

        Args:
            search_term: The term to find similar matches for
            column: Column to search in
            top_k: Maximum number of matches to return
            threshold: Similarity threshold (uses default if not specified)

        Returns:
            List of (term, similarity_score) tuples, sorted by similarity desc
        """
        if not search_term or not search_term.strip():
            return []

        threshold = threshold if threshold is not None else self.similarity_threshold

        # Ensure column is cached
        self._ensure_column_cached(column)

        cache = self._cache.get(column)
        if not cache or not cache['values']:
            return []

        values = cache['values']
        embeddings = cache['embeddings']

        # Generate embedding for search term
        search_embedding = self.embedding_manager.generate_embedding(search_term)

        # Compute similarities
        similarities = []
        for i, (value, embedding) in enumerate(zip(values, embeddings)):
            sim = self.embedding_manager.compute_similarity(
                search_embedding, embedding
            )
            if sim >= threshold:
                similarities.append((value, sim))

        # Sort by similarity descending
        similarities.sort(key=lambda x: x[1], reverse=True)

        # Return top_k
        results = similarities[:top_k]

        if results:
            logger.info(
                f"Found {len(results)} similar terms for '{search_term}' in '{column}': "
                f"{[(t, f'{s:.3f}') for t, s in results[:3]]}"
            )

        return results

    def find_best_match(
        self,
        search_term: str,
        column: str,
        threshold: Optional[float] = None,
    ) -> Optional[Tuple[str, float]]:
        """Find the single best matching term.

        Args:
            search_term: The term to find a match for
            column: Column to search in
            threshold: Similarity threshold

        Returns:
            (term, similarity) tuple or None if no match above threshold
        """
        matches = self.find_similar_terms(
            search_term, column, top_k=1, threshold=threshold
        )
        return matches[0] if matches else None

    def suggest_corrections(
        self,
        search_term: str,
        columns: Optional[List[str]] = None,
    ) -> Dict[str, List[Tuple[str, float]]]:
        """Find similar terms across multiple columns.

        Args:
            search_term: The term to find corrections for
            columns: Columns to search in. Defaults to common searchable columns.

        Returns:
            Dict mapping column names to list of (term, similarity) matches
        """
        if columns is None:
            # Default searchable columns
            columns = [
                'technologies',
                'parent_project',
                'created_by',
                'title',
            ]

        results = {}
        for col in columns:
            # Try both original and _lower version
            matches = self.find_similar_terms(search_term, col)
            if matches:
                results[col] = matches

        return results

    def clear_cache(self, column: Optional[str] = None) -> None:
        """Clear cached embeddings.

        Args:
            column: Specific column to clear, or None for all
        """
        if column:
            self._cache.pop(column, None)
            logger.info(f"Cleared cache for column '{column}'")
        else:
            self._cache.clear()
            logger.info("Cleared all FuzzyMatcher cache")


def extract_search_terms_from_query(query: str) -> List[str]:
    """Extract string literals from a pandas query.

    Finds quoted strings and values being compared in the query.

    Args:
        query: Pandas query string

    Returns:
        List of search terms found in the query

    Example:
        >>> extract_search_terms_from_query("df[df['col'] == 'xgbost']")
        ['xgbost']
        >>> extract_search_terms_from_query("df[df['col'].str.contains('pyth')]")
        ['pyth']
    """
    terms = []

    # Pattern 1: Single quoted strings
    single_quotes = re.findall(r"'([^']+)'", query)
    terms.extend(single_quotes)

    # Pattern 2: Double quoted strings
    double_quotes = re.findall(r'"([^"]+)"', query)
    terms.extend(double_quotes)

    # Filter out column names and pandas keywords
    pandas_keywords = {
        'records', 'index', 'columns', 'values', 'na', 'nan',
        'ascending', 'descending', 'inplace', 'axis',
    }

    # Filter out things that look like column names (contain _lower, etc.)
    filtered = []
    for term in terms:
        term_lower = term.lower()
        # Skip pandas keywords
        if term_lower in pandas_keywords:
            continue
        # Skip column references (contain _ and common suffixes)
        if '_lower' in term or '_id' in term:
            continue
        # Skip very short terms (likely operators)
        if len(term) < 2:
            continue
        filtered.append(term)

    # Deduplicate while preserving order
    seen = set()
    unique = []
    for term in filtered:
        if term.lower() not in seen:
            seen.add(term.lower())
            unique.append(term)

    return unique


def detect_searchable_column(query: str) -> Optional[str]:
    """Detect which column is being searched in a query.

    Args:
        query: Pandas query string

    Returns:
        Column name being searched, or None if not detected
    """
    # Pattern: df['column_name'] or df["column_name"]
    col_match = re.search(r"df\[(['\"])(\w+)\1\]", query)
    if col_match:
        return col_match.group(2)

    # Pattern: df.column_name
    dot_match = re.search(r"df\.(\w+)", query)
    if dot_match:
        col = dot_match.group(1)
        # Skip DataFrame methods
        if col not in {'groupby', 'apply', 'shape', 'head', 'tail', 'sort_values'}:
            return col

    return None
