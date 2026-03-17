"""
DataFrame loader for Confluence JSON data.

This module loads preprocessed Confluence JSON data into a pandas DataFrame
with proper type handling and column normalization.

Example:
    >>> from database.dataframe_loader import DataFrameLoader
    >>> loader = DataFrameLoader("Data_Storage/confluence_pages.json")
    >>> df = loader.load()
    >>> print(df.columns.tolist())
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd
from loguru import logger


class DataFrameLoader:
    """
    Load Confluence JSON data into pandas DataFrame.

    Handles JSON loading, type conversion, and column normalization
    for structured queries.

    Attributes:
        json_path: Path to the Confluence JSON file
        df: Loaded DataFrame (populated after load())

    Example:
        >>> loader = DataFrameLoader("pages.json")
        >>> df = loader.load()
        >>> print(f"Loaded {len(df)} pages")
    """

    # Columns to include in the DataFrame
    STANDARD_COLUMNS = [
        "id",
        "title",
        "space_key",
        "created_date",
        "last_modified",
        "created_by",
        "depth",
        "parent_project",
        "technologies",
        "completeness_score",
        "completeness_summary",
        "url",
        "content_text",
    ]

    # Columns that get lowercase versions for case-insensitive search
    LOWERCASE_COLUMNS = [
        "title",
        "parent_project",
        "created_by",
    ]

    # List columns that get lowercase versions
    LOWERCASE_LIST_COLUMNS = [
        "technologies",
    ]

    def __init__(self, json_path: Union[str, Path]) -> None:
        """Initialize the loader.

        Args:
            json_path: Path to Confluence JSON file
        """
        self.json_path = Path(json_path)
        self.df: Optional[pd.DataFrame] = None

        logger.info(f"Initialized DataFrameLoader with: {self.json_path}")

    def load(self, include_content: bool = False) -> pd.DataFrame:
        """
        Load JSON data into a pandas DataFrame.

        Args:
            include_content: Whether to include content_text column (large)

        Returns:
            DataFrame with Confluence page data

        Raises:
            FileNotFoundError: If JSON file doesn't exist
            ValueError: If JSON is invalid
        """
        if not self.json_path.exists():
            raise FileNotFoundError(f"JSON file not found: {self.json_path}")

        logger.info(f"Loading data from {self.json_path}")

        with open(self.json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if not isinstance(data, list):
            raise ValueError("Expected JSON array of page objects")

        # Normalize nested data
        normalized = self._normalize_pages(data)

        # Create DataFrame
        self.df = pd.DataFrame(normalized)

        # Select and order columns
        columns = [c for c in self.STANDARD_COLUMNS if c in self.df.columns]
        if not include_content and "content_text" in columns:
            columns.remove("content_text")

        self.df = self.df[columns]

        # Convert types
        self._convert_types()

        # Create lowercase columns for case-insensitive searching
        self._create_lowercase_columns()

        logger.info(f"Loaded {len(self.df)} pages with {len(self.df.columns)} columns")

        return self.df

    def _normalize_pages(self, pages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Normalize page data for DataFrame creation.

        Args:
            pages: List of page dictionaries

        Returns:
            Normalized list ready for DataFrame
        """
        normalized = []

        for page in pages:
            row = {
                "id": page.get("id", ""),
                "title": page.get("title", ""),
                "space_key": page.get("space_key", ""),
                "created_date": page.get("created_date", ""),
                "last_modified": page.get("modified_date", page.get("last_modified", "")),
                "created_by": page.get("author", page.get("created_by", "")),  # JSON uses 'author'
                "depth": page.get("depth"),
                "parent_project": page.get("parent_project"),
                "technologies": page.get("technologies", []),
                "completeness_score": page.get("completeness_score"),
                "completeness_summary": page.get("completeness_summary"),
                "url": page.get("url", ""),
                "content_text": page.get("content_text", ""),
            }
            normalized.append(row)

        return normalized

    def _convert_types(self) -> None:
        """Convert DataFrame columns to appropriate types."""
        if self.df is None:
            return

        # Convert date columns
        for col in ["created_date", "last_modified"]:
            if col in self.df.columns:
                self.df[col] = pd.to_datetime(self.df[col], errors="coerce")

        # Convert numeric columns
        if "depth" in self.df.columns:
            self.df["depth"] = pd.to_numeric(self.df["depth"], errors="coerce")

        if "completeness_score" in self.df.columns:
            self.df["completeness_score"] = pd.to_numeric(
                self.df["completeness_score"], errors="coerce"
            )

        logger.debug("Converted column types")

    def _create_lowercase_columns(self) -> None:
        """Create lowercase versions of text columns for case-insensitive search.

        Creates columns like 'title_lower', 'parent_project_lower' etc.
        that contain lowercase versions of the original text for matching.
        The original columns are preserved for display purposes.
        """
        if self.df is None:
            return

        # Create lowercase versions of string columns
        for col in self.LOWERCASE_COLUMNS:
            if col in self.df.columns:
                lower_col = f"{col}_lower"
                self.df[lower_col] = self.df[col].apply(
                    lambda x: x.lower() if isinstance(x, str) else x
                )

        # Create lowercase versions of list columns (e.g., technologies)
        for col in self.LOWERCASE_LIST_COLUMNS:
            if col in self.df.columns:
                lower_col = f"{col}_lower"
                self.df[lower_col] = self.df[col].apply(
                    lambda x: [item.lower() for item in x] if isinstance(x, list) else x
                )

        logger.debug(
            f"Created lowercase columns: "
            f"{[f'{c}_lower' for c in self.LOWERCASE_COLUMNS + self.LOWERCASE_LIST_COLUMNS]}"
        )

    def get_schema(self) -> Dict[str, str]:
        """
        Get DataFrame schema information.

        Returns:
            Dictionary mapping column names to dtype strings
        """
        if self.df is None:
            self.load()

        return {col: str(dtype) for col, dtype in self.df.dtypes.items()}

    def get_column_info(self) -> str:
        """
        Get detailed column information for query generation.

        Returns:
            Formatted string describing available columns
        """
        if self.df is None:
            self.load()

        info_lines = ["Available columns:"]

        # Document primary columns (exclude _lower columns from main list)
        primary_cols = [c for c in self.df.columns if not c.endswith("_lower")]

        for col in primary_cols:
            dtype = str(self.df[col].dtype)
            non_null = self.df[col].notna().sum()
            total = len(self.df)

            # Sample values for context
            if col == "technologies":
                # Flatten and get unique technologies
                all_techs = []
                for tech_list in self.df[col].dropna():
                    if isinstance(tech_list, list):
                        all_techs.extend(tech_list)
                unique_techs = list(set(all_techs))[:10]
                sample = f"Examples: {unique_techs}"
            elif dtype in ["object", "string"]:
                unique = self.df[col].nunique()
                sample = f"Unique values: {unique}"
            elif "datetime" in dtype:
                min_val = self.df[col].min()
                max_val = self.df[col].max()
                sample = f"Range: {min_val} to {max_val}"
            elif "float" in dtype or "int" in dtype:
                min_val = self.df[col].min()
                max_val = self.df[col].max()
                sample = f"Range: {min_val} to {max_val}"
            else:
                sample = ""

            info_lines.append(f"  - {col} ({dtype}): {non_null}/{total} non-null. {sample}")

        # Document lowercase columns
        lower_cols = [c for c in self.df.columns if c.endswith("_lower")]
        if lower_cols:
            info_lines.append("\nLowercase columns for case-insensitive search:")
            for col in lower_cols:
                original_col = col.replace("_lower", "")
                info_lines.append(f"  - {col}: lowercase version of '{original_col}' for case-insensitive matching")

        return "\n".join(info_lines)

    def get_technology_list(self) -> List[str]:
        """
        Get list of all unique technologies in the data.

        Returns:
            Sorted list of unique technology names
        """
        if self.df is None:
            self.load()

        all_techs = set()
        for tech_list in self.df["technologies"].dropna():
            if isinstance(tech_list, list):
                all_techs.update(tech_list)

        return sorted(all_techs)

    def get_project_list(self) -> List[str]:
        """
        Get list of all unique parent projects.

        Returns:
            Sorted list of unique project names
        """
        if self.df is None:
            self.load()

        projects = self.df["parent_project"].dropna().unique().tolist()
        return sorted(projects)

    def get_author_list(self) -> List[str]:
        """
        Get list of all unique authors.

        Returns:
            Sorted list of unique author names
        """
        if self.df is None:
            self.load()

        authors = self.df["created_by"].dropna().unique().tolist()
        return sorted(authors)
