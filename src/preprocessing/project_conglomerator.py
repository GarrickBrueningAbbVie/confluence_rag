"""
Project conglomeration for aggregated project-level retrieval.

This module combines all pages under a main_project into a single document,
enabling project-level similarity search before chunk-level retrieval.
This supports the two-stage RAG pipeline where we first identify relevant
projects, then search within those projects.

Example:
    >>> from preprocessing.project_conglomerator import ProjectConglomerator
    >>> conglomerator = ProjectConglomerator()
    >>>
    >>> # Load preprocessed pages
    >>> with open("confluence_pages_processed.json") as f:
    ...     pages = json.load(f)
    >>>
    >>> # Conglomerate by main_project
    >>> conglomerated = conglomerator.conglomerate_pages(pages)
    >>> conglomerator.save_conglomerated(conglomerated, "conglomerated_projects.json")
"""

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

from loguru import logger


class ProjectConglomerator:
    """
    Aggregate pages by main_project for project-level retrieval.

    Creates one entry per main_project containing combined content from
    the project page and all its subpages. This enables project-level
    similarity search in the two-stage RAG pipeline.

    Attributes:
        max_content_length: Maximum characters per conglomerated project
        include_page_headers: Whether to add page title headers in combined content

    Example:
        >>> conglomerator = ProjectConglomerator(max_content_length=500000)
        >>> projects = conglomerator.conglomerate_pages(pages)
    """

    def __init__(
        self,
        max_content_length: int = 500000,
        include_page_headers: bool = True,
    ) -> None:
        """Initialize the project conglomerator.

        Args:
            max_content_length: Maximum characters per project (~500K default)
            include_page_headers: Whether to add "=== Page Title ===" headers
        """
        self.max_content_length = max_content_length
        self.include_page_headers = include_page_headers

        logger.info(
            f"Initialized ProjectConglomerator "
            f"(max_length={max_content_length}, headers={include_page_headers})"
        )

    def conglomerate_pages(
        self,
        pages: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Aggregate pages by main_project.

        Creates one entry per main_project containing combined content_text
        from the project page and all subpages. Pages are sorted by depth
        so the main project page content comes first.

        Args:
            pages: List of page dictionaries with main_project field

        Returns:
            List of conglomerated project dictionaries with:
            - main_project: Project name
            - main_project_id: Project page ID
            - content_text: Combined content from all pages
            - page_count: Number of pages included
            - total_pages: Total pages in project
            - page_titles: List of included page titles
            - url: URL of main project page
            - space_key: Confluence space key

        Example:
            >>> projects = conglomerator.conglomerate_pages(pages)
            >>> for p in projects:
            ...     print(f"{p['main_project']}: {p['page_count']} pages")
        """
        logger.info(f"Conglomerating {len(pages)} pages by main_project")

        # Group pages by main_project
        project_pages: Dict[str, List[Dict]] = defaultdict(list)
        project_metadata: Dict[str, Dict] = {}

        for page in pages:
            main_project = page.get("main_project")
            main_project_id = page.get("main_project_id")

            if not main_project:
                continue

            project_pages[main_project].append(page)

            # Store metadata from the main project page itself
            if page.get("id") == main_project_id:
                project_metadata[main_project] = {
                    "id": page.get("id"),
                    "title": page.get("title"),
                    "url": page.get("url"),
                    "space_key": page.get("space_key", page.get("space", "")),
                }

        logger.info(f"Found {len(project_pages)} unique main_projects")

        # Create conglomerated entries
        conglomerated = []
        for project_name, pages_list in project_pages.items():
            # Sort by depth so main project page content comes first
            pages_list.sort(key=lambda p: p.get("depth", 999))

            # Combine content
            content_parts = []
            page_titles = []
            total_length = 0

            for page in pages_list:
                page_content = page.get("content_text", "") or ""
                if not page_content.strip():
                    continue

                page_title = page.get("title", "Unknown")

                # Add page header if enabled
                if self.include_page_headers:
                    page_header = f"\n\n=== {page_title} ===\n\n"
                    content_to_add = page_header + page_content
                else:
                    content_to_add = "\n\n" + page_content

                # Check if adding this would exceed limit
                if total_length + len(content_to_add) > self.max_content_length:
                    logger.warning(
                        f"Project '{project_name}' exceeded max length at "
                        f"{len(page_titles)} pages, truncating"
                    )
                    break

                content_parts.append(content_to_add)
                page_titles.append(page_title)
                total_length += len(content_to_add)

            combined_content = "".join(content_parts).strip()

            # Get metadata for the project
            meta = project_metadata.get(project_name, {})

            conglomerated.append({
                "main_project": project_name,
                "main_project_id": meta.get("id", ""),
                "content_text": combined_content,
                "page_count": len(page_titles),
                "total_pages": len(pages_list),
                "page_titles": page_titles,
                "url": meta.get("url", ""),
                "space_key": meta.get("space_key", ""),
                "content_length": len(combined_content),
            })

            logger.debug(
                f"Conglomerated '{project_name}': "
                f"{len(page_titles)}/{len(pages_list)} pages, "
                f"{len(combined_content)} chars"
            )

        # Sort by project name for consistent output
        conglomerated.sort(key=lambda p: p.get("main_project", ""))

        # Log summary
        total_pages_included = sum(p["page_count"] for p in conglomerated)
        total_chars = sum(p["content_length"] for p in conglomerated)
        logger.info(
            f"Conglomeration complete: {len(conglomerated)} projects, "
            f"{total_pages_included} pages, {total_chars:,} total chars"
        )

        return conglomerated

    def save_conglomerated(
        self,
        conglomerated: List[Dict[str, Any]],
        output_path: str,
    ) -> None:
        """Save conglomerated data to JSON file.

        Args:
            conglomerated: List of conglomerated project dictionaries
            output_path: Path to output JSON file

        Example:
            >>> conglomerator.save_conglomerated(projects, "projects.json")
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(conglomerated, f, indent=2, ensure_ascii=False)

        logger.info(f"Saved {len(conglomerated)} projects to {output_path}")

    def load_conglomerated(
        self,
        input_path: str,
    ) -> List[Dict[str, Any]]:
        """Load conglomerated data from JSON file.

        Args:
            input_path: Path to conglomerated JSON file

        Returns:
            List of conglomerated project dictionaries

        Example:
            >>> projects = conglomerator.load_conglomerated("projects.json")
        """
        input_path = Path(input_path)

        if not input_path.exists():
            raise FileNotFoundError(f"Conglomerated file not found: {input_path}")

        with open(input_path, "r", encoding="utf-8") as f:
            conglomerated = json.load(f)

        logger.info(f"Loaded {len(conglomerated)} projects from {input_path}")
        return conglomerated

    def get_project_summary(
        self,
        conglomerated: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Get summary statistics for conglomerated data.

        Args:
            conglomerated: List of conglomerated project dictionaries

        Returns:
            Dictionary with summary statistics

        Example:
            >>> summary = conglomerator.get_project_summary(projects)
            >>> print(f"Total projects: {summary['total_projects']}")
        """
        if not conglomerated:
            return {
                "total_projects": 0,
                "total_pages": 0,
                "total_chars": 0,
                "avg_pages_per_project": 0,
                "avg_chars_per_project": 0,
            }

        total_pages = sum(p["page_count"] for p in conglomerated)
        total_chars = sum(p["content_length"] for p in conglomerated)

        return {
            "total_projects": len(conglomerated),
            "total_pages": total_pages,
            "total_chars": total_chars,
            "avg_pages_per_project": total_pages / len(conglomerated),
            "avg_chars_per_project": total_chars / len(conglomerated),
            "largest_project": max(conglomerated, key=lambda p: p["page_count"])["main_project"],
            "smallest_project": min(conglomerated, key=lambda p: p["page_count"])["main_project"],
        }
