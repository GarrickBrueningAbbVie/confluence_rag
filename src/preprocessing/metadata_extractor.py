"""
Metadata extraction from Confluence pages.

This module extracts structured metadata from Confluence pages:
- Parent project name (for pages under DSA Projects/Products)
- Technologies used in the project

The extracted metadata is used for database queries and filtering.

Example:
    >>> from preprocessing.metadata_extractor import MetadataExtractor
    >>> from iliad.client import IliadClient, IliadClientConfig
    >>>
    >>> iliad = IliadClient(IliadClientConfig.from_env())
    >>> extractor = MetadataExtractor(iliad)
    >>>
    >>> page_data = {"title": "My Project", "ancestors": [...], "content_text": "..."}
    >>> updated = extractor.process_page(page_data)
    >>> print(updated["parent_project"], updated["technologies"])
"""

from typing import Any, Dict, List, Optional

from loguru import logger

# Import types for type hints
try:
    from iliad.client import IliadClient
    from iliad.analyze import DocumentAnalyzer
except ImportError:
    pass


# Known parent categories that contain projects
# Pages under these are considered project pages
DSA_PROJECT_ROOTS = [
    "DSA Products and Solutions",
    "DSA Projects",
    "DSA Trial Execution",
    "Products and Solutions",
    "Projects",
]


class MetadataExtractor:
    """
    Extract structured metadata from Confluence pages.

    Uses page hierarchy (ancestors) to determine the parent project
    and Iliad API to extract technologies from content.

    Attributes:
        iliad_client: Iliad API client for LLM-based extraction
        analyzer: Document analyzer for technology extraction
        model: Model to use for extraction (default: gpt-5-mini-global)

    Example:
        >>> extractor = MetadataExtractor(iliad_client)
        >>> parent = extractor.extract_parent_project(page_data)
        >>> techs = extractor.extract_technologies(content, title)
    """

    def __init__(
        self,
        iliad_client: "IliadClient",
        model: str = "gpt-5-mini-global",
    ) -> None:
        """Initialize metadata extractor.

        Args:
            iliad_client: Configured Iliad API client
            model: Model to use for LLM extraction
        """
        self.iliad_client = iliad_client
        self.model = model
        self.analyzer = DocumentAnalyzer(iliad_client, default_model=model)

        logger.info(f"Initialized MetadataExtractor with model: {model}")

    def _is_project_root(self, title: str) -> bool:
        """Check if a page title is a known project root.

        Args:
            title: Page title to check

        Returns:
            True if title matches a known project root
        """
        title_lower = title.lower()
        return any(root.lower() in title_lower for root in DSA_PROJECT_ROOTS)

    def extract_parent_project(
        self,
        page_data: Dict[str, Any],
    ) -> Optional[str]:
        """
        Extract parent project name from page hierarchy.

        For pages under DSA project roots, finds the immediate child
        of the root category - this is the project name.

        Logic:
        1. Walk ancestors from root to parent
        2. Find the first ancestor that is a project root
        3. Return the next ancestor's title (the project name)
        4. If page itself is directly under root, return page's title

        Args:
            page_data: Page dictionary with 'ancestors' list

        Returns:
            Parent project name, or None if not under a project

        Example:
            >>> # Page: "API Documentation" under "Code Doc Tool" under "DSA Projects"
            >>> parent = extractor.extract_parent_project(page_data)
            >>> print(parent)  # "Code Doc Tool"
        """
        ancestors = page_data.get("ancestors", [])
        title = page_data.get("title", "")

        if not ancestors:
            # No ancestors - check if this is a project root
            if self._is_project_root(title):
                return None  # This is a root, not a project
            return None

        # Find the project root in ancestors
        root_index = None
        for i, ancestor in enumerate(ancestors):
            ancestor_title = ancestor.get("title", "")
            if self._is_project_root(ancestor_title):
                root_index = i
                break

        if root_index is None:
            # No project root found in ancestors
            return None

        # The project name is the ancestor immediately after the root
        project_index = root_index + 1

        if project_index < len(ancestors):
            # There's an ancestor after the root - that's the project
            return ancestors[project_index].get("title", "")
        else:
            # This page is directly under the root - it IS a project page
            return title

    def extract_technologies(
        self,
        content: str,
        title: Optional[str] = None,
    ) -> List[str]:
        """
        Extract technologies mentioned in page content.

        Uses Iliad API to identify:
        - Programming languages (Python, R, Java, etc.)
        - Frameworks (Django, React, TensorFlow, etc.)
        - Databases (PostgreSQL, MongoDB, etc.)
        - Cloud services (AWS, Azure, etc.)
        - Tools (Docker, Airflow, Git, etc.)

        Args:
            content: Text content to analyze
            title: Optional page title for context

        Returns:
            List of technology names (deduplicated)

        Example:
            >>> techs = extractor.extract_technologies("We use Python and Airflow")
            >>> print(techs)  # ["Python", "Airflow"]
        """
        if not content or len(content.strip()) < 50:
            return []

        try:
            technologies = self.analyzer.extract_technologies(content, title)
            logger.debug(f"Extracted {len(technologies)} technologies from '{title or 'content'}'")
            return technologies
        except Exception as e:
            logger.warning(f"Technology extraction failed: {e}")
            return []

    def process_page(
        self,
        page_data: Dict[str, Any],
        extract_technologies: bool = True,
    ) -> Dict[str, Any]:
        """
        Process a single page to extract all metadata.

        Updates the page dictionary with:
        - parent_project: Extracted project name
        - technologies: List of technologies (if extract_technologies=True)

        Args:
            page_data: Page dictionary to process
            extract_technologies: Whether to extract technologies

        Returns:
            Updated page dictionary with metadata fields

        Example:
            >>> updated = extractor.process_page(page_data)
            >>> print(updated["parent_project"])
            >>> print(updated["technologies"])
        """
        title = page_data.get("title", "")

        # Extract parent project
        parent_project = self.extract_parent_project(page_data)
        page_data["parent_project"] = parent_project

        if parent_project:
            logger.debug(f"Page '{title}' belongs to project: {parent_project}")

        # Extract technologies if requested
        if extract_technologies:
            content = page_data.get("content_text", "") or ""

            # Also include attachment content if available
            attachment_content = page_data.get("attachment_content", "") or ""
            full_content = f"{content}\n\n{attachment_content}".strip()

            if full_content:
                technologies = self.extract_technologies(full_content, title)
                page_data["technologies"] = technologies
            else:
                page_data["technologies"] = []
        else:
            page_data.setdefault("technologies", [])

        return page_data

    def process_pages(
        self,
        pages: List[Dict[str, Any]],
        extract_technologies: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Process multiple pages to extract metadata.

        Args:
            pages: List of page dictionaries
            extract_technologies: Whether to extract technologies

        Returns:
            Updated list of page dictionaries

        Example:
            >>> pages = extractor.process_pages(all_pages)
            >>> for p in pages:
            ...     print(f"{p['title']}: {p['parent_project']}")
        """
        logger.info(f"Extracting metadata from {len(pages)} pages")

        for i, page in enumerate(pages):
            if (i + 1) % 10 == 0:
                logger.info(f"Processing page {i + 1}/{len(pages)}")

            try:
                self.process_page(page, extract_technologies=extract_technologies)
            except Exception as e:
                title = page.get("title", "unknown")
                logger.error(f"Failed to process page '{title}': {e}")
                page.setdefault("parent_project", None)
                page.setdefault("technologies", [])

        # Log summary
        with_project = sum(1 for p in pages if p.get("parent_project"))
        with_tech = sum(1 for p in pages if p.get("technologies"))

        logger.info(f"Metadata extraction complete:")
        logger.info(f"  - {with_project}/{len(pages)} pages have parent_project")
        logger.info(f"  - {with_tech}/{len(pages)} pages have technologies")

        return pages

    def propagate_project_technologies(
        self,
        pages: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Propagate technologies from project pages to all child pages.

        After initial extraction, this method ensures that all pages
        within a project share the same technology list (union of all
        pages in the project).

        Args:
            pages: List of page dictionaries (already processed)

        Returns:
            Updated pages with propagated technologies

        Example:
            >>> pages = extractor.propagate_project_technologies(pages)
        """
        # Group pages by project
        project_techs: Dict[str, set] = {}

        for page in pages:
            project = page.get("parent_project")
            if project:
                if project not in project_techs:
                    project_techs[project] = set()

                techs = page.get("technologies", [])
                project_techs[project].update(techs)

        # Propagate back to pages
        for page in pages:
            project = page.get("parent_project")
            if project and project in project_techs:
                # Merge with existing (in case page has unique techs)
                existing = set(page.get("technologies", []))
                combined = existing.union(project_techs[project])
                page["technologies"] = sorted(combined)

        logger.info(f"Propagated technologies across {len(project_techs)} projects")

        return pages
