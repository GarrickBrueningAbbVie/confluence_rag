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
    >>> page_data = {"title": "My Project", "parents": [...], "content_text": "..."}
    >>> updated = extractor.process_page(page_data)
    >>> print(updated["parent_project"], updated["technologies"])
"""

from typing import Any, Dict, List, Optional, Tuple

from loguru import logger

from .parallel import ParallelProcessor

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

    Uses page hierarchy (parents) to determine the parent project
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
        1. Walk parents from root to immediate parent
        2. Find the first parent that is a project root
        3. Return the next parent's title (the project name)
        4. If page itself is directly under root, return page's title

        Args:
            page_data: Page dictionary with 'parents' list

        Returns:
            Parent project name, or None if not under a project

        Example:
            >>> # Page: "API Documentation" under "Code Doc Tool" under "DSA Projects"
            >>> parent = extractor.extract_parent_project(page_data)
            >>> print(parent)  # "Code Doc Tool"
        """
        parents = page_data.get("parents", [])
        title = page_data.get("title", "")

        if not parents:
            # No parents - check if this is a project root
            if self._is_project_root(title):
                return None  # This is a root, not a project
            return None

        # Find the project root in parents
        root_index = None
        for i, parent in enumerate(parents):
            parent_title = parent.get("title", "")
            if self._is_project_root(parent_title):
                root_index = i
                break

        if root_index is None:
            # No project root found in parents
            return None

        # The project name is the parent immediately after the root
        project_index = root_index + 1

        if project_index < len(parents):
            # There's a parent after the root - that's the project
            return parents[project_index].get("title", "")
        else:
            # This page is directly under the root - it IS a project page
            return title

    def extract_main_project(
        self,
        page_data: Dict[str, Any],
    ) -> Tuple[Optional[str], Optional[str]]:
        """
        Extract main project name and ID (depth 3 ancestor).

        DSA Hierarchy:
        - Level 1: DSA (space root)
        - Level 2: DSA Areas, DSA Products and Solutions, DSA Projects, etc.
        - Level 3: Main project (e.g., ATLAS, ALFA) <- This is what we extract
        - Level 4+: Subpages

        Args:
            page_data: Page dictionary with 'parents' list and 'depth'

        Returns:
            Tuple of (main_project_name, main_project_id)
            Returns (None, None) for pages at depth 1-2

        Example:
            >>> name, id = extractor.extract_main_project(page_data)
            >>> print(name)  # "ATLAS"
        """
        parents = page_data.get("parents", [])
        depth = page_data.get("depth", len(parents) + 1)
        title = page_data.get("title", "")
        page_id = page_data.get("id", "")

        if depth <= 2:
            # Page is above project level (DSA root or category)
            return (None, None)

        if depth == 3:
            # Page IS the main project
            return (title, page_id)

        # Depth 4+: Get depth-3 ancestor (index 2 in 0-based parents list)
        # parents[0] = depth 1 (DSA root)
        # parents[1] = depth 2 (category)
        # parents[2] = depth 3 (main project)
        if len(parents) >= 3:
            main_project_parent = parents[2]
            return (
                main_project_parent.get("title", ""),
                main_project_parent.get("id", ""),
            )

        return (None, None)

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
        - parent_project: Extracted project name (from project root pattern)
        - main_project: Main project name (depth 3 ancestor)
        - main_project_id: Main project page ID
        - technologies: List of technologies (if extract_technologies=True)

        Args:
            page_data: Page dictionary to process
            extract_technologies: Whether to extract technologies

        Returns:
            Updated page dictionary with metadata fields

        Example:
            >>> updated = extractor.process_page(page_data)
            >>> print(updated["parent_project"])
            >>> print(updated["main_project"])
            >>> print(updated["technologies"])
        """
        title = page_data.get("title", "")

        # Extract parent project (from project root pattern)
        parent_project = self.extract_parent_project(page_data)
        page_data["parent_project"] = parent_project

        # Extract main project (depth 3 ancestor)
        main_project, main_project_id = self.extract_main_project(page_data)
        page_data["main_project"] = main_project
        page_data["main_project_id"] = main_project_id

        if main_project:
            logger.debug(f"Page '{title}' main project: {main_project}")
        elif parent_project:
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
                page.setdefault("main_project", None)
                page.setdefault("main_project_id", None)
                page.setdefault("technologies", [])

        # Log summary
        with_project = sum(1 for p in pages if p.get("parent_project"))
        with_main = sum(1 for p in pages if p.get("main_project"))
        with_tech = sum(1 for p in pages if p.get("technologies"))

        logger.info(f"Metadata extraction complete:")
        logger.info(f"  - {with_project}/{len(pages)} pages have parent_project")
        logger.info(f"  - {with_main}/{len(pages)} pages have main_project")
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

    # -------------------------------------------------------------------------
    # Parallel Processing Methods
    # -------------------------------------------------------------------------

    def _process_page_wrapper(
        self,
        args: Tuple[Dict[str, Any], bool],
    ) -> Tuple[int, Dict[str, Any]]:
        """Wrapper for parallel page processing.

        Args:
            args: Tuple of (page_data, extract_technologies)

        Returns:
            Tuple of (original_index, processed_page_data)
        """
        page_data, extract_technologies = args
        return self.process_page(page_data, extract_technologies=extract_technologies)

    def process_pages_parallel(
        self,
        pages: List[Dict[str, Any]],
        extract_technologies: bool = True,
        max_workers: int = 8,
        rate_limit_rps: Optional[float] = 10.0,
        batch_size: int = 50,
    ) -> List[Dict[str, Any]]:
        """Process multiple pages in parallel to extract metadata.

        Parallel version of process_pages() that uses a thread pool to
        extract metadata (especially technologies) from multiple pages
        concurrently.

        Note: Parent project extraction is fast (no API calls) and is
        still done sequentially. Technology extraction (which requires
        LLM calls) is parallelized.

        Args:
            pages: List of page dictionaries
            extract_technologies: Whether to extract technologies
            max_workers: Maximum concurrent processing threads
            rate_limit_rps: Optional requests per second limit
            batch_size: Number of pages to process per batch

        Returns:
            Updated list of page dictionaries with metadata

        Example:
            >>> pages = extractor.process_pages_parallel(
            ...     all_pages,
            ...     max_workers=8,
            ...     rate_limit_rps=10.0,
            ...     batch_size=50,
            ... )
        """
        logger.info(
            f"Extracting metadata from {len(pages)} pages in parallel "
            f"(workers={max_workers}, rps={rate_limit_rps}, batch={batch_size})"
        )

        # First pass: Extract parent_project and main_project (fast, no API calls)
        # This can stay sequential since it's just dictionary lookups
        for page in pages:
            parent_project = self.extract_parent_project(page)
            page["parent_project"] = parent_project

            main_project, main_project_id = self.extract_main_project(page)
            page["main_project"] = main_project
            page["main_project_id"] = main_project_id

        with_project = sum(1 for p in pages if p.get("parent_project"))
        with_main = sum(1 for p in pages if p.get("main_project"))
        logger.info(f"Project extraction: {with_project}/{len(pages)} pages have parent_project")
        logger.info(f"Project extraction: {with_main}/{len(pages)} pages have main_project")

        # Second pass: Extract technologies in parallel (LLM calls)
        if extract_technologies:
            # Filter to pages that have content worth analyzing
            pages_to_analyze = []
            pages_indices = []

            for i, page in enumerate(pages):
                content = page.get("content_text", "") or ""
                attachment_content = page.get("attachment_content", "") or ""
                full_content = f"{content}\n\n{attachment_content}".strip()

                if full_content and len(full_content) >= 50:
                    pages_to_analyze.append((page, True))
                    pages_indices.append(i)
                else:
                    page["technologies"] = []

            if pages_to_analyze:
                logger.info(
                    f"Extracting technologies from {len(pages_to_analyze)} pages with content"
                )

                processor = ParallelProcessor(
                    max_workers=max_workers,
                    rate_limit_rps=rate_limit_rps,
                )

                try:
                    results = processor.map_batched(
                        self._process_page_wrapper,
                        pages_to_analyze,
                        batch_size=batch_size,
                        desc="Technology extraction",
                        pause_between_batches=1.0,
                    )

                    # Update pages with results
                    for proc_result, idx in zip(results, pages_indices):
                        if proc_result.success and proc_result.value:
                            pages[idx] = proc_result.value
                        else:
                            pages[idx].setdefault("technologies", [])

                finally:
                    processor.shutdown()
        else:
            for page in pages:
                page.setdefault("technologies", [])

        # Log summary
        with_tech = sum(1 for p in pages if p.get("technologies"))
        logger.info(f"Metadata extraction complete (parallel):")
        logger.info(f"  - {with_project}/{len(pages)} pages have parent_project")
        logger.info(f"  - {with_main}/{len(pages)} pages have main_project")
        logger.info(f"  - {with_tech}/{len(pages)} pages have technologies")

        return pages
