"""
Preprocessing pipeline orchestration.

This module orchestrates all preprocessing steps for Confluence pages:
1. Fetch and process attachments
2. Extract metadata (parent_project, technologies)
3. Assess completeness

The pipeline can be run standalone or integrated into the data acquisition flow.

Example:
    >>> from preprocessing.processor import PreprocessingPipeline
    >>> pipeline = PreprocessingPipeline.from_env()
    >>> pages = pipeline.process("Data_Storage/confluence_pages.json")

Usage as standalone script:
    python -m preprocessing.processor [input_json] [output_json]
"""

import json
import math
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from loguru import logger


@dataclass
class ParallelConfig:
    """Configuration for parallel processing in the pipeline.

    Attributes:
        enabled: Whether to use parallel processing
        max_workers_pages: Maximum concurrent page processing
        max_workers_attachments: Maximum concurrent attachments per page
        max_workers_metadata: Maximum concurrent metadata extraction
        rate_limit_rps: Requests per second limit for API calls
        batch_size: Number of items to process per batch

    Example:
        >>> config = ParallelConfig(
        ...     enabled=True,
        ...     max_workers_pages=8,
        ...     rate_limit_rps=10.0,
        ... )
        >>> pipeline.process("input.json", parallel_config=config)
    """

    enabled: bool = True
    max_workers_pages: int = 8
    max_workers_attachments: int = 4
    max_workers_metadata: int = 8
    rate_limit_rps: Optional[float] = 10.0
    batch_size: int = 50

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from confluence.rest_client import ConfluenceRestClient
from iliad.client import IliadClient, IliadClientConfig
from preprocessing.attachment_fetcher import AttachmentFetcher
from preprocessing.metadata_extractor import MetadataExtractor
from preprocessing.completeness_assessor import CompletenessAssessor


class PreprocessingPipeline:
    """
    Complete preprocessing pipeline for Confluence pages.

    Orchestrates:
    1. Attachment processing (optional)
    2. Metadata extraction
    3. Completeness assessment

    Attributes:
        confluence_client: Confluence REST client (for attachments)
        iliad_client: Iliad API client
        attachment_fetcher: Attachment processing component
        metadata_extractor: Metadata extraction component
        completeness_assessor: Completeness assessment component
        process_attachments: Whether to process attachments

    Example:
        >>> pipeline = PreprocessingPipeline.from_env()
        >>> pages = pipeline.process("confluence_pages.json")
    """

    def __init__(
        self,
        confluence_client: Optional[ConfluenceRestClient] = None,
        iliad_client: Optional[IliadClient] = None,
        attachment_storage_path: str = "Data_Storage/attachments",
        process_attachments: bool = True,
        extract_technologies: bool = True,
        use_llm_completeness: bool = False,
    ) -> None:
        """Initialize preprocessing pipeline.

        Args:
            confluence_client: Confluence REST client (required for attachments)
            iliad_client: Iliad API client (required for LLM features)
            attachment_storage_path: Path to store downloaded attachments
            process_attachments: Whether to fetch and process attachments
            extract_technologies: Whether to extract technologies with LLM
            use_llm_completeness: Whether to use LLM for completeness assessment
        """
        self.confluence_client = confluence_client
        self.iliad_client = iliad_client
        self.process_attachments_enabled = process_attachments and confluence_client is not None
        self.extract_technologies = extract_technologies
        self.use_llm_completeness = use_llm_completeness

        # Initialize components
        if self.process_attachments_enabled and iliad_client:
            self.attachment_fetcher = AttachmentFetcher(
                confluence_client,
                iliad_client,
                storage_path=attachment_storage_path,
            )
        else:
            self.attachment_fetcher = None

        if iliad_client:
            self.metadata_extractor = MetadataExtractor(iliad_client)
        else:
            self.metadata_extractor = None

        self.completeness_assessor = CompletenessAssessor(iliad_client)

        logger.info(
            f"Initialized PreprocessingPipeline "
            f"(attachments: {self.process_attachments_enabled}, "
            f"technologies: {extract_technologies}, "
            f"llm_completeness: {use_llm_completeness})"
        )

    @classmethod
    def from_env(
        cls,
        process_attachments: bool = True,
        extract_technologies: bool = True,
        use_llm_completeness: bool = False,
    ) -> "PreprocessingPipeline":
        """Create pipeline from environment variables.

        Required environment variables:
        - ILIAD_API_KEY: API key for Iliad
        - ILIAD_API_URL: Base URL for Iliad API

        Optional (for attachment processing):
        - CONFLUENCE_URL: Confluence base URL
        - CONFLUENCE_USERNAME: Confluence username
        - CONFLUENCE_API_TOKEN: Confluence API token

        Args:
            process_attachments: Whether to process attachments
            extract_technologies: Whether to extract technologies
            use_llm_completeness: Whether to use LLM for completeness

        Returns:
            Configured PreprocessingPipeline instance

        Example:
            >>> pipeline = PreprocessingPipeline.from_env()
        """
        # Load environment variables
        from dotenv import load_dotenv
        load_dotenv()

        # Initialize Iliad client
        iliad_client = None
        try:
            iliad_config = IliadClientConfig.from_env()
            iliad_client = IliadClient(iliad_config)
        except ValueError as e:
            logger.warning(f"Iliad client not configured: {e}")

        # Initialize Confluence client (optional)
        confluence_client = None
        confluence_url = os.getenv("CONFLUENCE_URL")
        confluence_user = os.getenv("CONFLUENCE_USERNAME")
        confluence_token = os.getenv("CONFLUENCE_API_TOKEN")

        if all([confluence_url, confluence_user, confluence_token]):
            confluence_client = ConfluenceRestClient(
                base_url=confluence_url,
                username=confluence_user,
                api_token=confluence_token,
                verify_ssl=False,
            )
        else:
            logger.warning("Confluence client not configured - attachment processing disabled")

        # Get storage path from env
        attachment_path = os.getenv("ATTACHMENT_STORAGE_PATH", "Data_Storage/attachments")

        return cls(
            confluence_client=confluence_client,
            iliad_client=iliad_client,
            attachment_storage_path=attachment_path,
            process_attachments=process_attachments,
            extract_technologies=extract_technologies,
            use_llm_completeness=use_llm_completeness,
        )

    def process_page_attachments(
        self,
        pages: List[Dict[str, Any]],
        parallel_config: Optional[ParallelConfig] = None,
    ) -> List[Dict[str, Any]]:
        """
        Process attachments for all pages.

        Args:
            pages: List of page dictionaries
            parallel_config: Optional parallel processing configuration

        Returns:
            Updated pages with attachment_content field

        Example:
            >>> pages = pipeline.process_page_attachments(pages)
            >>> # With parallel processing
            >>> pages = pipeline.process_page_attachments(
            ...     pages,
            ...     parallel_config=ParallelConfig(enabled=True)
            ... )
        """
        if not self.attachment_fetcher:
            logger.warning("Attachment fetcher not available - skipping")
            for page in pages:
                page.setdefault("attachments", [])
                page.setdefault("attachment_content", "")
            return pages

        # Use parallel processing if configured
        if parallel_config and parallel_config.enabled:
            return self._process_page_attachments_parallel(pages, parallel_config)

        # Sequential processing (original implementation)
        logger.info(f"Processing attachments for {len(pages)} pages (sequential)")

        for i, page in enumerate(pages):
            page_id = page.get("id", "")
            title = page.get("title", "")

            if (i + 1) % 20 == 0:
                logger.info(f"Processing page {i + 1}/{len(pages)}")

            try:
                # Fetch attachment metadata
                attachments = self.attachment_fetcher.fetch_page_attachments(page_id)
                page["attachments"] = attachments

                # Process and extract text if there are attachments
                if attachments:
                    content = self.attachment_fetcher.process_all_page_attachments(page_id)
                    page["attachment_content"] = content
                else:
                    page["attachment_content"] = ""

            except Exception as e:
                logger.error(f"Failed to process attachments for '{title}': {e}")
                page.setdefault("attachments", [])
                page.setdefault("attachment_content", "")

        # Log summary
        with_attachments = sum(1 for p in pages if p.get("attachments"))
        with_content = sum(1 for p in pages if p.get("attachment_content"))

        logger.info(f"Attachment processing complete:")
        logger.info(f"  - {with_attachments}/{len(pages)} pages have attachments")
        logger.info(f"  - {with_content}/{len(pages)} pages have extracted content")

        return pages

    def _process_page_attachments_parallel(
        self,
        pages: List[Dict[str, Any]],
        config: ParallelConfig,
    ) -> List[Dict[str, Any]]:
        """
        Process attachments in parallel.

        Args:
            pages: List of page dictionaries
            config: Parallel processing configuration

        Returns:
            Updated pages with attachment_content field
        """
        logger.info(
            f"Processing attachments for {len(pages)} pages (parallel: "
            f"pages={config.max_workers_pages}, attachments={config.max_workers_attachments}, "
            f"rps={config.rate_limit_rps}, batch={config.batch_size})"
        )

        # Get page IDs
        page_ids = [p.get("id", "") for p in pages]
        page_id_to_index = {p.get("id", ""): i for i, p in enumerate(pages)}

        # First: Fetch attachment metadata for all pages (relatively fast)
        logger.info("Fetching attachment metadata for all pages...")
        for page in pages:
            page_id = page.get("id", "")
            try:
                attachments = self.attachment_fetcher.fetch_page_attachments(page_id)
                page["attachments"] = attachments
            except Exception as e:
                logger.warning(f"Failed to fetch attachments for page {page_id}: {e}")
                page["attachments"] = []

        # Filter to pages with attachments
        pages_with_attachments = [
            (p.get("id", ""), i) for i, p in enumerate(pages) if p.get("attachments")
        ]

        if not pages_with_attachments:
            logger.info("No pages have attachments - skipping parallel processing")
            for page in pages:
                page["attachment_content"] = ""
            return pages

        logger.info(f"{len(pages_with_attachments)} pages have attachments to process")

        # Process attachments in parallel
        page_ids_to_process = [pid for pid, _ in pages_with_attachments]

        results = self.attachment_fetcher.process_pages_parallel(
            page_ids_to_process,
            max_workers_pages=config.max_workers_pages,
            max_workers_attachments=config.max_workers_attachments,
            rate_limit_rps=config.rate_limit_rps,
            batch_size=config.batch_size,
            cleanup=True,
        )

        # Update pages with results
        for page_id, content in results.items():
            if page_id in page_id_to_index:
                pages[page_id_to_index[page_id]]["attachment_content"] = content

        # Ensure all pages have the field
        for page in pages:
            page.setdefault("attachment_content", "")

        # Log summary
        with_attachments = sum(1 for p in pages if p.get("attachments"))
        with_content = sum(1 for p in pages if p.get("attachment_content"))

        logger.info(f"Attachment processing complete (parallel):")
        logger.info(f"  - {with_attachments}/{len(pages)} pages have attachments")
        logger.info(f"  - {with_content}/{len(pages)} pages have extracted content")

        return pages

    def _extract_parent_project_basic(self, page_data: Dict[str, Any]) -> Optional[str]:
        """
        Extract parent project using hierarchy only (no LLM needed).

        Args:
            page_data: Page dictionary with parents

        Returns:
            Parent project name or None
        """
        DSA_PROJECT_ROOTS = [
            "DSA Products and Solutions",
            "DSA Projects",
            "DSA Trial Execution",
            "Products and Solutions",
            "Projects",
        ]

        def is_project_root(title: str) -> bool:
            title_lower = title.lower()
            return any(root.lower() in title_lower for root in DSA_PROJECT_ROOTS)

        parents = page_data.get("parents", [])
        title = page_data.get("title", "")

        if not parents:
            return None

        # Find the project root in parents
        root_index = None
        for i, parent in enumerate(parents):
            parent_title = parent.get("title", "")
            if is_project_root(parent_title):
                root_index = i
                break

        if root_index is None:
            return None

        # The project name is the parent immediately after the root
        project_index = root_index + 1
        if project_index < len(parents):
            return parents[project_index].get("title", "")
        else:
            return title

    def _extract_main_project_basic(
        self, page_data: Dict[str, Any]
    ) -> Tuple[Optional[str], Optional[str]]:
        """
        Extract main project (depth 3 ancestor) using hierarchy only.

        Args:
            page_data: Page dictionary with parents

        Returns:
            Tuple of (main_project_name, main_project_id)
        """
        parents = page_data.get("parents", [])
        depth = page_data.get("depth", len(parents) + 1)
        title = page_data.get("title", "")
        page_id = page_data.get("id", "")

        if depth <= 2:
            return (None, None)
        if depth == 3:
            return (title, page_id)
        if len(parents) >= 3:
            return (parents[2].get("title", ""), parents[2].get("id", ""))
        return (None, None)

    def process_metadata(
        self,
        pages: List[Dict[str, Any]],
        parallel_config: Optional[ParallelConfig] = None,
    ) -> List[Dict[str, Any]]:
        """
        Extract metadata for all pages.

        Extracts:
        - parent_project: Project name from hierarchy (no LLM needed)
        - technologies: Technologies mentioned in content (requires LLM)

        Args:
            pages: List of page dictionaries
            parallel_config: Optional parallel processing configuration

        Returns:
            Updated pages with metadata fields

        Example:
            >>> pages = pipeline.process_metadata(pages)
            >>> # With parallel processing
            >>> pages = pipeline.process_metadata(
            ...     pages,
            ...     parallel_config=ParallelConfig(enabled=True)
            ... )
        """
        logger.info(f"Extracting metadata for {len(pages)} pages")

        # Always extract parent_project and main_project (doesn't need LLM)
        for page in pages:
            if page.get("parent_project") is None:
                page["parent_project"] = self._extract_parent_project_basic(page)

            if page.get("main_project") is None:
                main_project, main_project_id = self._extract_main_project_basic(page)
                page["main_project"] = main_project
                page["main_project_id"] = main_project_id

        with_project = sum(1 for p in pages if p.get("parent_project"))
        with_main = sum(1 for p in pages if p.get("main_project"))
        logger.info(f"  - {with_project}/{len(pages)} pages have parent_project")
        logger.info(f"  - {with_main}/{len(pages)} pages have main_project")

        # Extract technologies only if LLM available and enabled
        if self.metadata_extractor and self.extract_technologies:
            # Use parallel processing if configured
            if parallel_config and parallel_config.enabled:
                pages = self.metadata_extractor.process_pages_parallel(
                    pages,
                    extract_technologies=True,
                    max_workers=parallel_config.max_workers_metadata,
                    rate_limit_rps=parallel_config.rate_limit_rps,
                    batch_size=parallel_config.batch_size,
                )
            else:
                pages = self.metadata_extractor.process_pages(
                    pages,
                    extract_technologies=True,
                )
        else:
            if self.extract_technologies:
                logger.warning("Technology extraction skipped - metadata extractor not available")
            for page in pages:
                page.setdefault("technologies", [])

        return pages

    def process_completeness(
        self,
        pages: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Assess completeness for all pages.

        Args:
            pages: List of page dictionaries

        Returns:
            Updated pages with completeness fields

        Example:
            >>> pages = pipeline.process_completeness(pages)
        """
        pages = self.completeness_assessor.process_pages(
            pages,
            use_llm=self.use_llm_completeness,
        )

        return pages

    def _recalculate_depth(self, pages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Recalculate depth for all pages from parents.

        This ensures depth is always valid even if the original JSON
        has depth=None or incorrect values.

        Args:
            pages: List of page dictionaries

        Returns:
            Updated pages with correct depth values
        """
        updated = 0
        for page in pages:
            parents = page.get("parents", [])
            calculated_depth = len(parents) + 1
            current_depth = page.get("depth")

            if current_depth is None or current_depth != calculated_depth:
                page["depth"] = calculated_depth
                updated += 1

        if updated > 0:
            logger.info(f"Recalculated depth for {updated} pages")

        return pages

    def process(
        self,
        input_path: str,
        output_path: Optional[str] = None,
        skip_attachments: bool = False,
        skip_technologies: bool = False,
        skip_completeness: bool = False,
        parallel: bool = False,
        parallel_config: Optional[ParallelConfig] = None,
    ) -> List[Dict[str, Any]]:
        """
        Run complete preprocessing pipeline on a JSON file.

        Steps:
        1. Load pages from JSON
        2. Recalculate depth from parents
        3. Process attachments (optional, can be parallelized)
        4. Extract metadata (can be parallelized)
        5. Assess completeness
        6. Save enhanced JSON

        Args:
            input_path: Path to input JSON file
            output_path: Path to output JSON file (default: overwrite input)
            skip_attachments: Skip attachment processing
            skip_technologies: Skip technology extraction
            skip_completeness: Skip completeness assessment
            parallel: Enable parallel processing with default config
            parallel_config: Custom parallel processing configuration

        Returns:
            List of processed page dictionaries

        Example:
            >>> pages = pipeline.process("Data_Storage/confluence_pages.json")
            >>> # With parallel processing
            >>> pages = pipeline.process(
            ...     "input.json",
            ...     parallel=True,
            ...     parallel_config=ParallelConfig(max_workers_pages=8)
            ... )
        """
        input_path = Path(input_path)

        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")

        # Set up parallel config
        if parallel and parallel_config is None:
            parallel_config = ParallelConfig(enabled=True)
        elif parallel_config is not None:
            parallel_config.enabled = True

        # Load pages
        logger.info(f"Loading pages from {input_path}")
        with open(input_path, "r", encoding="utf-8") as f:
            pages = json.load(f)

        logger.info(f"Loaded {len(pages)} pages")

        # Step 0: Recalculate depth for all pages
        pages = self._recalculate_depth(pages)

        # Step 1: Process attachments
        if not skip_attachments and self.process_attachments_enabled:
            pages = self.process_page_attachments(pages, parallel_config=parallel_config)
        else:
            # Ensure fields exist
            for page in pages:
                page.setdefault("attachments", [])
                page.setdefault("attachment_content", "")

        # Step 2: Extract metadata
        old_extract = self.extract_technologies
        if skip_technologies:
            self.extract_technologies = False

        pages = self.process_metadata(pages, parallel_config=parallel_config)

        self.extract_technologies = old_extract

        # Step 3: Assess completeness
        if not skip_completeness:
            pages = self.process_completeness(pages)
        else:
            for page in pages:
                page.setdefault("completeness_score", float("nan"))
                page.setdefault("completeness_summary", "Not assessed")

        # Save results
        if output_path is None:
            output_path = input_path

        output_path = Path(output_path)

        logger.info(f"Saving {len(pages)} pages to {output_path}")

        # Handle NaN for JSON serialization
        def nan_handler(obj):
            if isinstance(obj, float) and math.isnan(obj):
                return None
            raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(pages, f, indent=2, ensure_ascii=False, default=nan_handler)

        logger.info("Preprocessing complete!")

        return pages

    def process_pages_in_memory(
        self,
        pages: List[Dict[str, Any]],
        skip_attachments: bool = False,
        skip_technologies: bool = False,
        skip_completeness: bool = False,
        parallel: bool = False,
        parallel_config: Optional[ParallelConfig] = None,
    ) -> List[Dict[str, Any]]:
        """
        Run preprocessing on pages already in memory.

        Args:
            pages: List of page dictionaries
            skip_attachments: Skip attachment processing
            skip_technologies: Skip technology extraction
            skip_completeness: Skip completeness assessment
            parallel: Enable parallel processing with default config
            parallel_config: Custom parallel processing configuration

        Returns:
            List of processed page dictionaries

        Example:
            >>> pages = pipeline.process_pages_in_memory(pages)
            >>> # With parallel processing
            >>> pages = pipeline.process_pages_in_memory(pages, parallel=True)
        """
        # Set up parallel config
        if parallel and parallel_config is None:
            parallel_config = ParallelConfig(enabled=True)
        elif parallel_config is not None:
            parallel_config.enabled = True

        # Step 0: Recalculate depth for all pages
        pages = self._recalculate_depth(pages)

        # Step 1: Process attachments
        if not skip_attachments and self.process_attachments_enabled:
            pages = self.process_page_attachments(pages, parallel_config=parallel_config)
        else:
            for page in pages:
                page.setdefault("attachments", [])
                page.setdefault("attachment_content", "")

        # Step 2: Extract metadata
        old_extract = self.extract_technologies
        if skip_technologies:
            self.extract_technologies = False

        pages = self.process_metadata(pages, parallel_config=parallel_config)

        self.extract_technologies = old_extract

        # Step 3: Assess completeness
        if not skip_completeness:
            pages = self.process_completeness(pages)
        else:
            for page in pages:
                page.setdefault("completeness_score", float("nan"))
                page.setdefault("completeness_summary", "Not assessed")

        return pages


def main():
    """Run preprocessing pipeline as standalone script.

    Usage:
        python -m preprocessing.processor [input_json] [output_json] [flags]

    Flags:
        --skip-attachments  Skip attachment processing
        --skip-technologies Skip technology extraction
        --skip-completeness Skip completeness assessment
        --parallel          Enable parallel processing
        --workers-pages N   Max concurrent page processing (default: 8)
        --workers-att N     Max concurrent attachments per page (default: 4)
        --rate-limit N      Requests per second limit (default: 10.0)
        --batch-size N      Batch size for parallel processing (default: 50)

    Examples:
        python -m preprocessing.processor
        python -m preprocessing.processor Data_Storage/confluence_pages.json
        python -m preprocessing.processor input.json output.json --parallel
        python -m preprocessing.processor input.json --parallel --workers-pages 16
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="Preprocess Confluence pages for RAG pipeline"
    )
    parser.add_argument(
        "input_path",
        nargs="?",
        default="Data_Storage/confluence_pages.json",
        help="Path to input JSON file (default: Data_Storage/confluence_pages.json)",
    )
    parser.add_argument(
        "output_path",
        nargs="?",
        default=None,
        help="Path to output JSON file (default: same as input)",
    )
    parser.add_argument(
        "--skip-attachments",
        action="store_true",
        help="Skip attachment processing",
    )
    parser.add_argument(
        "--skip-technologies",
        action="store_true",
        help="Skip technology extraction",
    )
    parser.add_argument(
        "--skip-completeness",
        action="store_true",
        help="Skip completeness assessment",
    )
    parser.add_argument(
        "--parallel",
        action="store_true",
        help="Enable parallel processing",
    )
    parser.add_argument(
        "--workers-pages",
        type=int,
        default=8,
        help="Max concurrent page processing (default: 8)",
    )
    parser.add_argument(
        "--workers-att",
        type=int,
        default=4,
        help="Max concurrent attachments per page (default: 4)",
    )
    parser.add_argument(
        "--rate-limit",
        type=float,
        default=10.0,
        help="Requests per second limit (default: 10.0)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=50,
        help="Batch size for parallel processing (default: 50)",
    )

    args = parser.parse_args()

    input_path = args.input_path
    output_path = args.output_path
    skip_attachments = args.skip_attachments
    skip_technologies = args.skip_technologies
    skip_completeness = args.skip_completeness
    use_parallel = args.parallel

    # Build parallel config if enabled
    parallel_config = None
    if use_parallel:
        parallel_config = ParallelConfig(
            enabled=True,
            max_workers_pages=args.workers_pages,
            max_workers_attachments=args.workers_att,
            max_workers_metadata=args.workers_pages,  # Same as pages
            rate_limit_rps=args.rate_limit,
            batch_size=args.batch_size,
        )

    logger.info("=" * 60)
    logger.info("CONFLUENCE RAG PREPROCESSING PIPELINE")
    logger.info("=" * 60)

    # Create pipeline from environment
    try:
        pipeline = PreprocessingPipeline.from_env(
            process_attachments=not skip_attachments,
            extract_technologies=not skip_technologies,
        )
    except Exception as e:
        logger.error(f"Failed to initialize pipeline: {e}")
        sys.exit(1)

    # Run preprocessing
    try:
        pages = pipeline.process(
            input_path,
            output_path,
            skip_attachments=skip_attachments,
            skip_technologies=skip_technologies,
            skip_completeness=skip_completeness,
            parallel=use_parallel,
            parallel_config=parallel_config,
        )

        # Print summary
        print("\n" + "=" * 60)
        print("PREPROCESSING SUMMARY")
        print("=" * 60)
        print(f"\nTotal pages processed: {len(pages)}")

        # Attachment stats
        with_attachments = sum(1 for p in pages if p.get("attachments"))
        with_content = sum(1 for p in pages if p.get("attachment_content"))
        print(f"Pages with attachments: {with_attachments}")
        print(f"Pages with extracted content: {with_content}")

        # Metadata stats
        with_project = sum(1 for p in pages if p.get("parent_project"))
        with_tech = sum(1 for p in pages if p.get("technologies"))
        print(f"Pages with parent_project: {with_project}")
        print(f"Pages with technologies: {with_tech}")

        # Completeness stats
        scored = [p for p in pages if not math.isnan(p.get("completeness_score", float("nan")))]
        if scored:
            avg_score = sum(p["completeness_score"] for p in scored) / len(scored)
            print(f"Pages with completeness score: {len(scored)}")
            print(f"Average completeness score: {avg_score:.1f}/100")

        print(f"\nOutput saved to: {output_path or input_path}")
        print("=" * 60)

    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
