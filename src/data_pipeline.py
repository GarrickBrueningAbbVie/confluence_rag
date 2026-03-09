#!/usr/bin/env python
"""
End-to-end pipeline for Confluence RAG data acquisition and preprocessing.

This script performs the complete data pipeline:
1. Fetch all pages from Confluence
2. Preprocess pages (metadata, completeness scores)
3. Vectorize for RAG

Usage:
    python -m src.data_pipeline                      # Full pipeline (fast mode)
    python -m src.data_pipeline --fetch-only         # Only fetch from Confluence
    python -m src.data_pipeline --preprocess-only    # Only preprocess existing data
    python -m src.data_pipeline --vectorize-only     # Only vectorize existing data
    python -m src.data_pipeline --with-technologies  # Enable technology extraction
    python -m src.data_pipeline --with-completeness  # Enable completeness assessment
    python -m src.data_pipeline --with-attachments   # Enable attachment processing
"""

import argparse
import json
import math
import os
import sys
from pathlib import Path

# Determine project root (parent of src/)
SRC_DIR = Path(__file__).parent
PROJECT_ROOT = SRC_DIR.parent

# Add src to path for imports
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from dotenv import load_dotenv
from loguru import logger

# Load environment variables
load_dotenv()


def fetch_confluence_pages(space_key: str = None) -> list:
    """Fetch all pages from Confluence."""
    from confluence.rest_client import ConfluenceRestClient

    # Get credentials from environment
    confluence_url = os.getenv("CONFLUENCE_URL")
    username = os.getenv("CONFLUENCE_USERNAME")
    api_token = os.getenv("CONFLUENCE_API_TOKEN")
    space_key = space_key or os.getenv("CONFLUENCE_SPACE_KEY", "DSA")

    # Validate credentials
    if not all([confluence_url, username, api_token]):
        logger.error("Missing required environment variables")
        logger.error("Please ensure the following are set in your .env file:")
        logger.error("  - CONFLUENCE_URL")
        logger.error("  - CONFLUENCE_USERNAME")
        logger.error("  - CONFLUENCE_API_TOKEN")
        sys.exit(1)

    logger.info("=" * 60)
    logger.info("STEP 1: FETCHING CONFLUENCE PAGES")
    logger.info("=" * 60)
    logger.info(f"Confluence URL: {confluence_url}")
    logger.info(f"Space Key: {space_key}")

    # Try Bearer token authentication first
    logger.info("Attempting Bearer token authentication...")
    client = ConfluenceRestClient(
        base_url=confluence_url,
        username=username,
        api_token=api_token,
        verify_ssl=False,
        auth_type="bearer",
    )

    if not client.test_connection():
        logger.info("Bearer token failed. Trying Basic authentication...")
        client = ConfluenceRestClient(
            base_url=confluence_url,
            username=username,
            api_token=api_token,
            verify_ssl=False,
            auth_type="basic",
        )
        if not client.test_connection():
            logger.error("Authentication failed with both methods")
            sys.exit(1)

    logger.info("Authentication successful!")

    # Fetch all pages
    pages = client.get_all_pages_from_space(space_key)
    logger.info(f"Retrieved {len(pages)} pages from Confluence")

    # Save to JSON
    output_dir = PROJECT_ROOT / "Data_Storage"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "confluence_pages.json"

    logger.info(f"Saving pages to: {output_file}")
    client.save_pages_to_json(pages, str(output_file))

    logger.info("Confluence fetch complete!")
    return pages


def preprocess_pages(
    input_path: str = None,
    output_path: str = None,
    skip_attachments: bool = True,
    skip_technologies: bool = True,
    skip_completeness: bool = False,
) -> list:
    """Preprocess Confluence pages with metadata extraction."""

    logger.info("=" * 60)
    logger.info("STEP 2: PREPROCESSING PAGES")
    logger.info("=" * 60)

    # Default paths
    data_dir = PROJECT_ROOT / "Data_Storage"
    input_path = Path(input_path) if input_path else data_dir / "confluence_pages.json"
    output_path = (
        Path(output_path) if output_path else data_dir / "confluence_pages_processed.json"
    )

    if not input_path.exists():
        logger.error(f"Input file not found: {input_path}")
        sys.exit(1)

    # Load pages
    logger.info(f"Loading pages from {input_path}")
    with open(input_path, "r", encoding="utf-8") as f:
        pages = json.load(f)

    logger.info(f"Loaded {len(pages)} pages")

    # Recalculate depth for all pages
    logger.info("Recalculating page depths...")
    for page in pages:
        parents = page.get("parents", [])
        page["depth"] = len(parents) + 1

    # Extract parent project (no LLM needed)
    logger.info("Extracting parent projects...")
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

    for page in pages:
        parents = page.get("parents", [])
        title = page.get("title", "")

        parent_project = None
        if parents:
            root_index = None
            for i, parent in enumerate(parents):
                if is_project_root(parent.get("title", "")):
                    root_index = i
                    break

            if root_index is not None:
                project_index = root_index + 1
                if project_index < len(parents):
                    parent_project = parents[project_index].get("title", "")
                else:
                    parent_project = title

        page["parent_project"] = parent_project
        page.setdefault("technologies", [])
        page.setdefault("attachments", [])
        page.setdefault("attachment_content", "")

    # Count pages with parent projects
    with_project = sum(1 for p in pages if p.get("parent_project"))
    logger.info(f"  {with_project}/{len(pages)} pages have parent_project")

    # Technology extraction
    if not skip_technologies:
        logger.info("Extracting technologies from pages...")
        from preprocessing.metadata_extractor import MetadataExtractor

        try:
            from iliad.client import IliadClient, IliadClientConfig

            iliad_config = IliadClientConfig.from_env()
            iliad_client = IliadClient(iliad_config)
            extractor = MetadataExtractor(iliad_client)

            for i, page in enumerate(pages):
                if (i + 1) % 50 == 0:
                    logger.info(f"  Processing page {i + 1}/{len(pages)}")

                content = page.get("content_text", "")
                if content and len(content.strip()) > 100:
                    try:
                        techs = extractor.extract_technologies(content, page.get("title", ""))
                        page["technologies"] = techs
                    except Exception as e:
                        logger.debug(f"Tech extraction failed for '{page.get('title', '')}': {e}")
                        page["technologies"] = []

            with_tech = sum(1 for p in pages if p.get("technologies"))
            logger.info(f"  {with_tech}/{len(pages)} pages have technologies")

        except Exception as e:
            logger.warning(f"Technology extraction failed: {e}")
            logger.warning("Skipping technology extraction")
    else:
        logger.info("Skipping technology extraction")

    # Completeness assessment
    if not skip_completeness:
        logger.info("Assessing page completeness...")
        from preprocessing.completeness_assessor import CompletenessAssessor

        try:
            from iliad.client import IliadClient, IliadClientConfig

            iliad_config = IliadClientConfig.from_env()
            iliad_client = IliadClient(iliad_config)
            assessor = CompletenessAssessor(iliad_client)
            pages = assessor.process_pages(pages)
        except Exception as e:
            logger.warning(f"Completeness assessment failed: {e}")
            logger.warning("Skipping completeness assessment")
            for page in pages:
                page.setdefault("completeness_score", None)
                page.setdefault("completeness_summary", "Not assessed")
    else:
        logger.info("Skipping completeness assessment")
        for page in pages:
            page.setdefault("completeness_score", None)
            page.setdefault("completeness_summary", "Not assessed")

    # Save results
    logger.info(f"Saving {len(pages)} pages to {output_path}")

    def nan_handler(obj):
        if isinstance(obj, float) and math.isnan(obj):
            return None
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(pages, f, indent=2, ensure_ascii=False, default=nan_handler)

    logger.info("Preprocessing complete!")
    return pages


def vectorize_data(input_path: str = None) -> None:
    """Vectorize preprocessed pages for RAG."""
    from config import ConfigConfluenceRag
    from rag.vectorstore import VectorStore

    logger.info("=" * 60)
    logger.info("STEP 3: VECTORIZING DATA")
    logger.info("=" * 60)

    # Default path
    data_dir = PROJECT_ROOT / "Data_Storage"
    input_path = (
        Path(input_path) if input_path else data_dir / "confluence_pages_processed.json"
    )

    if not input_path.exists():
        # Fall back to non-processed version
        input_path = data_dir / "confluence_pages.json"

    if not input_path.exists():
        logger.error(f"Input file not found: {input_path}")
        sys.exit(1)

    # Load pages
    logger.info(f"Loading pages from {input_path}")
    with open(input_path, "r", encoding="utf-8") as f:
        pages = json.load(f)

    logger.info(f"Loaded {len(pages)} pages")

    # Initialize vector store (handles embeddings internally)
    vector_store = VectorStore(
        persist_directory=ConfigConfluenceRag.VECTOR_DB_PATH,
        collection_name="confluence_docs",
        embedding_model=ConfigConfluenceRag.EMBEDDING_MODEL,
    )

    # Prepare documents for vectorization
    logger.info("Preparing documents for vectorization...")
    documents = []
    metadatas = []
    ids = []

    for page in pages:
        page_id = page.get("id", "")
        title = page.get("title", "")
        content = page.get("content_text", "")

        if not content or len(content.strip()) < 50:
            continue

        # Create document with metadata
        documents.append(content)
        metadatas.append({
            "id": page_id,
            "title": title,
            "url": page.get("url", ""),
            "space_key": page.get("space_key", ""),
            "parent_project": page.get("parent_project", ""),
            "type": "confluence_page",
        })
        ids.append(f"conf_{page_id}")

    logger.info(f"Prepared {len(documents)} documents for vectorization")

    # Add documents to vector store (embeddings generated internally)
    logger.info("Adding documents to vector store (generating embeddings)...")
    vector_store.add_documents(
        texts=documents,
        metadatas=metadatas,
        ids=ids,
    )

    logger.info(f"Vector store now contains {vector_store.count()} documents")
    logger.info("Vectorization complete!")


def main():
    """Run the end-to-end pipeline."""
    parser = argparse.ArgumentParser(
        description="End-to-end Confluence RAG data pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python -m src.data_pipeline                      # Full pipeline (fast mode)
    python -m src.data_pipeline --fetch-only         # Only fetch from Confluence
    python -m src.data_pipeline --preprocess-only    # Only preprocess existing data
    python -m src.data_pipeline --vectorize-only     # Only vectorize existing data
    python -m src.data_pipeline --with-completeness  # Enable completeness scoring
    python -m src.data_pipeline --with-technologies  # Enable technology extraction
        """,
    )

    parser.add_argument(
        "--fetch-only",
        action="store_true",
        help="Only fetch pages from Confluence",
    )
    parser.add_argument(
        "--preprocess-only",
        action="store_true",
        help="Only preprocess existing data",
    )
    parser.add_argument(
        "--vectorize-only",
        action="store_true",
        help="Only vectorize existing data",
    )
    parser.add_argument(
        "--with-attachments",
        action="store_true",
        help="Enable attachment processing (slow, uses Iliad API)",
    )
    parser.add_argument(
        "--with-technologies",
        action="store_true",
        help="Enable technology extraction (slow, uses Iliad API)",
    )
    parser.add_argument(
        "--with-completeness",
        action="store_true",
        help="Enable completeness assessment (slow, uses Iliad API)",
    )
    parser.add_argument(
        "--skip-vectorize",
        action="store_true",
        help="Skip vectorization step",
    )
    parser.add_argument(
        "--space-key",
        type=str,
        default=None,
        help="Confluence space key (default: from .env)",
    )

    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("CONFLUENCE RAG PIPELINE")
    logger.info("=" * 60)

    # Determine what to run
    run_fetch = not (args.preprocess_only or args.vectorize_only)
    run_preprocess = not (args.fetch_only or args.vectorize_only)
    run_vectorize = not (args.fetch_only or args.preprocess_only or args.skip_vectorize)

    if args.fetch_only:
        run_fetch = True
        run_preprocess = False
        run_vectorize = False

    if args.preprocess_only:
        run_fetch = False
        run_preprocess = True
        run_vectorize = False

    if args.vectorize_only:
        run_fetch = False
        run_preprocess = False
        run_vectorize = True

    # Run pipeline steps
    try:
        if run_fetch:
            fetch_confluence_pages(space_key=args.space_key)

        if run_preprocess:
            preprocess_pages(
                skip_attachments=not args.with_attachments,
                skip_technologies=not args.with_technologies,
                skip_completeness=not args.with_completeness,
            )

        if run_vectorize:
            vectorize_data()

        logger.info("=" * 60)
        logger.info("PIPELINE COMPLETE!")
        logger.info("=" * 60)

    except KeyboardInterrupt:
        logger.warning("Pipeline interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
