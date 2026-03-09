"""
Preprocessing package for Confluence RAG system.

This package provides components for preprocessing Confluence page data
before vectorization, including:
- Attachment fetching and text extraction (with parallel processing)
- Attachment deduplication using LLM
- Metadata extraction (parent project, main project, technologies)
- Project completeness assessment
- Full preprocessing orchestration
- Parallel processing utilities

Modules:
    attachment_fetcher: Fetch and process Confluence attachments
    attachment_deduplicator: Detect and compact duplicate attachments
    metadata_extractor: Extract parent_project, main_project, and technologies
    completeness_assessor: Assess project page completeness
    processor: Orchestrate all preprocessing steps
    parallel: Parallel processing utilities (RateLimiter, ParallelProcessor)

Example:
    >>> from preprocessing import PreprocessingPipeline, ParallelConfig
    >>> pipeline = PreprocessingPipeline.from_env()
    >>> # Sequential processing
    >>> pipeline.process("Data_Storage/confluence_pages.json")
    >>> # Parallel processing
    >>> pipeline.process("input.json", parallel=True)
"""

from .attachment_fetcher import AttachmentFetcher
from .attachment_deduplicator import AttachmentDeduplicator
from .metadata_extractor import MetadataExtractor
from .completeness_assessor import CompletenessAssessor
from .processor import PreprocessingPipeline, ParallelConfig
from .parallel import ParallelProcessor, RateLimiter, ProcessingResult

__all__ = [
    "AttachmentFetcher",
    "AttachmentDeduplicator",
    "MetadataExtractor",
    "CompletenessAssessor",
    "PreprocessingPipeline",
    "ParallelConfig",
    "ParallelProcessor",
    "RateLimiter",
    "ProcessingResult",
]
