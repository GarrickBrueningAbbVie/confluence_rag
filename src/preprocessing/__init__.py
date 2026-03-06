"""
Preprocessing package for Confluence RAG system.

This package provides components for preprocessing Confluence page data
before vectorization, including:
- Attachment fetching and text extraction
- Metadata extraction (parent project, technologies)
- Project completeness assessment
- Full preprocessing orchestration

Modules:
    attachment_fetcher: Fetch and process Confluence attachments
    metadata_extractor: Extract parent_project and technologies
    completeness_assessor: Assess project page completeness
    processor: Orchestrate all preprocessing steps

Example:
    >>> from preprocessing import PreprocessingPipeline
    >>> pipeline = PreprocessingPipeline.from_env()
    >>> pipeline.process("Data_Storage/confluence_pages.json")
"""

from .attachment_fetcher import AttachmentFetcher
from .metadata_extractor import MetadataExtractor
from .completeness_assessor import CompletenessAssessor
from .processor import PreprocessingPipeline

__all__ = [
    "AttachmentFetcher",
    "MetadataExtractor",
    "CompletenessAssessor",
    "PreprocessingPipeline",
]
