"""
Iliad API client package.

This package provides a unified interface for interacting with AbbVie's Iliad API,
including endpoints for:
- Text recognition and OCR
- Document analysis
- Chat completions
- Router/code generation

Modules:
    client: Base client with authentication and retry logic
    recognize: Text extraction from documents and images
    analyze: Document analysis and content extraction
"""

from .client import IliadClient, IliadClientConfig, IliadModel
from .recognize import TextRecognizer
from .analyze import DocumentAnalyzer

__all__ = [
    "IliadClient",
    "IliadClientConfig",
    "IliadModel",
    "TextRecognizer",
    "DocumentAnalyzer",
]
