"""
Prompt engineering utilities for the RAG system.

This package provides prompt templates and splitting utilities
for various components of the system.

Example:
    >>> from prompts import PromptSplitter
    >>> splitter = PromptSplitter()
    >>> result = splitter.split("What is X? Please be concise.")
"""

from .prompt_splitter import PromptSplitter
from .templates import PromptTemplates

__all__ = [
    "PromptSplitter",
    "PromptTemplates",
]
