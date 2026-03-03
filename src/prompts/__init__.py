"""
Prompt engineering utilities for the RAG system.

This package provides prompt templates, splitting, and few-shot
examples for various components of the system.

Example:
    >>> from prompts import PromptSplitter
    >>> splitter = PromptSplitter()
    >>> result = splitter.split("What is X? Please be concise.")
"""

from .prompt_splitter import PromptSplitter
from .templates import PromptTemplates
from .few_shot_examples import FewShotExamples

__all__ = [
    "PromptSplitter",
    "PromptTemplates",
    "FewShotExamples",
]
