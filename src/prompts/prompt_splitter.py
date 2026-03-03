"""
Prompt splitting for hybrid retrieval.

This module separates user prompts into core questions (for retrieval)
and extra instructions (for response formatting).

Example:
    >>> from prompts.prompt_splitter import PromptSplitter
    >>> splitter = PromptSplitter()
    >>> result = splitter.split("What is the RAG pipeline? Be concise and use bullet points.")
    >>> print(result.core_question)  # "What is the RAG pipeline"
    >>> print(result.instructions)   # "Be concise and use bullet points."
"""

import re
from dataclasses import dataclass
from typing import List, Optional

from loguru import logger

# Import types for type hints
try:
    from iliad.client import IliadClient
except ImportError:
    pass


@dataclass
class SplitResult:
    """Result of prompt splitting."""

    original: str  # Original prompt
    core_question: str  # Main question for retrieval
    instructions: List[str]  # Formatting/response instructions
    has_instructions: bool  # Whether instructions were found


# Instruction indicators - phrases that signal formatting/style instructions
INSTRUCTION_INDICATORS = [
    # Format instructions
    "be concise",
    "be brief",
    "be detailed",
    "in detail",
    "keep it short",
    "summarize",
    "use bullet points",
    "use numbered list",
    "in a table",
    "as a list",
    "format as",
    "present as",
    # Length instructions
    "in one sentence",
    "in a few sentences",
    "in a paragraph",
    "no more than",
    "at least",
    "maximum",
    "minimum",
    # Style instructions
    "explain like",
    "as if",
    "in simple terms",
    "for beginners",
    "for experts",
    "technically",
    "non-technically",
    # Content instructions
    "include examples",
    "with examples",
    "include code",
    "with code",
    "step by step",
    "include links",
    "cite sources",
    # Exclusion instructions
    "don't include",
    "do not include",
    "without",
    "exclude",
    "skip",
    "ignore",
    # Response instructions
    "please",
    "make sure",
    "ensure",
    "remember to",
]

# Sentence starters that indicate instructions
INSTRUCTION_STARTERS = [
    "please ",
    "make sure ",
    "ensure ",
    "remember to ",
    "don't forget ",
    "be sure to ",
    "keep ",
    "use ",
    "include ",
    "provide ",
    "give ",
    "show ",
    "list ",
    "format ",
]


class PromptSplitter:
    """
    Split prompts into core questions and instructions.

    Separates the retrieval-relevant question from formatting
    and style instructions.

    Attributes:
        iliad_client: Optional Iliad client for LLM-based splitting

    Example:
        >>> splitter = PromptSplitter()
        >>> result = splitter.split("What is X? Please be concise.")
        >>> print(result.core_question)
    """

    def __init__(
        self,
        iliad_client: Optional["IliadClient"] = None,
        use_llm: bool = False,
        model: str = "gpt-5-mini-global",
    ) -> None:
        """Initialize prompt splitter.

        Args:
            iliad_client: Optional Iliad client for LLM splitting
            use_llm: Whether to use LLM for complex cases
            model: Model to use for LLM splitting
        """
        self.iliad_client = iliad_client
        self.use_llm = use_llm and iliad_client is not None
        self.model = model

        logger.info(f"Initialized PromptSplitter (LLM: {self.use_llm})")

    def split(self, prompt: str) -> SplitResult:
        """
        Split a prompt into core question and instructions.

        Args:
            prompt: User prompt to split

        Returns:
            SplitResult with separated components

        Example:
            >>> result = splitter.split("What is Python? Be concise.")
            >>> print(result.core_question)  # "What is Python"
            >>> print(result.instructions)   # ["Be concise."]
        """
        prompt = prompt.strip()

        if not prompt:
            return SplitResult(
                original="",
                core_question="",
                instructions=[],
                has_instructions=False,
            )

        # Try rule-based splitting first
        result = self._split_rule_based(prompt)

        # If complex and LLM available, use LLM
        if self.use_llm and self._is_complex(prompt, result):
            return self._split_with_llm(prompt)

        return result

    def _split_rule_based(self, prompt: str) -> SplitResult:
        """Split using rule-based approach.

        Args:
            prompt: User prompt

        Returns:
            SplitResult
        """
        instructions = []
        core_parts = []

        # Split into sentences
        sentences = self._split_sentences(prompt)

        for sentence in sentences:
            sentence_lower = sentence.lower().strip()

            # Check if this sentence is an instruction
            is_instruction = False

            # Check for instruction indicators
            for indicator in INSTRUCTION_INDICATORS:
                if indicator in sentence_lower:
                    is_instruction = True
                    break

            # Check for instruction starters
            if not is_instruction:
                for starter in INSTRUCTION_STARTERS:
                    if sentence_lower.startswith(starter):
                        is_instruction = True
                        break

            # Check if sentence is a question (less likely to be instruction)
            is_question = sentence.strip().endswith("?")

            if is_instruction and not is_question:
                instructions.append(sentence.strip())
            else:
                core_parts.append(sentence.strip())

        # Reconstruct core question
        core_question = " ".join(core_parts)

        # Clean up core question
        core_question = self._clean_question(core_question)

        return SplitResult(
            original=prompt,
            core_question=core_question,
            instructions=instructions,
            has_instructions=len(instructions) > 0,
        )

    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences.

        Args:
            text: Text to split

        Returns:
            List of sentences
        """
        # Handle common sentence endings
        # Be careful not to split on abbreviations
        pattern = r"(?<=[.!?])\s+(?=[A-Z])"
        sentences = re.split(pattern, text)

        # Also split on newlines
        result = []
        for sent in sentences:
            result.extend(sent.split("\n"))

        return [s.strip() for s in result if s.strip()]

    def _clean_question(self, question: str) -> str:
        """Clean up the core question.

        Args:
            question: Raw question text

        Returns:
            Cleaned question
        """
        # Remove trailing punctuation except ?
        question = question.strip()

        # Remove common instruction prefixes that might remain
        prefixes_to_remove = [
            "please ",
            "can you ",
            "could you ",
            "would you ",
            "i need you to ",
            "i want you to ",
        ]

        question_lower = question.lower()
        for prefix in prefixes_to_remove:
            if question_lower.startswith(prefix):
                question = question[len(prefix):]
                break

        return question.strip()

    def _is_complex(self, prompt: str, result: SplitResult) -> bool:
        """Determine if prompt needs LLM splitting.

        Args:
            prompt: Original prompt
            result: Rule-based split result

        Returns:
            True if LLM splitting is recommended
        """
        # If rule-based found no instructions in a long prompt, might be complex
        if len(prompt) > 200 and not result.has_instructions:
            return True

        # If many potential instructions were found
        if len(result.instructions) > 3:
            return True

        # If core question seems too short relative to original
        if len(result.core_question) < len(prompt) * 0.3:
            return True

        return False

    def _split_with_llm(self, prompt: str) -> SplitResult:
        """Split using LLM.

        Args:
            prompt: User prompt

        Returns:
            SplitResult from LLM
        """
        split_prompt = f"""Analyze this user query and separate it into:
1. CORE QUESTION: The main question or request that should be used for information retrieval
2. INSTRUCTIONS: Any formatting, style, or response instructions

User Query: "{prompt}"

Respond in this exact format:
CORE QUESTION: <the main question>
INSTRUCTIONS: <comma-separated list of instructions, or "none">"""

        try:
            messages = [{"role": "user", "content": split_prompt}]
            response = self.iliad_client.chat(messages=messages, model=self.model)
            content = self.iliad_client.extract_content(response)

            # Parse response
            core_question = ""
            instructions = []

            for line in content.strip().split("\n"):
                if line.startswith("CORE QUESTION:"):
                    core_question = line.replace("CORE QUESTION:", "").strip()
                elif line.startswith("INSTRUCTIONS:"):
                    instr_text = line.replace("INSTRUCTIONS:", "").strip()
                    if instr_text.lower() != "none":
                        instructions = [i.strip() for i in instr_text.split(",")]

            return SplitResult(
                original=prompt,
                core_question=core_question or prompt,
                instructions=instructions,
                has_instructions=len(instructions) > 0,
            )

        except Exception as e:
            logger.warning(f"LLM splitting failed: {e}")
            return self._split_rule_based(prompt)

    def reconstruct(
        self,
        core_question: str,
        instructions: List[str],
        context: str = "",
    ) -> str:
        """Reconstruct a prompt with context and instructions.

        Args:
            core_question: Core question
            instructions: List of instructions
            context: Retrieved context to include

        Returns:
            Reconstructed prompt for LLM
        """
        parts = []

        if context:
            parts.append(f"Context:\n{context}\n")

        parts.append(f"Question: {core_question}")

        if instructions:
            parts.append(f"\nInstructions: {', '.join(instructions)}")

        return "\n".join(parts)

    def get_retrieval_query(self, prompt: str) -> str:
        """Get optimized query for retrieval.

        Args:
            prompt: User prompt

        Returns:
            Query optimized for semantic search
        """
        result = self.split(prompt)
        return result.core_question or prompt
