"""
Duplicate attachment detection and compaction.

This module uses Iliad LLM to identify semantically duplicate attachments
(common in Confluence where the same images are uploaded multiple times)
and compacts them to preserve unique information while reducing noise.

Example:
    >>> from preprocessing.attachment_deduplicator import AttachmentDeduplicator
    >>> from iliad.client import IliadClient, IliadClientConfig
    >>>
    >>> iliad = IliadClient(IliadClientConfig.from_env())
    >>> deduplicator = AttachmentDeduplicator(iliad)
    >>>
    >>> results = [{"filename": "img1.png", "extracted_text": "..."}, ...]
    >>> deduplicated = deduplicator.process_attachments(results)
"""

import re
from typing import Any, Dict, List, Optional, Tuple

from loguru import logger

# Import types for type hints
try:
    from iliad.client import IliadClient
    from iliad.analyze import DocumentAnalyzer
except ImportError:
    pass


class AttachmentDeduplicator:
    """
    Detect and compact duplicate attachments using Iliad LLM.

    Confluence often stores multiple versions of the same image or document.
    This class identifies semantically similar attachments and compacts them
    to reduce noise in the RAG retrieval pipeline.

    Attributes:
        iliad_client: Iliad API client for LLM calls
        model: Model to use for duplicate detection
        analyzer: Document analyzer for LLM operations

    Example:
        >>> deduplicator = AttachmentDeduplicator(iliad_client)
        >>> deduplicated = deduplicator.process_attachments(attachment_results)
    """

    def __init__(
        self,
        iliad_client: "IliadClient",
        model: str = "gpt-4o-mini-global",
    ) -> None:
        """Initialize the attachment deduplicator.

        Args:
            iliad_client: Configured Iliad API client
            model: Model to use for LLM operations
        """
        self.iliad_client = iliad_client
        self.model = model
        self.analyzer = DocumentAnalyzer(iliad_client, default_model=model)

        logger.info(f"Initialized AttachmentDeduplicator with model: {model}")

    def _build_duplicate_detection_prompt(
        self,
        attachments: List[Dict[str, Any]],
    ) -> str:
        """Build prompt for LLM duplicate detection.

        Args:
            attachments: List of attachment results with extracted_text

        Returns:
            Formatted prompt for duplicate detection
        """
        prompt_parts = [
            "Analyze the following attachment contents and identify which ones are duplicates or very similar.",
            "Return ONLY a JSON array of groups, where each group contains the indices (0-based) of similar attachments.",
            "If an attachment is unique, put it in its own group.",
            "",
            "Example output format:",
            '[[0, 2, 5], [1], [3, 4]]',
            "",
            "This means: attachments 0, 2, 5 are similar; attachment 1 is unique; attachments 3 and 4 are similar.",
            "",
            "ATTACHMENTS:",
            "",
        ]

        for i, att in enumerate(attachments):
            filename = att.get("filename", f"attachment_{i}")
            text = att.get("extracted_text", "")[:500]  # Truncate for prompt
            prompt_parts.append(f"[{i}] {filename}:")
            prompt_parts.append(f"{text}")
            prompt_parts.append("")

        prompt_parts.append("Respond with ONLY the JSON array, no other text.")

        return "\n".join(prompt_parts)

    def _parse_duplicate_groups(
        self,
        response: str,
        total_count: int,
    ) -> List[List[int]]:
        """Parse LLM response into duplicate groups.

        Args:
            response: LLM response text
            total_count: Total number of attachments

        Returns:
            List of duplicate groups (each group is list of indices)
        """
        # Try to extract JSON array from response
        import json

        try:
            # Find JSON array in response
            match = re.search(r'\[[\s\S]*\]', response)
            if match:
                groups = json.loads(match.group())
                # Validate groups
                all_indices = set()
                valid_groups = []
                for group in groups:
                    if isinstance(group, list):
                        valid_indices = [i for i in group if isinstance(i, int) and 0 <= i < total_count]
                        if valid_indices:
                            valid_groups.append(valid_indices)
                            all_indices.update(valid_indices)

                # Add any missing indices as singleton groups
                for i in range(total_count):
                    if i not in all_indices:
                        valid_groups.append([i])

                return valid_groups

        except (json.JSONDecodeError, Exception) as e:
            logger.warning(f"Failed to parse duplicate groups: {e}")

        # Fallback: each attachment is its own group
        return [[i] for i in range(total_count)]

    def identify_duplicates(
        self,
        attachments: List[Dict[str, Any]],
    ) -> List[List[int]]:
        """Identify groups of duplicate attachments.

        Uses LLM to compare attachment contents and group semantically
        similar attachments together.

        Args:
            attachments: List of processed attachment results with:
                - filename: Attachment filename
                - extracted_text: Extracted text content
                - file_type: Type of file (image, document, text)

        Returns:
            List of duplicate groups where each group is a list of indices

        Example:
            >>> groups = deduplicator.identify_duplicates(attachments)
            >>> print(groups)  # [[0, 2], [1], [3, 4, 5]]
        """
        if len(attachments) <= 1:
            return [[0]] if attachments else []

        # Build prompt and get LLM response
        prompt = self._build_duplicate_detection_prompt(attachments)

        try:
            response = self.iliad_client.chat(
                messages=[{"role": "user", "content": prompt}],
                model=self.model,
            )

            groups = self._parse_duplicate_groups(response, len(attachments))
            logger.debug(f"Identified {len(groups)} groups from {len(attachments)} attachments")
            return groups

        except Exception as e:
            logger.warning(f"Duplicate detection failed: {e}")
            # Fallback: each attachment is its own group
            return [[i] for i in range(len(attachments))]

    def _build_compaction_prompt(self, duplicate_texts: List[str]) -> str:
        """Build prompt for compacting duplicate attachments.

        Args:
            duplicate_texts: List of extracted texts from duplicate attachments

        Returns:
            Formatted prompt for compaction
        """
        prompt_parts = [
            "The following attachment contents are similar or duplicate.",
            "Create a single, concise summary that preserves ALL unique information from all versions.",
            "Do NOT include redundant information - merge and deduplicate the content.",
            "Keep technical details, specific values, and unique phrases from each version.",
            "",
            "ATTACHMENT CONTENTS:",
            "",
        ]

        for i, text in enumerate(duplicate_texts):
            prompt_parts.append(f"--- Version {i + 1} ---")
            prompt_parts.append(text)
            prompt_parts.append("")

        prompt_parts.append("--- COMBINED SUMMARY ---")

        return "\n".join(prompt_parts)

    def compact_duplicates(
        self,
        contents: List[str],
        group_indices: List[int],
    ) -> str:
        """Compact a group of duplicate attachments into single representation.

        Uses LLM to identify unique information across duplicates and create
        a compact representation that preserves all meaningful content.

        Args:
            contents: List of all attachment contents
            group_indices: Indices of duplicate attachments to compact

        Returns:
            Compacted content string

        Example:
            >>> compacted = deduplicator.compact_duplicates(
            ...     contents=["Text A...", "Text B...", "Text C..."],
            ...     group_indices=[0, 2]  # Compact texts at indices 0 and 2
            ... )
        """
        if len(group_indices) == 1:
            return contents[group_indices[0]]

        # Get duplicate texts
        duplicate_texts = [contents[i] for i in group_indices if i < len(contents)]

        if len(duplicate_texts) <= 1:
            return duplicate_texts[0] if duplicate_texts else ""

        # Build prompt and get compacted result
        prompt = self._build_compaction_prompt(duplicate_texts)

        try:
            compacted = self.iliad_client.chat(
                messages=[{"role": "user", "content": prompt}],
                model=self.model,
            )
            logger.debug(f"Compacted {len(group_indices)} duplicates into single summary")
            return compacted

        except Exception as e:
            logger.warning(f"Compaction failed: {e}")
            # Fallback: return first text
            return duplicate_texts[0]

    def _format_single_attachment(self, attachment: Dict[str, Any]) -> str:
        """Format a single attachment for output.

        Args:
            attachment: Attachment result dictionary

        Returns:
            Formatted attachment text
        """
        filename = attachment.get("filename", "unknown")
        text = attachment.get("extracted_text", "")
        file_type = attachment.get("file_type", "unknown")

        if not text.strip():
            return ""

        return f"[{file_type.upper()}: {filename}]\n{text}"

    def process_attachments(
        self,
        attachment_results: List[Dict[str, Any]],
    ) -> str:
        """Process all attachments with deduplication.

        Main entry point for attachment deduplication. Identifies duplicates,
        compacts similar attachments, and returns combined content.

        Args:
            attachment_results: List of processed attachment dictionaries with:
                - filename: Attachment filename
                - extracted_text: Extracted text content
                - file_type: Type of file
                - success: Whether extraction succeeded

        Returns:
            Combined, deduplicated content string

        Example:
            >>> results = attachment_fetcher.process_all_attachments(page_id)
            >>> deduplicated = deduplicator.process_attachments(results)
            >>> page["attachment_content"] = deduplicated
        """
        # Filter to successful extractions with content
        valid_results = [
            r for r in attachment_results
            if r.get("success") and r.get("extracted_text", "").strip()
        ]

        if not valid_results:
            logger.debug("No valid attachment results to process")
            return ""

        if len(valid_results) == 1:
            return self._format_single_attachment(valid_results[0])

        # Identify duplicates
        logger.info(f"Checking {len(valid_results)} attachments for duplicates")
        duplicate_groups = self.identify_duplicates(valid_results)

        # Count duplicates found
        duplicate_count = sum(1 for g in duplicate_groups if len(g) > 1)
        if duplicate_count > 0:
            logger.info(f"Found {duplicate_count} groups of duplicate attachments")

        # Compact each group and combine
        compacted_parts = []
        contents = [r.get("extracted_text", "") for r in valid_results]

        for group in duplicate_groups:
            if len(group) == 1:
                # Unique attachment - use as-is
                att = valid_results[group[0]]
                formatted = self._format_single_attachment(att)
                if formatted:
                    compacted_parts.append(formatted)
            else:
                # Duplicate group - compact
                compacted = self.compact_duplicates(contents, group)
                filenames = [valid_results[i].get("filename", "unknown") for i in group]
                header = f"[COMPACTED FROM {len(group)} SIMILAR ATTACHMENTS: {', '.join(filenames)}]"
                compacted_parts.append(f"{header}\n{compacted}")

        result = "\n\n".join(compacted_parts)
        logger.info(
            f"Deduplication complete: {len(valid_results)} attachments -> "
            f"{len(duplicate_groups)} unique entries"
        )

        return result
