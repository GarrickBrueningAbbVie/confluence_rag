"""
Confluence attachment fetcher and processor.

This module downloads attachments from Confluence pages and processes them
using Iliad API for text extraction and analysis.

Supported attachment types:
- Documents: PDF, DOCX, PPTX, XLSX, DOC, PPT, XLS
- Images: PNG, JPG, JPEG, GIF, TIFF (analyzed via multimodal LLM for detailed descriptions)
- Text: TXT, CSV, JSON, XML, MD

Example:
    >>> from preprocessing.attachment_fetcher import AttachmentFetcher
    >>> from confluence.rest_client import ConfluenceRestClient
    >>> from iliad.client import IliadClient, IliadClientConfig
    >>>
    >>> confluence = ConfluenceRestClient(...)
    >>> iliad = IliadClient(IliadClientConfig.from_env())
    >>> fetcher = AttachmentFetcher(confluence, iliad)
    >>>
    >>> # Process all attachments for a page
    >>> content = fetcher.process_all_page_attachments("123456")
"""

import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from loguru import logger

from .parallel import ParallelProcessor, ProcessingResult

# Import types for type hints
try:
    from confluence.rest_client import ConfluenceRestClient
    from iliad.client import IliadClient
    from iliad.recognize import TextRecognizer
    from iliad.analyze import DocumentAnalyzer
except ImportError:
    # Allow module to be imported for type checking
    pass


# File type categorization
DOCUMENT_EXTENSIONS = {".pdf", ".docx", ".pptx", ".xlsx", ".doc", ".ppt", ".xls", ".rtf", ".odt"}
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".tiff", ".tif", ".bmp"}
TEXT_EXTENSIONS = {".txt", ".csv", ".json", ".xml", ".md", ".html"}

# Maximum file size to process (in bytes) - 50MB default
DEFAULT_MAX_FILE_SIZE = 50 * 1024 * 1024


class AttachmentFetcher:
    """
    Fetch and process Confluence attachments.

    Downloads attachments from Confluence pages and processes them using
    Iliad API for text extraction. Supports documents, images (via multimodal
    LLM analysis for detailed descriptions), and text files.

    Attributes:
        confluence_client: Confluence REST client for fetching attachments
        iliad_client: Iliad API client for text extraction
        storage_path: Path to store downloaded attachments
        max_file_size: Maximum file size to process (bytes)

    Example:
        >>> fetcher = AttachmentFetcher(confluence_client, iliad_client)
        >>> attachments = fetcher.fetch_page_attachments("123456")
        >>> content = fetcher.process_all_page_attachments("123456")
    """

    def __init__(
        self,
        confluence_client: "ConfluenceRestClient",
        iliad_client: "IliadClient",
        storage_path: str = "Data_Storage/attachments",
        max_file_size: int = DEFAULT_MAX_FILE_SIZE,
    ) -> None:
        """Initialize attachment fetcher.

        Args:
            confluence_client: Configured Confluence REST client
            iliad_client: Configured Iliad API client
            storage_path: Directory to store downloaded attachments
            max_file_size: Maximum file size to process (bytes)
        """
        self.confluence_client = confluence_client
        self.iliad_client = iliad_client
        self.storage_path = Path(storage_path)
        self.max_file_size = max_file_size

        # Initialize Iliad helper classes
        self.recognizer = TextRecognizer(iliad_client)
        self.analyzer = DocumentAnalyzer(iliad_client)

        # Ensure storage directory exists
        self.storage_path.mkdir(parents=True, exist_ok=True)

        logger.info(f"Initialized AttachmentFetcher with storage: {self.storage_path}")

    def _get_file_category(self, filename: str) -> str:
        """Determine file category from filename.

        Args:
            filename: Attachment filename

        Returns:
            Category string: 'document', 'image', 'text', or 'unsupported'
        """
        ext = Path(filename).suffix.lower()

        if ext in DOCUMENT_EXTENSIONS:
            return "document"
        elif ext in IMAGE_EXTENSIONS:
            return "image"
        elif ext in TEXT_EXTENSIONS:
            return "text"
        else:
            return "unsupported"

    def fetch_page_attachments(self, page_id: str) -> List[Dict[str, Any]]:
        """
        Fetch attachment metadata for a page.

        Args:
            page_id: Confluence page ID

        Returns:
            List of attachment metadata dictionaries

        Example:
            >>> attachments = fetcher.fetch_page_attachments("123456")
            >>> for att in attachments:
            ...     print(f"{att['title']} - {att['fileSize']} bytes")
        """
        return self.confluence_client.get_attachments(page_id)

    def download_attachment(
        self,
        attachment: Dict[str, Any],
        page_id: Optional[str] = None,
    ) -> Optional[Path]:
        """
        Download a single attachment to local storage.

        Args:
            attachment: Attachment metadata dictionary
            page_id: Optional page ID for organizing storage

        Returns:
            Path to downloaded file, or None if failed

        Example:
            >>> path = fetcher.download_attachment(attachment, page_id="123456")
            >>> if path:
            ...     print(f"Downloaded to {path}")
        """
        download_url = attachment.get("download_url", "")
        filename = attachment.get("title", "unknown")
        file_size = attachment.get("fileSize", 0)

        if not download_url:
            logger.warning(f"No download URL for attachment: {filename}")
            return None

        # Check file size
        if file_size > self.max_file_size:
            logger.warning(
                f"Attachment {filename} exceeds max size "
                f"({file_size / 1024 / 1024:.1f}MB > {self.max_file_size / 1024 / 1024:.1f}MB)"
            )
            return None

        # Determine output path
        if page_id:
            output_dir = self.storage_path / page_id
        else:
            output_dir = self.storage_path

        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / filename

        # Download
        try:
            self.confluence_client.download_attachment(download_url, str(output_path))
            logger.debug(f"Downloaded attachment: {filename}")
            return output_path

        except Exception as e:
            logger.error(f"Failed to download {filename}: {e}")
            return None

    def process_attachment(
        self,
        file_path: Union[str, Path],
    ) -> Dict[str, Any]:
        """
        Process a downloaded attachment using Iliad API.

        Extracts text from the attachment based on file type:
        - Documents: Use /recognize endpoint
        - Images: Use multimodal chat for detailed analysis and descriptions
        - Text files: Read directly

        Args:
            file_path: Path to the attachment file

        Returns:
            Dict with:
            - extracted_text: Text content from file
            - description: Brief description (for complex files)
            - file_type: Category of file
            - success: Whether extraction succeeded

        Example:
            >>> result = fetcher.process_attachment("/path/to/file.pdf")
            >>> if result['success']:
            ...     print(result['extracted_text'][:100])
        """
        path = Path(file_path)
        filename = path.name
        category = self._get_file_category(filename)

        result = {
            "filename": filename,
            "file_type": category,
            "extracted_text": "",
            "description": "",
            "success": False,
        }

        if not path.exists():
            logger.warning(f"File not found: {file_path}")
            return result

        try:
            if category == "document":
                # Use Iliad recognize for documents
                result["extracted_text"] = self.recognizer.recognize_file(str(path))
                result["success"] = True
                logger.debug(f"Extracted {len(result['extracted_text'])} chars from {filename}")

            elif category == "image":
                # Use multimodal LLM for detailed image analysis (not just OCR)
                result["extracted_text"] = self.recognizer.analyze_image(str(path))
                result["success"] = True
                logger.debug(f"Image analysis extracted {len(result['extracted_text'])} chars from {filename}")

            elif category == "text":
                # Read text files directly
                with open(path, "r", encoding="utf-8", errors="ignore") as f:
                    result["extracted_text"] = f.read()
                result["success"] = True
                logger.debug(f"Read {len(result['extracted_text'])} chars from {filename}")

            else:
                logger.info(f"Skipping unsupported file type: {filename}")
                return result

            # Generate description for non-trivial content
            if result["extracted_text"] and len(result["extracted_text"]) > 100:
                try:
                    result["description"] = self._generate_description(
                        result["extracted_text"],
                        filename,
                    )
                except Exception as e:
                    logger.warning(f"Failed to generate description for {filename}: {e}")

        except Exception as e:
            logger.error(f"Failed to process attachment {filename}: {e}")

        return result

    def _generate_description(
        self,
        content: str,
        filename: str,
        max_length: int = 200,
    ) -> str:
        """Generate a brief description of attachment content.

        Args:
            content: Extracted text content
            filename: Original filename for context
            max_length: Maximum description length

        Returns:
            Brief description of the content
        """
        # Use first portion of content for analysis
        sample = content[:4000]

        prompt = f"""Provide a one-sentence description of this attachment content.
Filename: {filename}

Content:
{sample}

Description (keep under {max_length} characters):"""

        try:
            response = self.analyzer.analyze_text(sample, prompt)
            return response.strip()[:max_length]
        except Exception:
            return ""

    def process_all_page_attachments(
        self,
        page_id: str,
        cleanup: bool = True,
    ) -> str:
        """
        Process all attachments for a page and return combined content.

        Downloads all attachments, extracts text from each, and combines
        the results into a single string.

        Args:
            page_id: Confluence page ID
            cleanup: Whether to delete downloaded files after processing

        Returns:
            Combined text content from all attachments

        Example:
            >>> content = fetcher.process_all_page_attachments("123456")
            >>> print(f"Extracted {len(content)} characters from attachments")
        """
        # Fetch attachment metadata
        attachments = self.fetch_page_attachments(page_id)

        if not attachments:
            logger.debug(f"No attachments found for page {page_id}")
            return ""

        logger.info(f"Processing {len(attachments)} attachments for page {page_id}")

        all_content = []
        downloaded_files = []

        for attachment in attachments:
            # Download attachment
            file_path = self.download_attachment(attachment, page_id)

            if file_path:
                downloaded_files.append(file_path)

                # Process attachment
                result = self.process_attachment(file_path)

                if result["success"] and result["extracted_text"]:
                    # Add content with header
                    header = f"\n\n--- Attachment: {result['filename']} ---\n"
                    if result["description"]:
                        header += f"Description: {result['description']}\n"
                    header += "\n"

                    all_content.append(header + result["extracted_text"])

        # Cleanup downloaded files if requested
        if cleanup:
            for file_path in downloaded_files:
                try:
                    file_path.unlink()
                    logger.debug(f"Cleaned up: {file_path}")
                except Exception as e:
                    logger.warning(f"Failed to cleanup {file_path}: {e}")

            # Try to remove page directory if empty
            try:
                page_dir = self.storage_path / page_id
                if page_dir.exists() and not any(page_dir.iterdir()):
                    page_dir.rmdir()
            except Exception:
                pass

        combined = "\n".join(all_content)
        logger.info(
            f"Extracted {len(combined)} chars from {len(all_content)} attachments for page {page_id}"
        )

        return combined

    def process_attachments_batch(
        self,
        page_ids: List[str],
        cleanup: bool = True,
    ) -> Dict[str, str]:
        """
        Process attachments for multiple pages.

        Args:
            page_ids: List of page IDs to process
            cleanup: Whether to delete downloaded files after processing

        Returns:
            Dictionary mapping page IDs to extracted content

        Example:
            >>> results = fetcher.process_attachments_batch(["123", "456", "789"])
            >>> for page_id, content in results.items():
            ...     print(f"Page {page_id}: {len(content)} chars")
        """
        results = {}

        logger.info(f"Processing attachments for {len(page_ids)} pages")

        for i, page_id in enumerate(page_ids, 1):
            logger.info(f"Processing page {i}/{len(page_ids)}: {page_id}")

            try:
                content = self.process_all_page_attachments(page_id, cleanup=cleanup)
                results[page_id] = content
            except Exception as e:
                logger.error(f"Failed to process page {page_id}: {e}")
                results[page_id] = ""

        successful = sum(1 for c in results.values() if c)
        logger.info(f"Batch complete: {successful}/{len(page_ids)} pages had extractable attachments")

        return results

    # -------------------------------------------------------------------------
    # Parallel Processing Methods
    # -------------------------------------------------------------------------

    def _download_and_process_attachment(
        self,
        args: Tuple[Dict[str, Any], str],
    ) -> Dict[str, Any]:
        """Download and process a single attachment (for parallel execution).

        Args:
            args: Tuple of (attachment_metadata, page_id)

        Returns:
            Processing result dictionary
        """
        attachment, page_id = args

        # Download
        file_path = self.download_attachment(attachment, page_id)
        if not file_path:
            return {
                "filename": attachment.get("title", "unknown"),
                "file_type": "unknown",
                "extracted_text": "",
                "description": "",
                "success": False,
                "file_path": None,
            }

        # Process
        result = self.process_attachment(file_path)
        result["file_path"] = file_path
        return result

    def process_attachments_parallel(
        self,
        attachments: List[Dict[str, Any]],
        page_id: str,
        max_workers: int = 4,
        rate_limit_rps: Optional[float] = None,
        cleanup: bool = True,
    ) -> List[Dict[str, Any]]:
        """Process multiple attachments in parallel within a single page.

        Downloads and processes attachments concurrently using a thread pool.
        This is useful when a page has multiple image attachments that each
        require LLM analysis.

        Args:
            attachments: List of attachment metadata dictionaries
            page_id: Confluence page ID for organizing storage
            max_workers: Maximum concurrent processing threads
            rate_limit_rps: Optional requests per second limit
            cleanup: Whether to delete downloaded files after processing

        Returns:
            List of processing result dictionaries

        Example:
            >>> attachments = fetcher.fetch_page_attachments("123456")
            >>> results = fetcher.process_attachments_parallel(attachments, "123456")
            >>> successful = [r for r in results if r['success']]
        """
        if not attachments:
            return []

        logger.info(
            f"Processing {len(attachments)} attachments in parallel "
            f"(workers={max_workers}, rps={rate_limit_rps})"
        )

        # Prepare arguments for parallel processing
        args_list = [(att, page_id) for att in attachments]

        # Process in parallel
        processor = ParallelProcessor(
            max_workers=max_workers,
            rate_limit_rps=rate_limit_rps,
        )

        try:
            results = processor.map(
                self._download_and_process_attachment,
                args_list,
                desc=f"Attachments for page {page_id}",
            )

            # Extract results and collect file paths for cleanup
            output = []
            downloaded_files = []

            for proc_result in results:
                if proc_result.success and proc_result.value:
                    result = proc_result.value
                    if result.get("file_path"):
                        downloaded_files.append(result["file_path"])
                    output.append(result)
                else:
                    # Failed processing - create error result
                    output.append({
                        "filename": "unknown",
                        "file_type": "unknown",
                        "extracted_text": "",
                        "description": "",
                        "success": False,
                        "error": str(proc_result.error) if proc_result.error else "Unknown error",
                    })

            # Cleanup
            if cleanup:
                for file_path in downloaded_files:
                    try:
                        if file_path and Path(file_path).exists():
                            Path(file_path).unlink()
                    except Exception as e:
                        logger.warning(f"Failed to cleanup {file_path}: {e}")

                # Remove page directory if empty
                try:
                    page_dir = self.storage_path / page_id
                    if page_dir.exists() and not any(page_dir.iterdir()):
                        page_dir.rmdir()
                except Exception:
                    pass

            return output

        finally:
            processor.shutdown()

    def process_all_page_attachments_parallel(
        self,
        page_id: str,
        max_workers: int = 4,
        rate_limit_rps: Optional[float] = None,
        cleanup: bool = True,
    ) -> str:
        """Process all attachments for a page in parallel.

        Parallel version of process_all_page_attachments() that processes
        multiple attachments concurrently.

        Args:
            page_id: Confluence page ID
            max_workers: Maximum concurrent processing threads
            rate_limit_rps: Optional requests per second limit
            cleanup: Whether to delete downloaded files after processing

        Returns:
            Combined text content from all attachments

        Example:
            >>> content = fetcher.process_all_page_attachments_parallel("123456")
            >>> print(f"Extracted {len(content)} characters")
        """
        # Fetch attachment metadata
        attachments = self.fetch_page_attachments(page_id)

        if not attachments:
            logger.debug(f"No attachments found for page {page_id}")
            return ""

        # Process in parallel
        results = self.process_attachments_parallel(
            attachments,
            page_id,
            max_workers=max_workers,
            rate_limit_rps=rate_limit_rps,
            cleanup=cleanup,
        )

        # Combine results
        all_content = []
        for result in results:
            if result.get("success") and result.get("extracted_text"):
                header = f"\n\n--- Attachment: {result['filename']} ---\n"
                if result.get("description"):
                    header += f"Description: {result['description']}\n"
                header += "\n"
                all_content.append(header + result["extracted_text"])

        combined = "\n".join(all_content)
        logger.info(
            f"Extracted {len(combined)} chars from {len(all_content)} attachments "
            f"for page {page_id} (parallel)"
        )

        return combined

    def _process_page_attachments_wrapper(self, page_id: str) -> Tuple[str, str]:
        """Wrapper for parallel page processing.

        Args:
            page_id: Confluence page ID

        Returns:
            Tuple of (page_id, extracted_content)
        """
        try:
            content = self.process_all_page_attachments_parallel(
                page_id,
                max_workers=4,  # Within-page parallelization
                cleanup=True,
            )
            return (page_id, content)
        except Exception as e:
            logger.error(f"Failed to process page {page_id}: {e}")
            return (page_id, "")

    def process_pages_parallel(
        self,
        page_ids: List[str],
        max_workers_pages: int = 8,
        max_workers_attachments: int = 4,
        rate_limit_rps: Optional[float] = 10.0,
        batch_size: int = 50,
        cleanup: bool = True,
    ) -> Dict[str, str]:
        """Process attachments for multiple pages in parallel.

        Two-level parallelization:
        1. Process multiple pages concurrently (max_workers_pages)
        2. Within each page, process attachments concurrently (max_workers_attachments)

        Includes batch processing to limit concurrent API load and optional
        rate limiting for API calls.

        Args:
            page_ids: List of Confluence page IDs
            max_workers_pages: Maximum concurrent page processing
            max_workers_attachments: Maximum concurrent attachments per page
            rate_limit_rps: Optional requests per second limit
            batch_size: Number of pages to process per batch (default 50)
            cleanup: Whether to delete downloaded files after processing

        Returns:
            Dictionary mapping page IDs to extracted content

        Example:
            >>> results = fetcher.process_pages_parallel(
            ...     page_ids[:100],
            ...     max_workers_pages=8,
            ...     batch_size=50,
            ...     rate_limit_rps=10.0,
            ... )
            >>> successful = sum(1 for c in results.values() if c)
        """
        if not page_ids:
            return {}

        logger.info(
            f"Processing {len(page_ids)} pages in parallel "
            f"(page_workers={max_workers_pages}, att_workers={max_workers_attachments}, "
            f"batch_size={batch_size}, rps={rate_limit_rps})"
        )

        # Store attachment worker config for nested processing
        self._parallel_attachment_workers = max_workers_attachments

        # Create processor for page-level parallelization
        processor = ParallelProcessor(
            max_workers=max_workers_pages,
            rate_limit_rps=rate_limit_rps,
        )

        try:
            # Process in batches
            results = processor.map_batched(
                self._process_page_attachments_wrapper,
                page_ids,
                batch_size=batch_size,
                desc="Pages",
                pause_between_batches=2.0,  # Give API time to recover
            )

            # Convert to dictionary
            output = {}
            for proc_result in results:
                if proc_result.success and proc_result.value:
                    page_id, content = proc_result.value
                    output[page_id] = content
                else:
                    # Use original item as page_id for failed results
                    output[proc_result.item] = ""

            successful = sum(1 for c in output.values() if c)
            logger.info(
                f"Parallel batch complete: {successful}/{len(page_ids)} pages "
                f"had extractable attachments"
            )

            return output

        finally:
            processor.shutdown()
            if hasattr(self, "_parallel_attachment_workers"):
                delattr(self, "_parallel_attachment_workers")
