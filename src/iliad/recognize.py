"""
Text recognition using Iliad API.

This module provides high-level text extraction from various file types
including documents, images, and audio files.

Supported file types:
- Documents: PDF, DOCX, PPTX, RTF, DOC, PPT, XLS, XLSX, ODS, ODT, ODP
- Text: TXT, MD, JSON, XML, HTML, CSV, TSV
- Images: JPG, JPEG, PNG, TIFF (via OCR)
- Audio: AMR, FLAC, M4A, MP3, MP4, OGG, WEBM, WAV (via transcription)

Example:
    >>> from iliad.recognize import TextRecognizer
    >>> from iliad.client import IliadClient, IliadClientConfig
    >>> client = IliadClient(IliadClientConfig.from_env())
    >>> recognizer = TextRecognizer(client)
    >>> text = recognizer.recognize_file("/path/to/document.pdf")
"""

from pathlib import Path
from typing import Dict, List, Optional, Union

from loguru import logger

from .client import IliadClient


# File type mappings
DOCUMENT_EXTENSIONS = {
    ".pdf", ".docx", ".pptx", ".rtf", ".doc", ".ppt", ".xls",
    ".xlsx", ".xlsm", ".odt", ".ods", ".odp", ".odg",
}

TEXT_EXTENSIONS = {
    ".txt", ".md", ".json", ".xml", ".html", ".xhtml",
    ".csv", ".tsv", ".epub", ".eml",
}

IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".tiff", ".tif",
}

AUDIO_EXTENSIONS = {
    ".amr", ".flac", ".m4a", ".mp3", ".mp4",
    ".ogg", ".webm", ".wav",
}


class TextRecognizer:
    """
    High-level text extraction and image analysis from various file formats.

    Automatically selects the appropriate Iliad API endpoint based on
    file type:
    - Documents/text: /recognize endpoint
    - Images: Multimodal chat for detailed analysis (vision LLM)
    - Audio: /transcribe/aws (async)

    For images, this class uses multimodal LLM capabilities to provide
    detailed analysis including descriptions of diagrams, charts, and
    visual elements - not just OCR text extraction.

    Attributes:
        client: IliadClient instance for API calls
        default_language: Default source language for recognition

    Example:
        >>> recognizer = TextRecognizer(iliad_client)
        >>> text = recognizer.recognize_file("document.pdf")
        >>> analysis = recognizer.analyze_image("diagram.png")
    """

    def __init__(
        self,
        client: IliadClient,
        default_language: str = "en-US",
    ) -> None:
        """Initialize text recognizer.

        Args:
            client: Configured IliadClient instance
            default_language: Default source language code
        """
        self.client = client
        self.default_language = default_language
        logger.info("Initialized TextRecognizer")

    def _get_file_type(self, file_path: Union[str, Path]) -> str:
        """Determine file type category from path.

        Args:
            file_path: Path to file

        Returns:
            Category string: 'document', 'text', 'image', 'audio', or 'unknown'
        """
        path = Path(file_path)
        ext = path.suffix.lower()

        if ext in DOCUMENT_EXTENSIONS:
            return "document"
        elif ext in TEXT_EXTENSIONS:
            return "text"
        elif ext in IMAGE_EXTENSIONS:
            return "image"
        elif ext in AUDIO_EXTENSIONS:
            return "audio"
        else:
            return "unknown"

    def recognize_file(
        self,
        file_path: Union[str, Path],
        source_language: Optional[str] = None,
    ) -> str:
        """Extract text from a single file.

        Automatically selects the appropriate API endpoint based on file type.

        Args:
            file_path: Path to file
            source_language: Language code (defaults to self.default_language)

        Returns:
            Extracted text content

        Raises:
            ValueError: If file type is not supported
            FileNotFoundError: If file does not exist

        Example:
            >>> text = recognizer.recognize_file("report.pdf")
            >>> print(text[:100])
        """
        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        file_type = self._get_file_type(path)
        language = source_language or self.default_language

        logger.info(f"Recognizing {file_type} file: {path.name}")

        if file_type in ("document", "text"):
            return self.client.recognize(str(path), source_language=language)

        elif file_type == "image":
            # Use multimodal chat for detailed image analysis instead of basic OCR
            return self.client.analyze_image(str(path))

        elif file_type == "audio":
            # Audio transcription is async, so we need to poll
            logger.warning("Audio transcription not yet implemented - returning empty")
            return ""

        else:
            # Try document recognition as fallback
            logger.warning(f"Unknown file type {path.suffix}, attempting document recognition")
            try:
                return self.client.recognize(str(path), source_language=language)
            except Exception as e:
                logger.error(f"Failed to recognize file {path.name}: {e}")
                raise ValueError(f"Unsupported file type: {path.suffix}")

    def recognize_image(
        self,
        image_path: Union[str, Path],
    ) -> str:
        """Extract text from image using OCR.

        Note: For detailed image analysis and descriptions (not just text extraction),
        use analyze_image() instead.

        Args:
            image_path: Path to image file (JPEG, PNG, or TIFF)

        Returns:
            Extracted text content

        Example:
            >>> text = recognizer.recognize_image("screenshot.png")
        """
        path = Path(image_path)

        if not path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        ext = path.suffix.lower()
        if ext not in IMAGE_EXTENSIONS:
            raise ValueError(f"Unsupported image format: {ext}. Use JPEG, PNG, or TIFF.")

        logger.info(f"Running OCR on image: {path.name}")
        return self.client.recognize_ocr(str(path))

    def analyze_image(
        self,
        image_path: Union[str, Path],
        prompt: Optional[str] = None,
        model: Optional[str] = None,
    ) -> str:
        """Analyze an image using multimodal LLM capabilities.

        Unlike recognize_image() which only extracts text via OCR, this method
        uses a vision-capable LLM to understand and describe the full content
        of the image including diagrams, charts, visual elements, and context.

        Args:
            image_path: Path to image file (JPEG, PNG, WebP, or GIF)
            prompt: Custom prompt for analysis. If None, uses a default prompt
                   that requests a detailed explanation and summary.
            model: Model to use (defaults to gpt-4o for vision capabilities)

        Returns:
            Detailed analysis and description of the image

        Example:
            >>> analysis = recognizer.analyze_image("architecture_diagram.png")
            >>> print(analysis)
            "This diagram illustrates a microservices architecture..."

            >>> analysis = recognizer.analyze_image(
            ...     "chart.png",
            ...     prompt="What trends are shown in this chart?"
            ... )
        """
        path = Path(image_path)

        if not path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        ext = path.suffix.lower()
        supported_extensions = {".jpg", ".jpeg", ".png", ".webp", ".gif"}
        if ext not in supported_extensions:
            raise ValueError(
                f"Unsupported image format: {ext}. "
                f"Use JPEG, PNG, WebP, or GIF."
            )

        logger.info(f"Analyzing image with LLM: {path.name}")

        kwargs = {}
        if prompt:
            kwargs["prompt"] = prompt
        if model:
            kwargs["model"] = model

        return self.client.analyze_image(str(path), **kwargs)

    def recognize_batch(
        self,
        file_paths: List[Union[str, Path]],
        source_language: Optional[str] = None,
    ) -> Dict[str, str]:
        """Process multiple files and extract text from each.

        Args:
            file_paths: List of file paths
            source_language: Language code for text recognition

        Returns:
            Dictionary mapping file paths to extracted text

        Example:
            >>> results = recognizer.recognize_batch(["doc1.pdf", "doc2.pdf"])
            >>> for path, text in results.items():
            ...     print(f"{path}: {len(text)} chars")
        """
        results = {}

        logger.info(f"Batch processing {len(file_paths)} files")

        for file_path in file_paths:
            path_str = str(file_path)
            try:
                text = self.recognize_file(file_path, source_language)
                results[path_str] = text
                logger.debug(f"Processed {path_str}: {len(text)} chars")
            except Exception as e:
                logger.error(f"Failed to process {path_str}: {e}")
                results[path_str] = ""

        successful = sum(1 for t in results.values() if t)
        logger.info(f"Batch complete: {successful}/{len(file_paths)} files processed successfully")

        return results

    def to_markdown(
        self,
        file_path: Union[str, Path],
        max_workers: int = 16,
    ) -> str:
        """Convert document to markdown format.

        Useful for preserving structure from presentations and complex documents.

        Args:
            file_path: Path to document
            max_workers: Concurrent LLM image captioning calls (1-32)

        Returns:
            Markdown-formatted content

        Example:
            >>> md = recognizer.to_markdown("presentation.pptx")
            >>> print(md)
        """
        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        logger.info(f"Converting to markdown: {path.name}")
        return self.client.recognize_markdown(str(path), max_workers=max_workers)

    def get_supported_extensions(self) -> Dict[str, List[str]]:
        """Get dictionary of supported file extensions by category.

        Returns:
            Dictionary with categories as keys and extension lists as values

        Example:
            >>> extensions = recognizer.get_supported_extensions()
            >>> print(extensions["image"])
            ['.jpg', '.jpeg', '.png', '.tiff', '.tif']
        """
        return {
            "document": sorted(DOCUMENT_EXTENSIONS),
            "text": sorted(TEXT_EXTENSIONS),
            "image": sorted(IMAGE_EXTENSIONS),
            "audio": sorted(AUDIO_EXTENSIONS),
        }

    def is_supported(self, file_path: Union[str, Path]) -> bool:
        """Check if a file type is supported for text extraction.

        Args:
            file_path: Path to file

        Returns:
            True if file type is supported

        Example:
            >>> recognizer.is_supported("doc.pdf")
            True
            >>> recognizer.is_supported("video.avi")
            False
        """
        file_type = self._get_file_type(file_path)
        return file_type != "unknown"
