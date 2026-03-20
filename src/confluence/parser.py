"""Parser for Confluence HTML content."""

from typing import List, Dict, Any
from loguru import logger

from confluence.html_utils import (
    html_to_text as _html_to_text,
    extract_tables as _extract_tables,
    extract_links as _extract_links,
    extract_headings as _extract_headings,
)


class ConfluenceParser:
    """
    Parser for processing Confluence page content.

    This class converts Confluence HTML storage format to clean text,
    extracts structured information, and prepares content for vectorization.
    """

    def __init__(self) -> None:
        """Initialize the Confluence parser."""
        logger.debug("Initialized Confluence parser")

    def html_to_text(self, html_content: str) -> str:
        """
        Convert Confluence HTML storage format to plain text.

        Args:
            html_content: HTML content from Confluence page.

        Returns:
            Clean text extracted from HTML.
        """
        try:
            return _html_to_text(html_content)
        except Exception as e:
            logger.error(f"Error parsing HTML content: {str(e)}")
            return ""

    def extract_tables(self, html_content: str) -> List[List[str]]:
        """
        Extract tables from Confluence HTML content.

        Args:
            html_content: HTML content from Confluence page.

        Returns:
            List of tables, where each table is a list of rows (list of cells).
        """
        try:
            tables = _extract_tables(html_content)
            logger.debug(f"Extracted {len(tables)} tables from content")
            return tables
        except Exception as e:
            logger.error(f"Error extracting tables: {str(e)}")
            return []

    def extract_links(self, html_content: str) -> List[Dict[str, str]]:
        """
        Extract all hyperlinks from Confluence content.

        Args:
            html_content: HTML content from Confluence page.

        Returns:
            List of dictionaries with 'text' and 'url' keys for each link.
        """
        try:
            links = _extract_links(html_content, unique_only=False)
            logger.debug(f"Extracted {len(links)} links from content")
            return links
        except Exception as e:
            logger.error(f"Error extracting links: {str(e)}")
            return []

    def extract_headers(self, html_content: str) -> List[Dict[str, str]]:
        """
        Extract section headers from Confluence content.

        Args:
            html_content: HTML content from Confluence page.

        Returns:
            List of dictionaries with 'level' and 'text' for each header.
        """
        try:
            headers = _extract_headings(html_content)
            logger.debug(f"Extracted {len(headers)} headers from content")
            return headers
        except Exception as e:
            logger.error(f"Error extracting headers: {str(e)}")
            return []

    def parse_page(self, page_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Parse a complete Confluence page and extract structured information.

        Args:
            page_data: Dictionary containing page content and metadata.

        Returns:
            Dictionary with parsed content including text, tables, links, and headers.
        """
        logger.debug(f"Parsing page: {page_data.get('title', 'Unknown')}")

        html_content = page_data.get("content", "")

        parsed_data = {
            "id": page_data.get("id"),
            "title": page_data.get("title"),
            "url": page_data.get("url"),
            "space": page_data.get("space"),
            "version": page_data.get("version"),
            "last_updated": page_data.get("last_updated"),
            "text": self.html_to_text(html_content),
            "tables": self.extract_tables(html_content),
            "links": self.extract_links(html_content),
            "headers": self.extract_headers(html_content),
            "raw_html": html_content,
        }

        logger.debug(f"Successfully parsed page: {parsed_data['title']}")
        return parsed_data

    def parse_pages(self, pages_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Parse multiple Confluence pages.

        Args:
            pages_data: List of page dictionaries with content and metadata.

        Returns:
            List of parsed page dictionaries.
        """
        logger.info(f"Parsing {len(pages_data)} pages")
        parsed_pages = []

        for page in pages_data:
            try:
                parsed = self.parse_page(page)
                parsed_pages.append(parsed)
            except Exception as e:
                logger.warning(f"Failed to parse page {page.get('title')}: {str(e)}")
                continue

        logger.info(f"Successfully parsed {len(parsed_pages)} pages")
        return parsed_pages

    def chunk_text(
        self, text: str, chunk_size: int = 1000, chunk_overlap: int = 200
    ) -> List[str]:
        """
        Split text into overlapping chunks for vectorization.

        Args:
            text: Text to chunk.
            chunk_size: Maximum size of each chunk in characters. Defaults to 1000.
            chunk_overlap: Number of overlapping characters between chunks. Defaults to 200.

        Returns:
            List of text chunks.
        """
        if not text:
            return []

        chunks = []
        start = 0

        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]
            chunks.append(chunk)
            start = end - chunk_overlap

            if end >= len(text):
                break

        logger.debug(f"Split text into {len(chunks)} chunks")
        return chunks
