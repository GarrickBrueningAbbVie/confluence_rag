"""
Shared HTML utilities for the Confluence module.

This module provides the canonical implementations for HTML processing
functions used across the Confluence integration. All HTML extraction
should use these utilities to ensure consistency.

Example:
    >>> from confluence.html_utils import html_to_text, extract_links
    >>> text = html_to_text("<p>Hello <b>world</b></p>")
    >>> links = extract_links("<a href='http://example.com'>Example</a>")
"""

import re
from typing import Dict, List, Optional, Set
from urllib.parse import urlparse

from bs4 import BeautifulSoup
from loguru import logger


def html_to_text(html_content: str, preserve_newlines: bool = False) -> str:
    """
    Convert HTML content to clean plain text.

    Removes scripts, styles, and other non-content elements, then
    extracts and normalizes the text content.

    Args:
        html_content: Raw HTML string
        preserve_newlines: If True, preserves paragraph breaks as newlines

    Returns:
        Clean plain text string

    Example:
        >>> html_to_text("<p>Hello</p><p>World</p>")
        'Hello World'
        >>> html_to_text("<p>Hello</p><p>World</p>", preserve_newlines=True)
        'Hello\\nWorld'
    """
    if not html_content:
        return ""

    soup = BeautifulSoup(html_content, "html.parser")

    # Remove non-content elements
    for element in soup(["script", "style", "head", "meta", "link", "noscript"]):
        element.decompose()

    if preserve_newlines:
        # Add newlines for block-level elements
        for tag in soup.find_all(["p", "div", "br", "h1", "h2", "h3", "h4", "h5", "h6", "li"]):
            tag.insert_after(soup.new_string("\n"))

        text = soup.get_text(separator=" ")
        # Normalize whitespace while preserving intentional newlines
        lines = (line.strip() for line in text.splitlines())
        text = "\n".join(line for line in lines if line)
    else:
        # Extract text with space separator
        text = soup.get_text(separator=" ")

        # Normalize whitespace
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = " ".join(chunk for chunk in chunks if chunk)

    return text.strip()


def extract_links(
    html_content: str,
    base_url: Optional[str] = None,
    include_internal: bool = True,
    include_external: bool = True,
    unique_only: bool = True,
) -> List[Dict[str, str]]:
    """
    Extract links from HTML content.

    Parses HTML to find all anchor tags and extracts their href and text.
    Can filter by internal/external links and deduplicate results.

    Args:
        html_content: Raw HTML string
        base_url: Base URL for determining internal vs external links
        include_internal: Whether to include internal links
        include_external: Whether to include external links
        unique_only: Whether to deduplicate links by URL

    Returns:
        List of dicts with 'url' and 'text' keys

    Example:
        >>> html = '<a href="http://example.com">Example</a>'
        >>> extract_links(html)
        [{'url': 'http://example.com', 'text': 'Example'}]
    """
    if not html_content:
        return []

    soup = BeautifulSoup(html_content, "html.parser")
    links: List[Dict[str, str]] = []
    seen_urls: Set[str] = set()

    base_domain = None
    if base_url:
        parsed_base = urlparse(base_url)
        base_domain = parsed_base.netloc

    for anchor in soup.find_all("a", href=True):
        href = anchor.get("href", "").strip()

        # Skip empty hrefs and internal anchors
        if not href or href.startswith("#"):
            continue

        # Skip javascript links
        if href.startswith("javascript:"):
            continue

        # Deduplicate if requested
        if unique_only and href in seen_urls:
            continue

        # Determine if internal or external
        try:
            parsed = urlparse(href)
            is_external = bool(parsed.netloc) and (
                base_domain is None or parsed.netloc != base_domain
            )
        except Exception:
            is_external = True

        # Filter based on include flags
        if is_external and not include_external:
            continue
        if not is_external and not include_internal:
            continue

        # Extract link text
        text = anchor.get_text(strip=True)
        if not text:
            text = href  # Use URL as text if no text content

        links.append({
            "url": href,
            "text": text,
        })

        if unique_only:
            seen_urls.add(href)

    return links


def extract_tables(html_content: str) -> List[List[List[str]]]:
    """
    Extract tables from HTML content.

    Parses HTML tables and returns them as lists of rows,
    where each row is a list of cell values.

    Args:
        html_content: Raw HTML string

    Returns:
        List of tables, each table is a list of rows

    Example:
        >>> html = '<table><tr><td>A</td><td>B</td></tr></table>'
        >>> extract_tables(html)
        [[['A', 'B']]]
    """
    if not html_content:
        return []

    soup = BeautifulSoup(html_content, "html.parser")
    tables = []

    for table in soup.find_all("table"):
        rows = []
        for tr in table.find_all("tr"):
            cells = []
            for cell in tr.find_all(["td", "th"]):
                cells.append(cell.get_text(strip=True))
            if cells:
                rows.append(cells)
        if rows:
            tables.append(rows)

    return tables


def extract_headings(html_content: str) -> List[Dict[str, str]]:
    """
    Extract headings from HTML content.

    Finds all heading tags (h1-h6) and returns their level and text.

    Args:
        html_content: Raw HTML string

    Returns:
        List of dicts with 'level' (int) and 'text' keys

    Example:
        >>> html = '<h1>Title</h1><h2>Section</h2>'
        >>> extract_headings(html)
        [{'level': 1, 'text': 'Title'}, {'level': 2, 'text': 'Section'}]
    """
    if not html_content:
        return []

    soup = BeautifulSoup(html_content, "html.parser")
    headings = []

    for level in range(1, 7):
        for heading in soup.find_all(f"h{level}"):
            text = heading.get_text(strip=True)
            if text:
                headings.append({
                    "level": level,
                    "text": text,
                })

    # Sort by document order (approximate via position)
    # This works because find_all returns in document order
    return headings


def extract_code_blocks(html_content: str) -> List[Dict[str, str]]:
    """
    Extract code blocks from HTML content.

    Finds pre and code tags and extracts their content along
    with any language hints.

    Args:
        html_content: Raw HTML string

    Returns:
        List of dicts with 'code' and 'language' (optional) keys

    Example:
        >>> html = '<pre class="language-python">print("hello")</pre>'
        >>> extract_code_blocks(html)
        [{'code': 'print("hello")', 'language': 'python'}]
    """
    if not html_content:
        return []

    soup = BeautifulSoup(html_content, "html.parser")
    blocks = []

    for pre in soup.find_all(["pre", "code"]):
        code = pre.get_text()

        # Try to detect language from class
        language = None
        classes = pre.get("class", [])
        for cls in classes:
            if cls.startswith("language-"):
                language = cls[9:]
                break
            if cls.startswith("lang-"):
                language = cls[5:]
                break

        if code.strip():
            blocks.append({
                "code": code.strip(),
                "language": language,
            })

    return blocks


def clean_confluence_html(html_content: str) -> str:
    """
    Clean Confluence-specific HTML artifacts.

    Removes Confluence macros, editor artifacts, and other
    Confluence-specific elements that don't contribute to content.

    Args:
        html_content: Raw Confluence HTML

    Returns:
        Cleaned HTML string

    Example:
        >>> html = '<ac:structured-macro ac:name="info">...</ac:structured-macro>'
        >>> clean_confluence_html(html)
        '...'
    """
    if not html_content:
        return ""

    soup = BeautifulSoup(html_content, "html.parser")

    # Remove Confluence macros (ac:* tags)
    for macro in soup.find_all(re.compile(r"^ac:")):
        # Keep the content of the macro
        macro.unwrap()

    # Remove editor artifacts
    for artifact in soup.find_all(class_=re.compile(r"^(confluence-)?embedded")):
        artifact.decompose()

    # Remove empty divs and spans
    for tag in soup.find_all(["div", "span"]):
        if not tag.get_text(strip=True) and not tag.find_all():
            tag.decompose()

    return str(soup)


def estimate_reading_time(html_content: str, words_per_minute: int = 200) -> int:
    """
    Estimate reading time for HTML content.

    Args:
        html_content: Raw HTML string
        words_per_minute: Average reading speed

    Returns:
        Estimated reading time in minutes (rounded up)

    Example:
        >>> html = '<p>' + ' '.join(['word'] * 400) + '</p>'
        >>> estimate_reading_time(html)
        2
    """
    text = html_to_text(html_content)
    word_count = len(text.split())
    minutes = (word_count + words_per_minute - 1) // words_per_minute
    return max(1, minutes)
