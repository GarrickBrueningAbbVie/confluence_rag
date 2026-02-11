"""Unit tests for ConfluenceParser class."""

import pytest
from src.confluence.parser import ConfluenceParser


@pytest.fixture
def parser() -> ConfluenceParser:
    """Create a ConfluenceParser instance for testing."""
    return ConfluenceParser()


def test_html_to_text(parser: ConfluenceParser) -> None:
    """Test HTML to text conversion."""
    html = """
    <html>
        <body>
            <h1>Test Header</h1>
            <p>This is a test paragraph.</p>
            <script>alert('test');</script>
        </body>
    </html>
    """
    text = parser.html_to_text(html)

    assert "Test Header" in text
    assert "test paragraph" in text
    assert "alert" not in text  # Script should be removed


def test_extract_tables(parser: ConfluenceParser) -> None:
    """Test table extraction from HTML."""
    html = """
    <table>
        <tr><th>Header 1</th><th>Header 2</th></tr>
        <tr><td>Data 1</td><td>Data 2</td></tr>
    </table>
    """
    tables = parser.extract_tables(html)

    assert len(tables) == 1
    assert len(tables[0]) == 2  # Two rows
    assert tables[0][0] == ["Header 1", "Header 2"]
    assert tables[0][1] == ["Data 1", "Data 2"]


def test_extract_links(parser: ConfluenceParser) -> None:
    """Test link extraction from HTML."""
    html = """
    <a href="https://example.com">Example Link</a>
    <a href="https://test.com">Test Link</a>
    """
    links = parser.extract_links(html)

    assert len(links) == 2
    assert links[0]["text"] == "Example Link"
    assert links[0]["url"] == "https://example.com"


def test_extract_headers(parser: ConfluenceParser) -> None:
    """Test header extraction from HTML."""
    html = """
    <h1>Main Header</h1>
    <h2>Sub Header</h2>
    <h3>Sub-sub Header</h3>
    """
    headers = parser.extract_headers(html)

    assert len(headers) == 3
    assert headers[0]["level"] == 1
    assert headers[0]["text"] == "Main Header"
    assert headers[1]["level"] == 2
    assert headers[1]["text"] == "Sub Header"


def test_chunk_text(parser: ConfluenceParser) -> None:
    """Test text chunking."""
    text = "A" * 2500  # Long text

    chunks = parser.chunk_text(text, chunk_size=1000, chunk_overlap=200)

    assert len(chunks) > 1
    assert len(chunks[0]) == 1000
    # Check overlap
    assert chunks[0][-200:] == chunks[1][:200]


def test_chunk_text_empty(parser: ConfluenceParser) -> None:
    """Test chunking empty text."""
    chunks = parser.chunk_text("", chunk_size=1000)
    assert len(chunks) == 0


def test_parse_page(parser: ConfluenceParser) -> None:
    """Test parsing a complete page."""
    page_data = {
        "id": "123",
        "title": "Test Page",
        "url": "https://confluence.example.com/page",
        "space": "TEST",
        "version": 1,
        "last_updated": "2024-01-01",
        "content": "<h1>Test</h1><p>Content here</p>",
    }

    parsed = parser.parse_page(page_data)

    assert parsed["id"] == "123"
    assert parsed["title"] == "Test Page"
    assert "Test" in parsed["text"]
    assert "Content here" in parsed["text"]
    assert len(parsed["headers"]) == 1
