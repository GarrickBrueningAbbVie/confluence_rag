# Confluence REST API Test

This folder contains a test implementation for fetching Confluence pages using the direct REST API instead of the Atlassian Python API.

## Purpose

The Atlassian Python API has proven unreliable, so this implementation uses direct HTTP requests to the Confluence REST API v1 for better reliability and control.

## Files

- **confluence_rest_client.py** - Main REST API client with authentication and page retrieval methods
- **fetch_pages.py** - Script to fetch all pages from the DSA space and save to JSON
- **README.md** - This file

## Features

The REST API client provides:

- **Authentication**: Bearer token and Basic auth support with automatic fallback
- **SSL Handling**: Bypasses SSL verification for internal servers with self-signed certificates
- **Pagination**: Automatically handles paginated results
- **Page Retrieval**: Fetch all pages from a space with full content
- **Content Processing**:
  - Automatic HTML to plain text conversion
  - External link extraction from page content
- **Tree Structure**: Captures parent/child relationships and ancestor hierarchy
- **Search**: CQL-based search capabilities
- **Data Export**: Save pages to JSON format with rich metadata
- **Error Handling**: Robust error handling and logging

## Usage

### Quick Start

From the project root directory:

```bash
python rest_api_test/fetch_pages.py
```

This will:
1. Load credentials from `.env` file
2. Connect to Confluence using REST API
3. Fetch all pages from the DSA space
4. Display page information
5. Save results to `rest_api_test/confluence_pages.json`

### Programmatic Usage

```python
from rest_api_test.confluence_rest_client import ConfluenceRestClient
import os
from dotenv import load_dotenv

load_dotenv()

# Initialize client
client = ConfluenceRestClient(
    base_url=os.getenv('CONFLUENCE_URL'),
    username=os.getenv('CONFLUENCE_USERNAME'),
    api_token=os.getenv('CONFLUENCE_API_TOKEN')
)

# Fetch all pages from a space
pages = client.get_all_pages_from_space('DSA')

# Get a specific page by ID
page = client.get_page_by_id('123456789')

# Search using CQL
results = client.search_pages('space = DSA AND type = page AND title ~ "analytics"')

# Save to JSON
client.save_pages_to_json(pages, 'output.json')
```

## API Methods

### `get_all_pages_from_space(space_key, expand=None)`

Retrieves all pages from a Confluence space with pagination.

**Parameters:**
- `space_key` (str): The space key (e.g., 'DSA')
- `expand` (list, optional): Fields to expand (default: ['body.storage', 'version', 'space', 'history'])

**Returns:**
- List of `ConfluencePage` objects

### `get_page_by_id(page_id, expand=None)`

Retrieves a specific page by its ID.

**Parameters:**
- `page_id` (str): The Confluence page ID
- `expand` (list, optional): Fields to expand

**Returns:**
- `ConfluencePage` object or None

### `search_pages(cql, limit=100)`

Searches for pages using Confluence Query Language (CQL).

**Parameters:**
- `cql` (str): CQL query string
- `limit` (int): Maximum results per page (default: 100)

**Returns:**
- List of `ConfluencePage` objects

### `save_pages_to_json(pages, output_file)`

Saves pages to a JSON file.

**Parameters:**
- `pages` (list): List of `ConfluencePage` objects
- `output_file` (str): Path to output file

## ConfluencePage Object

Each page contains:

### Basic Information
- `id` - Unique page ID
- `title` - Page title
- `space_key` - Space key (e.g., 'DSA')
- `space_name` - Full space name
- `url` - Full URL to the page
- `created_date` - Creation timestamp
- `modified_date` - Last modification timestamp
- `author` - Page author name
- `version` - Current version number

### Content
- `content_html` - Full HTML content (storage format)
- `content_text` - Plain text content (HTML tags removed). If Confluence doesn't provide view text, automatically extracts from HTML
- `external_links` - List of external URLs found in the page content

### Tree Structure
- `parent_id` - ID of the parent page (if any)
- `parent_title` - Title of the parent page (if any)
- `ancestors` - List of ancestor pages from root to parent, each containing:
  - `id` - Ancestor page ID
  - `title` - Ancestor page title
  - `type` - Content type (usually "page")
- `children` - List of child pages, each containing:
  - `id` - Child page ID
  - `title` - Child page title
  - `type` - Content type (usually "page")

## Configuration

Credentials are loaded from the `.env` file in the project root:

```env
CONFLUENCE_URL=https://confluence.abbvienet.com/
CONFLUENCE_USERNAME=your.email@abbvie.com
CONFLUENCE_API_TOKEN=your_api_token
CONFLUENCE_SPACE_KEY=DSA
```

## Error Handling

The client includes:
- Request exception handling
- HTTP status code validation
- Detailed error messages
- Graceful degradation on partial failures

## Next Steps

After validating this REST API approach works reliably:

1. Compare performance and reliability with Atlassian Python API
2. Test web scraping approach if needed
3. Consider replacing the existing `src/confluence/client.py` implementation
4. Update data acquisition notebook to use new client

## Notes

- Uses Confluence REST API v1 (more stable than v2 for content retrieval)
- Handles pagination automatically (50 pages per request)
- Expands all necessary fields by default (content, version, space, history)
- Content is retrieved in storage format (HTML)
