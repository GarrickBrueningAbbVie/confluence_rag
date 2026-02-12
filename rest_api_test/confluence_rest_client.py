"""
Confluence REST API client for retrieving pages.

This module uses the Confluence REST API v2 to fetch all accessible pages
from a Confluence space. It's designed as an alternative to the unreliable
Atlassian Python API.
"""

import os
import requests
import json
import urllib3
import re
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from dotenv import load_dotenv
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse

# Disable SSL warnings when verification is disabled
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


@dataclass
class ConfluencePage:
    """Represents a Confluence page with its metadata."""

    id: str
    title: str
    space_key: str
    space_name: str
    url: str
    created_date: str
    modified_date: str
    author: str
    version: int
    content_html: Optional[str] = None
    content_text: Optional[str] = None
    external_links: List[str] = field(default_factory=list)
    parent_id: Optional[str] = None
    parent_title: Optional[str] = None
    ancestors: List[Dict[str, str]] = field(default_factory=list)
    children: List[Dict[str, str]] = field(default_factory=list)


class ConfluenceRestClient:
    """
    Client for interacting with Confluence REST API.

    This client handles authentication and provides methods to retrieve
    pages from Confluence spaces using the REST API v2.
    """

    def __init__(
        self,
        base_url: str,
        username: str,
        api_token: str,
        verify_ssl: bool = False,
        auth_type: str = "bearer",
    ):
        """
        Initialize the Confluence REST API client.

        Args:
            base_url: The base URL of the Confluence instance (e.g., https://confluence.abbvienet.com)
            username: Confluence username (email)
            api_token: Confluence API token for authentication
            verify_ssl: Whether to verify SSL certificates (default: False for internal servers)
            auth_type: Authentication type - 'bearer' (default) or 'basic'
        """
        self.base_url = base_url.rstrip("/")
        self.username = username
        self.api_token = api_token
        self.verify_ssl = verify_ssl
        self.auth_type = auth_type
        self.session = requests.Session()
        self.session.verify = verify_ssl

        # Set up authentication based on type
        if auth_type == "bearer":
            # Use Bearer token authentication (common for on-premise Confluence PATs)
            self.session.headers.update(
                {
                    "Authorization": f"Bearer {api_token}",
                    "Accept": "application/json",
                    "Content-Type": "application/json",
                }
            )
        else:
            # Use Basic authentication
            self.session.auth = (username, api_token)
            self.session.headers.update(
                {"Accept": "application/json", "Content-Type": "application/json"}
            )

    def _extract_text_from_html(self, html: str) -> str:
        """
        Extract plain text from HTML content by removing all tags.

        Args:
            html: HTML content string

        Returns:
            Plain text with HTML tags removed
        """
        if not html:
            return ""

        # Parse HTML
        soup = BeautifulSoup(html, 'html.parser')

        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()

        # Get text and clean up whitespace
        text = soup.get_text(separator=' ')
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = ' '.join(chunk for chunk in chunks if chunk)

        return text

    def _extract_external_links(self, html: str) -> List[str]:
        """
        Extract all external links from HTML content.

        Args:
            html: HTML content string

        Returns:
            List of unique external URLs
        """
        if not html:
            return []

        soup = BeautifulSoup(html, 'html.parser')
        links = []

        # Find all anchor tags with href
        for link in soup.find_all('a', href=True):
            href = link['href']

            # Skip internal anchors and relative links
            if href.startswith('#'):
                continue

            # Check if it's an external link (has a domain)
            parsed = urlparse(href)
            if parsed.scheme in ('http', 'https'):
                # External link
                links.append(href)
            elif href.startswith('/') and not href.startswith('/wiki'):
                # Absolute path but might be external
                full_url = urljoin(self.base_url, href)
                if urlparse(full_url).netloc != urlparse(self.base_url).netloc:
                    links.append(full_url)

        # Return unique links
        return list(set(links))

    def test_connection(self) -> bool:
        """
        Test the connection and authentication to Confluence.

        Returns:
            True if connection successful, False otherwise
        """
        url = f"{self.base_url}/rest/api/space"
        try:
            print(f"Testing connection to: {url}")
            print(f"Auth type: {self.auth_type}")
            response = self.session.get(url, params={"limit": 1})
            print(f"Response status: {response.status_code}")

            if response.status_code == 200:
                print("✓ Connection successful!")
                return True
            else:
                print(f"✗ Connection failed: {response.status_code}")
                print(f"Response: {response.text[:500]}")
                return False
        except requests.exceptions.RequestException as e:
            print(f"✗ Connection error: {e}")
            return False

    def get_all_pages_from_space(
        self, space_key: str, expand: Optional[List[str]] = None
    ) -> List[ConfluencePage]:
        """
        Retrieve all pages from a specific Confluence space.

        Args:
            space_key: The key of the Confluence space (e.g., 'DSA')
            expand: Optional list of fields to expand (e.g., ['body.storage', 'version', 'space'])

        Returns:
            List of ConfluencePage objects containing page metadata and content
        """
        if expand is None:
            expand = ["body.storage", "body.view", "version", "space", "history", "ancestors", "children.page"]

        # First, try to get the total count of pages using CQL search
        print("Counting total pages in space...")
        total_pages = 0

        try:
            # Use CQL search to get accurate count
            search_url = f"{self.base_url}/rest/api/content/search"
            cql_query = f"space = {space_key} AND type = page"
            search_params = {
                "cql": cql_query,
                "limit": 0,  # We just want the total count
            }

            response = self.session.get(search_url, params=search_params)
            response.raise_for_status()
            data = response.json()

            # Check for totalSize in response
            total_pages = data.get("totalSize", 0)

            # If totalSize not available, try the regular content endpoint
            if total_pages == 0:
                url = f"{self.base_url}/rest/api/content"
                count_params = {
                    "spaceKey": space_key,
                    "type": "page",
                    "status": "current",
                    "limit": 0,
                }
                response = self.session.get(url, params=count_params)
                response.raise_for_status()
                data = response.json()
                # Look for total in different possible locations
                total_pages = data.get("total", data.get("totalSize", 0))

        except requests.exceptions.RequestException as e:
            print(f"Could not get total count: {e}")
            total_pages = 0

        if total_pages > 0:
            print(f"Found {total_pages} total pages in space '{space_key}'")
        else:
            print("Could not determine total page count, will fetch all pages...")
        print()

        pages = []
        start = 0
        limit = 50  # Confluence API typically returns 25 by default, we'll use 50

        while True:
            # Using Confluence REST API v1 (more reliable than v2 for some operations)
            url = f"{self.base_url}/rest/api/content"
            params = {
                "spaceKey": space_key,
                "type": "page",
                "status": "current",
                "expand": ",".join(expand),
                "start": start,
                "limit": limit,
            }

            if total_pages > 0:
                print(f"Fetching pages {start} to {min(start + limit, total_pages)} (of {total_pages})...")
            else:
                print(f"Fetching pages {start} to {start + limit}...")

            try:
                response = self.session.get(url, params=params)
                response.raise_for_status()
                data = response.json()

                results = data.get("results", [])
                if not results:
                    break

                for page_data in results:
                    page = self._parse_page_data(page_data)
                    pages.append(page)

                # Check if there are more pages
                if len(results) < limit:
                    break

                start += limit

            except requests.exceptions.RequestException as e:
                print(f"Error fetching pages: {e}")
                if hasattr(e.response, "text"):
                    print(f"Response: {e.response.text}")
                break

        print(f"Total pages retrieved: {len(pages)}")
        return pages

    def get_page_by_id(
        self, page_id: str, expand: Optional[List[str]] = None
    ) -> Optional[ConfluencePage]:
        """
        Retrieve a specific page by its ID.

        Args:
            page_id: The ID of the Confluence page
            expand: Optional list of fields to expand

        Returns:
            ConfluencePage object or None if not found
        """
        if expand is None:
            expand = ["body.storage", "body.view", "version", "space", "history", "ancestors", "children.page"]

        url = f"{self.base_url}/rest/api/content/{page_id}"
        params = {"expand": ",".join(expand)}

        try:
            response = self.session.get(url, params=params)
            response.raise_for_status()
            page_data = response.json()
            return self._parse_page_data(page_data)
        except requests.exceptions.RequestException as e:
            print(f"Error fetching page {page_id}: {e}")
            return None

    def search_pages(self, cql: str, limit: int = 100) -> List[ConfluencePage]:
        """
        Search for pages using Confluence Query Language (CQL).

        Args:
            cql: CQL query string (e.g., 'space = DSA AND type = page')
            limit: Maximum number of results to return

        Returns:
            List of ConfluencePage objects matching the search
        """
        url = f"{self.base_url}/rest/api/content/search"
        params = {
            "cql": cql,
            "limit": limit,
            "expand": "body.storage,version,space,history",
        }

        pages = []
        start = 0

        while True:
            params["start"] = start

            try:
                response = self.session.get(url, params=params)
                response.raise_for_status()
                data = response.json()

                results = data.get("results", [])
                if not results:
                    break

                for page_data in results:
                    page = self._parse_page_data(page_data)
                    pages.append(page)

                if len(results) < limit:
                    break

                start += limit

            except requests.exceptions.RequestException as e:
                print(f"Error searching pages: {e}")
                break

        return pages

    def _parse_page_data(self, page_data: Dict[str, Any]) -> ConfluencePage:
        """
        Parse raw page data from API response into ConfluencePage object.

        Args:
            page_data: Raw page data dictionary from API

        Returns:
            ConfluencePage object
        """
        page_id = page_data.get("id", "")
        title = page_data.get("title", "Untitled")

        # Extract space information
        space = page_data.get("space", {})
        space_key = space.get("key", "")
        space_name = space.get("name", "")

        # Extract version and history information
        version_data = page_data.get("version", {})
        version = version_data.get("number", 1)

        history = page_data.get("history", {})
        created_date = history.get("createdDate", "")

        version_data_when = version_data.get("when", "")
        modified_date = version_data_when if version_data_when else created_date

        # Extract author information
        version_by = version_data.get("by", {})
        author = version_by.get("displayName", "Unknown")

        # Construct page URL
        url = f"{self.base_url}/wiki/spaces/{space_key}/pages/{page_id}/{title.replace(' ', '+')}"
        if "_links" in page_data:
            webui = page_data["_links"].get("webui", "")
            if webui:
                url = f"{self.base_url}{webui}"

        # Extract content
        content_html = None
        content_text = None
        body = page_data.get("body", {})

        # Get HTML content (storage format)
        if "storage" in body:
            content_html = body["storage"].get("value", "")

        # Always extract plain text from HTML to ensure no tags remain
        # The "view" format can still contain HTML, so we parse the storage format instead
        if content_html:
            content_text = self._extract_text_from_html(content_html)

        # Extract external links from HTML content
        external_links = []
        if content_html:
            external_links = self._extract_external_links(content_html)

        # Extract tree structure information (ancestors and children)
        parent_id = None
        parent_title = None
        ancestors = []

        # Get ancestors (parent pages)
        ancestors_data = page_data.get("ancestors", [])
        if ancestors_data:
            # Ancestors are ordered from root to immediate parent
            ancestors = [
                {
                    "id": ancestor.get("id", ""),
                    "title": ancestor.get("title", ""),
                    "type": ancestor.get("type", "page")
                }
                for ancestor in ancestors_data
            ]
            # The last ancestor is the immediate parent
            if ancestors:
                parent_id = ancestors[-1]["id"]
                parent_title = ancestors[-1]["title"]

        # Get children pages
        children = []
        children_data = page_data.get("children", {}).get("page", {})
        if children_data:
            results = children_data.get("results", [])
            children = [
                {
                    "id": child.get("id", ""),
                    "title": child.get("title", ""),
                    "type": child.get("type", "page")
                }
                for child in results
            ]

        return ConfluencePage(
            id=page_id,
            title=title,
            space_key=space_key,
            space_name=space_name,
            url=url,
            created_date=created_date,
            modified_date=modified_date,
            author=author,
            version=version,
            content_html=content_html,
            content_text=content_text,
            external_links=external_links,
            parent_id=parent_id,
            parent_title=parent_title,
            ancestors=ancestors,
            children=children,
        )

    def save_pages_to_json(self, pages: List[ConfluencePage], output_file: str):
        """
        Save retrieved pages to a JSON file.

        Args:
            pages: List of ConfluencePage objects
            output_file: Path to output JSON file
        """
        pages_dict = [
            {
                "id": page.id,
                "title": page.title,
                "space_key": page.space_key,
                "space_name": page.space_name,
                "url": page.url,
                "created_date": page.created_date,
                "modified_date": page.modified_date,
                "author": page.author,
                "version": page.version,
                "content_html": page.content_html,
                "content_text": page.content_text,
                "external_links": page.external_links,
                "parent_id": page.parent_id,
                "parent_title": page.parent_title,
                "ancestors": page.ancestors,
                "children": page.children,
            }
            for page in pages
        ]

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(pages_dict, f, indent=2, ensure_ascii=False)

        print(f"Saved {len(pages)} pages to {output_file}")


def main():
    """Main function to test the Confluence REST client."""
    # Load environment variables
    load_dotenv()

    # Get credentials from environment
    confluence_url = os.getenv("CONFLUENCE_URL")
    username = os.getenv("CONFLUENCE_USERNAME")
    api_token = os.getenv("CONFLUENCE_API_TOKEN")
    space_key = os.getenv("CONFLUENCE_SPACE_KEY", "DSA")

    if not all([confluence_url, username, api_token]):
        print("Error: Missing required environment variables")
        print(
            "Please ensure CONFLUENCE_URL, CONFLUENCE_USERNAME, and CONFLUENCE_API_TOKEN are set in .env"
        )
        return

    print(f"Connecting to Confluence at {confluence_url}")
    print(f"Username: {username}")
    print(f"Space: {space_key}")
    print("-" * 80)

    # Try Bearer token first (common for on-premise Confluence)
    print("\nAttempt 1: Testing Bearer token authentication...")
    client = ConfluenceRestClient(
        confluence_url, username, api_token, verify_ssl=False, auth_type="bearer"
    )
    if not client.test_connection():
        print("\nAttempt 2: Testing Basic authentication...")
        client = ConfluenceRestClient(
            confluence_url, username, api_token, verify_ssl=False, auth_type="basic"
        )
        if not client.test_connection():
            print(
                "\n✗ Both authentication methods failed. Please check your credentials."
            )
            print("Possible issues:")
            print("  1. API token may be invalid or expired")
            print("  2. Username may need to be in different format")
            print("  3. Additional authentication may be required (SSO, etc.)")
            return

    print("\n" + "=" * 80)
    print("Authentication successful! Proceeding to fetch pages...")
    print("=" * 80)

    # Fetch all pages from the space
    pages = client.get_all_pages_from_space(space_key)

    # Print summary
    print("-" * 80)
    print(f"\nRetrieved {len(pages)} pages from space '{space_key}'")
    print("\nSample pages:")
    for i, page in enumerate(pages[:5], 1):
        print(f"{i}. {page.title}")
        print(f"   ID: {page.id}")
        print(f"   URL: {page.url}")
        print(f"   Modified: {page.modified_date}")
        print(f"   Content length: {len(page.content_html or '')} chars")
        print()

    # Save to JSON
    output_file = os.path.join("rest_api_test", "confluence_pages.json")
    client.save_pages_to_json(pages, output_file)

    return pages


if __name__ == "__main__":
    main()
