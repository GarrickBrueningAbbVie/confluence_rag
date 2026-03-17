"""
Confluence REST API client for retrieving pages.

This module uses the Confluence REST API to fetch all accessible pages
from a Confluence space.
"""

import os
import requests
import json
import urllib3
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from pathlib import Path
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from loguru import logger

# Disable SSL warnings when verification is disabled
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


@dataclass
class ConfluencePage:
    """Represents a Confluence page with its metadata.

    Attributes:
        id: Unique page identifier
        title: Page title
        space_key: Confluence space key (e.g., 'DSA')
        space_name: Full space name
        url: Web URL to the page
        created_date: ISO format creation date
        modified_date: ISO format last modification date
        author: Display name of last author
        version: Page version number
        content_html: Raw HTML content (storage format)
        content_text: Extracted plain text content
        external_links: List of external URLs found in content
        github_links: List of parsed GitHub link info (owner, repo, path, etc.)
        parents: List of parent pages from root to immediate parent
        children: List of child pages
        depth: Page hierarchy depth (1 = top-level)
        attachments: List of attachment metadata dicts
        attachment_content: Extracted text from all attachments
        parent_project: Name of the parent project (for DSA pages)
        main_project: Name of the main project (depth 3 ancestor for DSA pages)
        main_project_id: ID of the main project page
        technologies: List of technologies used in this project
        completeness_score: Project completeness score (0-100, NaN for subpages)
        completeness_summary: Summary of completeness assessment
    """

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
    github_links: List[Dict[str, Any]] = field(default_factory=list)
    parents: List[Dict[str, str]] = field(default_factory=list)
    children: List[Dict[str, str]] = field(default_factory=list)
    depth: int = 1  # Page hierarchy depth: 1 = top-level, higher = deeper in tree
    # New fields for preprocessing enhancements
    attachments: List[Dict[str, Any]] = field(default_factory=list)
    attachment_content: str = ""
    parent_project: Optional[str] = None
    main_project: Optional[str] = None
    main_project_id: Optional[str] = None
    technologies: List[str] = field(default_factory=list)
    completeness_score: Optional[float] = None
    completeness_summary: Optional[str] = None


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

    def _categorize_external_links(
        self, links: List[str]
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Categorize external links by type and extract metadata.

        Identifies GitHub, Jira, SharePoint, and other link types.
        For GitHub links, extracts owner, repo, and path information.

        Args:
            links: List of external URLs

        Returns:
            Dictionary with categorized links:
            - github: List of GitHub link info dicts
            - jira: List of Jira link info dicts
            - sharepoint: List of SharePoint links
            - other: List of uncategorized links
        """
        categorized = {
            "github": [],
            "jira": [],
            "sharepoint": [],
            "other": [],
        }

        for link in links:
            parsed = urlparse(link)
            domain = parsed.netloc.lower()

            # GitHub links (public and enterprise)
            if "github" in domain:
                github_info = self._parse_github_url(link)
                if github_info:
                    categorized["github"].append(github_info)

            # Jira links
            elif "jira" in domain:
                categorized["jira"].append({
                    "url": link,
                    "domain": domain,
                })

            # SharePoint links
            elif "sharepoint" in domain or "abbvienet.sharepoint" in domain:
                categorized["sharepoint"].append({
                    "url": link,
                    "domain": domain,
                })

            # Other links
            else:
                categorized["other"].append({
                    "url": link,
                    "domain": domain,
                })

        return categorized

    def _parse_github_url(self, url: str) -> Optional[Dict[str, Any]]:
        """
        Parse a GitHub URL and extract repository information.

        Handles both github.com and enterprise GitHub URLs.
        Extracts owner, repo, branch, file path, and link type.

        Args:
            url: GitHub URL to parse

        Returns:
            Dictionary with parsed info, or None if not a valid GitHub URL:
            - url: Original URL
            - domain: GitHub domain (github.com or enterprise)
            - owner: Repository owner/organization
            - repo: Repository name
            - type: Link type (repo, blob, tree, issues, pulls, etc.)
            - branch: Branch name (if applicable)
            - path: File/folder path (if applicable)
            - line_start: Starting line number (if applicable)
            - line_end: Ending line number (if applicable)
        """
        import re

        parsed = urlparse(url)
        domain = parsed.netloc.lower()

        if "github" not in domain:
            return None

        path_parts = parsed.path.strip("/").split("/")

        if len(path_parts) < 2:
            # Not enough path parts for owner/repo
            return {
                "url": url,
                "domain": domain,
                "owner": path_parts[0] if path_parts else None,
                "repo": None,
                "type": "profile" if path_parts else "home",
            }

        owner = path_parts[0]
        repo = path_parts[1]

        result = {
            "url": url,
            "domain": domain,
            "owner": owner,
            "repo": repo,
            "type": "repo",  # Default to repo root
            "branch": None,
            "path": None,
            "line_start": None,
            "line_end": None,
        }

        # Parse additional path components
        if len(path_parts) > 2:
            link_type = path_parts[2]
            result["type"] = link_type

            if link_type in ("blob", "tree") and len(path_parts) > 3:
                # File or directory link: /owner/repo/blob/branch/path/to/file
                result["branch"] = path_parts[3]
                if len(path_parts) > 4:
                    result["path"] = "/".join(path_parts[4:])

            elif link_type == "issues" and len(path_parts) > 3:
                # Issue link: /owner/repo/issues/123
                result["issue_number"] = path_parts[3]

            elif link_type == "pull" and len(path_parts) > 3:
                # PR link: /owner/repo/pull/123
                result["pr_number"] = path_parts[3]

            elif link_type == "commit" and len(path_parts) > 3:
                # Commit link: /owner/repo/commit/abc123
                result["commit_sha"] = path_parts[3]

        # Check for line number references in fragment (e.g., #L10-L20)
        if parsed.fragment:
            line_match = re.match(r"L(\d+)(?:-L(\d+))?", parsed.fragment)
            if line_match:
                result["line_start"] = int(line_match.group(1))
                if line_match.group(2):
                    result["line_end"] = int(line_match.group(2))

        return result

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

    def get_attachments(self, page_id: str) -> List[Dict[str, Any]]:
        """
        Fetch attachments for a specific page.

        Uses the Confluence REST API endpoint:
        /rest/api/content/{id}/child/attachment

        Args:
            page_id: The ID of the Confluence page

        Returns:
            List of attachment metadata dictionaries, each containing:
            - id: Attachment ID
            - title: Filename
            - mediaType: MIME type
            - fileSize: Size in bytes
            - download_url: Full URL to download the attachment

        Example:
            >>> attachments = client.get_attachments("123456")
            >>> for att in attachments:
            ...     print(f"{att['title']} ({att['mediaType']})")
        """
        url = f"{self.base_url}/rest/api/content/{page_id}/child/attachment"
        attachments = []

        try:
            response = self.session.get(url)
            response.raise_for_status()
            data = response.json()

            results = data.get("results", [])
            for attachment in results:
                # Extract download link from _links
                links = attachment.get("_links", {})
                download_path = links.get("download", "")

                # Construct full download URL
                # The download_path typically looks like: /download/attachments/{pageId}/{filename}
                # Some Confluence instances need /wiki prefix, others don't
                download_url = ""
                if download_path:
                    # If path already starts with /wiki or is absolute, use as-is
                    if download_path.startswith("/wiki") or download_path.startswith("http"):
                        download_url = f"{self.base_url}{download_path}" if not download_path.startswith("http") else download_path
                    else:
                        # Try without /wiki prefix first (more common for on-premise)
                        download_url = f"{self.base_url}{download_path}"

                att_data = {
                    "id": attachment.get("id", ""),
                    "title": attachment.get("title", ""),
                    "mediaType": attachment.get("metadata", {}).get("mediaType", ""),
                    "fileSize": attachment.get("extensions", {}).get("fileSize", 0),
                    "download_url": download_url,
                    "created_date": attachment.get("history", {}).get("createdDate", ""),
                    "comment": attachment.get("metadata", {}).get("comment", ""),
                }
                attachments.append(att_data)

            logger.debug(f"Found {len(attachments)} attachments for page {page_id}")

        except requests.exceptions.RequestException as e:
            logger.warning(f"Failed to fetch attachments for page {page_id}: {e}")

        return attachments

    def download_attachment(
        self,
        download_url: str,
        output_path: Optional[str] = None,
    ) -> Optional[bytes]:
        """
        Download an attachment file.

        Args:
            download_url: Full URL to download the attachment
            output_path: Optional path to save the file. If None, returns bytes.

        Returns:
            File content as bytes if output_path is None, else None after saving.

        Example:
            >>> content = client.download_attachment(att['download_url'])
            >>> # or save to file
            >>> client.download_attachment(att['download_url'], '/tmp/file.pdf')
        """
        # Build list of URLs to try (handles different Confluence configurations)
        urls_to_try = [download_url]

        # If URL contains /wiki/, also try without it
        if "/wiki/" in download_url:
            urls_to_try.append(download_url.replace("/wiki/", "/"))

        # If URL doesn't contain /wiki/, also try with it
        elif "/download/attachments/" in download_url:
            urls_to_try.append(download_url.replace("/download/", "/wiki/download/"))

        for url in urls_to_try:
            try:
                logger.debug(f"Trying to download from: {url}")
                response = self.session.get(url)
                response.raise_for_status()

                if output_path:
                    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
                    with open(output_path, 'wb') as f:
                        f.write(response.content)
                    logger.debug(f"Downloaded attachment to {output_path}")
                    return None
                else:
                    return response.content

            except requests.exceptions.RequestException as e:
                logger.debug(f"Failed to download from {url}: {e}")
                continue

        logger.error(f"Failed to download attachment from all attempted URLs: {urls_to_try}")
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
        github_links = []
        if content_html:
            external_links = self._extract_external_links(content_html)
            # Parse and categorize GitHub links for easier agent integration
            categorized = self._categorize_external_links(external_links)
            github_links = categorized.get("github", [])

        # Extract tree structure information (parents and children)
        parents = []

        # Get parents (ancestor pages from root to immediate parent)
        parents_data = page_data.get("ancestors", [])
        if parents_data:
            # Parents are ordered from root to immediate parent
            parents = [
                {
                    "id": parent.get("id", ""),
                    "title": parent.get("title", ""),
                    "type": parent.get("type", "page")
                }
                for parent in parents_data
            ]

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

        # Calculate page depth from parents
        # Depth 1 = root/top-level pages (no parents or only space home)
        # Depth increases with each parent level
        depth = len(parents) + 1

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
            github_links=github_links,
            parents=parents,
            children=children,
            depth=depth,
        )

    def save_pages_to_json(
        self, pages: List[ConfluencePage], output_file: Optional[str] = None
    ):
        """
        Save retrieved pages to a JSON file in the Data_Storage directory.

        Args:
            pages: List of ConfluencePage objects
            output_file: Optional path to output JSON file. If not provided, saves to Data_Storage/confluence_pages.json
        """
        if output_file is None:
            # Default to Data_Storage directory
            project_root = Path(__file__).parent.parent.parent
            data_storage = project_root / "Data_Storage"
            data_storage.mkdir(exist_ok=True)
            output_file = str(data_storage / "confluence_pages.json")

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
                "github_links": page.github_links,
                "parents": page.parents,
                "children": page.children,
                "depth": page.depth,
                # Preprocessing fields
                "attachments": page.attachments,
                "attachment_content": page.attachment_content,
                "parent_project": page.parent_project,
                "main_project": page.main_project,
                "main_project_id": page.main_project_id,
                "technologies": page.technologies,
                "completeness_score": page.completeness_score,
                "completeness_summary": page.completeness_summary,
            }
            for page in pages
        ]

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(pages_dict, f, indent=2, ensure_ascii=False)

        logger.info(f"Saved {len(pages)} pages to {output_file}")


