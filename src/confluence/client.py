"""Confluence API client for retrieving project documentation."""

from typing import List, Dict, Optional, Any
from atlassian import Confluence
from loguru import logger
import re
import urllib3

# Suppress SSL warnings when verification is disabled (internal network use)
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


class ConfluenceClient:
    """
    Client for interacting with Confluence API.

    This class provides methods to connect to Confluence, retrieve pages from
    a specified space, and extract content and metadata.
    """

    def __init__(
        self,
        url: str,
        username: str,
        api_token: str,
        space_key: str = "DSA",
        verify_ssl: bool = False,
    ) -> None:
        """
        Initialize Confluence client.

        Args:
            url: Confluence instance URL.
            username: Confluence username/email.
            api_token: Confluence API token for authentication.
            space_key: Confluence space key to query. Defaults to 'DSA'.
            verify_ssl: Whether to verify SSL certificates. Defaults to False
                       for internal network compatibility.
        """
        self.url = url
        self.username = username
        self.space_key = space_key
        self.verify_ssl = verify_ssl
        self.client = Confluence(
            url=url,
            username=username,
            password=api_token,
            cloud=False,  # Set to False for on-premise Confluence servers
            verify_ssl=verify_ssl,
        )
        logger.info(f"Initialized Confluence client for space: {space_key}")
        if not verify_ssl:
            logger.warning("SSL verification is disabled - use only on trusted internal networks")

    def get_all_pages(self, limit: int = 500) -> List[Dict[str, Any]]:
        """
        Retrieve all pages from the configured Confluence space.

        Args:
            limit: Maximum number of pages to retrieve per request. Defaults to 500.

        Returns:
            List of page dictionaries containing page metadata and IDs.
        """
        logger.info(f"Retrieving pages from space: {self.space_key}")
        try:
            pages = []
            start = 0

            while True:
                response = self.client.get_all_pages_from_space(
                    space=self.space_key, start=start, limit=limit, expand="version"
                )

                if not response:
                    break

                pages.extend(response)
                logger.debug(f"Retrieved {len(response)} pages (total: {len(pages)})")

                if len(response) < limit:
                    break

                start += limit

            logger.info(f"Successfully retrieved {len(pages)} pages from Confluence")
            return pages

        except Exception as e:
            logger.error(f"Error retrieving pages from Confluence: {str(e)}")
            raise

    def get_page_content(self, page_id: str) -> Dict[str, Any]:
        """
        Get detailed content for a specific page.

        Args:
            page_id: The ID of the Confluence page.

        Returns:
            Dictionary containing page title, content, URL, and metadata.
        """
        try:
            page = self.client.get_page_by_id(
                page_id=page_id, expand="body.storage,version,space"
            )

            content = {
                "id": page["id"],
                "title": page["title"],
                "content": page["body"]["storage"]["value"],
                "url": f"{self.url}/wiki{page['_links']['webui']}",
                "space": page["space"]["key"],
                "version": page["version"]["number"],
                "last_updated": page["version"]["when"],
            }

            logger.debug(f"Retrieved content for page: {page['title']}")
            return content

        except Exception as e:
            logger.error(f"Error retrieving page {page_id}: {str(e)}")
            raise

    def get_all_pages_content(self, page_limit: int = 500) -> List[Dict[str, Any]]:
        """
        Retrieve full content for all pages in the space.

        Args:
            page_limit: Maximum number of pages to retrieve. Defaults to 500.

        Returns:
            List of dictionaries containing complete page content and metadata.
        """
        logger.info("Retrieving full content for all pages")
        pages = self.get_all_pages(limit=page_limit)
        pages_content = []

        for idx, page in enumerate(pages, 1):
            try:
                page_id = page["id"]
                content = self.get_page_content(page_id)
                pages_content.append(content)
                logger.debug(f"Processed page {idx}/{len(pages)}: {content['title']}")
            except Exception as e:
                logger.warning(f"Failed to retrieve content for page {page.get('id')}: {str(e)}")
                continue

        logger.info(f"Successfully retrieved content for {len(pages_content)} pages")
        return pages_content

    def search_pages(self, query: str, limit: int = 20) -> List[Dict[str, Any]]:
        """
        Search for pages matching a query string.

        Args:
            query: Search query string.
            limit: Maximum number of results to return. Defaults to 20.

        Returns:
            List of page dictionaries matching the search query.
        """
        logger.info(f"Searching for pages with query: {query}")
        try:
            cql = f'space = {self.space_key} AND text ~ "{query}"'
            results = self.client.cql(cql=cql, limit=limit)

            pages = []
            if "results" in results:
                for result in results["results"]:
                    if result["content"]["type"] == "page":
                        pages.append(result["content"])

            logger.info(f"Found {len(pages)} pages matching query")
            return pages

        except Exception as e:
            logger.error(f"Error searching pages: {str(e)}")
            raise

