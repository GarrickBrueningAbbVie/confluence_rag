"""
Confluence REST API test module.

This module provides a reliable alternative to the Atlassian Python API
for retrieving Confluence pages using direct REST API calls.
"""

from .confluence_rest_client import ConfluenceRestClient, ConfluencePage

__all__ = ['ConfluenceRestClient', 'ConfluencePage']
