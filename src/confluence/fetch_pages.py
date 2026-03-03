"""
Script to fetch all accessible Confluence pages using the REST API.

This script demonstrates how to use the ConfluenceRestClient to retrieve
all pages from the DSA space and save them for further processing.
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Add src directory to path to import confluence module
sys.path.insert(0, str(Path(__file__).parent.parent))

from confluence.rest_client import ConfluenceRestClient


def fetch_all_pages():
    """Fetch all pages from Confluence space and save to JSON."""
    # Load environment variables from .env file
    load_dotenv()

    # Get credentials from environment
    confluence_url = os.getenv('CONFLUENCE_URL')
    username = os.getenv('CONFLUENCE_USERNAME')
    api_token = os.getenv('CONFLUENCE_API_TOKEN')
    space_key = os.getenv('CONFLUENCE_SPACE_KEY', 'DSA')

    # Validate credentials
    if not all([confluence_url, username, api_token]):
        print("❌ Error: Missing required environment variables")
        print("Please ensure the following are set in your .env file:")
        print("  - CONFLUENCE_URL")
        print("  - CONFLUENCE_USERNAME")
        print("  - CONFLUENCE_API_TOKEN")
        sys.exit(1)

    print("=" * 80)
    print("Confluence REST API - Page Fetcher")
    print("=" * 80)
    print(f"Confluence URL: {confluence_url}")
    print(f"Username: {username}")
    print(f"Space Key: {space_key}")
    print("=" * 80)
    print()

    # Try Bearer token authentication first (common for on-premise Confluence with PATs)
    print("Attempting Bearer token authentication...")
    print()
    client = ConfluenceRestClient(
        base_url=confluence_url,
        username=username,
        api_token=api_token,
        verify_ssl=False,
        auth_type='bearer'
    )

    if not client.test_connection():
        print()
        print("Bearer token failed. Trying Basic authentication...")
        print()
        client = ConfluenceRestClient(
            base_url=confluence_url,
            username=username,
            api_token=api_token,
            verify_ssl=False,
            auth_type='basic'
        )
        if not client.test_connection():
            print()
            print("=" * 80)
            print("❌ Authentication Failed")
            print("=" * 80)
            print()
            print("Both authentication methods failed. Possible issues:")
            print("  1. API token may be invalid or expired")
            print("  2. Username format may be incorrect")
            print("  3. Additional authentication (SSO, VPN) may be required")
            print("  4. You may need to generate a new Personal Access Token (PAT)")
            print()
            sys.exit(1)

    print()
    print("=" * 80)
    print("✓ Authentication successful!")
    print("=" * 80)
    print()

    # Fetch all pages from the space
    pages = client.get_all_pages_from_space(space_key)

    # Display results
    print()
    print("=" * 80)
    print(f"✓ Successfully retrieved {len(pages)} pages")
    print("=" * 80)
    print()

    if pages:
        print("Sample of retrieved pages:")
        print("-" * 80)
        for i, page in enumerate(pages[:10], 1):
            print(f"{i}. {page.title}")
            print(f"   ID: {page.id}")
            print(f"   Space: {page.space_key} - {page.space_name}")
            print(f"   Author: {page.author}")
            print(f"   Modified: {page.modified_date}")
            print(f"   Version: {page.version}")
            print(f"   Content HTML size: {len(page.content_html or '')} characters")
            print(f"   Content Text size: {len(page.content_text or '')} characters")
            print(f"   External links: {len(page.external_links)} links")
            if page.parent_title:
                print(f"   Parent: {page.parent_title} (ID: {page.parent_id})")
            print(f"   Ancestors: {len(page.ancestors)} level(s)")
            print(f"   Children: {len(page.children)} page(s)")
            print(f"   URL: {page.url}")
            print()

        if len(pages) > 10:
            print(f"... and {len(pages) - 10} more pages")
            print()

        # Save to JSON in root Data_Storage directory
        project_root = Path(__file__).parent.parent.parent
        output_dir = project_root / 'Data_Storage'
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / 'confluence_pages.json'

        print("-" * 80)
        print(f"Saving pages to: {output_file}")
        client.save_pages_to_json(pages, str(output_file))

        print()
        print("=" * 80)
        print("✓ Done! Pages saved successfully")
        print("=" * 80)
    else:
        print("⚠️  No pages found in the space")

    return pages


if __name__ == '__main__':
    try:
        pages = fetch_all_pages()
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
