"""
Script to generate embeddings and store Confluence pages in vector database.

This script loads the Confluence pages JSON file, chunks the content,
generates embeddings, and stores them in ChromaDB for RAG queries.
"""

import sys
import json
from pathlib import Path
from typing import List, Dict, Any

# Add src directory to path to import modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import ConfigConfluenceRag
from rag.embeddings import EmbeddingManager
from rag.vectorstore import VectorStore
from confluence.parser import ConfluenceParser


def load_confluence_pages(json_path: str) -> List[Dict[str, Any]]:
    """Load Confluence pages from JSON file."""
    print(f"Loading Confluence pages from: {json_path}")
    with open(json_path, 'r', encoding='utf-8') as f:
        pages_data = json.load(f)

    print(f"✓ Loaded {len(pages_data)} pages from JSON")
    return pages_data


def chunk_pages(pages_data: List[Dict[str, Any]],
                chunk_size: int,
                chunk_overlap: int) -> List[Dict[str, Any]]:
    """Chunk pages into smaller segments for vectorization."""
    print("\nChunking documents...")
    parser = ConfluenceParser()
    chunks = []

    for page in pages_data:
        # Extract text content from the page
        text = page.get('content_text', '') or page.get('text', '')

        if not text:
            print(f"⚠️  Skipping page '{page.get('title', 'Unknown')}' - no text content")
            continue

        # Chunk the text
        text_chunks = parser.chunk_text(
            text,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )

        # Create chunk objects with metadata including hierarchy info for re-ranking
        for i, chunk in enumerate(text_chunks):
            # Get children list (needed for re-ranking)
            children = page.get('children', [])

            chunks.append({
                'text': chunk,
                'metadata': {
                    'title': page.get('title', 'Unknown'),
                    'url': page.get('url', ''),
                    'source_type': 'confluence',
                    'chunk_id': i,
                    'space': page.get('space', page.get('space_key', '')),
                    'author': page.get('author', ''),
                    'version': page.get('version', ''),
                    'page_id': page.get('id', ''),
                    # Hierarchy metadata for re-ranking
                    'depth': page.get('depth', len(page.get('ancestors', [])) + 1),
                    'parent_id': page.get('parent_id', ''),
                    'parent_title': page.get('parent_title', ''),
                    'children': children,
                    'has_children': len(children) > 0,
                    'children_count': len(children),
                }
            })

    print(f"✓ Created {len(chunks)} chunks from {len(pages_data)} pages")
    return chunks


def store_in_vectordb(chunks: List[Dict[str, Any]],
                      persist_directory: str,
                      collection_name: str,
                      embedding_model: str,
                      clear_existing: bool = True) -> None:
    """Generate embeddings and store in vector database."""
    print("\n" + "=" * 80)
    print("Initializing Vector Storage")
    print("=" * 80)

    # Initialize embedding manager
    print(f"\nLoading embedding model: {embedding_model}")
    embedding_manager = EmbeddingManager(model_name=embedding_model)
    print(f"✓ Embedding model loaded")
    print(f"  - Model: {embedding_manager.model_name}")
    print(f"  - Dimension: {embedding_manager.embedding_dimension}")

    # Ensure persist directory exists
    persist_path = Path(persist_directory)
    persist_path.mkdir(parents=True, exist_ok=True)

    # Initialize vector store
    print(f"\nInitializing vector store at: {persist_directory}")
    vector_store = VectorStore(
        persist_directory=persist_directory,
        collection_name=collection_name
    )

    # Clear existing data if requested
    existing_count = vector_store.count()
    if existing_count > 0:
        if clear_existing:
            print(f"Clearing existing {existing_count} documents from vector store...")
            vector_store.clear_collection()
            print("✓ Vector store cleared")
        else:
            print(f"⚠️  Vector store already contains {existing_count} documents")
            print("   Appending new documents...")

    # Prepare data for storage
    print("\nPreparing documents for vectorization...")
    texts = [chunk['text'] for chunk in chunks]
    metadatas = [chunk['metadata'] for chunk in chunks]
    ids = [f"doc_{i}" for i in range(len(chunks))]

    # Add documents to vector store
    print(f"Adding {len(chunks)} documents to vector store...")
    print("(This may take a few minutes depending on the number of documents)")
    vector_store.add_documents(texts=texts, metadatas=metadatas, ids=ids)

    final_count = vector_store.count()
    print(f"\n✓ Successfully added documents to vector store")
    print(f"  - Total documents in store: {final_count}")

    # Test query
    print("\n" + "=" * 80)
    print("Running test query to verify vector store")
    print("=" * 80)
    test_query = "What data science projects are documented?"
    print(f"\nTest query: {test_query}")

    results = vector_store.query(query_text=test_query, n_results=3)

    print("\nTop 3 results:")
    for i, (doc, meta, dist) in enumerate(zip(
        results['documents'],
        results['metadatas'],
        results['distances']
    ), 1):
        print(f"\n{i}. Title: {meta['title']}")
        print(f"   Source: {meta['source_type']}")
        print(f"   Distance: {dist:.4f}")
        print(f"   Preview: {doc[:150]}...")


def main():
    """Main function to vectorize Confluence pages."""
    print("=" * 80)
    print("Confluence RAG - Data Vectorization")
    print("=" * 80)
    print()

    # Validate configuration
    if not ConfigConfluenceRag.validate():
        print("❌ Configuration validation failed. Please check your .env file.")
        sys.exit(1)

    print("✓ Configuration validated successfully")
    print(f"  - Embedding Model: {ConfigConfluenceRag.EMBEDDING_MODEL}")
    print(f"  - Chunk Size: {ConfigConfluenceRag.CHUNK_SIZE}")
    print(f"  - Chunk Overlap: {ConfigConfluenceRag.CHUNK_OVERLAP}")
    print()

    # Determine paths
    project_root = Path(__file__).parent.parent.parent
    json_path = project_root / 'Data_Storage' / 'confluence_pages.json'
    vector_db_path = project_root / 'Data_Storage' / 'vector_db'

    # Check if JSON file exists
    if not json_path.exists():
        print("=" * 80)
        print("❌ Error: Confluence pages JSON file not found")
        print("=" * 80)
        print()
        print(f"Expected location: {json_path}")
        print()
        print("Please run the fetch_pages.py script first to download Confluence pages:")
        print("  python src/confluence/fetch_pages.py")
        print()
        sys.exit(1)

    print("=" * 80)
    print("Step 1: Load Confluence Pages")
    print("=" * 80)
    print()
    pages_data = load_confluence_pages(str(json_path))

    print()
    print("=" * 80)
    print("Step 2: Chunk Documents")
    print("=" * 80)
    print()
    chunks = chunk_pages(
        pages_data,
        chunk_size=ConfigConfluenceRag.CHUNK_SIZE,
        chunk_overlap=ConfigConfluenceRag.CHUNK_OVERLAP
    )

    print()
    print("=" * 80)
    print("Step 3: Generate Embeddings and Store in Vector Database")
    print("=" * 80)
    store_in_vectordb(
        chunks,
        persist_directory=str(vector_db_path),
        collection_name='confluence_docs',
        embedding_model=ConfigConfluenceRag.EMBEDDING_MODEL,
        clear_existing=True
    )

    print()
    print("=" * 80)
    print("✓ Vectorization Complete!")
    print("=" * 80)
    print()
    print(f"Vector database location: {vector_db_path}")
    print("The RAG pipeline is now ready to answer questions.")
    print()


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
