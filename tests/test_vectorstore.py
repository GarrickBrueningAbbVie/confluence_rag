"""Unit tests for VectorStore class."""

import pytest
from src.rag.vectorstore import VectorStore
import tempfile
import shutil


@pytest.fixture
def temp_db_path() -> str:
    """Create a temporary directory for test database."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture
def vector_store(temp_db_path: str) -> VectorStore:
    """Create a VectorStore instance for testing."""
    return VectorStore(persist_directory=temp_db_path, collection_name="test_collection")


def test_vector_store_initialization(temp_db_path: str) -> None:
    """Test VectorStore initialization."""
    store = VectorStore(persist_directory=temp_db_path, collection_name="test")
    assert store.persist_directory == temp_db_path
    assert store.collection_name == "test"
    assert store.count() == 0


def test_add_documents(vector_store: VectorStore) -> None:
    """Test adding documents to vector store."""
    texts = ["This is test document 1", "This is test document 2"]
    metadatas = [{"source": "test1"}, {"source": "test2"}]
    ids = ["doc1", "doc2"]

    vector_store.add_documents(texts=texts, metadatas=metadatas, ids=ids)
    assert vector_store.count() == 2


def test_query_documents(vector_store: VectorStore) -> None:
    """Test querying documents."""
    texts = [
        "Machine learning is a subset of artificial intelligence",
        "Python is a popular programming language",
        "Data science involves statistics and programming",
    ]
    metadatas = [{"topic": "ML"}, {"topic": "Programming"}, {"topic": "Data Science"}]
    ids = ["doc1", "doc2", "doc3"]

    vector_store.add_documents(texts=texts, metadatas=metadatas, ids=ids)

    results = vector_store.query("artificial intelligence", n_results=2)
    assert len(results["documents"]) == 2
    # The ML document should be the most relevant
    assert "machine learning" in results["documents"][0].lower() or "artificial" in results["documents"][0].lower()


def test_get_by_ids(vector_store: VectorStore) -> None:
    """Test retrieving documents by IDs."""
    texts = ["Document 1", "Document 2"]
    metadatas = [{"id": 1}, {"id": 2}]
    ids = ["doc1", "doc2"]

    vector_store.add_documents(texts=texts, metadatas=metadatas, ids=ids)

    results = vector_store.get_by_ids(["doc1"])
    assert len(results["documents"]) == 1
    assert results["documents"][0] == "Document 1"


def test_delete_documents(vector_store: VectorStore) -> None:
    """Test deleting documents."""
    texts = ["Document 1", "Document 2"]
    metadatas = [{"id": 1}, {"id": 2}]
    ids = ["doc1", "doc2"]

    vector_store.add_documents(texts=texts, metadatas=metadatas, ids=ids)
    assert vector_store.count() == 2

    vector_store.delete_documents(["doc1"])
    assert vector_store.count() == 1


def test_clear_collection(vector_store: VectorStore) -> None:
    """Test clearing all documents."""
    texts = ["Document 1", "Document 2", "Document 3"]
    metadatas = [{"id": i} for i in range(3)]
    ids = [f"doc{i}" for i in range(3)]

    vector_store.add_documents(texts=texts, metadatas=metadatas, ids=ids)
    assert vector_store.count() == 3

    vector_store.clear_collection()
    assert vector_store.count() == 0


def test_peek(vector_store: VectorStore) -> None:
    """Test peeking at documents."""
    texts = ["Document 1", "Document 2", "Document 3"]
    metadatas = [{"id": i} for i in range(3)]
    ids = [f"doc{i}" for i in range(3)]

    vector_store.add_documents(texts=texts, metadatas=metadatas, ids=ids)

    results = vector_store.peek(limit=2)
    assert len(results["documents"]) == 2


def test_get_all_documents(vector_store: VectorStore) -> None:
    """Test retrieving all documents."""
    texts = ["Document 1", "Document 2"]
    metadatas = [{"id": 1}, {"id": 2}]
    ids = ["doc1", "doc2"]

    vector_store.add_documents(texts=texts, metadatas=metadatas, ids=ids)

    results = vector_store.get_all_documents()
    assert len(results["documents"]) == 2
    assert len(results["metadatas"]) == 2
    assert len(results["ids"]) == 2
