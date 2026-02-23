"""Vector storage using numpy and pickle for the RAG pipeline."""

from typing import List, Dict, Optional, Any
import numpy as np
import pickle
import os
from loguru import logger
from rag.embeddings import EmbeddingManager


class VectorStore:
    """
    Vector database manager using numpy arrays and pickle persistence.

    This class handles storage and retrieval of document embeddings
    for the RAG pipeline using a simple but effective approach that
    avoids heavy dependencies.
    """

    def __init__(
        self,
        persist_directory: str = "./Data_Storage/vector_db",
        collection_name: str = "confluence_docs",
        embedding_model: str = "all-MiniLM-L6-v2",
    ) -> None:
        """
        Initialize the vector store.

        Args:
            persist_directory: Directory path for persisting the database.
                              Defaults to './Data_Storage/vector_db'.
            collection_name: Name of the collection to use. Defaults to 'confluence_docs'.
            embedding_model: Name of the embedding model. Defaults to 'all-MiniLM-L6-v2'.
        """
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.embedding_model = embedding_model

        os.makedirs(persist_directory, exist_ok=True)

        self.data_file = os.path.join(persist_directory, f"{collection_name}.pkl")
        self.embeddings_file = os.path.join(persist_directory, f"{collection_name}_embeddings.npy")

        logger.info(f"Initializing VectorStore at: {persist_directory}")

        # Initialize embedding manager
        self._embedding_manager: Optional[EmbeddingManager] = None

        # Load existing data or initialize empty
        self._load_or_initialize()

        logger.info(f"Vector store initialized with collection: {collection_name}")

    @property
    def embedding_manager(self) -> EmbeddingManager:
        """Lazy load embedding manager to avoid loading model until needed."""
        if self._embedding_manager is None:
            self._embedding_manager = EmbeddingManager(model_name=self.embedding_model)
        return self._embedding_manager

    def _load_or_initialize(self) -> None:
        """Load existing data from disk or initialize empty storage."""
        if os.path.exists(self.data_file) and os.path.exists(self.embeddings_file):
            try:
                with open(self.data_file, "rb") as f:
                    data = pickle.load(f)
                self.documents: List[str] = data["documents"]
                self.metadatas: List[Dict[str, Any]] = data["metadatas"]
                self.ids: List[str] = data["ids"]
                self.embeddings: np.ndarray = np.load(self.embeddings_file)
                logger.info(f"Loaded {len(self.documents)} documents from disk")
            except Exception as e:
                logger.warning(f"Error loading data, initializing empty: {str(e)}")
                self._initialize_empty()
        else:
            self._initialize_empty()

    def _initialize_empty(self) -> None:
        """Initialize empty storage."""
        self.documents = []
        self.metadatas = []
        self.ids = []
        self.embeddings = np.array([])

    def _save(self) -> None:
        """Save current data to disk."""
        try:
            data = {
                "documents": self.documents,
                "metadatas": self.metadatas,
                "ids": self.ids,
            }
            with open(self.data_file, "wb") as f:
                pickle.dump(data, f)

            if len(self.embeddings) > 0:
                np.save(self.embeddings_file, self.embeddings)

            logger.debug(f"Saved {len(self.documents)} documents to disk")
        except Exception as e:
            logger.error(f"Error saving data: {str(e)}")
            raise

    def add_documents(
        self,
        texts: List[str],
        metadatas: List[Dict[str, Any]],
        ids: Optional[List[str]] = None,
    ) -> None:
        """
        Add documents to the vector store.

        Args:
            texts: List of text chunks to add.
            metadatas: List of metadata dictionaries for each text.
            ids: Optional list of unique IDs for each document.
                If None, IDs will be auto-generated.
        """
        if not texts:
            logger.warning("No texts provided to add to vector store")
            return

        if len(texts) != len(metadatas):
            raise ValueError("Number of texts and metadatas must match")

        if ids is None:
            start_idx = len(self.documents)
            ids = [f"doc_{start_idx + i}" for i in range(len(texts))]

        if len(ids) != len(texts):
            raise ValueError("Number of IDs must match number of texts")

        logger.info(f"Adding {len(texts)} documents to vector store")

        try:
            # Generate embeddings for new texts
            new_embeddings = self.embedding_manager.generate_embeddings(texts)

            # Add to storage
            self.documents.extend(texts)
            self.metadatas.extend(metadatas)
            self.ids.extend(ids)

            # Concatenate embeddings
            if len(self.embeddings) == 0:
                self.embeddings = new_embeddings
            else:
                self.embeddings = np.vstack([self.embeddings, new_embeddings])

            # Persist to disk
            self._save()

            logger.info(f"Successfully added {len(texts)} documents")
        except Exception as e:
            logger.error(f"Error adding documents to vector store: {str(e)}")
            raise

    def query(
        self,
        query_text: str,
        n_results: int = 5,
        where: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Query the vector store for similar documents.

        Args:
            query_text: Query text to search for.
            n_results: Number of results to return. Defaults to 5.
            where: Optional metadata filter conditions (not fully implemented).

        Returns:
            Dictionary containing matched documents, distances, and metadatas.
        """
        if len(self.documents) == 0:
            logger.warning("Vector store is empty")
            return {
                "documents": [],
                "metadatas": [],
                "distances": [],
                "ids": [],
            }

        logger.info(f"Querying vector store for: '{query_text[:50]}...'")

        try:
            # Generate embedding for query
            query_embedding = self.embedding_manager.generate_embedding(query_text)

            # Calculate cosine similarities
            similarities = self._cosine_similarity(query_embedding, self.embeddings)

            # Get top k indices
            top_k = min(n_results, len(self.documents))
            top_indices = np.argsort(similarities)[::-1][:top_k]

            # Convert similarity to distance (1 - similarity)
            distances = [float(1 - similarities[i]) for i in top_indices]

            result = {
                "documents": [self.documents[i] for i in top_indices],
                "metadatas": [self.metadatas[i] for i in top_indices],
                "distances": distances,
                "ids": [self.ids[i] for i in top_indices],
            }

            logger.info(f"Found {len(result['documents'])} matching documents")
            return result

        except Exception as e:
            logger.error(f"Error querying vector store: {str(e)}")
            raise

    def _cosine_similarity(
        self, query_embedding: np.ndarray, embeddings: np.ndarray
    ) -> np.ndarray:
        """
        Calculate cosine similarity between query and all embeddings.

        Args:
            query_embedding: Query embedding vector.
            embeddings: Array of document embeddings.

        Returns:
            Array of similarity scores.
        """
        # Normalize query
        query_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-10)

        # Normalize embeddings
        embedding_norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-10
        normalized_embeddings = embeddings / embedding_norms

        # Calculate dot product (cosine similarity for normalized vectors)
        similarities = np.dot(normalized_embeddings, query_norm)

        return similarities

    def get_by_ids(self, ids: List[str]) -> Dict[str, Any]:
        """
        Retrieve documents by their IDs.

        Args:
            ids: List of document IDs to retrieve.

        Returns:
            Dictionary containing documents and their metadatas.
        """
        logger.debug(f"Retrieving {len(ids)} documents by ID")

        try:
            indices = [self.ids.index(doc_id) for doc_id in ids if doc_id in self.ids]
            return {
                "documents": [self.documents[i] for i in indices],
                "metadatas": [self.metadatas[i] for i in indices],
                "ids": [self.ids[i] for i in indices],
            }
        except Exception as e:
            logger.error(f"Error retrieving documents by ID: {str(e)}")
            raise

    def delete_documents(self, ids: List[str]) -> None:
        """
        Delete documents from the vector store.

        Args:
            ids: List of document IDs to delete.
        """
        logger.info(f"Deleting {len(ids)} documents from vector store")

        try:
            indices_to_delete = [
                i for i, doc_id in enumerate(self.ids) if doc_id in ids
            ]

            # Remove in reverse order to maintain indices
            for i in sorted(indices_to_delete, reverse=True):
                del self.documents[i]
                del self.metadatas[i]
                del self.ids[i]

            # Update embeddings
            if indices_to_delete:
                mask = np.ones(len(self.embeddings), dtype=bool)
                mask[indices_to_delete] = False
                self.embeddings = self.embeddings[mask]

            self._save()
            logger.info(f"Successfully deleted {len(indices_to_delete)} documents")
        except Exception as e:
            logger.error(f"Error deleting documents: {str(e)}")
            raise

    def update_document(
        self, doc_id: str, text: str, metadata: Dict[str, Any]
    ) -> None:
        """
        Update a document in the vector store.

        Args:
            doc_id: ID of the document to update.
            text: New text content.
            metadata: New metadata dictionary.
        """
        logger.info(f"Updating document: {doc_id}")

        try:
            if doc_id not in self.ids:
                raise ValueError(f"Document ID {doc_id} not found")

            idx = self.ids.index(doc_id)

            # Update document and metadata
            self.documents[idx] = text
            self.metadatas[idx] = metadata

            # Regenerate embedding
            new_embedding = self.embedding_manager.generate_embedding(text)
            self.embeddings[idx] = new_embedding

            self._save()
            logger.info(f"Successfully updated document: {doc_id}")
        except Exception as e:
            logger.error(f"Error updating document: {str(e)}")
            raise

    def count(self) -> int:
        """
        Get the number of documents in the collection.

        Returns:
            Number of documents stored.
        """
        count = len(self.documents)
        logger.debug(f"Vector store contains {count} documents")
        return count

    def clear_collection(self) -> None:
        """Delete all documents from the collection."""
        logger.warning("Clearing all documents from collection")
        try:
            self._initialize_empty()
            self._save()
            logger.info("Collection cleared successfully")
        except Exception as e:
            logger.error(f"Error clearing collection: {str(e)}")
            raise

    def get_all_documents(self) -> Dict[str, Any]:
        """
        Retrieve all documents from the collection.

        Returns:
            Dictionary containing all documents and their metadatas.
        """
        logger.info("Retrieving all documents from collection")
        return {
            "documents": self.documents.copy(),
            "metadatas": self.metadatas.copy(),
            "ids": self.ids.copy(),
        }

    def peek(self, limit: int = 10) -> Dict[str, Any]:
        """
        Peek at a few documents from the collection.

        Args:
            limit: Number of documents to retrieve. Defaults to 10.

        Returns:
            Dictionary containing sample documents and metadatas.
        """
        logger.debug(f"Peeking at {limit} documents")
        limit = min(limit, len(self.documents))
        return {
            "documents": self.documents[:limit],
            "metadatas": self.metadatas[:limit],
            "ids": self.ids[:limit],
        }
