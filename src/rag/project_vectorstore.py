"""
Project-level vector store for main_project retrieval.

This module provides a separate vector database optimized for project-level
similarity search. It's used in the two-stage RAG pipeline to first identify
relevant projects before searching within those projects at the chunk level.

Example:
    >>> from rag.project_vectorstore import ProjectVectorStore
    >>>
    >>> # Initialize store
    >>> store = ProjectVectorStore(persist_directory="./Data_Storage/project_vector_db")
    >>>
    >>> # Add conglomerated projects
    >>> store.add_projects(conglomerated_projects)
    >>>
    >>> # Query for similar projects
    >>> results = store.query_projects("What is ATLAS?", n_results=3)
    >>> for r in results:
    ...     print(f"{r['main_project']}: {r['similarity']:.3f}")
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pickle

from loguru import logger

# Import embedding manager for type hints
try:
    from rag.embeddings import EmbeddingManager
except ImportError:
    pass


class ProjectVectorStore:
    """
    Vector store for project-level retrieval.

    Maintains a separate vector database for conglomerated project documents,
    enabling fast project identification before chunk-level search.

    Attributes:
        persist_directory: Directory for storing vector data
        collection_name: Name of the vector collection
        embedding_model: Name of the embedding model

    Example:
        >>> store = ProjectVectorStore()
        >>> store.add_projects(projects)
        >>> results = store.query_projects("machine learning project")
    """

    def __init__(
        self,
        persist_directory: str = "./Data_Storage/project_vector_db",
        collection_name: str = "project_docs",
        embedding_model: str = "all-MiniLM-L6-v2",
    ) -> None:
        """Initialize the project vector store.

        Args:
            persist_directory: Directory to persist vector data
            collection_name: Name for the vector collection
            embedding_model: Sentence transformer model name
        """
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.embedding_model = embedding_model

        # Create directory
        Path(persist_directory).mkdir(parents=True, exist_ok=True)

        # File paths
        self.data_file = Path(persist_directory) / f"{collection_name}.pkl"
        self.embeddings_file = Path(persist_directory) / f"{collection_name}_embeddings.npy"

        # Lazy-load embedding manager
        self._embedding_manager: Optional["EmbeddingManager"] = None

        # Load or initialize data
        self._load_or_initialize()

        logger.info(
            f"Initialized ProjectVectorStore "
            f"(dir={persist_directory}, model={embedding_model})"
        )

    @property
    def embedding_manager(self) -> "EmbeddingManager":
        """Lazy-load embedding manager on first use."""
        if self._embedding_manager is None:
            from rag.embeddings import EmbeddingManager
            self._embedding_manager = EmbeddingManager(model_name=self.embedding_model)
        return self._embedding_manager

    def _load_or_initialize(self) -> None:
        """Load existing data or initialize empty store."""
        if self.data_file.exists() and self.embeddings_file.exists():
            try:
                with open(self.data_file, "rb") as f:
                    data = pickle.load(f)
                self.documents = data["documents"]
                self.metadatas = data["metadatas"]
                self.ids = data["ids"]
                self.embeddings = np.load(self.embeddings_file)
                logger.info(f"Loaded {len(self.documents)} projects from disk")
            except Exception as e:
                logger.warning(f"Error loading project store, initializing empty: {e}")
                self._initialize_empty()
        else:
            self._initialize_empty()

    def _initialize_empty(self) -> None:
        """Initialize empty data structures."""
        self.documents: List[str] = []
        self.metadatas: List[Dict[str, Any]] = []
        self.ids: List[str] = []
        self.embeddings = np.array([])

    def _save(self) -> None:
        """Save data to disk."""
        data = {
            "documents": self.documents,
            "metadatas": self.metadatas,
            "ids": self.ids,
        }
        with open(self.data_file, "wb") as f:
            pickle.dump(data, f)
        if len(self.embeddings) > 0:
            np.save(self.embeddings_file, self.embeddings)
        logger.debug(f"Saved {len(self.documents)} projects to disk")

    def clear(self) -> None:
        """Clear all data from the store."""
        self._initialize_empty()
        if self.data_file.exists():
            self.data_file.unlink()
        if self.embeddings_file.exists():
            self.embeddings_file.unlink()
        logger.info("Cleared project vector store")

    def add_projects(
        self,
        projects: List[Dict[str, Any]],
        clear_existing: bool = True,
    ) -> None:
        """Add conglomerated projects to vector store.

        Args:
            projects: List of conglomerated project dictionaries with:
                - main_project: Project name
                - main_project_id: Project page ID
                - content_text: Combined content
                - page_count: Number of pages
                - url: Project URL
            clear_existing: Whether to clear existing data first

        Example:
            >>> store.add_projects(conglomerated_projects)
        """
        if clear_existing:
            self.clear()

        logger.info(f"Adding {len(projects)} projects to vector store")

        texts = []
        metadatas = []
        ids = []

        for project in projects:
            project_name = project.get("main_project", "")
            content = project.get("content_text", "")

            if not content.strip():
                logger.warning(f"Skipping project '{project_name}' - no content")
                continue

            # Use truncated content for embedding (first 8000 chars)
            # This captures the main project info without exceeding model limits
            embedding_text = content[:8000]
            texts.append(embedding_text)

            metadatas.append({
                "main_project": project_name,
                "main_project_id": project.get("main_project_id", ""),
                "page_count": project.get("page_count", 0),
                "total_pages": project.get("total_pages", 0),
                "content_length": project.get("content_length", len(content)),
                "url": project.get("url", ""),
                "space_key": project.get("space_key", ""),
            })

            # Create unique ID from project name
            project_id = f"proj_{project_name.lower().replace(' ', '_').replace('-', '_')}"
            ids.append(project_id)

        if not texts:
            logger.warning("No projects with content to add")
            return

        # Generate embeddings
        logger.info(f"Generating embeddings for {len(texts)} projects...")
        new_embeddings = self.embedding_manager.generate_embeddings(texts)

        # Store data
        self.documents.extend(texts)
        self.metadatas.extend(metadatas)
        self.ids.extend(ids)

        if len(self.embeddings) == 0:
            self.embeddings = new_embeddings
        else:
            self.embeddings = np.vstack([self.embeddings, new_embeddings])

        # Save to disk
        self._save()
        logger.info(f"Added {len(texts)} projects to vector store")

    def query_projects(
        self,
        query: str,
        n_results: int = 3,
    ) -> List[Dict[str, Any]]:
        """Query for similar projects.

        Performs cosine similarity search to find projects most relevant
        to the query.

        Args:
            query: Query text
            n_results: Maximum number of results to return

        Returns:
            List of matching projects with:
            - main_project: Project name
            - main_project_id: Project page ID
            - similarity: Cosine similarity score (0-1)
            - content_preview: First 500 chars of content
            - page_count: Number of pages in project
            - url: Project URL

        Example:
            >>> results = store.query_projects("data pipeline", n_results=5)
            >>> for r in results:
            ...     print(f"{r['main_project']}: {r['similarity']:.3f}")
        """
        if len(self.documents) == 0:
            logger.warning("Project store is empty")
            return []

        # Generate query embedding
        query_embedding = self.embedding_manager.generate_embedding(query)

        # Compute cosine similarity
        query_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-10)
        embedding_norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True) + 1e-10
        normalized_embeddings = self.embeddings / embedding_norms
        similarities = np.dot(normalized_embeddings, query_norm)

        # Get top-k results
        top_k = min(n_results, len(self.documents))
        top_indices = np.argsort(similarities)[::-1][:top_k]

        results = []
        for idx in top_indices:
            results.append({
                "main_project": self.metadatas[idx].get("main_project"),
                "main_project_id": self.metadatas[idx].get("main_project_id"),
                "similarity": float(similarities[idx]),
                "content_preview": self.documents[idx][:500],
                "page_count": self.metadatas[idx].get("page_count", 0),
                "url": self.metadatas[idx].get("url", ""),
                "space_key": self.metadatas[idx].get("space_key", ""),
            })

        return results

    def get_project_names(self) -> List[str]:
        """Get list of all project names in store.

        Returns:
            List of main_project names

        Example:
            >>> projects = store.get_project_names()
            >>> print(f"Projects: {', '.join(projects)}")
        """
        return [m.get("main_project", "") for m in self.metadatas]

    def get_project_by_name(
        self,
        project_name: str,
    ) -> Optional[Dict[str, Any]]:
        """Get a specific project by name.

        Args:
            project_name: Name of the project to retrieve

        Returns:
            Project dictionary or None if not found

        Example:
            >>> project = store.get_project_by_name("ATLAS")
        """
        for i, meta in enumerate(self.metadatas):
            if meta.get("main_project") == project_name:
                return {
                    "main_project": meta.get("main_project"),
                    "main_project_id": meta.get("main_project_id"),
                    "content": self.documents[i],
                    "page_count": meta.get("page_count", 0),
                    "url": meta.get("url", ""),
                }
        return None

    def __len__(self) -> int:
        """Return number of projects in store."""
        return len(self.documents)

    def __contains__(self, project_name: str) -> bool:
        """Check if a project exists in store."""
        return project_name in self.get_project_names()
