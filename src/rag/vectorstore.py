"""Vector storage using numpy and pickle for the RAG pipeline."""

from typing import List, Dict, Optional, Any
import numpy as np
import pickle
import os
from loguru import logger
from rag.embeddings import EmbeddingManager
from rag.similarity import batch_cosine_similarity


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
            similarities = batch_cosine_similarity(query_embedding, self.embeddings)

            # DEBUG: Check similarity distribution
            logger.info(
                f"Similarity stats - min: {similarities.min():.4f}, "
                f"max: {similarities.max():.4f}, mean: {similarities.mean():.4f}, "
                f"std: {similarities.std():.4f}"
            )

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

            top_titles = [m.get('title', 'Unknown')[:50] for m in result['metadatas'][:3]]
            logger.info(f"Found {len(result['documents'])} matching documents, top results: {top_titles}")
            return result

        except Exception as e:
            logger.error(f"Error querying vector store: {str(e)}")
            raise

    def query_with_multi_filter(
        self,
        query_text: str,
        n_results: int = 5,
        filters: Optional[List[Dict[str, Any]]] = None,
        filter_logic: str = "OR",
    ) -> Dict[str, Any]:
        """
        Query the vector store with multiple metadata filters.

        Supports filtering on multiple fields (e.g., main_project AND/OR author)
        with configurable logic.

        Args:
            query_text: Query text to search for.
            n_results: Number of results to return. Defaults to 5.
            filters: List of filter dictionaries, each with:
                - field: Metadata field name (e.g., 'main_project', 'author')
                - values: List of allowed values for that field
            filter_logic: "OR" (match any filter) or "AND" (match all filters).

        Returns:
            Dictionary containing matched documents, distances, and metadatas.

        Example:
            >>> results = store.query_with_multi_filter(
            ...     query_text="data pipeline",
            ...     filters=[
            ...         {"field": "main_project", "values": ["ATLAS", "DataOps"]},
            ...         {"field": "author", "values": ["John Smith"]},
            ...     ],
            ...     filter_logic="OR"
            ... )
        """
        if len(self.documents) == 0:
            logger.warning("Vector store is empty")
            return {
                "documents": [],
                "metadatas": [],
                "distances": [],
                "ids": [],
            }

        logger.info(f"Querying with multi-filter (logic={filter_logic}): {filters}")

        try:
            # Generate embedding for query
            query_embedding = self.embedding_manager.generate_embedding(query_text)

            # Calculate cosine similarities
            similarities = batch_cosine_similarity(query_embedding, self.embeddings)

            # DEBUG: Check similarity distribution before filtering
            logger.info(
                f"Similarity stats (before filter) - min: {similarities.min():.4f}, "
                f"max: {similarities.max():.4f}, mean: {similarities.mean():.4f}"
            )

            # Apply metadata filters if specified
            if filters and len(filters) > 0:
                # Build individual masks for each filter
                filter_masks = []
                for f in filters:
                    field = f.get("field")
                    values = f.get("values", [])
                    if field and values:
                        # Normalize values for case-insensitive matching
                        values_lower = [v.lower() if isinstance(v, str) else v for v in values]
                        mask = np.array([
                            str(meta.get(field, "")).lower() in values_lower
                            for meta in self.metadatas
                        ])
                        filter_masks.append(mask)
                        matching_count = mask.sum()
                        logger.info(f"Filter '{field}' in {values}: matched {matching_count} documents")

                # Combine masks based on logic
                if filter_masks:
                    if filter_logic.upper() == "AND":
                        combined_mask = np.all(filter_masks, axis=0)
                    else:  # OR logic (default)
                        combined_mask = np.any(filter_masks, axis=0)

                    total_matching = combined_mask.sum()
                    logger.info(f"Combined filter ({filter_logic}): {total_matching} documents match")

                    # Set similarity to -inf for non-matching documents
                    filtered_similarities = np.where(combined_mask, similarities, -np.inf)
                else:
                    filtered_similarities = similarities
            else:
                filtered_similarities = similarities

            # Get top k indices from filtered results
            top_k = min(n_results, len(self.documents))
            top_indices = np.argsort(filtered_similarities)[::-1][:top_k]

            # Filter out indices with -inf similarity (filtered out)
            valid_indices = [i for i in top_indices if filtered_similarities[i] != -np.inf]

            # Convert similarity to distance (1 - similarity)
            distances = [float(1 - similarities[i]) for i in valid_indices]

            result = {
                "documents": [self.documents[i] for i in valid_indices],
                "metadatas": [self.metadatas[i] for i in valid_indices],
                "distances": distances,
                "ids": [self.ids[i] for i in valid_indices],
            }

            top_titles = [m.get('title', 'Unknown')[:50] for m in result['metadatas'][:3]]
            logger.info(f"Found {len(result['documents'])} documents after multi-filtering, top results: {top_titles}")
            return result

        except Exception as e:
            logger.error(f"Error querying vector store with multi-filter: {str(e)}")
            raise

    def query_with_filter(
        self,
        query_text: str,
        n_results: int = 5,
        filter_field: str = None,
        filter_values: List[str] = None,
    ) -> Dict[str, Any]:
        """
        Query the vector store with metadata filtering.

        Args:
            query_text: Query text to search for.
            n_results: Number of results to return. Defaults to 5.
            filter_field: Metadata field to filter on (e.g., 'main_project').
            filter_values: List of allowed values for the filter field.

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

        logger.info(f"Querying with filter: {filter_field}={filter_values}")

        try:
            # Generate embedding for query
            query_embedding = self.embedding_manager.generate_embedding(query_text)

            # Calculate cosine similarities
            similarities = batch_cosine_similarity(query_embedding, self.embeddings)

            # DEBUG: Check similarity distribution before filtering
            logger.info(
                f"Similarity stats (before filter) - min: {similarities.min():.4f}, "
                f"max: {similarities.max():.4f}, mean: {similarities.mean():.4f}, "
                f"std: {similarities.std():.4f}"
            )

            # Apply metadata filter if specified
            if filter_field and filter_values:
                # Create mask for documents matching filter
                mask = np.array([
                    meta.get(filter_field, "") in filter_values
                    for meta in self.metadatas
                ])
                matching_count = mask.sum()
                logger.info(f"Filter matched {matching_count}/{len(self.documents)} documents")

                # Show what values are in the store when no matches found
                if matching_count == 0:
                    unique_values = set(meta.get(filter_field, "") for meta in self.metadatas)
                    sample_values = sorted(list(unique_values))[:10]
                    logger.warning(
                        f"No matches for {filter_field}={filter_values}. "
                        f"Sample values in store: {sample_values}"
                    )

                # Set similarity to -inf for non-matching documents
                filtered_similarities = np.where(mask, similarities, -np.inf)
            else:
                filtered_similarities = similarities

            # Get top k indices from filtered results
            top_k = min(n_results, len(self.documents))
            top_indices = np.argsort(filtered_similarities)[::-1][:top_k]

            # Filter out indices with -inf similarity (filtered out)
            valid_indices = [i for i in top_indices if filtered_similarities[i] != -np.inf]

            # Convert similarity to distance (1 - similarity)
            distances = [float(1 - similarities[i]) for i in valid_indices]

            result = {
                "documents": [self.documents[i] for i in valid_indices],
                "metadatas": [self.metadatas[i] for i in valid_indices],
                "distances": distances,
                "ids": [self.ids[i] for i in valid_indices],
            }

            top_titles = [m.get('title', 'Unknown')[:50] for m in result['metadatas'][:3]]
            logger.info(f"Found {len(result['documents'])} matching documents after filtering, top results: {top_titles}")
            return result

        except Exception as e:
            logger.error(f"Error querying vector store with filter: {str(e)}")
            raise

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

    def find_main_projects_by_similarity(
        self,
        project_names: List[str],
        similarity_threshold: float = 0.8,
    ) -> List[str]:
        """
        Find main_project values that are similar to the given project names.

        Uses embedding similarity to match project names that may not be exact matches.

        Args:
            project_names: List of project names extracted from query.
            similarity_threshold: Minimum similarity score (0-1). Defaults to 0.8.

        Returns:
            List of matching main_project values from the store.
        """
        if not project_names or len(self.documents) == 0:
            return []

        # Get unique main_project values from the store
        unique_projects = list(set(
            meta.get('main_project', '') for meta in self.metadatas
            if meta.get('main_project')
        ))

        if not unique_projects:
            return []

        logger.info(f"Finding main_projects similar to: {project_names}")

        # Generate embeddings for project names and unique main_projects
        query_embeddings = self.embedding_manager.generate_embeddings(project_names)
        project_embeddings = self.embedding_manager.generate_embeddings(unique_projects)

        # Calculate similarities using batch operation
        matched_projects = set()
        for i, query_name in enumerate(project_names):
            # Use batch_cosine_similarity for vectorized computation
            similarities = batch_cosine_similarity(query_embeddings[i], project_embeddings)

            for j, proj_name in enumerate(unique_projects):
                similarity = float(similarities[j])
                if similarity >= similarity_threshold:
                    matched_projects.add(proj_name)
                    logger.debug(
                        f"  '{query_name}' matched '{proj_name}' "
                        f"(similarity: {similarity:.3f})"
                    )

        logger.info(f"Found {len(matched_projects)} matching main_projects: {list(matched_projects)}")
        return list(matched_projects)

    def find_pages_by_title_similarity(
        self,
        search_terms: List[str],
        similarity_threshold: float = 0.6,
        max_depth: int = 8,
    ) -> List[Dict[str, Any]]:
        """
        Find pages with titles similar to the search terms, prioritizing by depth.

        Searches shallower pages first (depth 1, then 2, etc.) to find the
        "main" page for a topic, but continues through all depths up to max_depth.

        Args:
            search_terms: List of terms to search for in titles (e.g., project names).
            similarity_threshold: Minimum similarity score (0-1). Defaults to 0.6.
            max_depth: Maximum depth to search. Defaults to 8 (covers most content).

        Returns:
            List of matching page info dicts with: page_id, title, depth, similarity.
            Results are sorted by depth (ascending) then similarity (descending).
        """
        if not search_terms or len(self.documents) == 0:
            return []

        logger.info(f"Searching titles similar to: {search_terms} (threshold={similarity_threshold}, max_depth={max_depth})")

        # Get unique pages (by page_id) with their titles and depths
        page_info = {}
        for meta in self.metadatas:
            page_id = meta.get('page_id', '')
            if page_id and page_id not in page_info:
                page_info[page_id] = {
                    'page_id': page_id,
                    'title': meta.get('title', ''),
                    'depth': meta.get('depth', 99),
                    'main_project': meta.get('main_project', ''),
                    'children_ids': meta.get('children_ids', ''),
                }

        if not page_info:
            return []

        # Generate embeddings for search terms
        term_embeddings = self.embedding_manager.generate_embeddings(search_terms)

        # Get unique titles and their embeddings
        titles = [p['title'] for p in page_info.values() if p['title']]
        page_ids_by_title = {p['title']: p['page_id'] for p in page_info.values()}

        if not titles:
            return []

        title_embeddings = self.embedding_manager.generate_embeddings(titles)

        # Calculate all similarities using batch operations
        all_matches = []
        for i, term in enumerate(search_terms):
            # Use batch_cosine_similarity for vectorized computation
            similarities = batch_cosine_similarity(term_embeddings[i], title_embeddings)

            for j, title in enumerate(titles):
                page_id = page_ids_by_title[title]
                page = page_info[page_id]

                # Skip pages beyond max_depth
                if page['depth'] > max_depth:
                    continue

                similarity = float(similarities[j])

                if similarity >= similarity_threshold:
                    all_matches.append({
                        'page_id': page_id,
                        'title': title,
                        'depth': page['depth'],
                        'similarity': similarity,
                        'matched_term': term,
                        'main_project': page['main_project'],
                        'children_ids': page['children_ids'],
                    })

        # Sort by depth (ascending) then similarity (descending)
        # This prioritizes shallower pages but includes all matches
        all_matches.sort(key=lambda x: (x['depth'], -x['similarity']))

        # Remove duplicates (keep first occurrence = shallowest + highest similarity)
        seen_pages = set()
        unique_matches = []
        for match in all_matches:
            if match['page_id'] not in seen_pages:
                seen_pages.add(match['page_id'])
                unique_matches.append(match)

        if unique_matches:
            # Log matches by depth
            depth_counts = {}
            for m in unique_matches:
                d = m['depth']
                depth_counts[d] = depth_counts.get(d, 0) + 1
            logger.info(f"Title matches by depth: {depth_counts}")

        matched_titles = [m['title'] for m in unique_matches]
        logger.info(f"Total title matches found: {len(unique_matches)} - {matched_titles}")
        return unique_matches

    def get_descendant_page_ids(self, parent_page_id: str) -> List[str]:
        """
        Get all descendant page IDs for a given parent page.

        Recursively finds all children, grandchildren, etc.

        Args:
            parent_page_id: The page ID to find descendants for.

        Returns:
            List of all descendant page IDs (including the parent).
        """
        if len(self.documents) == 0:
            return [parent_page_id]

        # Build mappings of page_id -> children_ids and page_id -> title
        page_children = {}
        page_titles = {}
        for meta in self.metadatas:
            page_id = meta.get('page_id', '')
            if page_id:
                if page_id not in page_children:
                    children_str = meta.get('children_ids', '')
                    children_ids = [c.strip() for c in children_str.split(',') if c.strip()]
                    page_children[page_id] = children_ids
                if page_id not in page_titles:
                    page_titles[page_id] = meta.get('title', 'Unknown')

        # BFS to find all descendants
        descendants = set([parent_page_id])
        queue = [parent_page_id]

        while queue:
            current = queue.pop(0)
            children = page_children.get(current, [])
            for child_id in children:
                if child_id not in descendants:
                    descendants.add(child_id)
                    queue.append(child_id)

        parent_title = page_titles.get(parent_page_id, 'Unknown')
        logger.info(f"Found {len(descendants)} descendants for page '{parent_title}' (ID: {parent_page_id})")
        return list(descendants)

    def query_with_page_ids(
        self,
        query_text: str,
        page_ids: List[str],
        n_results: int = 5,
    ) -> Dict[str, Any]:
        """
        Query the vector store filtered to specific page IDs.

        Args:
            query_text: Query text to search for.
            page_ids: List of page IDs to include.
            n_results: Number of results to return.

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

        query_preview = query_text[:80] + "..." if len(query_text) > 80 else query_text
        logger.info(f"Querying '{query_preview}' with {len(page_ids)} page IDs filter")

        try:
            # Generate embedding for query
            query_embedding = self.embedding_manager.generate_embedding(query_text)

            # Calculate cosine similarities
            similarities = batch_cosine_similarity(query_embedding, self.embeddings)

            # Create mask for matching page IDs
            page_ids_set = set(str(pid) for pid in page_ids)
            mask = np.array([
                str(meta.get('page_id', '')) in page_ids_set
                for meta in self.metadatas
            ])

            matching_count = mask.sum()
            logger.info(f"Page ID filter matched {matching_count} documents")

            # Set similarity to -inf for non-matching documents
            filtered_similarities = np.where(mask, similarities, -np.inf)

            # Get top k indices
            top_k = min(n_results, len(self.documents))
            top_indices = np.argsort(filtered_similarities)[::-1][:top_k]

            # Filter out indices with -inf similarity
            valid_indices = [i for i in top_indices if filtered_similarities[i] != -np.inf]

            # Convert similarity to distance
            distances = [float(1 - similarities[i]) for i in valid_indices]

            result = {
                "documents": [self.documents[i] for i in valid_indices],
                "metadatas": [self.metadatas[i] for i in valid_indices],
                "distances": distances,
                "ids": [self.ids[i] for i in valid_indices],
            }

            top_titles = [m.get('title', 'Unknown')[:50] for m in result['metadatas'][:3]]
            logger.info(f"Found {len(result['documents'])} documents after page ID filtering, top results: {top_titles}")
            return result

        except Exception as e:
            logger.error(f"Error querying with page IDs: {str(e)}")
            raise
