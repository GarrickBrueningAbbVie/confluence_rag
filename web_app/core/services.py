"""Pipeline integration service for Django web application.

This module provides a singleton service that manages pipeline instances
and query execution. It integrates with the existing RAG pipeline code
without modifications.

Example usage:
    from core.services import get_pipeline_service

    service = get_pipeline_service()
    result = service.execute_query("What is ALFA?")
"""

import sys
import time
import threading
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional
from dataclasses import dataclass, field
from functools import lru_cache

from loguru import logger

# Add src to path for pipeline imports
SRC_DIR = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(SRC_DIR))

# Import configuration
from config import ConfigConfluenceRag  # noqa: E402


@dataclass
class QueryResult:
    """Result of a query execution.

    Attributes:
        success: Whether the query executed successfully.
        answer: The generated answer.
        sources: List of source documents.
        intent: The detected query intent.
        query: Generated database query (if applicable).
        execution_time: Time taken in seconds.
        metadata: Additional metadata about the query.
        figures: List of chart figures (if applicable).
        tables: List of table data (if applicable).
        error: Error message if query failed.
    """

    success: bool = False
    answer: str = ""
    sources: List[Dict] = field(default_factory=list)
    intent: str = "pending"
    query: Optional[str] = None
    execution_time: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    figures: List[Dict] = field(default_factory=list)
    tables: List[Dict] = field(default_factory=list)
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary for JSON serialization.

        Returns:
            Dict containing all result fields.
        """
        return {
            "success": self.success,
            "answer": self.answer,
            "sources": self.sources,
            "intent": self.intent,
            "query": self.query,
            "execution_time": self.execution_time,
            "metadata": self.metadata,
            "figures": self.figures,
            "tables": self.tables,
            "error": self.error,
        }


class PipelineService:
    """Singleton service managing RAG pipeline instances.

    This service initializes and manages all pipeline components:
    - VectorStore for document retrieval
    - EmbeddingManager for query embedding
    - RAGPipeline for answer generation
    - DatabasePipeline for structured queries
    - QueryRouter for intelligent routing

    Thread-safe singleton pattern ensures a single instance across
    all Django requests.
    """

    _instance: Optional["PipelineService"] = None
    _lock = threading.Lock()
    _initialized = False

    def __new__(cls) -> "PipelineService":
        """Create or return the singleton instance.

        Returns:
            The singleton PipelineService instance.
        """
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        """Initialize the service (only runs once due to singleton)."""
        # Skip if already initialized
        if self._initialized:
            return

        self.vector_store = None
        self.embedding_manager = None
        self.project_store = None
        self.rag_pipeline = None
        self.db_pipeline = None
        self.query_router = None
        self.iliad_client = None
        self.chart_generator = None

        self._init_error: Optional[str] = None

    def initialize(self, force: bool = False) -> bool:
        """Initialize all pipeline components.

        Args:
            force: If True, reinitialize even if already initialized.

        Returns:
            True if initialization succeeded, False otherwise.
        """
        if self._initialized and not force:
            return True

        with self._lock:
            if self._initialized and not force:
                return True

            logger.info("Initializing pipeline service...")
            start_time = time.time()

            try:
                self._init_vector_store()
                self._init_embedding_manager()
                self._init_project_store()
                self._init_iliad_client()
                self._init_rag_pipeline()
                self._init_db_pipeline()
                self._init_query_router()
                self._init_chart_generator()

                self._initialized = True
                self._init_error = None

                elapsed = time.time() - start_time
                logger.info(f"Pipeline service initialized in {elapsed:.2f}s")
                return True

            except Exception as e:
                self._init_error = str(e)
                logger.error(f"Failed to initialize pipeline service: {e}")
                return False

    def _init_vector_store(self) -> None:
        """Initialize the vector store."""
        from rag.vectorstore import VectorStore

        self.vector_store = VectorStore(
            persist_directory=ConfigConfluenceRag.VECTOR_DB_PATH,
            collection_name="confluence_docs",
        )
        logger.debug(f"Vector store initialized with {self.vector_store.count()} documents")

    def _init_embedding_manager(self) -> None:
        """Initialize the embedding manager."""
        from rag.embeddings import EmbeddingManager

        self.embedding_manager = EmbeddingManager(
            model_name=ConfigConfluenceRag.EMBEDDING_MODEL
        )
        logger.debug(f"Embedding manager initialized with {ConfigConfluenceRag.EMBEDDING_MODEL}")

    def _init_project_store(self) -> None:
        """Initialize the project vector store for two-stage RAG."""
        if not ConfigConfluenceRag.ENABLE_TWO_STAGE_RAG:
            logger.debug("Two-stage RAG disabled, skipping project store")
            return

        try:
            from rag.project_vectorstore import ProjectVectorStore

            self.project_store = ProjectVectorStore(
                persist_directory=ConfigConfluenceRag.PROJECT_VECTOR_DB_PATH
            )
            logger.debug(f"Project store initialized with {self.project_store.count()} projects")
        except Exception as e:
            logger.warning(f"Failed to initialize project store: {e}")
            self.project_store = None

    def _init_iliad_client(self) -> None:
        """Initialize the Iliad API client."""
        try:
            from iliad.client import IliadClient, IliadClientConfig

            config = IliadClientConfig(
                api_key=ConfigConfluenceRag.ILIAD_API_KEY,
                base_url=ConfigConfluenceRag.ILIAD_API_URL,
            )
            self.iliad_client = IliadClient(config)
            logger.debug("Iliad client initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize Iliad client: {e}")
            self.iliad_client = None

    def _init_rag_pipeline(self) -> None:
        """Initialize the RAG pipeline."""
        from rag.pipeline import RAGPipeline

        self.rag_pipeline = RAGPipeline(
            vector_store=self.vector_store,
            embedding_manager=self.embedding_manager,
            iliad_api_key=ConfigConfluenceRag.ILIAD_API_KEY,
            iliad_api_url=ConfigConfluenceRag.ILIAD_API_URL,
            project_store=self.project_store,
            enable_two_stage_rag=ConfigConfluenceRag.ENABLE_TWO_STAGE_RAG,
        )
        logger.debug("RAG pipeline initialized")

    def _init_db_pipeline(self) -> None:
        """Initialize the database pipeline."""
        try:
            from database.pipeline import DatabasePipeline

            # Find JSON data path (same logic as Streamlit app)
            json_path = Path(SRC_DIR.parent / "Data_Storage/confluence_pages_processed.json")
            if not json_path.exists():
                json_path = Path(SRC_DIR.parent / "Data_Storage/confluence_pages.json")

            if not json_path.exists():
                logger.warning("No JSON data found for database pipeline")
                self.db_pipeline = None
                return

            if not self.iliad_client:
                logger.warning("Iliad client not available for database pipeline")
                self.db_pipeline = None
                return

            self.db_pipeline = DatabasePipeline(
                json_path=str(json_path),
                iliad_client=self.iliad_client,
                model=ConfigConfluenceRag.ILIAD_DEFAULT_MODEL,
            )
            logger.debug("Database pipeline initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize database pipeline: {e}")
            self.db_pipeline = None

    def _init_query_router(self) -> None:
        """Initialize the query router."""
        if not self.db_pipeline:
            logger.debug("Database pipeline not available, skipping query router")
            return

        try:
            from routing.query_router import QueryRouter

            self.query_router = QueryRouter(
                rag_pipeline=self.rag_pipeline,
                db_pipeline=self.db_pipeline,
                iliad_client=self.iliad_client,
            )
            logger.debug("Query router initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize query router: {e}")
            self.query_router = None

    def _init_chart_generator(self) -> None:
        """Initialize the chart generator."""
        try:
            from visualization.chart_generator import ChartGenerator

            if not self.iliad_client:
                logger.warning("Iliad client not available for chart generator")
                self.chart_generator = None
                return

            self.chart_generator = ChartGenerator(
                iliad_client=self.iliad_client,
                model=ConfigConfluenceRag.ILIAD_DEFAULT_MODEL,
            )
            logger.debug("Chart generator initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize chart generator: {e}")
            self.chart_generator = None

    def execute_query(
        self,
        query: str,
        mode: str = "auto",
        top_k: int = 10,
        progress_callback: Optional[Callable[[int, str], None]] = None,
    ) -> QueryResult:
        """Execute a query through the appropriate pipeline.

        Args:
            query: The query text.
            mode: Routing mode (auto, rag, database, hybrid, smart).
            top_k: Number of documents to retrieve.
            progress_callback: Optional callback for progress updates.
                Signature: callback(percent: int, step: str)

        Returns:
            QueryResult containing the answer and metadata.
        """
        if not self._initialized:
            if not self.initialize():
                return QueryResult(
                    success=False,
                    error=f"Pipeline not initialized: {self._init_error}",
                )

        start_time = time.time()

        # Define pipeline stages with percentages
        STAGES = {
            "analyzing": {"percent": 10, "name": "Analyzing"},
            "classifying": {"percent": 25, "name": "Classifying"},
            "retrieving": {"percent": 45, "name": "Retrieving"},
            "generating": {"percent": 70, "name": "Generating"},
            "aggregating": {"percent": 85, "name": "Aggregating"},
            "complete": {"percent": 100, "name": "Complete"},
        }

        def emit_progress(stage: str, description: str = "") -> None:
            if progress_callback:
                stage_info = STAGES.get(stage, {"percent": 0, "name": stage})
                progress_callback(stage_info["percent"], stage_info["name"])

        try:
            emit_progress("analyzing")

            # Route based on mode
            if mode == "rag" or not self.query_router:
                emit_progress("classifying")
                emit_progress("retrieving")
                result = self._execute_rag(query, top_k)
                emit_progress("generating")
            elif mode == "database" and self.db_pipeline:
                emit_progress("classifying")
                emit_progress("retrieving")
                result = self._execute_database(query)
                emit_progress("generating")
            elif mode == "smart" and self.query_router:
                result = self._execute_smart(query, top_k, emit_progress)
            else:
                # Auto mode with query router
                result = self._execute_routed(query, top_k, emit_progress)

            result.execution_time = time.time() - start_time

            emit_progress("complete")

            return result

        except Exception as e:
            logger.exception(f"Query execution failed: {e}")
            return QueryResult(
                success=False,
                error=str(e),
                execution_time=time.time() - start_time,
            )

    def _execute_rag(self, query: str, top_k: int) -> QueryResult:
        """Execute query through RAG pipeline.

        Args:
            query: The query text.
            top_k: Number of documents to retrieve.

        Returns:
            QueryResult from RAG pipeline.
        """
        result = self.rag_pipeline.query(query, n_results=top_k)

        return QueryResult(
            success=result.get("success", True),
            answer=result.get("answer", ""),
            sources=result.get("sources", []),
            intent="rag",
            metadata=result.get("metadata", {}),
        )

    def _execute_database(self, query: str) -> QueryResult:
        """Execute query through database pipeline.

        Args:
            query: The query text.

        Returns:
            QueryResult from database pipeline.
        """
        result = self.db_pipeline.query(query)

        return QueryResult(
            success=result.get("success", True),
            answer=result.get("answer", ""),
            query=result.get("query", ""),
            intent="database",
            metadata=result.get("metadata", {}),
        )

    def _execute_routed(
        self,
        query: str,
        top_k: int,
        emit_progress: Callable[[str, str], None],
    ) -> QueryResult:
        """Execute query through query router.

        Args:
            query: The query text.
            top_k: Number of documents to retrieve (used by internal pipelines).
            emit_progress: Progress callback function (stage, description).

        Returns:
            QueryResult from appropriate pipeline.
        """
        emit_progress("classifying")

        # Note: query_router.route() doesn't take top_k parameter directly
        # It uses the pipelines' default settings internally
        emit_progress("retrieving")
        result = self.query_router.route(query)

        emit_progress("generating")

        # Extract intent from result
        intent = result.get("intent", "unknown")

        # Extract figures directly from result (query_router puts them here)
        figures = result.get("figures", [])

        # Extract tables directly from result
        tables = result.get("tables", [])

        return QueryResult(
            success=result.get("success", True),
            answer=result.get("answer", ""),
            sources=result.get("sources", []),
            query=result.get("query"),
            intent=intent,
            metadata=result.get("metadata", {}),
            figures=figures,
            tables=tables,
        )

    def _execute_smart(
        self,
        query: str,
        top_k: int,
        emit_progress: Callable[[str, str], None],
    ) -> QueryResult:
        """Execute query through smart router with decomposition.

        Args:
            query: The query text.
            top_k: Number of documents to retrieve (used by internal pipelines).
            emit_progress: Progress callback function (stage, description).

        Returns:
            QueryResult from smart routing.
        """
        emit_progress("classifying")
        emit_progress("retrieving")

        # Use route_multistep for complex queries
        # Note: route_multistep doesn't take top_k parameter directly
        result = self.query_router.route_multistep(query)

        emit_progress("generating")
        emit_progress("aggregating")

        # SmartRouteResult has different attributes, convert to dict
        result_dict = {
            "success": getattr(result, "success", True),
            "answer": getattr(result, "answer", ""),
            "sources": getattr(result, "sources", []),
            "metadata": getattr(result, "metadata", {}),
        }

        return QueryResult(
            success=result_dict.get("success", True),
            answer=result_dict.get("answer", ""),
            sources=result_dict.get("sources", []),
            intent="smart",
            metadata=result_dict.get("metadata", {}),
        )

    def get_status(self) -> Dict[str, Any]:
        """Get the current status of pipeline components.

        Returns:
            Dictionary with status of each component.
        """
        return {
            "initialized": self._initialized,
            "init_error": self._init_error,
            "components": {
                "vector_store": {
                    "ready": self.vector_store is not None,
                    "document_count": (
                        self.vector_store.count() if self.vector_store else 0
                    ),
                },
                "embedding_manager": {
                    "ready": self.embedding_manager is not None,
                    "model": ConfigConfluenceRag.EMBEDDING_MODEL,
                },
                "project_store": {
                    "ready": self.project_store is not None,
                    "project_count": (
                        self.project_store.count() if self.project_store else 0
                    ),
                },
                "rag_pipeline": {
                    "ready": self.rag_pipeline is not None,
                },
                "db_pipeline": {
                    "ready": self.db_pipeline is not None,
                },
                "query_router": {
                    "ready": self.query_router is not None,
                },
                "chart_generator": {
                    "ready": self.chart_generator is not None,
                },
            },
        }

    def reload(self) -> bool:
        """Reload all pipeline components.

        Returns:
            True if reload succeeded, False otherwise.
        """
        logger.info("Reloading pipeline service...")
        self._initialized = False
        return self.initialize(force=True)


@lru_cache(maxsize=1)
def get_pipeline_service() -> PipelineService:
    """Get the singleton pipeline service instance.

    This function is cached to ensure the same instance is returned
    across all calls.

    Returns:
        The singleton PipelineService instance.
    """
    return PipelineService()
