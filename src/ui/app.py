"""Streamlit application for Confluence RAG question answering."""

import sys
from pathlib import Path
from typing import Dict, Any, Optional
import json
from loguru import logger

# Configure loguru BEFORE importing anything else that uses it
# This prevents "missing ScriptRunContext" warnings in Streamlit
logger.remove()  # Remove default stderr handler
logger.add(
    sys.stderr,
    format="{time:HH:mm:ss} | {level: <8} | {message}",
    level="INFO",
    enqueue=True,  # Thread-safe async logging
    backtrace=False,
)

import streamlit as st

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import ConfigConfluenceRag
from rag.vectorstore import VectorStore
from rag.embeddings import EmbeddingManager
from rag.pipeline import RAGPipeline
from rag.project_vectorstore import ProjectVectorStore

# Conditional imports for new features
try:
    from iliad.client import IliadClient, IliadClientConfig
    from database.pipeline import DatabasePipeline
    from routing.query_router import QueryRouter
    from routing.intent_classifier import QueryIntent
    from visualization.chart_generator import ChartGenerator

    ADVANCED_FEATURES_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Advanced features not available: {e}")
    ADVANCED_FEATURES_AVAILABLE = False


# AbbVie Color Palette
ABBVIE_COLORS = {
    "dark_blue": "#071D49",
    "white": "#FFFFFF",
    "medium_blue": "#A6B5E0",
    "light_blue": "#EDF0FF",
    "purple": "#8A2ECC",
    "cobalt": "#0066F5",
    "red": "#CF451C",
    "green": "#338700",
    "medium_gray": "#B9B4B4",
    "light_gray": "#E8E8E8",
}


def load_custom_css() -> None:
    """Load custom CSS for AbbVie styling with dark mode."""
    custom_css = f"""
    <style>
    /* Main app styling - Dark mode */
    .stApp {{
        background-color: #0E1117;
    }}

    /* Main content area */
    .main .block-container {{
        background-color: #0E1117;
    }}

    /* Header styling - Light colors for dark background */
    h1 {{
        color: {ABBVIE_COLORS['white']};
        font-family: 'Arial', sans-serif;
        font-weight: bold;
    }}

    h2, h3 {{
        color: {ABBVIE_COLORS['medium_blue']};
        font-family: 'Arial', sans-serif;
    }}

    /* Regular text */
    p, .stMarkdown, label {{
        color: {ABBVIE_COLORS['white']};
    }}

    /* Sidebar styling - Darker with AbbVie dark blue */
    [data-testid="stSidebar"] {{
        background-color: {ABBVIE_COLORS['dark_blue']};
    }}

    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {{
        color: {ABBVIE_COLORS['white']};
    }}

    [data-testid="stSidebar"] p, [data-testid="stSidebar"] label, [data-testid="stSidebar"] .stMarkdown {{
        color: {ABBVIE_COLORS['medium_blue']};
    }}

    /* Button styling */
    .stButton>button {{
        background-color: {ABBVIE_COLORS['purple']};
        color: {ABBVIE_COLORS['white']};
        border-radius: 5px;
        border: none;
        padding: 10px 24px;
        font-weight: bold;
    }}

    .stButton>button:hover {{
        background-color: {ABBVIE_COLORS['cobalt']};
    }}

    /* Text input styling */
    .stTextInput>div>div>input {{
        background-color: #262730;
        color: {ABBVIE_COLORS['white']};
        border-color: {ABBVIE_COLORS['medium_blue']};
    }}

    .stTextInput>div>div>input::placeholder {{
        color: {ABBVIE_COLORS['medium_gray']};
    }}

    /* Info/Warning/Error box styling */
    .stAlert {{
        background-color: rgba(7, 29, 73, 0.3);
        border-left: 4px solid {ABBVIE_COLORS['cobalt']};
        color: {ABBVIE_COLORS['white']};
    }}

    div[data-baseweb="notification"] {{
        background-color: rgba(7, 29, 73, 0.5);
        color: {ABBVIE_COLORS['white']};
    }}

    /* Source links */
    a {{
        color: {ABBVIE_COLORS['cobalt']};
        text-decoration: none;
    }}

    a:hover {{
        color: {ABBVIE_COLORS['purple']};
    }}

    /* Expander styling */
    .streamlit-expanderHeader {{
        background-color: rgba(7, 29, 73, 0.5);
        color: {ABBVIE_COLORS['white']};
        border: 1px solid {ABBVIE_COLORS['medium_blue']};
    }}

    .streamlit-expanderHeader:hover {{
        background-color: rgba(138, 46, 204, 0.3);
    }}

    /* Spinner text */
    .stSpinner > div {{
        color: {ABBVIE_COLORS['white']};
    }}

    /* Progress bar */
    .stProgress > div > div > div {{
        background-color: {ABBVIE_COLORS['purple']};
    }}

    /* Slider styling */
    .stSlider {{
        color: {ABBVIE_COLORS['white']};
    }}

    /* Code blocks */
    code {{
        background-color: #262730;
        color: {ABBVIE_COLORS['medium_blue']};
    }}

    pre {{
        background-color: #262730;
        border: 1px solid {ABBVIE_COLORS['dark_blue']};
    }}

    /* JSON viewer */
    .json-content {{
        background-color: #262730;
        color: {ABBVIE_COLORS['white']};
    }}

    /* Pipeline mode badges */
    .intent-badge {{
        display: inline-block;
        padding: 4px 12px;
        border-radius: 12px;
        font-size: 12px;
        font-weight: bold;
        margin-right: 8px;
    }}

    .intent-rag {{
        background-color: {ABBVIE_COLORS['cobalt']};
        color: white;
    }}

    .intent-database {{
        background-color: {ABBVIE_COLORS['green']};
        color: white;
    }}

    .intent-hybrid {{
        background-color: {ABBVIE_COLORS['purple']};
        color: white;
    }}

    .intent-chart {{
        background-color: {ABBVIE_COLORS['red']};
        color: white;
    }}

    /* Chart container */
    .chart-container {{
        background-color: #1E1E1E;
        border-radius: 8px;
        padding: 16px;
        margin: 16px 0;
    }}
    </style>
    """
    st.markdown(custom_css, unsafe_allow_html=True)


@st.cache_resource
def initialize_vector_store() -> VectorStore:
    """Initialize and cache the vector store."""
    logger.info("Initializing vector store")
    return VectorStore(
        persist_directory=ConfigConfluenceRag.VECTOR_DB_PATH,
        collection_name="confluence_docs",
    )


@st.cache_resource
def initialize_project_store() -> Optional[ProjectVectorStore]:
    """Initialize and cache the project vector store for two-stage RAG."""
    logger.info("Initializing project vector store")
    try:
        project_store = ProjectVectorStore(
            persist_directory=ConfigConfluenceRag.PROJECT_VECTOR_DB_PATH,
            embedding_model=ConfigConfluenceRag.EMBEDDING_MODEL,
        )
        if project_store.count() > 0:
            logger.info(f"Project store loaded with {project_store.count()} projects")
            return project_store
        else:
            logger.warning("Project store is empty - two-stage RAG not available")
            return None
    except Exception as e:
        logger.warning(f"Could not initialize project store: {e}")
        return None


@st.cache_resource
def initialize_embedding_manager() -> EmbeddingManager:
    """Initialize and cache the embedding manager."""
    return EmbeddingManager(model_name=ConfigConfluenceRag.EMBEDDING_MODEL)


def create_rag_pipeline(
    vector_store: VectorStore,
    embedding_manager: EmbeddingManager,
    project_store: Optional[ProjectVectorStore] = None,
    enable_two_stage: bool = True,
    enable_reranking: bool = True,
    project_top_k: int = 3,
    top_k: int = 10,
) -> RAGPipeline:
    """
    Create a RAG pipeline with configurable parameters.

    Args:
        vector_store: The vector store instance.
        embedding_manager: The embedding manager instance.
        project_store: Optional project vector store for two-stage RAG.
        enable_two_stage: Whether to enable two-stage retrieval.
        enable_reranking: Whether to enable document re-ranking.
        project_top_k: Number of projects to identify in stage 1.
        top_k: Number of documents to retrieve.

    Returns:
        Configured RAGPipeline instance.
    """
    # Validate configuration
    if not ConfigConfluenceRag.validate():
        st.error("Configuration validation failed. Please check your .env file.")
        st.stop()

    pipeline = RAGPipeline(
        vector_store=vector_store,
        embedding_manager=embedding_manager,
        iliad_api_key=ConfigConfluenceRag.ILIAD_API_KEY,
        iliad_api_url=ConfigConfluenceRag.ILIAD_API_URL,
        top_k=top_k,
        use_reranking=enable_reranking,
        project_store=project_store if enable_two_stage else None,
        enable_two_stage_rag=enable_two_stage,
        project_retrieval_top_k=project_top_k,
    )

    return pipeline


@st.cache_resource
def initialize_rag_pipeline() -> RAGPipeline:
    """
    Initialize and cache the default RAG pipeline components.

    Returns:
        Initialized RAGPipeline instance.
    """
    logger.info("Initializing default RAG pipeline for Streamlit app")

    vector_store = initialize_vector_store()
    embedding_manager = initialize_embedding_manager()
    project_store = initialize_project_store()

    pipeline = create_rag_pipeline(
        vector_store=vector_store,
        embedding_manager=embedding_manager,
        project_store=project_store,
        enable_two_stage=ConfigConfluenceRag.ENABLE_TWO_STAGE_RAG,
        enable_reranking=True,
        project_top_k=ConfigConfluenceRag.PROJECT_RETRIEVAL_TOP_K,
        top_k=10,
    )

    logger.info("RAG pipeline initialized successfully")
    return pipeline


@st.cache_resource
def initialize_database_pipeline() -> Optional["DatabasePipeline"]:
    """
    Initialize and cache the Database pipeline.

    Returns:
        DatabasePipeline instance or None if not available.
    """
    if not ADVANCED_FEATURES_AVAILABLE:
        return None

    if not ConfigConfluenceRag.ENABLE_DATABASE_PIPELINE:
        return None

    try:
        # Check if processed JSON exists
        json_path = Path("Data_Storage/confluence_pages_processed.json")
        if not json_path.exists():
            json_path = Path("Data_Storage/confluence_pages.json")

        if not json_path.exists():
            logger.warning("No JSON data found for database pipeline")
            return None

        # Initialize Iliad client
        iliad_config = IliadClientConfig.from_env()
        iliad_client = IliadClient(iliad_config)

        pipeline = DatabasePipeline(
            json_path=str(json_path),
            iliad_client=iliad_client,
            model=ConfigConfluenceRag.ILIAD_DEFAULT_MODEL,
        )

        logger.info("Database pipeline initialized successfully")
        return pipeline

    except Exception as e:
        logger.error(f"Failed to initialize database pipeline: {e}")
        return None


@st.cache_resource
def initialize_query_router(
    _rag_pipeline: RAGPipeline,
    _db_pipeline: Optional["DatabasePipeline"],
) -> Optional["QueryRouter"]:
    """
    Initialize the query router.

    Args:
        _rag_pipeline: RAG pipeline instance
        _db_pipeline: Database pipeline instance

    Returns:
        QueryRouter instance or None
    """
    if not ADVANCED_FEATURES_AVAILABLE:
        return None

    try:
        iliad_config = IliadClientConfig.from_env()
        iliad_client = IliadClient(iliad_config)

        router = QueryRouter(
            rag_pipeline=_rag_pipeline,
            db_pipeline=_db_pipeline,
            iliad_client=iliad_client,
            use_llm_fallback=False,  # Use rule-based for speed
        )

        logger.info("Query router initialized successfully")
        return router

    except Exception as e:
        logger.error(f"Failed to initialize query router: {e}")
        return None


def get_intent_badge(intent: str) -> str:
    """Get HTML badge for intent type."""
    badge_classes = {
        "rag": "intent-rag",
        "database": "intent-database",
        "hybrid": "intent-hybrid",
        "chart": "intent-chart",
    }

    css_class = badge_classes.get(intent, "intent-rag")
    label = intent.upper()

    return f'<span class="intent-badge {css_class}">{label}</span>'


def display_answer_as_table(answer: Any) -> bool:
    """
    Try to display answer as a table if it's tabular data.

    Args:
        answer: The answer data to display

    Returns:
        True if displayed as table, False otherwise
    """
    import pandas as pd

    try:
        # List of dicts -> table
        if isinstance(answer, list) and len(answer) > 0 and isinstance(answer[0], dict):
            df = pd.DataFrame(answer)
            row_count = len(df)

            if row_count > 10:
                # Scrollable table for large datasets
                st.dataframe(df, use_container_width=True, height=400)
            else:
                # Regular table for small datasets
                st.dataframe(df, use_container_width=True)

            st.caption(f"Showing {row_count} results")
            return True

        # Dict with consistent structure -> table
        if isinstance(answer, dict) and len(answer) > 0:
            # Check if it's a key-value mapping (like counts)
            values = list(answer.values())

            # If values are simple types, show as key-value table
            if all(isinstance(v, (int, float, str, bool, type(None))) for v in values):
                df = pd.DataFrame(
                    list(answer.items()),
                    columns=["Key", "Value"]
                )
                row_count = len(df)

                if row_count > 10:
                    st.dataframe(df, use_container_width=True, height=400)
                else:
                    st.dataframe(df, use_container_width=True)

                st.caption(f"Showing {row_count} results")
                return True

        return False

    except Exception as e:
        logger.debug(f"Could not display as table: {e}")
        return False


def display_answer(result: Dict[str, Any], show_routing: bool = False) -> None:
    """
    Display the answer and sources in formatted sections.

    Args:
        result: Result dictionary from RAG pipeline.
        show_routing: Whether to show routing information.
    """
    # Display intent badge if available
    if show_routing and result.get("intent"):
        intent_html = get_intent_badge(result["intent"])
        st.markdown(intent_html, unsafe_allow_html=True)

    # Display answer
    st.markdown("### Answer")

    # Extract content from answer - handle both string and dict responses
    answer = result.get("answer")

    # Debug: Log the answer type and value
    logger.debug(f"Answer type: {type(answer)}, value: {answer}")

    # Check if answer is tabular data (from database queries)
    # Try to display as table first
    if isinstance(answer, (list, dict)) and not isinstance(answer, str):
        # Skip if it's an Iliad API response dict (has 'completion' or 'content' key)
        is_api_response = (
            isinstance(answer, dict) and
            ("completion" in answer or "content" in answer)
        )

        if not is_api_response and display_answer_as_table(answer):
            # Successfully displayed as table
            pass
        elif is_api_response:
            # Extract text from API response
            if "completion" in answer and isinstance(answer["completion"], dict):
                answer_text = answer["completion"].get("content", "")
            else:
                answer_text = answer.get("content", str(answer))

            if answer_text and answer_text.strip():
                st.markdown(answer_text)
        else:
            # Fallback to JSON display
            st.json(answer)
    elif answer is None:
        st.warning("No answer received.")
    elif isinstance(answer, (int, float)):
        # Scalar numeric answer
        st.markdown(f"**{answer}**")
    else:
        # String or other type
        answer_text = str(answer)
        if answer_text and answer_text.strip() and answer_text != "None":
            st.markdown(answer_text)
        else:
            st.warning(f"No answer content received.")

    # Display sources (deduplicated by URL, showing which documents reference each)
    if result.get("sources"):
        st.markdown("### Sources")

        sources = result["sources"]
        logger.debug(f"Displaying {len(sources)} sources")

        # Group sources by URL to deduplicate (same page can have multiple chunks)
        unique_sources = {}
        for i, source in enumerate(sources, 1):
            doc_idx = source.get("document_index", i)
            url = source.get("url", "")
            title = source.get("title", "Unknown")

            if url not in unique_sources:
                unique_sources[url] = {
                    "title": title,
                    "url": url,
                    "type": source.get("type", ""),
                    "document_indices": [],
                }
            unique_sources[url]["document_indices"].append(doc_idx)

        # Display unique sources with document reference info
        for idx, (url, source_info) in enumerate(unique_sources.items(), 1):
            title = source_info["title"]
            doc_indices = source_info["document_indices"]

            # Show which document numbers reference this source
            if len(doc_indices) == 1:
                doc_ref = f"(Document {doc_indices[0]})"
            else:
                doc_ref = f"(Documents {', '.join(map(str, doc_indices))})"

            expander_label = f"📄 {idx}. {title} {doc_ref}"

            with st.expander(expander_label):
                if source_info.get("type"):
                    st.markdown(f"**Type:** {source_info['type']}")
                if source_info.get("url"):
                    st.markdown(f"**Link:** [{source_info['url']}]({source_info['url']})")
                if len(doc_indices) > 1:
                    st.markdown(f"**Note:** This page appears in {len(doc_indices)} retrieved chunks")

    # Display relevance scores in sidebar (grouped by unique source)
    with st.sidebar:
        if result.get("distances") and result.get("sources"):
            st.markdown("### Relevance Scores")

            # Group distances by URL (same as sources display)
            sources = result["sources"]
            distances = result["distances"]
            url_scores = {}

            for i, (source, distance) in enumerate(zip(sources, distances)):
                url = source.get("url", f"source_{i}")
                title = source.get("title", "Unknown")
                score = 1 - distance

                if url not in url_scores:
                    url_scores[url] = {"title": title, "best_score": score, "count": 0}
                url_scores[url]["count"] += 1
                # Keep best (highest) score for the page
                if score > url_scores[url]["best_score"]:
                    url_scores[url]["best_score"] = score

            # Display grouped scores
            for idx, (url, info) in enumerate(url_scores.items(), 1):
                title_short = info["title"][:30] + "..." if len(info["title"]) > 30 else info["title"]
                chunk_info = f" ({info['count']} chunks)" if info["count"] > 1 else ""
                st.progress(info["best_score"], text=f"{idx}. {title_short}{chunk_info}: {info['best_score']:.1%}")


def is_chartable_data(data: Any) -> bool:
    """Check if data can be visualized as a chart."""
    if data is None:
        return False

    # Dict with string keys and numeric values (e.g., counts, aggregations)
    if isinstance(data, dict):
        if len(data) == 0:
            return False
        # Check if values are numeric
        values = list(data.values())
        if all(isinstance(v, (int, float)) for v in values):
            return True
        return False

    # List of dicts with consistent structure
    if isinstance(data, list) and len(data) > 0:
        if isinstance(data[0], dict) and len(data) <= 50:
            # Check if has numeric columns
            keys = set(data[0].keys())
            for item in data:
                if isinstance(item, dict):
                    for v in item.values():
                        if isinstance(v, (int, float)):
                            return True
        return False

    return False


def display_chart(result: Dict[str, Any]) -> None:
    """Display chart visualization if available."""
    if not ADVANCED_FEATURES_AVAILABLE:
        return

    # Get chart data from metadata or from the answer itself
    chart_data = result.get("metadata", {}).get("chart_data")
    requires_viz = result.get("metadata", {}).get("requires_visualization", False)

    # If no explicit chart data, check if the answer itself is chartable
    if not chart_data and not requires_viz:
        answer = result.get("answer")
        if is_chartable_data(answer):
            chart_data = answer
            requires_viz = True

    if not chart_data or not requires_viz:
        return

    try:
        iliad_config = IliadClientConfig.from_env()
        iliad_client = IliadClient(iliad_config)
        generator = ChartGenerator(iliad_client)

        # Determine best chart type based on data
        chart_type = "bar"
        if isinstance(chart_data, dict):
            if len(chart_data) <= 6:
                chart_type = "pie"

        chart_result = generator.generate_quick_chart(
            data=chart_data,
            chart_type=chart_type,
            title="Query Results",
        )

        if chart_result["success"] and chart_result.get("figure"):
            st.markdown("### Visualization")
            # Use native Streamlit Plotly support for better rendering
            st.plotly_chart(chart_result["figure"], use_container_width=True)
        elif chart_result.get("error"):
            logger.warning(f"Chart generation failed: {chart_result['error']}")

    except Exception as e:
        logger.warning(f"Failed to generate chart: {e}")


def display_routing_info(result: Dict[str, Any]) -> None:
    """Display routing information in an expander."""
    metadata = result.get("metadata", {})

    if not metadata:
        return

    with st.expander("🔀 Query Analysis"):
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Detected Intent:**")
            intent = metadata.get("intent", "unknown")
            st.markdown(get_intent_badge(intent), unsafe_allow_html=True)

            st.markdown(f"**Confidence:** {metadata.get('confidence', 0):.0%}")

        with col2:
            st.markdown("**Reasoning:**")
            st.markdown(metadata.get("reasoning", "N/A"))

        # Show generated query if available
        if result.get("query"):
            st.markdown("---")
            st.markdown("**Generated Query:**")
            st.code(result["query"], language="python")


def main() -> None:
    """Main Streamlit application function."""
    # Page configuration with dark mode
    st.set_page_config(
        page_title="AbbVie DSA Project Assistant",
        page_icon="🔬",
        layout="wide",
        initial_sidebar_state="expanded",
        menu_items={
            'About': "AbbVie DSA Project Assistant - RAG-powered question answering"
        }
    )

    # Load custom CSS
    load_custom_css()

    # Header
    st.title("🔬 AbbVie DSA Project Assistant")
    st.markdown(
        "Ask questions about Data Science & Analytics projects using Confluence documentation."
    )

    # Sidebar
    with st.sidebar:
        st.markdown("## About")
        st.markdown(
            """
        This application uses Retrieval Augmented Generation (RAG) to answer
        questions about AbbVie's Data Science & Analytics projects.

        **Data Sources:**
        - Confluence project documentation

        **Powered by:**
        - AbbVie Iliad API
        - ChromaDB vector database
        - Sentence Transformers
        """
        )

        st.markdown("---")

        # Settings
        st.markdown("## Settings")

        # Clear cache button for development/debugging
        if st.button("🔄 Reload Pipelines", help="Clear cached pipelines and reload"):
            st.cache_resource.clear()
            st.rerun()

        top_k = st.slider(
            "Number of documents to retrieve",
            min_value=1,
            max_value=20,
            value=10,
            help="Higher values provide more context but may be slower",
        )

        # RAG Configuration Section
        st.markdown("---")
        st.markdown("## RAG Configuration")

        # Check if project store is available
        project_store = initialize_project_store()
        project_store_available = project_store is not None and project_store.count() > 0

        # Two-stage RAG toggle
        enable_two_stage = st.checkbox(
            "Enable Two-Stage RAG",
            value=project_store_available and ConfigConfluenceRag.ENABLE_TWO_STAGE_RAG,
            disabled=not project_store_available,
            help="First identifies relevant projects, then searches within them" if project_store_available
                 else "Not available - run data pipeline with conglomeration first",
        )

        # Project top-k slider (only show if two-stage enabled)
        project_top_k = 3
        if enable_two_stage and project_store_available:
            project_top_k = st.slider(
                "Projects to identify (Stage 1)",
                min_value=1,
                max_value=10,
                value=ConfigConfluenceRag.PROJECT_RETRIEVAL_TOP_K,
                help="Number of projects to identify before filtering chunks",
            )

        # Reranking toggle
        enable_reranking = st.checkbox(
            "Enable Re-ranking",
            value=True,
            help="Apply composite scoring to re-rank retrieved documents",
        )

        # Comparison mode toggle
        st.markdown("---")
        st.markdown("## Comparison Mode")
        comparison_mode = st.checkbox(
            "Enable Comparison Mode",
            value=False,
            help="Run query with different settings side-by-side",
        )

        if comparison_mode:
            st.markdown("**Compare settings:**")
            compare_two_stage = st.checkbox(
                "Compare: Two-Stage vs Standard",
                value=True,
                disabled=not project_store_available,
            )
            compare_reranking = st.checkbox(
                "Compare: With vs Without Re-ranking",
                value=False,
            )

        # Pipeline mode selection (if advanced features available)
        pipeline_mode = "auto"
        if ADVANCED_FEATURES_AVAILABLE:
            st.markdown("---")
            st.markdown("## Query Mode")
            pipeline_mode = st.selectbox(
                "Select pipeline mode:",
                options=["auto", "rag", "database", "hybrid"],
                index=0,
                help="Auto: Automatically detect query type. RAG: Semantic search. Database: Structured queries.",
            )

            st.markdown(
                """
            **Mode Descriptions:**
            - **Auto**: Automatically routes based on query
            - **RAG**: Semantic search for conceptual questions
            - **Database**: Structured queries (counts, lists)
            - **Hybrid**: Combined semantic + structured
            """
            )

    # Initialize base components
    try:
        vector_store = initialize_vector_store()
        embedding_manager = initialize_embedding_manager()
        # project_store already initialized in sidebar
    except Exception as e:
        st.error(f"Error initializing components: {str(e)}")
        logger.error(f"Failed to initialize components: {str(e)}")
        st.stop()

    # Create RAG pipeline with current settings
    try:
        rag_pipeline = create_rag_pipeline(
            vector_store=vector_store,
            embedding_manager=embedding_manager,
            project_store=project_store if enable_two_stage else None,
            enable_two_stage=enable_two_stage,
            enable_reranking=enable_reranking,
            project_top_k=project_top_k,
            top_k=top_k,
        )
    except Exception as e:
        st.error(f"Error initializing RAG pipeline: {str(e)}")
        logger.error(f"Failed to initialize pipeline: {str(e)}")
        st.stop()

    # Initialize optional components
    db_pipeline = None
    query_router = None

    if ADVANCED_FEATURES_AVAILABLE and ConfigConfluenceRag.ENABLE_DATABASE_PIPELINE:
        db_pipeline = initialize_database_pipeline()
        if db_pipeline:
            query_router = initialize_query_router(rag_pipeline, db_pipeline)

    # Check if vector store has documents
    doc_count = rag_pipeline.vector_store.count()
    if doc_count == 0:
        st.warning(
            """
        ⚠️ The vector database is empty. Please run the data acquisition notebook
        to populate it with Confluence data.
        """
        )
        st.stop()

    # Status indicators
    status_col1, status_col2, status_col3 = st.columns(3)
    with status_col1:
        st.info(f"📚 Chunks: {doc_count}")
    with status_col2:
        if project_store_available:
            st.success(f"🎯 Projects: {project_store.count()}")
        else:
            st.warning("🎯 Projects: Not available")
    with status_col3:
        if db_pipeline:
            stats = db_pipeline.get_stats()
            st.info(f"📊 Pages: {stats['total_pages']}")
        elif ADVANCED_FEATURES_AVAILABLE:
            st.info("📊 DB: Not enabled")

    # Main query interface
    st.markdown("---")
    st.markdown("## Ask a Question")

    # Query input
    question = st.text_input(
        "Enter your question:",
        placeholder="e.g., What is the purpose of the customer segmentation project?",
        help="Ask any question about DSA projects documented in Confluence",
    )

    # Example questions based on mode
    with st.expander("💡 Example Questions"):
        if pipeline_mode in ["auto", "rag"]:
            st.markdown("**Semantic Search (RAG):**")
            rag_examples = [
                "What data science projects are documented?",
                "Which projects use machine learning?",
                "Explain the purpose of the Code Doc Tool",
                "What technologies are used in DSA projects?",
            ]
            for example in rag_examples:
                if st.button(example, key=f"rag_{example}"):
                    question = example

        if pipeline_mode in ["auto", "database", "hybrid"] and db_pipeline:
            st.markdown("**Database Queries:**")
            db_examples = [
                "How many pages are there in total?",
                "Who has created the most pages?",
                "What projects have completeness score above 50?",
                "How many pages use Python?",
            ]
            for example in db_examples:
                if st.button(example, key=f"db_{example}"):
                    question = example

    # Process query
    if question:
        # Check if comparison mode is enabled
        if comparison_mode and (compare_two_stage or compare_reranking):
            st.markdown("---")
            st.markdown("## Comparison Results")

            # Build list of configurations to compare
            configs = []

            if compare_two_stage and project_store_available:
                configs.append({
                    "name": "Two-Stage RAG",
                    "two_stage": True,
                    "reranking": enable_reranking,
                })
                configs.append({
                    "name": "Standard RAG",
                    "two_stage": False,
                    "reranking": enable_reranking,
                })
            elif compare_reranking:
                configs.append({
                    "name": "With Re-ranking",
                    "two_stage": enable_two_stage,
                    "reranking": True,
                })
                configs.append({
                    "name": "Without Re-ranking",
                    "two_stage": enable_two_stage,
                    "reranking": False,
                })
            else:
                # Fallback to current settings
                configs.append({
                    "name": "Current Settings",
                    "two_stage": enable_two_stage,
                    "reranking": enable_reranking,
                })

            # Create columns for side-by-side comparison
            cols = st.columns(len(configs))

            for idx, (col, config) in enumerate(zip(cols, configs)):
                with col:
                    st.markdown(f"### {config['name']}")

                    # Show config details
                    config_details = []
                    if config["two_stage"]:
                        config_details.append(f"Two-Stage (top {project_top_k} projects)")
                    else:
                        config_details.append("Standard retrieval")
                    if config["reranking"]:
                        config_details.append("Re-ranking ON")
                    else:
                        config_details.append("Re-ranking OFF")
                    st.caption(" | ".join(config_details))

                    with st.spinner(f"Running {config['name']}..."):
                        try:
                            # Create pipeline with this config
                            test_pipeline = create_rag_pipeline(
                                vector_store=vector_store,
                                embedding_manager=embedding_manager,
                                project_store=project_store if config["two_stage"] else None,
                                enable_two_stage=config["two_stage"],
                                enable_reranking=config["reranking"],
                                project_top_k=project_top_k,
                                top_k=top_k,
                            )

                            # Run query
                            result = test_pipeline.query(question, n_results=top_k)

                            # Display answer (simplified for comparison)
                            st.markdown("**Answer:**")
                            answer = result.get("answer", "")
                            if len(answer) > 500:
                                st.markdown(answer[:500] + "...")
                            else:
                                st.markdown(answer)

                            # Show identified projects if two-stage
                            if config["two_stage"] and result.get("identified_projects"):
                                st.markdown("**Projects identified:**")
                                st.write(result["identified_projects"])

                            # Show top sources
                            if result.get("sources"):
                                st.markdown("**Top Sources:**")
                                for i, src in enumerate(result["sources"][:3], 1):
                                    st.markdown(f"{i}. {src.get('title', 'Unknown')[:40]}")

                            # Show distances
                            if result.get("distances"):
                                avg_dist = sum(result["distances"][:5]) / min(5, len(result["distances"]))
                                st.metric("Avg Distance (top 5)", f"{avg_dist:.3f}")

                        except Exception as e:
                            st.error(f"Error: {str(e)}")

        else:
            # Standard single-query mode
            with st.spinner("🔍 Searching for relevant information..."):
                try:
                    # Route based on selected mode
                    if query_router and pipeline_mode != "rag":
                        # Use query router
                        force_intent = None
                        if pipeline_mode == "database":
                            force_intent = QueryIntent.DATABASE
                        elif pipeline_mode == "hybrid":
                            force_intent = QueryIntent.HYBRID

                        result = query_router.route(
                            question,
                            force_intent=force_intent,
                            return_metadata=True,
                        )

                        st.markdown("---")

                        # Display routing info
                        display_routing_info(result)

                        # Display answer
                        display_answer(result, show_routing=True)

                        # Display chart if applicable
                        display_chart(result)

                    else:
                        # Use RAG pipeline directly
                        result = rag_pipeline.query(question, n_results=top_k)

                        st.markdown("---")

                        # Show current RAG config
                        config_info = []
                        if enable_two_stage:
                            config_info.append(f"Two-Stage RAG (top {project_top_k} projects)")
                            if result.get("identified_projects"):
                                config_info.append(f"Identified: {result['identified_projects']}")
                        else:
                            config_info.append("Standard RAG")
                        if enable_reranking:
                            config_info.append("Re-ranking enabled")

                        st.info(" | ".join(config_info))

                        display_answer(result)

                    # Debug information
                    with st.expander("🔧 Debug Information"):
                        debug_info = {
                            "question": question,
                            "pipeline_mode": pipeline_mode,
                            "rag_config": {
                                "two_stage_enabled": enable_two_stage,
                                "reranking_enabled": enable_reranking,
                                "project_top_k": project_top_k if enable_two_stage else "N/A",
                                "top_k": top_k,
                            },
                            "num_sources": len(result.get("sources", [])),
                        }

                        if result.get("identified_projects"):
                            debug_info["identified_projects"] = result["identified_projects"]

                        if result.get("distances"):
                            debug_info["distances"] = result["distances"]

                        if result.get("query"):
                            debug_info["generated_query"] = result["query"]

                        if result.get("metadata"):
                            debug_info["routing_metadata"] = result["metadata"]

                        # Show source structure for debugging
                        sources = result.get("sources", [])
                        if sources:
                            debug_info["sources_sample"] = [
                                {
                                    "has_document_index": "document_index" in s,
                                    "document_index": s.get("document_index"),
                                    "title": s.get("title", "")[:50],
                                    "main_project": s.get("main_project", ""),
                                    "keys": list(s.keys()),
                                }
                                for s in sources[:5]
                            ]

                        st.json(debug_info)

                except Exception as e:
                    st.error(f"Error processing query: {str(e)}")
                    logger.error(f"Query processing error: {str(e)}")

    # Footer
    st.markdown("---")
    st.markdown(
        """
    <div style='text-align: center; color: #B9B4B4;'>
        <small>AbbVie Data Science & Analytics | Powered by Iliad API</small>
    </div>
    """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
