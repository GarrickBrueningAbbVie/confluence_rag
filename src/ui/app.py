"""Streamlit application for Confluence RAG question answering."""

import sys
from pathlib import Path
import streamlit as st
from typing import Dict, Any, Optional
import json
from loguru import logger

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import ConfigConfluenceRag
from rag.vectorstore import VectorStore
from rag.embeddings import EmbeddingManager
from rag.pipeline import RAGPipeline

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
def initialize_rag_pipeline() -> RAGPipeline:
    """
    Initialize and cache the RAG pipeline components.

    Returns:
        Initialized RAGPipeline instance.
    """
    logger.info("Initializing RAG pipeline for Streamlit app")

    # Validate configuration
    if not ConfigConfluenceRag.validate():
        st.error("Configuration validation failed. Please check your .env file.")
        st.stop()

    # Initialize components
    vector_store = VectorStore(
        persist_directory=ConfigConfluenceRag.VECTOR_DB_PATH,
        collection_name="confluence_docs",
    )

    embedding_manager = EmbeddingManager(model_name=ConfigConfluenceRag.EMBEDDING_MODEL)

    pipeline = RAGPipeline(
        vector_store=vector_store,
        embedding_manager=embedding_manager,
        iliad_api_key=ConfigConfluenceRag.ILIAD_API_KEY,
        iliad_api_url=ConfigConfluenceRag.ILIAD_API_URL,
        top_k=10,  # Default to 10 results as per requirements
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

    # If answer is a dict (from Iliad API), extract the content
    if answer is None:
        answer_text = ""
    elif isinstance(answer, dict):
        if "completion" in answer and isinstance(answer["completion"], dict):
            answer_text = answer["completion"].get("content", "")
        else:
            answer_text = answer.get("content", str(answer))
    else:
        answer_text = str(answer)

    # Display the extracted answer text
    if answer_text and answer_text.strip() and answer_text != "None":
        st.markdown(answer_text)
    else:
        # Show what we got for debugging
        st.warning(f"No answer content received. Raw answer: {repr(answer)}")

    # Display sources
    if result.get("sources"):
        st.markdown("### Sources")
        for i, source in enumerate(result["sources"], 1):
            with st.expander(f"📄 {source.get('title', f'Source {i}')}"):
                if source.get("type"):
                    st.markdown(f"**Type:** {source['type']}")
                if source.get("url"):
                    st.markdown(f"**Link:** [{source['url']}]({source['url']})")

    # Display relevance scores in sidebar
    with st.sidebar:
        if result.get("distances"):
            st.markdown("### Relevance Scores")
            for i, distance in enumerate(result["distances"], 1):
                score = 1 - distance  # Convert distance to similarity
                st.progress(score, text=f"Document {i}: {score:.2%}")


def display_chart(result: Dict[str, Any]) -> None:
    """Display chart visualization if available."""
    if not result.get("metadata", {}).get("requires_visualization"):
        return

    chart_data = result.get("metadata", {}).get("chart_data")
    if not chart_data:
        return

    try:
        # Try to generate a quick chart
        if ADVANCED_FEATURES_AVAILABLE:
            iliad_config = IliadClientConfig.from_env()
            iliad_client = IliadClient(iliad_config)
            generator = ChartGenerator(iliad_client)

            chart_result = generator.generate_quick_chart(
                data=chart_data,
                chart_type="bar",
                title="Query Results",
            )

            if chart_result["success"] and chart_result.get("html"):
                st.markdown("### Visualization")
                st.components.v1.html(chart_result["html"], height=400)

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
        top_k = st.slider(
            "Number of documents to retrieve",
            min_value=1,
            max_value=20,
            value=10,
            help="Higher values provide more context but may be slower",
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

    # Initialize RAG pipeline
    try:
        rag_pipeline = initialize_rag_pipeline()
        rag_pipeline.top_k = top_k
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
    status_col1, status_col2 = st.columns(2)
    with status_col1:
        st.info(f"📚 Vector database: {doc_count} document chunks")
    with status_col2:
        if db_pipeline:
            stats = db_pipeline.get_stats()
            st.info(f"📊 Database: {stats['total_pages']} pages")
        elif ADVANCED_FEATURES_AVAILABLE:
            st.info("📊 Database pipeline: Not enabled")

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
                    display_answer(result)

                # Debug information
                with st.expander("🔧 Debug Information"):
                    debug_info = {
                        "question": question,
                        "pipeline_mode": pipeline_mode,
                        "num_sources": len(result.get("sources", [])),
                    }

                    if result.get("distances"):
                        debug_info["distances"] = result["distances"]

                    if result.get("query"):
                        debug_info["generated_query"] = result["query"]

                    if result.get("metadata"):
                        debug_info["routing_metadata"] = result["metadata"]

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
