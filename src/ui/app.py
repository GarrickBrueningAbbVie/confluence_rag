"""Streamlit application for Confluence RAG question answering."""

import sys
from pathlib import Path
import streamlit as st
from typing import Dict, Any
import json
from loguru import logger

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import ConfigConfluenceRag
from rag.vectorstore import VectorStore
from rag.embeddings import EmbeddingManager
from rag.pipeline import RAGPipeline


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


def display_answer(result: Dict[str, Any]) -> None:
    """
    Display the answer and sources in formatted sections.

    Args:
        result: Result dictionary from RAG pipeline.
    """
    # Display answer
    st.markdown("### Answer")

    # Extract content from answer - handle both string and dict responses
    answer = result.get("answer", "")

    # If answer is a dict (from Iliad API), extract the content
    if isinstance(answer, dict):
        if "completion" in answer and isinstance(answer["completion"], dict):
            answer_text = answer["completion"].get("content", "")
        else:
            answer_text = answer.get("content", str(answer))
    else:
        answer_text = str(answer)

    # Display the extracted answer text
    if answer_text:
        st.markdown(answer_text)
    else:
        st.warning("No answer content received from the API.")

    # Display sources
    if result.get("sources"):
        st.markdown("### Sources")
        for i, source in enumerate(result["sources"], 1):
            with st.expander(f"📄 {source['title']}"):
                st.markdown(f"**Type:** {source['type']}")
                st.markdown(f"**Link:** [{source['url']}]({source['url']})")

    # Display relevance scores in sidebar
    with st.sidebar:
        if result.get("distances"):
            st.markdown("### Relevance Scores")
            for i, distance in enumerate(result["distances"], 1):
                score = 1 - distance  # Convert distance to similarity
                st.progress(score, text=f"Document {i}: {score:.2%}")


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

    # Initialize RAG pipeline
    try:
        pipeline = initialize_rag_pipeline()
        pipeline.top_k = top_k
    except Exception as e:
        st.error(f"Error initializing RAG pipeline: {str(e)}")
        logger.error(f"Failed to initialize pipeline: {str(e)}")
        st.stop()

    # Check if vector store has documents
    doc_count = pipeline.vector_store.count()
    if doc_count == 0:
        st.warning(
            """
        ⚠️ The vector database is empty. Please run the data acquisition notebook
        to populate it with Confluence data.
        """
        )
        st.stop()

    st.info(f"📚 Vector database contains {doc_count} document chunks")

    # Main query interface
    st.markdown("---")
    st.markdown("## Ask a Question")

    # Query input
    question = st.text_input(
        "Enter your question:",
        placeholder="e.g., What is the purpose of the customer segmentation project?",
        help="Ask any question about DSA projects documented in Confluence",
    )

    # Example questions
    with st.expander("💡 Example Questions"):
        examples = [
            "What data science projects are documented?",
            "Which projects use machine learning?",
            "What is the purpose of the [project name]?",
            "What technologies are used in [project name]?",
        ]
        for example in examples:
            if st.button(example, key=example):
                question = example

    # Process query
    if question:
        with st.spinner("🔍 Searching for relevant information..."):
            try:
                result = pipeline.query(question, n_results=top_k)

                st.markdown("---")
                display_answer(result)

                # Debug information (optional)
                with st.expander("🔧 Debug Information"):
                    st.json(
                        {
                            "question": result["question"],
                            "num_sources": len(result["sources"]),
                            "distances": result["distances"],
                        }
                    )

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
