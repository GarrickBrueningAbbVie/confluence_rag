# Confluence RAG Pipeline

A Retrieval Augmented Generation (RAG) system for answering questions about AbbVie Data Science & Analytics (DSA) projects using Confluence documentation and GitHub repositories.

## Overview

This project provides an intelligent question-answering system that:
- Retrieves project documentation from Confluence
- Preprocesses and enriches pages with metadata (parent project, completeness scores)
- Vectorizes and stores content in a searchable database
- Routes queries intelligently between semantic search (RAG) and structured queries (Database)
- Uses AbbVie's Iliad API to generate accurate, context-aware answers
- Provides an intuitive Streamlit UI with AbbVie branding

## Features

- **Confluence Integration**: Automatically fetches and parses project documentation from Confluence spaces
- **Intelligent Preprocessing**: Extracts metadata including parent projects and page completeness scores
- **Vector Search**: Uses ChromaDB and sentence transformers for efficient document retrieval
- **Database Pipeline**: Supports structured queries (counts, filters, aggregations) using pandas
- **Query Routing**: Automatically routes queries to RAG, Database, or Hybrid pipelines based on intent
- **RAG Pipeline**: Combines retrieval with Iliad API for accurate answer generation
- **Visualization**: Generate charts from query results using Plotly with native Streamlit integration
- **Smart Display**: Automatic table formatting for JSON/list results with scrolling support
- **Source Deduplication**: Intelligent grouping of sources by URL with document reference tracking
- **Modern UI**: Streamlit-based interface with AbbVie styling and pipeline mode selection
- **Extensible Architecture**: Well-structured, typed Python code following best practices
- **Advanced Query Processing**: NLTK-powered keyword extraction and lemmatization
- **Composite Re-ranking**: Multi-signal scoring using page hierarchy, keyword matching, and similarity metrics
- **Agent Framework**: Modular agent-based architecture for multi-step queries
- **Multi-Step Queries**: Support for complex queries requiring multiple steps with context passing
- **Feedback Loops**: Automatic query refinement when results are insufficient

## Project Structure

```
confluence_rag/
├── src/                              # Main source code
│   ├── data_pipeline.py              # End-to-end data pipeline script
│   ├── config.py                     # Configuration management
│   ├── confluence/                   # Confluence API integration
│   │   ├── rest_client.py            # REST API client for retrieving pages
│   │   ├── fetch_pages.py            # Standalone script to fetch Confluence pages
│   │   └── parser.py                 # HTML content parser
│   ├── iliad/                        # Iliad API client
│   │   ├── client.py                 # Unified Iliad API client
│   │   ├── recognize.py              # Text extraction (documents, OCR)
│   │   └── analyze.py                # Document analysis
│   ├── preprocessing/                # Data preprocessing
│   │   ├── processor.py              # Main preprocessing orchestrator
│   │   ├── attachment_fetcher.py     # Fetch and process attachments
│   │   ├── metadata_extractor.py     # Extract parent project, technologies
│   │   └── completeness_assessor.py  # Assess page completeness (0-100)
│   ├── database/                     # Database query pipeline
│   │   ├── dataframe_loader.py       # Load JSON into pandas DataFrame
│   │   ├── query_generator.py        # Natural language to pandas queries
│   │   ├── query_executor.py         # Safe pandas query execution
│   │   └── pipeline.py               # Database pipeline orchestrator
│   ├── routing/                      # Query routing
│   │   ├── intent_classifier.py      # Classify query intent (RAG/DB/Hybrid)
│   │   ├── query_router.py           # Route queries to pipelines
│   │   ├── smart_router.py           # LLM-based query decomposition and orchestrator
│   │   └── response_combiner.py      # Combine multi-pipeline responses
│   ├── agents/                       # Agent framework for multi-step queries
│   │   ├── base.py                   # BaseAgent, AgentContext, AgentResult
│   │   ├── rag_agent.py              # RAGAgent for semantic search
│   │   ├── database_agent.py         # DatabaseAgent for structured queries
│   │   ├── plotting_agent.py         # PlottingAgent for visualizations
│   │   ├── feedback_controller.py    # Manages feedback loops and refinement
│   │   └── orchestrator.py           # AgentOrchestrator coordinates agents
│   ├── prompts/                      # Prompt engineering
│   │   ├── prompt_splitter.py        # Split prompts into question + instructions
│   │   ├── templates.py              # Centralized prompt templates
│   │   └── few_shot_examples.py      # Few-shot examples for queries
│   ├── visualization/                # Chart generation
│   │   ├── chart_generator.py        # Generate Plotly charts
│   │   └── code_executor.py          # Safe code execution
│   ├── rag/                          # RAG pipeline components
│   │   ├── embeddings.py             # Embedding generation
│   │   ├── vectorstore.py            # Vector database management
│   │   ├── vectorize_data.py         # Standalone vectorization script
│   │   ├── pipeline.py               # Main RAG pipeline
│   │   ├── query_processor.py        # Query preprocessing
│   │   └── reranker.py               # Document re-ranker
│   └── ui/                           # Streamlit application
│       └── app.py                    # Main UI with pipeline mode selection
├── notebooks/                        # Jupyter notebooks
│   ├── 01_data_acquisition.ipynb     # Data collection and vectorization
│   └── 02_rag_queries.ipynb          # Query examples
├── tests/                            # Unit tests
├── config/                           # Configuration files
│   ├── flake8.cfg                    # Linting configuration
│   └── mypy.ini                      # Type checking configuration
├── Data_Storage/                     # Primary data storage (gitignored)
│   ├── confluence_pages.json         # Raw Confluence data
│   ├── confluence_pages_processed.json # Preprocessed data with metadata
│   └── vector_db/                    # Vector database files
├── requirements.txt                  # Python dependencies
├── Makefile                          # Development commands
├── .env.example                      # Environment variables template
└── README.md                         # This file
```

## Installation

### Prerequisites

- Python 3.9 or higher
- pip package manager
- Access to AbbVie's Iliad API
- Confluence API credentials

### Setup

1. **Clone the repository:**
   ```bash
   cd confluence_rag
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   make install
   # or
   pip install -r requirements.txt
   ```

4. **Configure environment variables:**
   ```bash
   cp .env.example .env
   # Edit .env with your credentials
   ```

   Required environment variables:
   ```
   # Iliad API
   ILIAD_API_KEY=your_iliad_api_key
   ILIAD_API_URL=https://your-iliad-api-url.com/endpoint
   ILIAD_DEFAULT_MODEL=gpt-5-mini-global

   # Confluence
   CONFLUENCE_URL=https://your-confluence-instance.atlassian.net
   CONFLUENCE_USERNAME=your_email@abbvie.com
   CONFLUENCE_API_TOKEN=your_confluence_api_token
   CONFLUENCE_SPACE_KEY=DSA

   # Optional: Enable database pipeline for structured queries
   ENABLE_DATABASE_PIPELINE=true
   ```

## Usage

### Quick Start: End-to-End Pipeline

The easiest way to set up the system is using the unified data pipeline:

```bash
python -m src.data_pipeline
```

This single command will:
1. Fetch all pages from Confluence
2. Preprocess pages (extract metadata, calculate completeness scores)
3. Generate embeddings and store in vector database

### Pipeline Options

```bash
# Full pipeline (fetch + preprocess + vectorize)
python -m src.data_pipeline

# Only fetch from Confluence
python -m src.data_pipeline --fetch-only

# Only preprocess existing data
python -m src.data_pipeline --preprocess-only

# Only vectorize existing data
python -m src.data_pipeline --vectorize-only

# Skip completeness scoring (faster)
python -m src.data_pipeline --skip-completeness

# Skip vectorization step
python -m src.data_pipeline --skip-vectorize

# Specify a different Confluence space
python -m src.data_pipeline --space-key MYSPACE
```

### Running the Streamlit App

Launch the web interface:

```bash
streamlit run src/ui/app.py
```

The app will be available at `http://localhost:8501`

**UI Features:**
- **Query Mode Selection**: Choose between Auto, RAG, Database, or Hybrid modes
- **Query Analysis**: See how your query was routed and what pandas query was generated
- **Source Documents**: View deduplicated sources with document reference mapping
- **Relevance Scores**: See similarity scores grouped by unique source page
- **Smart Tables**: JSON/list answers automatically displayed as scrollable tables
- **Chart Visualization**: Automatic chart generation for numeric data using Plotly
- **Cache Control**: Reload Pipelines button to clear cached components

### Query Types

The system supports different types of queries:

| Query Type | Examples | Pipeline Used |
|------------|----------|---------------|
| **Semantic** | "What is the RAG pipeline?", "Explain authentication" | RAG |
| **Structured** | "How many pages use Python?", "List all projects" | Database |
| **Hybrid** | "List projects and explain their purpose" | Both |
| **Visualization** | "Show me a chart of pages by author" | Database + Chart |
| **Multi-Step** | "What projects are similar to ALFA?" | Agent Orchestrator |

### Programmatic Usage

```python
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / 'src'))

from config import ConfigConfluenceRag
from rag.vectorstore import VectorStore
from rag.embeddings import EmbeddingManager
from rag.pipeline import RAGPipeline

# Initialize RAG pipeline
vector_store = VectorStore(persist_directory=ConfigConfluenceRag.VECTOR_DB_PATH)
embedding_manager = EmbeddingManager(model_name=ConfigConfluenceRag.EMBEDDING_MODEL)

pipeline = RAGPipeline(
    vector_store=vector_store,
    embedding_manager=embedding_manager,
    iliad_api_key=ConfigConfluenceRag.ILIAD_API_KEY,
    iliad_api_url=ConfigConfluenceRag.ILIAD_API_URL
)

# Ask a question
result = pipeline.query("What is the customer segmentation project?")
print(result['answer'])
```

#### Using the Query Router (for intelligent routing)

```python
from iliad.client import IliadClient, IliadClientConfig
from database.pipeline import DatabasePipeline
from routing.query_router import QueryRouter

# Initialize components
iliad_client = IliadClient(IliadClientConfig.from_env())
db_pipeline = DatabasePipeline("Data_Storage/confluence_pages_processed.json", iliad_client)

# Create router
router = QueryRouter(
    rag_pipeline=pipeline,
    db_pipeline=db_pipeline,
    iliad_client=iliad_client
)

# Query is automatically routed to the right pipeline
result = router.route("How many pages use Airflow?")
print(result['answer'])
print(result['intent'])  # "database"
print(result['query'])   # Generated pandas query
```

#### Using Multi-Step Queries (Agent Orchestrator)

For complex queries that require multiple steps with context passing:

```python
from routing.smart_router import SmartQueryRouter

# Initialize smart router
smart_router = SmartQueryRouter(
    rag_pipeline=pipeline,
    db_pipeline=db_pipeline,
    iliad_client=iliad_client,
)

# Multi-step query: First summarizes ALFA, then finds similar projects
result = smart_router.route_multistep("What projects are similar to ALFA?")
print(result.answer)
print(result.metadata)  # Shows execution steps and agents used

# Standard smart routing with query decomposition
result = smart_router.route("Describe ALFA and how many pages mention it")
```

**Multi-Step Query Examples:**
- "What projects are similar to ALFA?" - Summarizes ALFA first, then searches for similar projects
- "Which authors work on Python projects and how many pages have they created?" - Gets authors first, then counts their pages
- "Compare the data pipelines of Project A and Project B" - Gets info on both projects, then synthesizes comparison

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `ILIAD_API_KEY` | Iliad API authentication key | Required |
| `ILIAD_API_URL` | Iliad API base URL | Required |
| `ILIAD_DEFAULT_MODEL` | Default model for LLM tasks | `gpt-5-mini-global` |
| `CONFLUENCE_URL` | Confluence instance URL | Required |
| `CONFLUENCE_USERNAME` | Confluence username | Required |
| `CONFLUENCE_API_TOKEN` | Confluence API token | Required |
| `CONFLUENCE_SPACE_KEY` | Default space to fetch | `DSA` |
| `VECTOR_DB_PATH` | Vector database location | `./Data_Storage/vector_db` |
| `EMBEDDING_MODEL` | Sentence transformer model | `all-MiniLM-L6-v2` |
| `CHUNK_SIZE` | Text chunk size | `1000` |
| `CHUNK_OVERLAP` | Chunk overlap | `200` |
| `TOP_K_RESULTS` | Results to retrieve | `5` |
| `ENABLE_DATABASE_PIPELINE` | Enable structured queries | `false` |

## Architecture

### Data Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         CONFLUENCE RAG SYSTEM                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                     DATA PIPELINE (data_pipeline.py)                  │   │
│  │  Confluence API → Preprocessing → Metadata Extraction → Vectorization │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                   ↓                                          │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                      QUERY ROUTING LAYER                              │   │
│  │                                                                        │   │
│  │    User Query → Intent Classifier → Query Router                      │   │
│  │                        ↓                ↓                             │   │
│  │               ┌────────────────┐ ┌─────────────────┐                  │   │
│  │               │  RAG Pipeline  │ │Database Pipeline│                  │   │
│  │               │  (semantic)    │ │   (pandas)      │                  │   │
│  │               └────────────────┘ └─────────────────┘                  │   │
│  │                        ↓                ↓                             │   │
│  │                   Response Combiner + Chart Generator                 │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Key Components

| Component | Purpose |
|-----------|---------|
| `data_pipeline.py` | End-to-end data acquisition and processing |
| `ConfluenceRestClient` | REST API integration with Confluence |
| `IliadClient` | Unified client for Iliad API (chat, analyze, recognize) |
| `PreprocessingPipeline` | Metadata extraction and completeness assessment |
| `VectorStore` | Vector database with numpy and pickle persistence |
| `DatabasePipeline` | Structured queries using pandas |
| `QueryRouter` | Intelligent query routing based on intent |
| `RAGPipeline` | Semantic search and answer generation |
| `ChartGenerator` | Visualization generation using Plotly |
| `display_answer` | Smart answer display with tables, charts, source deduplication |
| `AgentOrchestrator` | Coordinates multi-agent execution for complex queries |
| `RAGAgent` | Agent wrapper for semantic search |
| `DatabaseAgent` | Agent wrapper for structured queries |
| `FeedbackController` | Manages query refinement and feedback loops |

## Development

### Code Quality

```bash
# Format code with Black
make format

# Run linting checks
make check-lint

# Run type checking
make check-types

# Run tests
make test

# Clean temporary files
make clean
```

### Code Style Guidelines

- **Formatting**: Black with 100 character line length
- **Linting**: flake8 configuration in `config/flake8.cfg`
- **Type Hints**: Required for all functions and methods
- **Testing**: pytest with comprehensive coverage
- **Docstrings**: Google-style docstrings for all public APIs

## Troubleshooting

### Common Issues

**Issue**: Vector database is empty
```bash
python -m src.data_pipeline
```

**Issue**: Iliad API authentication fails
```bash
# Check your .env file has correct ILIAD_API_KEY and ILIAD_API_URL
```

**Issue**: Confluence connection errors
```bash
# Verify CONFLUENCE_URL, CONFLUENCE_USERNAME, and CONFLUENCE_API_TOKEN
```

**Issue**: Import errors
```bash
# Ensure virtual environment is activated
source .venv/bin/activate
pip install -r requirements.txt
```

**Issue**: Database pipeline not working
```bash
# Ensure ENABLE_DATABASE_PIPELINE=true in .env
# Ensure confluence_pages_processed.json exists
python -m src.data_pipeline --preprocess-only
```

**Issue**: Sources appear duplicated
- This is expected behavior - documents are chunked, and multiple chunks from the same page may be relevant
- The UI now deduplicates sources by URL and shows which document numbers reference each source
- Example: "📄 1. Page Title (Documents 1, 2, 4)" indicates chunks 1, 2, 4 are from the same page

**Issue**: Charts not displaying
```bash
# Ensure Plotly is installed
pip install plotly
# Charts use st.plotly_chart() for native Streamlit integration
```

**Issue**: Streamlit showing stale results
```bash
# Click "🔄 Reload Pipelines" button in sidebar to clear cache
# Or restart the Streamlit app
```

## Security

- Never commit `.env` file or credentials
- Use environment variables for all sensitive data
- API tokens should have minimal required permissions
- All data storage directories are protected by `.gitignore`

## License

Internal AbbVie project - not for external distribution.

## Support

For questions or issues:
- Review this README and documentation
- Check existing issues in project tracker
- Contact the Data Science & Analytics team

## Acknowledgments

- AbbVie Data Science & Analytics team
- Iliad API platform team
- Open-source libraries: ChromaDB, sentence-transformers, Streamlit, Plotly
