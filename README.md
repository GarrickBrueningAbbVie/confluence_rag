# Confluence RAG Pipeline

A Retrieval Augmented Generation (RAG) system for answering questions about AbbVie Data Science & Analytics (DSA) projects using Confluence documentation and GitHub repositories.

## Overview

This project provides an intelligent question-answering system that:
- Retrieves project documentation from Confluence
- Vectorizes and stores content in a searchable database
- Uses AbbVie's Iliad API to generate accurate, context-aware answers
- Provides an intuitive Streamlit UI with AbbVie branding

## Features

- **Confluence Integration**: Automatically fetches and parses project documentation from Confluence spaces
- **Vector Search**: Uses ChromaDB and sentence transformers for efficient document retrieval
- **RAG Pipeline**: Combines retrieval with Iliad API for accurate answer generation
- **Modern UI**: Streamlit-based interface with AbbVie styling
- **Extensible Architecture**: Well-structured, typed Python code following best practices

## Project Structure

```
confluence_rag/
├── src/                          # Main source code
│   ├── confluence/              # Confluence API integration
│   │   ├── rest_client.py      # REST API client for retrieving pages
│   │   ├── fetch_pages.py      # Standalone script to fetch Confluence pages
│   │   ├── client.py           # Legacy API client (deprecated)
│   │   └── parser.py           # HTML content parser
│   ├── rag/                    # RAG pipeline components
│   │   ├── embeddings.py       # Embedding generation
│   │   ├── vectorstore.py      # Vector database management
│   │   ├── vectorize_data.py   # Standalone script to vectorize data
│   │   └── pipeline.py         # Main RAG pipeline
│   ├── ui/                     # Streamlit application
│   │   └── app.py              # Main UI application
│   └── config.py               # Configuration management
├── notebooks/                   # Jupyter notebooks
│   ├── 01_data_acquisition.ipynb  # Data collection and vectorization
│   ├── 02_rag_queries.ipynb       # Query examples
│   └── Data_Storage/           # Notebook-specific data storage (gitignored)
├── tests/                       # Unit tests
├── config/                      # Configuration files
│   ├── flake8.cfg              # Linting configuration
│   └── mypy.ini                # Type checking configuration
├── Data_Storage/               # Primary data storage (gitignored)
│   ├── confluence_pages.json   # Raw Confluence data (gitignored)
│   └── vector_db/              # Vector database files (gitignored)
├── claude_docs/                # Claude change documentation
├── requirements.txt            # Python dependencies
├── Makefile                    # Development commands
├── .env.example                # Environment variables template
└── README.md                   # This file
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
   ILIAD_API_KEY=your_iliad_api_key
   ILIAD_API_URL=https://your-iliad-api-url.com/endpoint
   CONFLUENCE_URL=https://your-confluence-instance.atlassian.net
   CONFLUENCE_USERNAME=your_email@abbvie.com
   CONFLUENCE_API_TOKEN=your_confluence_api_token
   CONFLUENCE_SPACE_KEY=DSA
   ```

## Usage

### 1. Data Acquisition

There are two ways to collect and vectorize data:

#### Option A: Using Standalone Scripts (Recommended for Production)

**Step 1: Fetch Confluence Pages**
```bash
python src/confluence/fetch_pages.py
```

This script will:
- Connect to Confluence using REST API
- Retrieve all pages from the configured space
- Save raw JSON data to `Data_Storage/confluence_pages.json`

**Step 2: Generate Embeddings and Store in Vector Database**
```bash
python src/rag/vectorize_data.py
```

This script will:
- Load the Confluence pages JSON
- Chunk documents for optimal retrieval
- Generate embeddings using sentence transformers
- Store vectors in `Data_Storage/vector_db/`

#### Option B: Using Jupyter Notebook (Recommended for Exploration)

```bash
jupyter notebook notebooks/01_data_acquisition.ipynb
```

This notebook provides an interactive way to:
- Connect to Confluence and retrieve all pages from the DSA space
- Save raw JSON data for backup
- Parse and clean the content
- Generate embeddings and store in ChromaDB
- Visualize the data and test queries

### 2. Running the Streamlit App

Launch the web interface:

```bash
streamlit run src/ui/app.py
```

The app will be available at `http://localhost:8501`

### 3. Querying via Notebook

For programmatic access and experimentation:

```bash
jupyter notebook notebooks/02_rag_queries.ipynb
```

### 4. Programmatic Usage

```python
import sys
from pathlib import Path

# Add src to path if running outside of package
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from config import config
from rag.vectorstore import VectorStore
from rag.embeddings import EmbeddingManager
from rag.pipeline import RAGPipeline

# Initialize components
vector_store = VectorStore(
    persist_directory=config.VECTOR_DB_PATH
)

embedding_manager = EmbeddingManager(
    model_name=config.EMBEDDING_MODEL
)

pipeline = RAGPipeline(
    vector_store=vector_store,
    embedding_manager=embedding_manager,
    iliad_api_key=config.ILIAD_API_KEY,
    iliad_api_url=config.ILIAD_API_URL
)

# Ask a question
result = pipeline.query("What is the customer segmentation project?")
print(result['answer'])
```

## Development

### Code Quality

The project follows strict code quality standards:

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

### Adding New Features

1. Create feature branch
2. Implement with type hints and docstrings
3. Add tests in `tests/` directory
4. Run code quality checks
5. Submit for review

## Configuration

### Vector Database Settings

Modify in `.env`:
```
VECTOR_DB_PATH=./Data_Storage/vector_db
EMBEDDING_MODEL=all-MiniLM-L6-v2
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
TOP_K_RESULTS=5
```

### Confluence Settings

```
CONFLUENCE_URL=https://your-instance.atlassian.net
CONFLUENCE_SPACE_KEY=DSA  # Change to your space
```

## Architecture

### Data Flow

1. **Acquisition**: Confluence pages are fetched via API
2. **Parsing**: HTML content is converted to structured text
3. **Chunking**: Documents are split into optimal-sized chunks
4. **Embedding**: Text chunks are converted to vectors using sentence transformers
5. **Storage**: Vectors are stored in ChromaDB with metadata
6. **Query**: User questions are embedded and matched against stored vectors
7. **Generation**: Retrieved context + question sent to Iliad API for answer generation

### Key Components

- **ConfluenceRestClient**: REST API integration with Confluence
- **ConfluenceClient**: Legacy API client (deprecated)
- **VectorStore**: Vector database with numpy and pickle persistence
- **EmbeddingManager**: Sentence transformer management
- **RAGPipeline**: Main orchestration of retrieval and generation
- **fetch_pages.py**: Standalone script for data acquisition
- **vectorize_data.py**: Standalone script for vectorization

## Troubleshooting

### Common Issues

**Issue**: Vector database is empty
```bash
# Option 1: Run standalone scripts
python src/confluence/fetch_pages.py
python src/rag/vectorize_data.py

# Option 2: Run the data acquisition notebook
jupyter notebook notebooks/01_data_acquisition.ipynb
```

**Issue**: Iliad API authentication fails
```bash
# Check your .env file has correct ILIAD_API_KEY and ILIAD_API_URL
# Verify credentials with your Iliad administrator
```

**Issue**: Confluence connection errors
```bash
# Verify CONFLUENCE_URL, CONFLUENCE_USERNAME, and CONFLUENCE_API_TOKEN
# Ensure API token has correct permissions
```

**Issue**: Import errors
```bash
# Ensure virtual environment is activated
source .venv/bin/activate
# Reinstall dependencies
pip install -r requirements.txt

# For standalone scripts, imports are handled automatically
# All scripts use: sys.path.insert(0, str(Path(__file__).parent.parent))
```

**Issue**: Data Storage directory missing
```bash
# The directory is created automatically when running scripts
# But you can create it manually if needed:
mkdir -p Data_Storage/vector_db
```

## Performance Optimization

- **Batch Processing**: Use batch embedding generation for large datasets
- **Chunk Size**: Adjust `CHUNK_SIZE` based on content characteristics
- **Top K**: Reduce `TOP_K_RESULTS` for faster responses
- **Model Selection**: Use smaller embedding models for speed vs accuracy tradeoff

## Security

- Never commit `.env` file or credentials
- Use environment variables for all sensitive data
- API tokens should have minimal required permissions
- Vector database and raw data are protected by `.gitignore`:
  - `Data_Storage/` - All data storage directories
  - `confluence_pages.json` - Raw Confluence exports
  - `vector_db/` - Vector database files
  - `*.pkl`, `*.npy` - Pickle and numpy array files
- Each data storage directory has its own `.gitignore` for extra protection

## Contributing

Follow the code guidelines in `claude_code_rules.md`:
- Use Black formatting
- Include type hints
- Write comprehensive tests
- Document all public APIs
- Follow existing patterns

## License

Internal AbbVie project - not for external distribution.

## Support

For questions or issues:
- Review this README and documentation
- Check existing issues in project tracker
- Contact the Data Science & Analytics team

## Changelog

See `claude_docs/` directory for detailed change logs.

## Acknowledgments

- AbbVie Data Science & Analytics team
- Iliad API platform team
- Open-source libraries: ChromaDB, sentence-transformers, Streamlit
