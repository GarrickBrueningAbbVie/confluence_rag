# Confluence RAG Pipeline

A RAG system for answering questions about AbbVie DSA projects using Confluence documentation.

## Quick Start

### Prerequisites
- Python 3.9+
- Access to Iliad API and Confluence API credentials

### Setup

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Configure credentials
cp .env.example .env
# Edit .env with your API keys
```

### Run Data Pipeline

```bash
# Full pipeline: fetch → preprocess → vectorize
python -m src.data_pipeline

# Or run individual steps
python -m src.data_pipeline --fetch-only
python -m src.data_pipeline --preprocess-only
python -m src.data_pipeline --vectorize-only
```

### Launch UI

```bash
streamlit run src/ui/app.py
```

## Features

- **Smart Query Routing**: Automatically routes to RAG, Database, or Hybrid pipelines
- **Two-Stage RAG**: Project-filtered retrieval for improved accuracy
- **LLM Query Decomposition**: Splits complex queries into parallel sub-queries
- **Structured Queries**: Supports counts, filters, and aggregations via pandas
- **Chart Generation**: Auto-generates Plotly visualizations
- **Agent Framework**: Multi-step queries with context passing

## Query Types

| Type | Example | Pipeline |
|------|---------|----------|
| Semantic | "What is ALFA?" | RAG |
| Structured | "How many pages use Python?" | Database |
| Hybrid | "List projects and describe them" | Both |
| Multi-step | "What projects are similar to ALFA?" | Agent Orchestrator |

## Project Structure

```
src/
├── confluence/      # Confluence API integration
├── preprocessing/   # Data enrichment and metadata extraction
├── rag/             # RAG pipeline, embeddings, vector store
├── database/        # Structured query pipeline
├── routing/         # Query routing and decomposition
├── agents/          # Multi-step query agents
├── visualization/   # Chart generation
└── ui/              # Streamlit application
```

## Configuration

Required environment variables:

```bash
ILIAD_API_KEY=your_key
ILIAD_API_URL=your_url
CONFLUENCE_URL=your_confluence_url
CONFLUENCE_USERNAME=your_email
CONFLUENCE_API_TOKEN=your_token
CONFLUENCE_SPACE_KEY=DSA
```

See `.env.example` for all options.

## Usage Example

```python
from routing.query_router import QueryRouter

router = QueryRouter(rag_pipeline, db_pipeline, iliad_client)
result = router.route("How many pages mention Airflow?")
print(result['answer'])
```

## Development

```bash
make format      # Format with Black
make check-lint  # Run linting
make test        # Run tests
```

## Documentation

For detailed architecture, component descriptions, and flow diagrams, see [ARCHITECTURE.md](ARCHITECTURE.md).

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Empty vector DB | Run `python -m src.data_pipeline` |
| Auth failures | Check `.env` credentials |
| Import errors | Activate venv: `source .venv/bin/activate` |
| Stale results | Click "Reload Pipelines" in UI sidebar |

## License

Internal AbbVie project - not for external distribution.
