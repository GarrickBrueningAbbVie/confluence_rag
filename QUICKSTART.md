# Quick Start Guide

Get up and running with the Confluence RAG Pipeline in minutes.

## Prerequisites

- Python 3.9+
- Confluence API credentials
- Iliad API access
- (Optional) GitHub personal access token

## Installation (5 minutes)

1. **Navigate to project directory:**
   ```bash
   cd confluence_rag
   ```

2. **Create virtual environment:**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Windows: .venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment:**
   ```bash
   cp .env.example .env
   # Edit .env with your credentials
   ```

   Minimum required in `.env`:
   ```
   ILIAD_API_KEY=your_key_here
   ILIAD_API_URL=https://your-api-url.com
   CONFLUENCE_URL=https://your-instance.atlassian.net
   CONFLUENCE_USERNAME=your@email.com
   CONFLUENCE_API_TOKEN=your_token
   CONFLUENCE_SPACE_KEY=DSA
   ```

## First-Time Setup (10-15 minutes)

### Step 1: Collect and Vectorize Data

Run the data acquisition notebook:

```bash
jupyter notebook notebooks/01_data_acquisition.ipynb
```

Or use Python directly:

```python
from src.config import config
from src.confluence.client import ConfluenceClient
from src.confluence.parser import ConfluenceParser
from src.rag.vectorstore import VectorStore
from src.rag.embeddings import EmbeddingManager

# Validate config
config.validate()

# Get Confluence pages
client = ConfluenceClient(
    url=config.CONFLUENCE_URL,
    username=config.CONFLUENCE_USERNAME,
    api_token=config.CONFLUENCE_API_TOKEN,
    space_key=config.CONFLUENCE_SPACE_KEY
)
pages = client.get_all_pages_content()

# Parse and chunk
parser = ConfluenceParser()
parsed = parser.parse_pages(pages)

chunks = []
for page in parsed:
    text_chunks = parser.chunk_text(page['text'])
    for i, chunk in enumerate(text_chunks):
        chunks.append({
            'text': chunk,
            'metadata': {
                'title': page['title'],
                'url': page['url'],
                'source_type': 'confluence'
            }
        })

# Store in vector database
vector_store = VectorStore()
texts = [c['text'] for c in chunks]
metadatas = [c['metadata'] for c in chunks]
ids = [f"doc_{i}" for i in range(len(chunks))]

vector_store.add_documents(texts, metadatas, ids)
print(f"✅ Added {vector_store.count()} documents")
```

### Step 2: Launch the UI

```bash
streamlit run src/ui/app.py
```

Open your browser to `http://localhost:8501`

## Using the Application

### Web Interface

1. Enter your question in the text box
2. Click example questions for inspiration
3. View the answer and source documents
4. Click source links to view original documentation

### Programmatic Usage

```python
from src.config import config
from src.rag.vectorstore import VectorStore
from src.rag.embeddings import EmbeddingManager
from src.rag.pipeline import RAGPipeline

# Initialize
vector_store = VectorStore()
embedding_manager = EmbeddingManager()
pipeline = RAGPipeline(
    vector_store=vector_store,
    embedding_manager=embedding_manager,
    iliad_api_key=config.ILIAD_API_KEY,
    iliad_api_url=config.ILIAD_API_URL
)

# Ask question
result = pipeline.query("What is the customer segmentation project?")
print(result['answer'])

# View sources
for source in result['sources']:
    print(f"- {source['title']}: {source['url']}")
```

### Notebook Exploration

```bash
jupyter notebook notebooks/02_rag_queries.ipynb
```

## Troubleshooting

**"Vector database is empty"**
→ Run the data acquisition notebook first

**"Configuration validation failed"**
→ Check your `.env` file has all required variables

**"Module not found"**
→ Activate virtual environment: `source .venv/bin/activate`

**"Confluence authentication failed"**
→ Verify API token and username in `.env`

## Common Commands

```bash
# Format code
make format

# Run tests
make test

# Check code quality
make check-lint
make check-types

# Clean temporary files
make clean

# Install/update dependencies
make install
```

## Example Questions

Try asking:
- "What data science projects are documented?"
- "Which projects use machine learning?"
- "What is the purpose of [project name]?"
- "What GitHub repositories are available?"
- "What technologies are used in [project name]?"

## Next Steps

1. Explore the notebooks in `notebooks/`
2. Read the full documentation in `README.md`
3. Review code in `src/` directories
4. Check the change log in `claude_docs/`
5. Customize the Streamlit UI styling if needed

## Getting Help

- Review `README.md` for detailed documentation
- Check `claude_docs/INITIAL_SETUP.md` for architecture details
- Examine test files in `tests/` for usage examples
- Review example notebooks for working code

## Key Files

- `src/config.py` - Configuration management
- `src/rag/pipeline.py` - Main RAG logic
- `src/ui/app.py` - Streamlit interface
- `.env` - Your credentials (create from `.env.example`)
- `README.md` - Full documentation

## Performance Tips

- Start with `TOP_K_RESULTS=5` for balanced speed/accuracy
- Use smaller `CHUNK_SIZE` (500-1000) for more precise retrieval
- Increase `CHUNK_SIZE` (1500-2000) for more context per chunk
- Adjust `CHUNK_OVERLAP` to maintain context between chunks

Enjoy using the Confluence RAG Pipeline! 🚀
