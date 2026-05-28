# FYP: Chat UI + RAG Pipeline

A comprehensive competitive programming platform with a ChatGPT-style UI and a RAG (Retrieval-Augmented Generation) pipeline for building high-quality knowledge bases.

## Project Overview

This project has two main components:

1. **Chat UI** - A web-based ChatGPT-style interface powered by Python/FastAPI backend
2. **RAG Pipeline** - A scraping and indexing system for competitive programming knowledge

## 📚 RAG Pipeline: CP Knowledge Base

A scraping and RAG pipeline for building a high-quality competitive programming knowledge base to improve LLM performance on algorithmic tasks.

### Data Sources

| Source | What we scrape | Volume | Quality |
|--------|---------------|--------|---------|
| **Codeforces** | Problems + editorials + tags + ratings | ~500 problems | ⭐⭐⭐⭐ |
| **USACO** | Problems + USACO.guide editorials | ~200 problems | ⭐⭐⭐⭐⭐ |
| **AtCoder** | ABC/ARC/AGC problems + official editorials | ~300 problems | ⭐⭐⭐⭐ |
| **CP-Algorithms** | Full algorithm reference (180+ articles) | ~180 articles | ⭐⭐⭐⭐⭐ |
| **CPH Book** | 30 structured chapters, CC BY-NC-SA 4.0 | 30 chapters | ⭐⭐⭐⭐⭐ |

### RAG Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run everything (scrape all sources + build RAG chunks as JSONL)
python main.py

# Specific sources only
python main.py --sources codeforces,cp_algorithms

# Ingest into ChromaDB (local vector store)
python main.py --vector-db chroma

# Only re-chunk already-scraped data (no network requests)
python main.py --pipeline-only
```

### RAG Project Structure

```
FYP/
├── main.py                    # Orchestrator — runs scrapers + pipeline
├── requirements.txt           # All dependencies
├── backend.py                 # FastAPI backend for Chat UI
├── server.js                  # Node.js server (alternative)
├── public/                    # Frontend (JavaScript + HTML)
│   ├── index.html
│   ├── settings.html
│   └── ...
├── RAG/
│   ├── rag_pipeline.py        # Chunking + embedding + vector DB ingestion
│   ├── scrapers/
│   │   ├── codeforces.py      # CF API + HTML scraping
│   │   ├── usaco.py           # USACO.org + USACO.guide editorials
│   │   ├── atcoder.py         # AtCoder via kenkoooo API
│   │   ├── cp_algorithms.py   # CP-algorithms.com full crawl
│   │   └── cph_book.py        # CPH LaTeX source + PDF download
│   └── data/                  # Created at runtime
│       ├── codeforces/        # Raw JSON per problem
│       ├── usaco/
│       ├── atcoder/
│       ├── cp_algorithms/
│       ├── cph/
│       └── rag_chunks/
│           ├── chunks.jsonl   # All chunks, portable format
│           ├── stats.json     # Chunk statistics
│           └── chroma_db/     # ChromaDB files (if --vector-db chroma)
└── logs/                      # Pipeline execution logs
```

### Chunk Format

Each chunk in `chunks.jsonl`:

```json
{
  "chunk_id": "uuid",
  "source_id": "1234A",
  "source": "codeforces",
  "chunk_type": "editorial",
  "title": "CF 1234A Editorial (part 1)",
  "text": "The key insight is...",
  "metadata": {
    "problem_id": "1234A",
    "rating": 1800,
    "tags": ["dp", "graphs"],
    "editorial_url": "..."
  },
  "token_estimate": 312
}
```

### Chunking Strategy

- **Problems**: Statement split by paragraphs (~400 tokens), examples kept together
- **Editorials**: Semantic paragraphs (~512 tokens) with 64-token overlap
- **Theory articles**: Paragraph chunks (~600 tokens), code blocks as separate chunks
- **Book chapters**: ~700 token chunks preserving section boundaries

### Vector Database Options

#### JSONL (default — no setup needed)
Raw output. Compatible with any downstream tool.

#### ChromaDB (local, easiest)
```bash
pip install chromadb openai
export OPENAI_API_KEY=sk-...
python main.py --vector-db chroma
```

---

## 💬 Chat UI

A ChatGPT-style web interface for interacting with the RAG pipeline or standalone LLMs.

### Chat UI Quick Start

#### Option 1: Run with Python Backend
1. Install Python 3.11+ and create a virtual environment
2. Open a terminal in `d:\Github\FYP`
3. Install dependencies:
   ```bash
   python -m pip install -r requirements.txt
   ```
4. Run the backend server:
   ```bash
   python backend.py
   ```
5. Open `http://127.0.0.1:8000` in your browser

#### Option 2: Static Preview Only
Open `public/index.html` directly in your browser for a static demo without a backend.

### Chat UI Features

- **Settings panel** - Configure:
  - Model selection (defaults to `mock-chat:1.0`)
  - API keys for external models
  - TextGrad loop count (when enabled)
- **Mock chat mode** - Test the UI without API keys
- **RAG integration** - Connect to vector databases for contextual responses
- **TextGrad support** - Optional gradient-based LLM optimization

### Backend Details

- The Python backend serves the frontend from `public/` directory
- REST API exposed at `/api/chat`
- Environment variables configurable via `.env` file
- Supports both local (Ollama) and remote (OpenAI, etc.) LLM backends

---

## 🔧 Configuration

### Environment Variables (.env)

```bash
# LLM Configuration
LLM_MODEL=gpt-4                    # Default LLM model
OPENAI_API_KEY=sk-...             # OpenAI API key (if using OpenAI models)

# Vector Database
VECTOR_DB=chroma                   # Options: chroma, qdrant, jsonl
CHROMA_HOST=localhost             # ChromaDB server (if remote)
CHROMA_PORT=8000

# Scraping
CODEFORCES_API_TIMEOUT=30
USACO_API_TIMEOUT=30

# Server
SERVER_PORT=8000
SERVER_HOST=127.0.0.1
```

---

## 📦 Dependencies

All dependencies are organized in a single `requirements.txt` file, grouped by functionality:

- **Core Web Framework**: FastAPI, Uvicorn
- **HTTP & Networking**: httpx, requests, beautifulsoup4
- **Data Processing**: pandas, numpy, pyarrow, datasets
- **LLM & Text Processing**: openai, ollama, textgrad, tiktoken
- **Vector Databases**: chromadb, qdrant-client
- **PDF & File Handling**: pillow, diskcache, fsspec
- **Development & Testing**: pytest, pytest-asyncio
- **And more...**

Install all dependencies:
```bash
pip install -r requirements.txt
```

---

## 📝 Notes

- The Chat UI includes a `TextGrad loop count` setting that is passed to the backend when TextGrad is enabled
- If you choose a model other than `mock-chat:1.0`, you must provide appropriate API credentials
- RAG pipeline logs are saved in `logs/` directory with timestamps
- Vector database files are stored in `RAG/data/rag_chunks/` for persistence
- The project supports both local LLM inference (via Ollama) and cloud-based APIs

---

## 🚀 Getting Started

1. Clone/setup the repository
2. Install dependencies: `pip install -r requirements.txt`
3. For Chat UI only: `python backend.py` then open http://127.0.0.1:8000
4. For RAG pipeline: `python main.py` to start scraping and building the knowledge base
5. Configure API keys in settings panel (Chat UI) or `.env` file (backend)

---

## 📄 License

The CPH Book content is provided under CC BY-NC-SA 4.0 license. All other code and content follow the repository's license.
