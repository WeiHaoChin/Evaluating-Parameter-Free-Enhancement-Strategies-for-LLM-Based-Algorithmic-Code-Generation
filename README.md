# Competitive Programming LLM Evaluation Platform

This work-in-progress project evaluates language models on competitive-programming tasks. It combines an interactive browser UI, retrieval-augmented generation (RAG), optional TextGrad prompt refinement, and LiveCodeBench-based benchmarking.

## Current status

Implemented workflows include:

- Streaming chat through FastAPI WebSockets.
- Local Ollama models and configured remote model providers.
- RAG retrieval from a persistent local ChromaDB collection.
- A UI-triggered, end-to-end RAG build: scrape, save, clean, chunk, embed, index, and reload.
- Optional TextGrad refinement for chat and benchmark generations.
- LiveCodeBench dataset download, readiness checks, background execution, progress reporting, cancellation, and saved results.
- Docker Compose services for the Nginx frontend and FastAPI backend.

The RAG build is designed to work in a fresh container with no existing `backend/data`. Scraped source records and generated artifacts are created there during the build.

## Architecture

```text
Browser UI (Nginx or FastAPI static files)
    -> FastAPI HTTP/WebSocket endpoints
        -> RAG retrieval from local ChromaDB
        -> model generation
        -> optional TextGrad refinement
        -> benchmark evaluation and saved results
```

Chat requests follow this path:

```text
Browser -> /ws/chat -> optional Chroma retrieval -> model generation
                    -> optional TextGrad refinement -> streamed UI result
```

## Project structure

```text
FYP/
├── Dockerfile                    Python backend image
├── docker-compose.yaml           Frontend/backend development stack
├── requirements.txt              Python dependencies
├── clearDocker.bat               Local Docker cleanup helper
├── README.md
├── public/
│   ├── Dockerfile                Nginx frontend image
│   ├── nginx.conf                Static-site configuration
│   ├── index.html                Chat page
│   ├── app.js                    Chat state and WebSocket client
│   ├── settings.html             Model, RAG, and dataset settings page
│   ├── settings.js               Settings UI and RAG-build trigger
│   ├── settings-store.js         Browser settings persistence/validation
│   ├── benchmark.html            Benchmark dashboard
│   ├── benchmark.js              Benchmark controls and result rendering
│   ├── examples.html             Saved-conversation browser
│   ├── new-chat.html             New-chat entry page
│   ├── page-transitions.js       Shared navigation behavior
│   └── styles.css                Shared UI styling
└── backend/
    ├── main.py                   FastAPI application and background RAG build
    ├── schemas.py                API request and settings models
    ├── solver.py                 Generation, prompting, and evaluation flow
    ├── TextGrad.py               TextGrad optimization integration
    ├── llm_clients.py            Local and remote model clients
    ├── rag_handler.py            Chroma loading, querying, and formatting
    ├── config/
    │   ├── models.py             Model/provider registry
    │   └── generation.py         Shared generation limits
    ├── routes/
    │   └── benchmark.py          Benchmark HTTP endpoints
    ├── RAG/
    │   ├── main.py               Full scrape-to-index orchestrator
    │   ├── pipeline/
    │   │   └── rag_pipeline.py   Load, clean, chunk, embed, and ingest logic
    │   └── scrapers/
    │       ├── atcoder.py
    │       ├── codeforces.py
    │       ├── cp_algorithms.py
    │       ├── cph_book.py
    │       └── usaco.py
    ├── benchmark/
    │   ├── fetch_lcb.py          Local LiveCodeBench loading
    │   ├── prefetch_lcb.py       Dataset download/cache handling
    │   ├── runner.py             Background benchmark orchestration
    │   ├── metrics.py            Aggregate metrics
    │   └── logger.py             Checkpoint and result persistence
    ├── lcb_runner/
    │   ├── benchmarks/           LiveCodeBench problem schemas/loaders
    │   ├── evaluation/           Correctness and scoring utilities
    │   └── utils/                Execution and extraction helpers
    ├── tests/                    Backend and RAG regression tests
    └── data/                     Runtime-generated and downloaded data
        ├── atcoder/              Scraped AtCoder records
        ├── codeforces/           Scraped Codeforces records
        ├── cp_algorithms/        Scraped CP-Algorithms articles
        ├── cph/                  Extracted handbook content
        ├── usaco/                Scraped USACO records
        ├── hf_cache/             Hugging Face model/dataset cache
        └── rag_chunks/
            ├── chunks.jsonl      Processed retrieval chunks
            └── chroma_db/        Persistent local vector index
```

`backend/data` and `logs` are runtime locations and may be absent in a fresh checkout. The application creates the required data directories as their workflows run.

## Quick start with Docker

From the repository root:

```bash
docker compose up --build
```

Open:

- Frontend: <http://localhost:3000>
- Backend API documentation: <http://localhost:5050/docs>

The backend bind-mounts the repository at `/workspace`, so RAG source data and indexes written to `/workspace/backend/data` persist in the host repository. The Hugging Face cache has an additional mount at `backend/data/hf_cache`. Local Ollama access from the container uses `host.docker.internal:11434`.

## Where the RAG data comes from

The knowledge base is assembled from five public competitive-programming sources:

| Source | Content collected | Saved under |
|---|---|---|
| Codeforces | Problem statements, metadata, tags, ratings, and available editorial material | `backend/data/codeforces/` |
| USACO | Problems and associated solution/editorial content obtained by the USACO scraper | `backend/data/usaco/` |
| AtCoder | ABC, ARC, and AGC problem metadata, statements, examples, difficulty data, and available editorials | `backend/data/atcoder/` |
| CP-Algorithms | Algorithm-reference articles and their metadata | `backend/data/cp_algorithms/` |
| Competitive Programmer's Handbook | Extracted and structured handbook chapters | `backend/data/cph/` |

The source-specific scrapers are in `backend/RAG/scrapers/`. Their saved records are intermediate inputs, not retrieval results. The RAG pipeline subsequently normalizes and chunks those records into `backend/data/rag_chunks/chunks.jsonl`, then embeds them into the local Chroma collection used by chat and benchmarks.

Dataset size is not fixed. It depends on scraper limits, source availability, cached records, and whether a full or selected-source build is run. Upstream content retains its original ownership and licensing; see [Data and licensing](#data-and-licensing).

## Build the RAG knowledge base

### From the frontend

Open **Settings** and select **Build RAG knowledge base**. This starts the complete build, not only chunk generation. The frontend calls `POST /api/rag/build`, and the backend runs the equivalent of:

```bash
python -m backend.RAG.main \
  --data-root backend/data \
  --embedder local \
  --vector-db chroma
```

The pipeline performs these steps:

1. Scrapes Codeforces, USACO, AtCoder, CP-Algorithms, and the Competitive Programmer's Handbook.
2. Saves source records below `backend/data` in source-specific directories.
3. Loads and normalizes the saved records.
4. Cleans and semantically chunks their content.
5. Writes `backend/data/rag_chunks/chunks.jsonl`.
6. Embeds chunks with `BAAI/bge-small-en-v1.5`.
7. Replaces the `cp_rag` collection in `backend/data/rag_chunks/chroma_db`.
8. Reloads the completed index into the running backend.

The initial build needs internet access for scraping and for the first embedding-model download. It can take a substantial amount of time. Progress and recent subprocess output are exposed by `GET /api/rag/build/status`; only one build can run at once.

Existing scraper output is reused where each scraper supports cached records. Rebuilding replaces the vector collection, preventing duplicate indexed chunks.

### Command-line options

Run the same complete flow as the UI:

```bash
python -m backend.RAG.main --data-root backend/data --embedder local --vector-db chroma
```

Reprocess already-saved data without scraping:

```bash
python -m backend.RAG.main --data-root backend/data --pipeline-only --embedder local --vector-db chroma
```

Scrape and process selected sources:

```bash
python -m backend.RAG.main --data-root backend/data --sources codeforces,cp_algorithms --embedder local --vector-db chroma
```

Supported source names are `codeforces`, `usaco`, `atcoder`, `cp_algorithms`, and `cph`.

At startup, `backend/rag_handler.py` attempts to load the local Chroma index. If none exists, the application remains available without RAG retrieval until the build finishes.

## Local development

Python 3.11 is the container runtime. On Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
$env:PYTHONPATH = "$PWD\backend"
uvicorn backend.main:app --host 127.0.0.1 --port 5050 --reload
```

Open <http://127.0.0.1:5050/>. FastAPI serves both the API and the static frontend in this mode.

This host-native setup supports frontend development, chat, model access, and RAG. The LiveCodeBench evaluator itself uses POSIX APIs including `SIGALRM`, `signal.alarm`, and `resource` for execution limits. Consequently, full benchmark execution is not supported directly by Windows Python; run the backend in its Linux Docker container or under WSL. Native Linux can run it without Docker when all dependencies are installed, although isolation is still strongly recommended because benchmark evaluation executes model-generated code.

## Model configuration

Use Settings to select the initial model and optional TextGrad model. Locally detected Ollama models are included in the choices returned by `GET /api/defaults`. Remote models require an API key in the UI.

The underlying integrations also recognize provider environment variables such as:

```dotenv
GOOGLE_API_KEY=...
OPENAI_API_KEY=...
OLLAMA_API_KEY=...
QDRANT_API_KEY=...
```

Do not commit `.env`; it is ignored by Git.

## Benchmarking

The benchmark page runs the configured pipeline against a locally cached LiveCodeBench release:

> **Execution environment:** On Windows, run the backend through Docker Compose (or WSL). Host-native Windows is suitable for the UI, chat, and RAG, but not the POSIX-based LiveCodeBench code evaluator. Docker is recommended on every platform because generated solutions are executed during scoring.

1. Download the selected dataset from Settings.
2. Complete the RAG build and enable RAG.
3. Select working initial and TextGrad models and provide required API keys.
4. Review the readiness checks and start the benchmark.
5. Follow progress and inspect saved summaries and per-test-case records under `logs/`.

The runner reports baseline, RAG-only, TextGrad-only, and full RAG-plus-TextGrad views while reusing the relevant initial generations.

## Main endpoints

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/api/status` | Backend health |
| `GET` | `/api/defaults` | Settings and detected models |
| `GET` | `/api/ollama/status` | Local Ollama status |
| `POST` | `/api/chat` | Non-streaming chat |
| WebSocket | `/ws/chat` | Streaming chat and TextGrad events |
| `POST` | `/api/rag/build` | Run the full scrape-to-index RAG pipeline |
| `GET` | `/api/rag/build/status` | RAG build progress and chunk count |
| `POST` | `/benchmark/dataset/download` | Cache a LiveCodeBench release |
| `GET` | `/benchmark/dataset/status` | Dataset download state |
| `POST` | `/benchmark/readiness` | Validate benchmark prerequisites |
| `POST` | `/benchmark/run` | Start a benchmark |
| `POST` | `/benchmark/stop` | Request benchmark cancellation |
| `GET` | `/benchmark/status` | Benchmark progress |
| `GET` | `/benchmark/results` | Latest saved results |

## Tests

From the repository root:

```powershell
$env:PYTHONPATH = "$PWD\backend;$PWD"
python -m pytest backend/tests
```

Some tests use local RAG data or evaluation subprocesses and take longer than small unit tests.

## Data and licensing

This repository is intended for research and educational use. Source content retains its original ownership and licensing. The Competitive Programmer's Handbook material is identified as CC BY-NC-SA 4.0. Verify every upstream source's terms before redistributing a generated corpus or index.
