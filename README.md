⚠️ Status: Work in Progress (WIP)
This repository is actively under development. Experimental results, code modules, and evaluation scripts are subject to ongoing updates and refinements.


# FYP: Competitive Programming LLM Evaluation Platform

This project combines a browser-based evaluation UI with a retrieval-augmented generation (RAG) pipeline for competitive programming data. It supports interactive prompting, optional TextGrad refinement, benchmarking against LiveCodeBench, and retrieval over a CP knowledge base built from public problem and editorial sources.

## Overview

The codebase has four main parts:

1. Frontend UI in `public/` for chatting, settings, and benchmark pages
2. FastAPI backend in `backend/` that serves the UI and provides API/WebSocket endpoints
3. RAG pipeline in `backend/RAG/` that scrapes, chunks, embeds, and indexes competitive programming content
4. Benchmark harness in `backend/benchmark/` and `backend/lcb_runner/` for evaluation runs

## Architecture

- `public/` contains the static HTML/JS UI
- `backend/main.py` is the main FastAPI app and service entry point
- `backend/solver.py` contains the main CP solving pipeline and prompt construction
- `backend/TextGrad.py` wraps TextGrad and LLM clients for iterative refinement
- `backend/rag_handler.py` exposes ChromaDB queries used by the app
- `backend/routes/benchmark.py` adds benchmark API endpoints
- `backend/RAG/main.py` builds the knowledge base from multiple sources
- `backend/benchmark/runner.py` runs benchmark jobs and tracks execution status

## Project Structure

```text
FYP/
├── Dockerfile
├── docker-compose.yaml
├── requirements.txt
├── package.json
├── server.js
├── .env
├── public/
│   ├── app.js
│   ├── benchmark.html
│   ├── benchmark.js
│   ├── examples.html
│   ├── index.html
│   ├── new-chat.html
│   ├── page-transitions.js
│   ├── settings.html
│   ├── settings.js
│   ├── styles.css
│   └── Dockerfile
├── backend/
│   ├── main.py
│   ├── schemas.py
│   ├── solver.py
│   ├── TextGrad.py
│   ├── llm_clients.py
│   ├── rag_handler.py
│   ├── data/
│   │   ├── hf_cache/
│   │   └── rag_chunks/
│   ├── benchmark/
│   │   ├── fetch_lcb.py
│   │   ├── logger.py
│   │   ├── metrics.py
│   │   ├── prefetch_lcb.py
│   │   └── runner.py
│   ├── lcb_runner/
│   │   ├── benchmarks/
│   │   ├── evaluation/
│   │   └── utils/
│   ├── routes/
│   │   └── benchmark.py
│   └── RAG/
│       ├── main.py
│       ├── pipeline/
│       ├── scrapers/
│       └── ...
├── logs/
└── README.md
```

## Features

- Chat UI for competitive-programming prompts and responses
- Optional RAG augmentation using a local ChromaDB knowledge base
- TextGrad support for iterative prompt refinement
- LLM model selection and settings panel in the browser
- LiveCodeBench-style benchmark orchestration and result tracking
- Local and cloud-compatible model backends (Ollama and Gemini/Gemini-style APIs in this codebase)
- Docker setup for frontend + backend + ChromaDB services

## Quick Start

### Recommended: run everything with Docker

This is the simplest way to run the app without manually installing all Python dependencies.

From the repository root:

```bash
docker compose up --build
```

This starts:

- frontend: http://localhost:3000
- backend API: http://localhost:5050/docs
- ChromaDB: http://localhost:8000

The app runs through the FastAPI backend and serves the static frontend from `public/`.

### Alternative: run locally with Python

If you want to work on the backend directly, install Python dependencies first:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Then start the backend:

```bash
uvicorn backend.main:app --host 127.0.0.1 --port 5050 --reload
```

Open the app at:

- http://127.0.0.1:5050/

### Optional: lightweight Node compatibility server

This project includes a simple Express wrapper, but it is not the main app backend:

```bash
npm install
npm start
```

It serves the frontend on port 3000 and points to the Python backend on port 5050.

## Docker

The repository includes a Docker Compose stack for the full application:

```bash
docker compose up --build
```

This starts:

- frontend container on port 3000
- backend container on port 5050
- ChromaDB container on port 8000

The backend image runs:

```bash
uvicorn backend.main:app --host 0.0.0.0 --port 5050
```

## API Endpoints

The backend exposes the following main endpoints:

- `POST /api/chat` — single-turn chat request
- `GET /api/status` — backend health check
- `GET /api/defaults` — default model/settings payload
- `WebSocket /ws/chat` — streaming chat responses
- `GET /benchmark/status` and `/benchmark/results` — benchmark status and saved results
- `POST /benchmark/run` — start a benchmark run

## RAG Data Pipeline

The RAG engine is built in `backend/RAG/` and creates the knowledge base by scraping and processing competitive programming material from public sources before indexing it in ChromaDB.

### Where the RAG data comes from

The dataset is built from the following public competitive-programming sources:

| Source | What is scraped | Volume | Quality |
|--------|-----------------|--------|---------|
| **Codeforces** | Problems, editorials, tags, and ratings | ~500 problems | ⭐⭐⭐⭐ |
| **USACO** | Problems and USACO Guide editorial content | ~200 problems | ⭐⭐⭐⭐⭐ |
| **AtCoder** | ABC/ARC/AGC problems and official editorials | ~300 problems | ⭐⭐⭐⭐ |
| **CP-Algorithms** | Full algorithm reference articles | ~180 articles | ⭐⭐⭐⭐⭐ |
| **CPH Book** | Structured chapters and theory explanations | 30 chapters | ⭐⭐⭐⭐⭐ |

These sources are scraped, chunked, and embedded before being stored in ChromaDB for retrieval during chat and benchmark runs.

### Supported sources

- Codeforces
- USACO
- AtCoder
- CP-Algorithms
- CPH Book

### Build the knowledge base

From the project root:

```bash
python -m backend.RAG.main
```

Subset a source list:

```bash
python -m backend.RAG.main --sources codeforces,cp_algorithms --vector-db chroma
```

Rebuild the vectorized chunks without re-scraping:

```bash
python -m backend.RAG.main --pipeline-only --embedder local --vector-db chroma
```

### RAG behavior

The backend initializes ChromaDB during startup via `backend/rag_handler.py`. If the database is missing, the app logs a warning and continues without RAG augmentation. This is intentional; the database should be generated first by running the RAG pipeline.

## Benchmarking

The benchmark routes are wired through `backend/routes/benchmark.py` and call the benchmark runner in `backend/benchmark/runner.py`.

Typical flow:

1. Download LiveCodeBench dataset via the UI or the benchmark route
2. Ensure the model/API settings are valid
3. Start a run from the benchmark page or API call
4. Review saved results in the benchmark UI or JSON files under the result directory

The monitoring flow is designed around a background task and a status endpoint so the UI can poll benchmark progress.

## Configuration

The app reads settings from the Pydantic model in `backend/schemas.py`, and the runtime can also use environment variables such as:

```bash
GOOGLE_API_KEY=...
OPENAI_API_KEY=...
QDRANT_API_KEY=...
OLLAMA_API_KEY=...
```

The app uses these values for model access and embedding pipelines depending on the selected configuration.

## Notes

- The primary backend entry point is `backend.main:app`, not a root-level `backend.py` file.
- `public/` is mounted as the static site by FastAPI; the browser UI interacts with the backend API.
- `server.js` is only a lightweight compatibility wrapper and is not the core application.
- RAG persistence is stored under `backend/data/rag_chunks/`.
- Benchmark results and logs are stored in the project logs/result folders.

## Development Notes

If you are working on the backend:

```bash
cd backend
uvicorn main:app --host 127.0.0.1 --port 5050 --reload
```

If you are working on the UI only, you can open static files directly in `public/` or serve them through the Python app.

## License

This repository includes third-party content and data sources for competitive programming research. The CPH Book content is explicitly noted as CC BY-NC-SA 4.0. Other source content and project code are provided for research and educational use within the repository context.
