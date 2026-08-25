import json
import logging
import asyncio
import re
from pathlib import Path
import sys
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from solver import call_llm, run_pipeline, build_rag_prompt
from TextGrad import run_textgrad, run_textgrad_sync
from rag_handler import (
    format_rag_context,
    get_rag_chunk_count,
    initialize_rag,
    is_rag_available,
    query_rag,
)
from routes.benchmark import router as benchmark_router
from schemas import Settings, ChatRequest, settings_defaults, validate_api_key_settings

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)

RAG_PIPELINE_PATH = Path(__file__).parent / "RAG" / "main.py"
RAG_DATA_ROOT = Path(__file__).parent / "data"
_rag_build_status = {
    "running": False, "error": None, "output": "", "percent": 0,
    "stage": "idle", "message": "Ready to build the RAG knowledge base.",
}
_rag_build_task: Optional[asyncio.Task] = None


# ── Lifespan ───────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("=" * 60)
    logger.info("Starting up FYP Backend...")
    logger.info("=" * 60)
    logger.info("Initializing RAG system...")
    success = initialize_rag(embedder="local", local_model="BAAI/bge-small-en-v1.5")
    if success:
        logger.info("✓ RAG system initialized successfully")
    else:
        logger.warning("✗ RAG system initialization failed or database not found")
    logger.info("=" * 60)
    logger.info("Backend ready!")
    logger.info("=" * 60)
    yield
    logger.info("Shutting down backend...")


# ── App ────────────────────────────────────────────────────────────────────────
app = FastAPI(lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def disable_development_caching(request, call_next):
    """Ensure mounted frontend changes are visible after an ordinary reload."""
    response = await call_next(request)
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response



# ── Helpers ────────────────────────────────────────────────────────────────────
def _build_rag_prompt_for_chat(message: str, settings: Settings) -> tuple[str, str]:
    """
    Attempt RAG augmentation.
    Returns (formatted_prompt, rag_context).
    Falls back to (original message, "") on failure.
    """
    if not (settings.includeRag and is_rag_available()):
        if settings.includeRag:
            logger.warning("RAG enabled but not available")
        else:
            logger.info("RAG disabled by user")
        return message, ""

    try:
        logger.info(f"Querying RAG for: '{message}'")
        rag_results = query_rag(message, n_results=5)
        rag_context = format_rag_context(rag_results, include_metadata=True)
        if rag_context:
            logger.info(f"✓ RAG context added ({len(rag_context)} chars)")
            return build_rag_prompt(message, rag_context), rag_context
        logger.warning("RAG returned no context")
    except Exception as e:
        logger.error(f"RAG query failed: {e}", exc_info=True)

    return message, ""


# ── HTTP endpoint ──────────────────────────────────────────────────────────────
@app.post("/api/chat")
async def chat(request: ChatRequest):
    settings = request.settings
    try:
        validate_api_key_settings(settings)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    formatted_prompt, rag_context = _build_rag_prompt_for_chat(
        request.message, settings
    )

    if settings.includeTextGrad:
        try:
            result = run_textgrad_sync(
                prompt_text=formatted_prompt,
                system_prompt=settings.systemPrompt,
                loops=settings.textGradLoops,
                model=settings.model,
                textGradModel=settings.textGradModel,
                api_key=settings.apiKey,
                textGrad_api_key=settings.textGradApiKey,
                loss_prompt=settings.textGradLossPrompt,
                temperature=settings.temperature
            )
            reply = f"TextGrad result after {settings.textGradLoops} loops:\n{result}"
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"TextGrad execution failed: {e}")
    else:
        try:
            reply = call_llm(
                message=formatted_prompt,
                system_prompt=settings.systemPrompt,
                model=settings.model,
                api_key=settings.apiKey,
                temperature=settings.temperature,
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"LLM call failed: {e}")

    return {
        "reply":                reply,
        "settings":             settings.dict(),
        "rag_enabled":          settings.includeRag and is_rag_available(),
        "rag_context_included": bool(rag_context),
        "formatted_prompt":     formatted_prompt if rag_context else "",
    }


# ── Status endpoint ────────────────────────────────────────────────────────────
@app.get("/api/status")
async def status():
    return {"status": "ok", "backend": "available"}


def _record_rag_progress(line: str) -> None:
    """Convert orchestrator output into a stable progress payload for the UI."""
    output = (_rag_build_status["output"] + line)[-12000:]
    _rag_build_status["output"] = output
    stages = (
        ("SCRAPER: Codeforces", 5, "codeforces", "Scraping Codeforces editorials..."),
        ("SCRAPER: USACO", 18, "usaco", "Scraping USACO solutions..."),
        ("SCRAPER: AtCoder", 31, "atcoder", "Scraping AtCoder editorials..."),
        ("SCRAPER: CP-Algorithms", 44, "cp_algorithms", "Scraping CP-Algorithms..."),
        ("SCRAPER: CPH Book", 57, "cph", "Extracting the Competitive Programmer's Handbook..."),
        ("PIPELINE: Chunking & Indexing", 70, "chunking", "Cleaning and chunking scraped content..."),
    )
    for marker, percent, stage, message in stages:
        if marker in line:
            _rag_build_status.update({"percent": percent, "stage": stage, "message": message})
            return
    match = re.search(r"ChromaDB:\s*(\d+)/(\d+) chunks ingested", line)
    if match:
        completed, total = map(int, match.groups())
        percent = 75 + round(24 * completed / total) if total else 75
        _rag_build_status.update({
            "percent": min(percent, 99), "stage": "indexing",
            "message": f"Indexing chunks in ChromaDB ({completed}/{total})...",
        })


async def _build_rag_chunks() -> None:
    """Run all scrapers, chunk/index their output, and reconnect RAG."""
    global _rag_build_task
    command = [
        sys.executable,
        str(RAG_PIPELINE_PATH),
        "--data-root", str(RAG_DATA_ROOT),
        "--embedder", "local",
        "--vector-db", "chroma",
    ]
    try:
        process = await asyncio.create_subprocess_exec(
            *command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            cwd=str(Path(__file__).parent.parent),
        )
        assert process.stdout is not None
        while True:
            raw_line = await process.stdout.readline()
            if not raw_line:
                break
            line = raw_line.decode("utf-8", errors="replace")
            _record_rag_progress(line)
            logger.info("RAG build: %s", line.rstrip())
        await process.wait()
        if process.returncode != 0:
            raise RuntimeError(f"RAG pipeline exited with code {process.returncode}.")

        _rag_build_status.update({
            "percent": 99, "stage": "loading",
            "message": "Reloading the completed RAG index...",
        })
        if not initialize_rag(embedder="local", local_model="BAAI/bge-small-en-v1.5"):
            raise RuntimeError("Chunks were created, but the RAG database could not be reloaded.")
        _rag_build_status.update({
            "percent": 100, "stage": "complete",
            "message": "RAG knowledge base is ready.",
        })
    except Exception as exc:
        logger.exception("RAG chunk build failed")
        _rag_build_status.update({"error": str(exc), "stage": "failed", "message": str(exc)})
    finally:
        _rag_build_status["running"] = False
        _rag_build_task = None


@app.post("/api/rag/build")
async def build_rag_chunks():
    """Rebuild RAG chunks, replacing the existing vector collection."""
    global _rag_build_task
    if _rag_build_status["running"]:
        raise HTTPException(status_code=409, detail="RAG chunk creation is already running.")
    _rag_build_status.update({
        "running": True, "error": None, "output": "", "percent": 1,
        "stage": "starting", "message": "Starting the full RAG pipeline...",
    })
    _rag_build_task = asyncio.create_task(_build_rag_chunks())
    return {"started": True, "mode": "replace"}


@app.get("/api/rag/build/status")
async def rag_build_status():
    chunk_count = get_rag_chunk_count()
    return {
        **_rag_build_status,
        "chunk_count": chunk_count,
        "chunks_exist": chunk_count > 0,
    }

# ── Default settings endpoint ──────────────────────────────────────────────────
@app.get("/api/defaults")
async def get_default_settings():
    """Get default settings from schemas.Settings"""
    return settings_defaults()

# ── WebSocket endpoint ─────────────────────────────────────────────────────────
@app.websocket("/ws/chat")
async def websocket_chat(websocket: WebSocket):
    """WebSocket endpoint for streaming TextGrad responses."""
    await websocket.accept()
    try:
        while True:
            data         = await websocket.receive_text()
            try:
                message_data = json.loads(data)
                message = message_data.get("message", "")
                settings = Settings(**message_data.get("settings", {}))
                validate_api_key_settings(settings)
            except (json.JSONDecodeError, TypeError, ValueError) as exc:
                await websocket.send_text(
                    json.dumps({"type": "error", "data": f"Invalid settings: {exc}"})
                )
                continue

            if not message:
                await websocket.send_text(
                    json.dumps({"type": "error", "data": "Empty message"})
                )
                continue

            try:
                # RAG augmentation
                formatted_prompt, rag_context = _build_rag_prompt_for_chat(
                    message, settings
                )

                if rag_context:
                    await websocket.send_text(
                        json.dumps({
                            "type":   "formatted_prompt",
                            "prompt": formatted_prompt,
                        })
                    )

                # start event
                await websocket.send_text(
                    json.dumps({"type": "start", "message": message})
                )

                if settings.includeTextGrad:
                    for event in run_textgrad(
                        prompt_text=formatted_prompt,
                        system_prompt=settings.systemPrompt,
                        loops=settings.textGradLoops,
                        model=settings.model,
                        textGradModel=settings.textGradModel,
                        api_key=settings.apiKey,
                        textGrad_api_key=settings.textGradApiKey,
                        loss_prompt=settings.textGradLossPrompt,
                        temperature=settings.temperature
                    ):
                        await websocket.send_text(json.dumps(event))

                else:
                    reply = call_llm(
                        message=formatted_prompt,
                        system_prompt=settings.systemPrompt,
                        model=settings.model,
                        api_key=settings.apiKey,
                        temperature=settings.temperature
                    )
                    await websocket.send_text(
                        json.dumps({"type": "complete", "answer": reply})
                    )

            except Exception as e:
                await websocket.send_text(
                    json.dumps({"type": "error", "data": f"Processing failed: {e}"})
                )

    except WebSocketDisconnect:
        logger.info("Client disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        try:
            await websocket.send_text(
                json.dumps({"type": "error", "data": f"WebSocket error: {e}"})
            )
        except Exception:
            pass


# ── Routers & static ───────────────────────────────────────────────────────────
app.include_router(benchmark_router)
app.mount("/", StaticFiles(directory="public", html=True), name="public")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=5050, reload=True)
