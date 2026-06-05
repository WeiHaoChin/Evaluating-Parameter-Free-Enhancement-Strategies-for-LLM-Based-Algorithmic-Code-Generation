import json
import logging
import sys
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from solver import call_llm, run_pipeline, build_rag_prompt
from TextGrad import run_textgrad, run_textgrad_sync
from rag_handler import initialize_rag, query_rag, format_rag_context, is_rag_available
from routes.benchmark import router as benchmark_router
from schemas import Settings, ChatRequest

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)


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
                api_key=settings.textGradApiKey,
                loss_prompt=settings.textGradLossPrompt,
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


# ── WebSocket endpoint ─────────────────────────────────────────────────────────
@app.websocket("/ws/chat")
async def websocket_chat(websocket: WebSocket):
    """WebSocket endpoint for streaming TextGrad responses."""
    await websocket.accept()
    try:
        while True:
            data         = await websocket.receive_text()
            message_data = json.loads(data)
            message      = message_data.get("message", "")
            settings     = Settings(**message_data.get("settings", {}))

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
                        api_key=settings.textGradApiKey,
                        loss_prompt=settings.textGradLossPrompt,
                    ):
                        await websocket.send_text(json.dumps(event))

                else:
                    reply = call_llm(
                        message=formatted_prompt,
                        system_prompt=settings.systemPrompt,
                        model=settings.model,
                        api_key=settings.apiKey,
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
    uvicorn.run("main:app", host="127.0.0.1", port=5500, reload=True)