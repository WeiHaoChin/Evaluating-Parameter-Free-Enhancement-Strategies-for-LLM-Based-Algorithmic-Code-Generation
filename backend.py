from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from typing import Optional
import json
import logging
import sys
from contextlib import asynccontextmanager
from TextGrad import run_textgrad, run_textgrad_sync, OllamaLLM, GoogleGenerativeAI
from rag_handler import initialize_rag, query_rag, format_rag_context, is_rag_available

# Configure logging to show all INFO messages
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)

logger = logging.getLogger(__name__)

# Lifespan context manager for startup/shutdown
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
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
    
    # Shutdown
    logger.info("Shutting down backend...")

app = FastAPI(lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Settings(BaseModel):
    model: str = Field(default="gemma3:4b")
    systemPrompt: str = Field(default="The explanation must be clear and beginner-friendly.")
    darkTheme: bool = Field(default=True)
    temperature: float = Field(default=0.7, ge=0.0, le=1.0)
    benchmark: str = Field(default="TruthfulQA")
    includeRag: bool = Field(default=True)
    includeTextGrad: bool = Field(default=True)
    textGradModel: str = Field(default="gemma3:4b")
    textGradLoops: int = Field(default=1, ge=1)
    textGradLossPrompt: str = Field(default="Evaluate this answer. It should be factual, clear, and directly answer the question.")
    apiKey: Optional[str] = Field(default=None)
    textGradApiKey: Optional[str] = Field(default=None)

class ChatRequest(BaseModel):
    message: str
    settings: Settings

def call_llm(message: str, system_prompt: str, model: str, api_key: Optional[str] = None) -> str:
    """Call the LLM directly without TextGrad."""
    if model.startswith('gemini-'):
        llm = GoogleGenerativeAI(model=model, api_key=api_key)
    elif model.startswith('gemma3:') or model.startswith('gpt-oss:') or model.startswith('deepseek-'):
        llm = OllamaLLM(model=model, api_key=api_key)
    else:
        raise ValueError(f"Unsupported model type: {model}")
    
    return llm(message, system_prompt=system_prompt)

@app.post('/api/chat')
async def chat(request: ChatRequest):
    settings = request.settings
    system_prompt = settings.systemPrompt
    
    # Track formatted prompt to send to frontend
    formatted_prompt = ""
    original_message = request.message
    
    # Augment system prompt with RAG context if enabled
    rag_context = ""
    if settings.includeRag:
        if is_rag_available():
            logger.info("📡 RAG is enabled and available")
            try:
                logger.info(f"Querying RAG for: '{request.message}'")
                rag_results = query_rag(request.message, n_results=5)
                logger.info(f"Got {len(rag_results)} RAG results")
                rag_context = format_rag_context(rag_results, include_metadata=True)
                if rag_context:
                    formatted_prompt = f"""## Problem
{original_message}
## Relevant Context (may or may not be useful)
{rag_context}
## Task
Solve the problem. Use the context above only if it's relevant."""
                    request.message = formatted_prompt
                    logger.info(f"✓ RAG context added to system prompt ({len(rag_context)} chars)")
                else:
                    logger.warning("⚠ RAG returned no formatted context")
            except Exception as exc:
                logger.error(f"✗ RAG query failed: {exc}", exc_info=True)
        else:
            logger.warning("RAG enabled but not available (database may not be initialized)")
    else:
        logger.info("RAG is disabled by user")

    if settings.includeTextGrad:
        try:
            textgrad_output = run_textgrad_sync(
                prompt_text=request.message,
                system_prompt=system_prompt,
                loops=settings.textGradLoops,
                model=settings.model,
                textGradModel=settings.textGradModel,
                api_key=settings.textGradApiKey,
                loss_prompt=settings.textGradLossPrompt,
            )
            reply = (
                f'TextGrad result after {settings.textGradLoops} loops:\n{textgrad_output}'
            )
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f'TextGrad execution failed: {exc}')
    else:
        try:
            reply = call_llm(
                message=request.message,
                system_prompt=system_prompt,
                model=settings.model,
                api_key=settings.apiKey
            )
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f'LLM call failed: {str(exc)}')

    return {
        'reply': reply,
        'settings': settings.dict(),
        'rag_enabled': settings.includeRag and is_rag_available(),
        'rag_context_included': bool(rag_context),
        'formatted_prompt': formatted_prompt,
    }


@app.get('/api/status')
async def status():
    return {'status': 'ok', 'backend': 'available'}


@app.websocket('/ws/chat')
async def websocket_chat(websocket: WebSocket):
    """WebSocket endpoint for streaming TextGrad responses."""
    await websocket.accept()
    try:
        while True:
            # Receive message and settings from client
            data = await websocket.receive_text()
            message_data = json.loads(data)
            
            message = message_data.get('message', '')
            settings_dict = message_data.get('settings', {})
            
            # Reconstruct Settings object
            settings = Settings(**settings_dict)
            
            if not message:
                await websocket.send_text(json.dumps({
                    'type': 'error',
                    'data': 'Empty message'
                }))
                continue
            
            try:
                # Prepare system prompt with RAG context if enabled
                system_prompt = settings.systemPrompt
                rag_context = ""
                formatted_prompt = ""
                
                if settings.includeRag:
                    if is_rag_available():
                        logger.info("[WS] 📡 RAG is enabled and available")
                        try:
                            logger.info(f"[WS] Querying RAG for: '{message}'")
                            rag_results = query_rag(message, n_results=5)
                            logger.info(f"[WS] Got {len(rag_results)} RAG results")
                            rag_context = format_rag_context(rag_results, include_metadata=True)
                            if rag_context:
                                formatted_prompt = f"""## Problem
{message}
## Relevant Context (may or may not be useful)
{rag_context}
## Task
Solve the problem. Use the context above only if it's relevant."""
                                system_prompt = f"{settings.systemPrompt}\n\n{rag_context}"
                                logger.info(f"[WS] ✓ RAG context added ({len(rag_context)} chars)")
                                await websocket.send_text(json.dumps({
                                    'type': 'formatted_prompt',
                                    'prompt': formatted_prompt
                                }))
                            else:
                                logger.warning("[WS] ⚠ RAG returned no formatted context")
                        except Exception as exc:
                            logger.error(f"[WS] ✗ RAG query failed: {exc}", exc_info=True)
                    else:
                        logger.warning("[WS] RAG enabled but not available")
                else:
                    logger.info("[WS] RAG is disabled by user")
                
                # Send start event
                await websocket.send_text(json.dumps({
                    'type': 'start',
                    'message': message
                }))
                
                if settings.includeTextGrad:
                    # Stream TextGrad events
                    for event in run_textgrad(
                        prompt_text=message,
                        system_prompt=system_prompt,
                        loops=settings.textGradLoops,
                        model=settings.model,
                        textGradModel=settings.textGradModel,
                        api_key=settings.textGradApiKey,
                        loss_prompt=settings.textGradLossPrompt,
                    ):
                        await websocket.send_text(json.dumps(event))
                else:
                    # Call LLM directly without TextGrad
                    try:
                        reply = call_llm(
                            message=message,
                            system_prompt=system_prompt,
                            model=settings.model,
                            api_key=settings.apiKey
                        )
                        
                        await websocket.send_text(json.dumps({
                            'type': 'complete',
                            'answer': reply
                        }))
                    except Exception as exc:
                        raise Exception(f'LLM call failed: {str(exc)}')
                
            except Exception as exc:
                await websocket.send_text(json.dumps({
                    'type': 'error',
                    'data': f'Processing failed: {str(exc)}'
                }))
    except WebSocketDisconnect:
        print("Client disconnected")
    except Exception as exc:
        print(f"WebSocket error: {exc}")
        try:
            await websocket.send_text(json.dumps({
                'type': 'error',
                'data': f'WebSocket error: {str(exc)}'
            }))
        except:
            pass


app.mount("/", StaticFiles(directory="public", html=True), name="public")


if __name__ == '__main__':
    import uvicorn
    uvicorn.run('backend:app', host='127.0.0.1', port=5500, reload=True)
