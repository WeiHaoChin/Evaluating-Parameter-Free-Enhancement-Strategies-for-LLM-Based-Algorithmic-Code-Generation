from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from typing import Optional
import json
from TextGrad import run_textgrad, run_textgrad_sync, OllamaLLM, GoogleGenerativeAI

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Settings(BaseModel):
    model: str = Field(default="mock-chat:1.0")
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
    elif model.startswith('gemma3:') or model.startswith('gpt-oss:') or model.startswith('mock-chat:'):
        llm = OllamaLLM(model=model, api_key=api_key)
    else:
        raise ValueError(f"Unsupported model type: {model}")
    
    return llm(message, system_prompt=system_prompt)

@app.post('/api/chat')
async def chat(request: ChatRequest):
    settings = request.settings

    if settings.includeTextGrad:
        try:
            textgrad_output = run_textgrad_sync(
                prompt_text=request.message,
                system_prompt=settings.systemPrompt,
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
                system_prompt=settings.systemPrompt,
                model=settings.model,
                api_key=settings.apiKey
            )
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f'LLM call failed: {str(exc)}')

    if settings.includeRag:
        reply += '\n\n[RAG enabled: retrieval content would be added here]'

    return {
        'reply': reply,
        'settings': settings.dict(),
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
                # Send start event
                await websocket.send_text(json.dumps({
                    'type': 'start',
                    'message': message
                }))
                
                if settings.includeTextGrad:
                    # Stream TextGrad events
                    for event in run_textgrad(
                        prompt_text=message,
                        system_prompt=settings.systemPrompt,
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
                            system_prompt=settings.systemPrompt,
                            model=settings.model,
                            api_key=settings.apiKey
                        )
                        
                        if settings.includeRag:
                            reply += '\n\n[RAG enabled: retrieval content would be added here]'
                        
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
