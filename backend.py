from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from typing import Optional
from demo import run_textgrad

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

@app.post('/api/chat')
async def chat(request: ChatRequest):
    settings = request.settings

    if settings.includeTextGrad:
        try:
            textgrad_output = run_textgrad(
                prompt_text=request.message,
                system_prompt=settings.systemPrompt,
                loops=settings.textGradLoops,
                model=settings.textGradModel,
                api_key=settings.textGradApiKey,
                loss_prompt=settings.textGradLossPrompt,
            )
            reply = (
                f'TextGrad result after {settings.textGradLoops} loops:\n{textgrad_output}'
            )
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f'TextGrad execution failed: {exc}')
    else:
        reply = f'Model {settings.model} received: {request.message}'

    if settings.includeRag:
        reply += '\n\n[RAG enabled: retrieval content would be added here]'

    return {
        'reply': reply,
        'settings': settings.dict(),
    }


@app.get('/api/status')
async def status():
    return {'status': 'ok', 'backend': 'available'}


app.mount("/", StaticFiles(directory="public", html=True), name="public")


if __name__ == '__main__':
    import uvicorn
    uvicorn.run('backend:app', host='127.0.0.1', port=5500, reload=True)
