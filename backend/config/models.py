"""Supported LLMs and the backend used to call each one."""

MODEL_CONFIG = {
    "gemini-2.5-pro": {"provider": "google"},
    "claude-sonnet-4-6": {"provider": "anthropic"},
    "deepseek-v4-flash": {"provider": "ollama_cloud"},
    "gemma4:cloud": {"provider": "ollama_cloud"},
    "gpt-oss:120b": {"provider": "ollama_cloud"},
    "qwen3.5:cloud": {"provider": "ollama_cloud"},
    "mistral-large-3:675b-cloud": {"provider": "ollama_cloud"},
    "kimi-k2.6:cloud": {"provider": "ollama_cloud"},
}

THINKING_MODES = frozenset({"default", "off", "on", "low", "medium", "high", "max"})


def unavailable_thinking_config(note: str | None = None) -> dict:
    """Return a conservative control when Ollama supplies no exact metadata."""
    return {
        "configurable": False,
        "default": "default",
        "modes": ["default"],
        "note": note or "Ollama did not report configurable thinking modes for this model.",
    }


def canonical_ollama_model_name(model: str) -> str:
    """Match local and cloud aliases without encoding model-family knowledge."""
    normalized = (model or "").strip().lower()
    if normalized.endswith(":cloud"):
        return normalized[:-6]
    if normalized.endswith("-cloud"):
        return normalized[:-6]
    return normalized

_LOCAL_OLLAMA_MODELS: set[str] = set()


def register_local_ollama_models(models: list[str]) -> None:
    """Replace the process-local set populated by Ollama model discovery."""
    _LOCAL_OLLAMA_MODELS.clear()
    _LOCAL_OLLAMA_MODELS.update(model for model in models if model)


def get_model_provider(model: str) -> str:
    """Route detected Ollama names locally and otherwise-unknown names to Cloud."""
    if not model or not model.strip():
        raise ValueError("Model name cannot be empty.")
    if model in MODEL_CONFIG:
        return MODEL_CONFIG[model]["provider"]
    if model in _LOCAL_OLLAMA_MODELS:
        return "ollama_local"
    return "ollama_cloud"


def model_requires_api_key(model: str) -> bool:
    """Local Ollama models do not require an API key."""
    return get_model_provider(model) != "ollama_local"
