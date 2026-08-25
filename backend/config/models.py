"""Supported LLMs and the backend used to call each one."""

MODEL_CONFIG = {
    "gemma3:4b": {"provider": "ollama"},
    "gpt-oss:120b": {"provider": "ollama"},
    "gemini-2.5-pro": {"provider": "google"},
    "qwen3:latest": {"provider": "ollama"},
    "deepseek-v3.2": {"provider": "deepseek"},
    "claude-sonnet-4-6": {"provider": "anthropic"},
    "deepseek-v4-flash": {"provider": "deepseek"},
}


def get_model_provider(model: str) -> str:
    """Return the configured provider for a supported model."""
    try:
        return MODEL_CONFIG[model]["provider"]
    except KeyError as exc:
        raise ValueError(f"Unsupported model: {model}") from exc
