import logging
import os
import random
import time

import httpx
from ollama import Client
from google import genai
from config.models import get_model_provider, register_local_ollama_models
from config.generation import DEFAULT_MAX_OUTPUT_TOKENS

logger = logging.getLogger(__name__)
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
OLLAMA_CLOUD_HOST = os.getenv("OLLAMA_CLOUD_HOST", "https://ollama.com").rstrip("/")
MAX_OUTPUT_TOKENS = DEFAULT_MAX_OUTPUT_TOKENS


def _value(source, name, default=None):
    if source is None:
        return default
    if isinstance(source, dict):
        return source.get(name, default)
    return getattr(source, name, default)


def _finish_reason(value):
    if value is None:
        return None
    value = getattr(value, "value", value)
    return str(value).lower().removeprefix("finishreason.")


def _is_truncated(finish_reason, output_tokens, max_output_tokens):
    """Prefer explicit provider reasons; use the token ceiling only if ambiguous."""
    reason = _finish_reason(finish_reason)
    if reason in {"stop", "end_turn", "stop_sequence", "complete", "completed"}:
        return False
    if reason in {"length", "max_tokens", "max_output_tokens", "token_limit"}:
        return True
    return bool(
        output_tokens is not None
        and max_output_tokens is not None
        and output_tokens >= max_output_tokens
    )


class GenerationTrackedLLM:
    def _init_tracking(
        self,
        max_output_tokens=MAX_OUTPUT_TOKENS,
        metadata_sink=None,
        mode=None,
        call_type="generation",
    ):
        self.max_output_tokens = int(max_output_tokens)
        self.metadata_sink = metadata_sink
        self.mode = mode
        self.call_type = call_type

    def set_generation_context(self, call_type, mode=None):
        self.call_type = call_type
        if mode is not None:
            self.mode = mode

    def _record(
        self,
        *,
        model=None,
        prompt_tokens=None,
        output_tokens=None,
        finish_reason=None,
        total_duration=None,
        prompt_eval_duration=None,
        eval_duration=None,
        client_duration_ns=None,
    ):
        if self.metadata_sink is None:
            return
        reason = _finish_reason(finish_reason)
        self.metadata_sink.append({
            "model": model or self.model,
            "mode": self.mode,
            "call_type": self.call_type,
            "prompt_tokens": prompt_tokens,
            "output_tokens": output_tokens,
            "max_output_tokens": self.max_output_tokens,
            "finish_reason": reason,
            "truncated": _is_truncated(reason, output_tokens, self.max_output_tokens),
            "total_duration": total_duration,
            "prompt_eval_duration": prompt_eval_duration,
            "eval_duration": eval_duration,
            "client_duration_ns": client_duration_ns,
        })


def get_ollama_status() -> dict:
    """Return local Ollama availability and the models installed on this machine."""
    try:
        response = Client(host=OLLAMA_HOST).list()
        entries = getattr(response, "models", None)
        if entries is None and isinstance(response, dict):
            entries = response.get("models", [])
        models = []
        for entry in entries or []:
            name = getattr(entry, "model", None) or getattr(entry, "name", None)
            if name is None and isinstance(entry, dict):
                name = entry.get("model") or entry.get("name")
            if name:
                models.append(str(name))
        models = sorted(set(models))
        register_local_ollama_models(models)
        return {"running": True, "host": OLLAMA_HOST, "models": models, "error": None}
    except Exception as exc:
        register_local_ollama_models([])
        logger.info("Local Ollama is unavailable at %s: %s", OLLAMA_HOST, exc)
        return {"running": False, "host": OLLAMA_HOST, "models": [], "error": str(exc)}

class OllamaLLM(GenerationTrackedLLM):
    def __init__(self, model, host=None, api_key=None, temperature=0.0, max_output_tokens=MAX_OUTPUT_TOKENS, metadata_sink=None, mode=None, call_type="generation"):
        self.model = model
        self.host = (host or OLLAMA_HOST).rstrip('/')
        self.client = Client(host=self.host)
        self.temperature = temperature
        self._init_tracking(max_output_tokens, metadata_sink, mode, call_type)

    def __call__(self, prompt, system_prompt=None, **kwargs):
        messages = []
        if system_prompt:
            messages.append({'role': 'system', 'content': str(system_prompt)})
        messages.append({'role': 'user', 'content': str(prompt)})

        call_started_ns = time.perf_counter_ns()
        response = self.client.chat(
            model=self.model,
            messages=messages,
            options={
                'temperature': self.temperature,
                'num_predict': self.max_output_tokens,
            },
        )
        message = getattr(response, 'message', None)
        content = getattr(message, 'content', None)
        self._record(
            model=_value(response, "model", self.model),
            prompt_tokens=_value(response, "prompt_eval_count"),
            output_tokens=_value(response, "eval_count"),
            finish_reason=_value(response, "done_reason"),
            total_duration=_value(response, "total_duration"),
            prompt_eval_duration=_value(response, "prompt_eval_duration"),
            eval_duration=_value(response, "eval_duration"),
            client_duration_ns=time.perf_counter_ns() - call_started_ns,
        )
        if content is not None:
            return content
        return response['message']['content']


class OllamaCloudLLM(OllamaLLM):
    """Authenticated Ollama Cloud client used for non-local model names."""

    def __init__(self, model, api_key=None, temperature=0.0, max_output_tokens=MAX_OUTPUT_TOKENS, metadata_sink=None, mode=None, call_type="generation", **kwargs):
        if not api_key:
            raise ValueError("An Ollama Cloud API key is required for this model.")
        self.model = model
        self.host = OLLAMA_CLOUD_HOST
        self.client = Client(
            host=self.host,
            headers={"Authorization": f"Bearer {api_key}"},
        )
        self.temperature = temperature
        self._init_tracking(max_output_tokens, metadata_sink, mode, call_type)

class GoogleGenerativeAI(GenerationTrackedLLM):
    def __init__(self, model, api_key,temperature=0.0, max_output_tokens=MAX_OUTPUT_TOKENS, metadata_sink=None, mode=None, call_type="generation"):
        self.model = model
        self.client = genai.Client(api_key=api_key)
        self.max_retries = 5
        self.initial_delay = 1  # Start with 1 second delay
        self.temperature = temperature
        self._init_tracking(max_output_tokens, metadata_sink, mode, call_type)

    def __call__(self, prompt, system_prompt=None, **kwargs):
        # For Google models, system_prompt is usually part of the prompt in a multi-turn conversation
        # or set as a safety setting. Here, we'll prepend it to the prompt if provided.
        # full_prompt = f"{system_prompt}\n{prompt}" if system_prompt else str(prompt)
        config = genai.types.GenerateContentConfig(
            temperature=self.temperature,
            max_output_tokens=self.max_output_tokens,
            system_instruction=system_prompt if system_prompt else None,
        )
        call_started_ns = time.perf_counter_ns()
        # Retry logic with exponential backoff
        for attempt in range(self.max_retries):
            try:
                response = self.client.models.generate_content(model=self.model, contents=str(prompt), config=config)
                usage = getattr(response, "usage_metadata", None)
                candidates = getattr(response, "candidates", None) or []
                self._record(
                    prompt_tokens=_value(usage, "prompt_token_count"),
                    output_tokens=_value(usage, "candidates_token_count"),
                    finish_reason=_value(candidates[0], "finish_reason") if candidates else None,
                    client_duration_ns=time.perf_counter_ns() - call_started_ns,
                )
                return response.text
            except Exception as e:
                error_str = str(e)
                # Check if it's a 503 or rate limit error
                if '503' in error_str or 'UNAVAILABLE' in error_str or 'high demand' in error_str:
                    if attempt < self.max_retries - 1:
                        # Exponential backoff with jitter
                        wait_time = self.initial_delay * (2 ** attempt) + random.uniform(0, 1)
                        print(f"API rate limited (attempt {attempt + 1}/{self.max_retries}). Waiting {wait_time:.2f}s before retry...")
                        time.sleep(wait_time)
                    else:
                        print(f"Max retries ({self.max_retries}) exceeded. Raising error.")
                        raise
                else:
                    # For other errors, don't retry
                    raise


class AnthropicLLM(GenerationTrackedLLM):
    """Minimal Anthropic Messages API adapter matching the other LLM clients."""

    def __init__(self, model, api_key, temperature=0.0, max_output_tokens=MAX_OUTPUT_TOKENS, metadata_sink=None, mode=None, call_type="generation"):
        if not api_key:
            raise ValueError("An Anthropic API key is required for Claude models.")
        self.model = model
        self.temperature = temperature
        self._init_tracking(max_output_tokens, metadata_sink, mode, call_type)
        self.client = httpx.Client(
            base_url="https://api.anthropic.com",
            headers={
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
            },
            timeout=120.0,
        )

    def __call__(self, prompt, system_prompt=None, **kwargs):
        payload = {
            "model": self.model,
            "max_tokens": self.max_output_tokens,
            "temperature": self.temperature,
            "messages": [{"role": "user", "content": str(prompt)}],
        }
        if system_prompt:
            payload["system"] = str(system_prompt)
        call_started_ns = time.perf_counter_ns()
        response = self.client.post("/v1/messages", json=payload)
        response.raise_for_status()
        response_data = response.json()
        usage = response_data.get("usage", {})
        self._record(
            prompt_tokens=usage.get("input_tokens"),
            output_tokens=usage.get("output_tokens"),
            finish_reason=response_data.get("stop_reason"),
            client_duration_ns=time.perf_counter_ns() - call_started_ns,
        )
        return "".join(
            block.get("text", "")
            for block in response_data.get("content", [])
            if block.get("type") == "text"
        )


class DeepSeekLLM(GenerationTrackedLLM):
    """DeepSeek's OpenAI-compatible chat-completions API adapter."""

    def __init__(self, model, api_key, temperature=0.0, max_output_tokens=MAX_OUTPUT_TOKENS, metadata_sink=None, mode=None, call_type="generation"):
        if not api_key:
            raise ValueError("A DeepSeek API key is required for DeepSeek models.")
        self.model = model
        self.temperature = temperature
        self._init_tracking(max_output_tokens, metadata_sink, mode, call_type)
        self.client = httpx.Client(
            base_url="https://api.deepseek.com",
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=120.0,
        )

    def __call__(self, prompt, system_prompt=None, **kwargs):
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": str(system_prompt)})
        messages.append({"role": "user", "content": str(prompt)})
        logger.info("Calling DeepSeek API with requested model '%s'", self.model)
        call_started_ns = time.perf_counter_ns()
        response = self.client.post(
            "/chat/completions",
            json={
                "model": self.model,
                "messages": messages,
                "temperature": self.temperature,
                "max_tokens": self.max_output_tokens,
            },
        )
        response.raise_for_status()
        response_data = response.json()
        usage = response_data.get("usage", {})
        choice = response_data["choices"][0]
        self._record(
            model=response_data.get("model", self.model),
            prompt_tokens=usage.get("prompt_tokens"),
            output_tokens=usage.get("completion_tokens"),
            finish_reason=choice.get("finish_reason"),
            client_duration_ns=time.perf_counter_ns() - call_started_ns,
        )
        returned_model = response_data.get("model", "unknown")
        logger.info(
            "DeepSeek API response model: requested='%s', returned='%s'",
            self.model,
            returned_model,
        )
        return choice["message"]["content"]


def create_llm_client(model, api_key=None, temperature=0.0, max_output_tokens=MAX_OUTPUT_TOKENS, metadata_sink=None, mode=None, call_type="generation"):
    """Create the correct LLM client from the central model configuration."""
    provider = get_model_provider(model)
    client_types = {
        "google": GoogleGenerativeAI,
        "ollama_local": OllamaLLM,
        "ollama_cloud": OllamaCloudLLM,
        "anthropic": AnthropicLLM,
        "deepseek": DeepSeekLLM,
    }
    try:
        return client_types[provider](
            model=model,
            api_key=api_key,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            metadata_sink=metadata_sink,
            mode=mode,
            call_type=call_type,
        )
    except KeyError as exc:
        raise ValueError(f"Unsupported provider '{provider}' configured for model '{model}'.") from exc
