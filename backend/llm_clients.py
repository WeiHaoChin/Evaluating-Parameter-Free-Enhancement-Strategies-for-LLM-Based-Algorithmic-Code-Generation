import logging
import random
import time

import httpx
from ollama import Client
from google import genai
from config.models import get_model_provider

logger = logging.getLogger(__name__)

class OllamaLLM:
    def __init__(self, model, host='https://ollama.com', api_key=None, temperature=0.0):
        self.model = model
        headers = {'Authorization': f'Bearer {api_key}'} if api_key else {}
        self.client = Client(host=host, headers=headers)
        self.temperature = temperature

    def __call__(self, prompt, system_prompt=None, **kwargs):
        messages = []
        if system_prompt:
            messages.append({'role': 'system', 'content': str(system_prompt)})
        messages.append({'role': 'user', 'content': str(prompt)})

        response = self.client.chat(model=self.model, messages=messages, options={'temperature': self.temperature})
        return response['message']['content']

class GoogleGenerativeAI:
    def __init__(self, model, api_key,temperature=0.0):
        self.model = model
        self.client = genai.Client(api_key=api_key)
        self.max_retries = 5
        self.initial_delay = 1  # Start with 1 second delay
        self.temperature = temperature

    def __call__(self, prompt, system_prompt=None, **kwargs):
        # For Google models, system_prompt is usually part of the prompt in a multi-turn conversation
        # or set as a safety setting. Here, we'll prepend it to the prompt if provided.
        # full_prompt = f"{system_prompt}\n{prompt}" if system_prompt else str(prompt)
        config = genai.types.GenerateContentConfig(
            temperature=self.temperature,
            system_instruction=system_prompt if system_prompt else None,
        )
        # Retry logic with exponential backoff
        for attempt in range(self.max_retries):
            try:
                response = self.client.models.generate_content(model=self.model, contents=str(prompt), config=config)
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


class AnthropicLLM:
    """Minimal Anthropic Messages API adapter matching the other LLM clients."""

    def __init__(self, model, api_key, temperature=0.0):
        if not api_key:
            raise ValueError("An Anthropic API key is required for Claude models.")
        self.model = model
        self.temperature = temperature
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
            "max_tokens": 8192,
            "temperature": self.temperature,
            "messages": [{"role": "user", "content": str(prompt)}],
        }
        if system_prompt:
            payload["system"] = str(system_prompt)
        response = self.client.post("/v1/messages", json=payload)
        response.raise_for_status()
        return "".join(
            block.get("text", "")
            for block in response.json().get("content", [])
            if block.get("type") == "text"
        )


class DeepSeekLLM:
    """DeepSeek's OpenAI-compatible chat-completions API adapter."""

    def __init__(self, model, api_key, temperature=0.0):
        if not api_key:
            raise ValueError("A DeepSeek API key is required for DeepSeek models.")
        self.model = model
        self.temperature = temperature
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
        response = self.client.post(
            "/chat/completions",
            json={
                "model": self.model,
                "messages": messages,
                "temperature": self.temperature,
            },
        )
        response.raise_for_status()
        response_data = response.json()
        returned_model = response_data.get("model", "unknown")
        logger.info(
            "DeepSeek API response model: requested='%s', returned='%s'",
            self.model,
            returned_model,
        )
        return response_data["choices"][0]["message"]["content"]


def create_llm_client(model, api_key=None, temperature=0.0):
    """Create the correct LLM client from the central model configuration."""
    provider = get_model_provider(model)
    client_types = {
        "google": GoogleGenerativeAI,
        "ollama": OllamaLLM,
        "anthropic": AnthropicLLM,
        "deepseek": DeepSeekLLM,
    }
    try:
        return client_types[provider](model=model, api_key=api_key, temperature=temperature)
    except KeyError as exc:
        raise ValueError(f"Unsupported provider '{provider}' configured for model '{model}'.") from exc
