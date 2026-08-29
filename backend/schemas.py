from typing import Optional
from pydantic import BaseModel, Field, field_validator
from config.models import (
    MODEL_CONFIG,
    get_model_provider,
    model_requires_api_key,
    register_local_ollama_models,
)
from config.generation import (
    DEFAULT_MAX_OUTPUT_TOKENS,
    DEFAULT_TEXTGRAD_INTERNAL_MAX_OUTPUT_TOKENS,
)
# 1. APPROACH: Brief explanation of your algorithm and why it's correct
# ── Schemas ────────────────────────────────────────────────────────────────────
# The frontend obtains this list from /api/defaults rather than maintaining a
# separate hard-coded copy. Keep model defaults and validation in this module.
SUPPORTED_MODELS = tuple(MODEL_CONFIG)


class Settings(BaseModel):
    model: str                  = Field(default="gemini-2.5-pro")
    systemPrompt: str           = Field(default="""You are an expert competitive programmer. Given a competitive programming problem, produce a correct and efficient solution that fits within the given time and memory constraints.""")
    temperature: float          = Field(default=0.0, ge=0.0, le=1.0)
    maxOutputTokens: int        = Field(default=DEFAULT_MAX_OUTPUT_TOKENS, ge=1)
    includeRag: bool            = Field(default=True)
    includeTextGrad: bool       = Field(default=True)
    textGradModel: str          = Field(default="gemini-2.5-pro")
    textGradLoops: int          = Field(default=1, ge=1, le=5)
    textGradInternalMaxOutputTokens: int = Field(
        default=DEFAULT_TEXTGRAD_INTERNAL_MAX_OUTPUT_TOKENS, ge=1
    )
    textGradLossPrompt: str     = Field(default="""You are evaluating a competitive programming solution. Your feedback will 
be used to improve the SYSTEM PROMPT that generated this solution.

Report at most 5 material failure risks:

1. ALGORITHMIC CORRECTNESS: Identify only concrete correctness defects that can be demonstrated directly from the submitted solution. Do not speculate about alternative algorithms or convert an uncertain concern into a required implementation.
2. COMPLEXITY: Report only a definite violation of the stated constraints. Recommend a prompt-level requirement to verify complexity, not a particular algorithm or data structure.
3. COMPLETENESS: Is the solution fully implemented and runnable?
4. INTERFACE AND IMPLEMENTATION: Python implementation correctness, including indexing, types, imports, and output format
5. Comment misuse, including comments containing reasoning, partial attempts,
   alternative approaches, uncertainty, or abandoned logic

Output format:
RISK: <one-sentence concrete failure risk>
PROMPT CHANGE: <one-sentence change to the SYSTEM PROMPT>

If there are no material risks, output:
NO MATERIAL RISKS

Do not recommend problem-specific algorithms, formulas, variable names, data structures, constants, or implementation steps in PROMPT CHANGE. Prompt changes must remain general and reusable across unrelated competitive-programming problems.

Constraints for your critique:
- Perform only the minimum analysis required and keep internal deliberation concise.
- Use at most 400 words.
- Do not restate the problem or solution.
- Do not explain correct parts of the solution.
- Do not derive an alternative algorithm.
- Do not rewrite or patch the submitted code.
- Do not discuss style, formatting, or verbosity unless they affect correctness.
- Prioritize definite defects over speculative concerns.
- Return only the risk and prompt-change pairs.""")
    apiKey: Optional[str]       = Field(default=None)
    textGradApiKey: Optional[str] = Field(default=None)

    @field_validator("model", "textGradModel")
    @classmethod
    def supported_model(cls, value: str) -> str:
        value = value.strip()
        get_model_provider(value)
        return value


def settings_defaults(local_models: list[str] | None = None) -> dict:
    """Settings defaults plus the supported choices for settings clients."""
    detected_local_models = local_models or []
    register_local_ollama_models(detected_local_models)
    models = list(dict.fromkeys([*detected_local_models, *SUPPORTED_MODELS]))
    defaults = Settings().model_dump()
    if detected_local_models:
        # Prefer an actually installed local model without hard-coding its name.
        defaults["model"] = detected_local_models[0]
        defaults["textGradModel"] = detected_local_models[0]
    return {
        **defaults,
        "models": models,
        "modelProviders": {model: get_model_provider(model) for model in models},
    }


def validate_api_key_settings(settings: Settings) -> None:
    """Reject runnable configurations that are missing required model keys."""
    if model_requires_api_key(settings.model) and not (settings.apiKey or "").strip():
        raise ValueError("API key is required for the selected primary model.")
    if (
        settings.includeTextGrad
        and model_requires_api_key(settings.textGradModel)
        and not (settings.textGradApiKey or "").strip()
    ):
        raise ValueError("API key is required for the selected TextGrad model.")


class ChatRequest(BaseModel):
    message: str
    settings: Settings

class BenchmarkRequest(BaseModel):
    version: str = "release_v6"
    n: int = Field(default=30, ge=1, le=100)
    difficulty: Optional[str] = None
    seed: int = 42
    startQuestion: int = Field(default=1, ge=1)
    settings: Optional[Settings] = Field(default_factory=Settings)
