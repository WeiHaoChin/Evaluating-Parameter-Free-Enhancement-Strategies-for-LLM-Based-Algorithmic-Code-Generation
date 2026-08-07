from typing import Optional
from pydantic import BaseModel, Field
# 1. APPROACH: Brief explanation of your algorithm and why it's correct
# ── Schemas ────────────────────────────────────────────────────────────────────
class Settings(BaseModel):
    model: str                  = Field(default="gemma3:4b")
    systemPrompt: str           = Field(default="""You are an expert competitive programmer. Given a competitive programming 
problem, produce a correct and efficient solution.

Your response must follow this exact structure:
1. COMPLEXITY: Time and space complexity analysis
2. CODE: Complete, runnable solution in Python

Requirements:
- Handle all edge cases explicitly
- Ensure your solution fits within the given time and memory constraints
- Output only the final solution code block, no partial attempts
- Do not include test scaffolding or input parsing beyond what is needed""")
    temperature: float          = Field(default=0.0, ge=0.0, le=1.0)
    includeRag: bool            = Field(default=True)
    includeTextGrad: bool       = Field(default=True)
    textGradModel: str          = Field(default="gemma3:4b")
    textGradLoops: int          = Field(default=1, ge=1)
    textGradLossPrompt: str     = Field(default="""You are evaluating a competitive programming solution. Your feedback will 
be used to improve the prompt that generated this solution.

Evaluate the solution on these criteria:
1. CORRECTNESS: Does the logic handle all cases including edge cases?
2. COMPLEXITY: Is the time/space complexity optimal for the constraints?
3. COMPLETENESS: Is the solution fully implemented and runnable?
4. CLARITY: Is the approach clearly explained?

For each criterion, state:
- What the solution did well
- What specific weakness exists
- How the PROMPT (not the code) should be changed to elicit a better solution

Focus your feedback on prompt-level issues — e.g. "the prompt should instruct 
the model to explicitly consider overflow", not "the code has a bug on line 5". 
The goal is to improve the instruction, not patch the output directly.""")
    apiKey: Optional[str]       = Field(default=None)
    textGradApiKey: Optional[str] = Field(default=None)


class ChatRequest(BaseModel):
    message: str
    settings: Settings

class BenchmarkRequest(BaseModel):
    version: str = "release_v5"
    n: int = 30
    difficulty: Optional[str] = None
    settings: Optional[Settings] = Field(default_factory=Settings)
