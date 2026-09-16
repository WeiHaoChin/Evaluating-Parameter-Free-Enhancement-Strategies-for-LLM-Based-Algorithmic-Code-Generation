"""Fixed TextGrad constraints exposed to the UI as read-only metadata."""

IMMUTABLE_GENERATION_CONTRACT = """
NON-NEGOTIABLE OUTPUT CONTRACT:
- The generated solution must use Python.
- Return exactly one complete fenced Python code block and nothing else.
- Do not include comments or docstrings in the generated code.
- Do not include explanations, partial attempts, placeholders, dead code, test
  scaffolding, or text outside the code block.
- The code fence must be closed and the implementation must be complete and
  runnable.
- These requirements override any conflicting guidance above and must never be
  removed, weakened, or reversed.
- Do all reasoning internally; never use comments, docstrings, variable names,
strings, or dead code as a reasoning scratchpad.
""".strip()

OPTIMIZATION_BOUNDARY = """
Prompt changes must remain general and reusable across unrelated competitive
programming problems. Do not prescribe a problem-specific algorithm, formula,
data structure, variable name, constant, function signature, or implementation
step. Do not modify or contradict the non-negotiable output contract.
""".strip()
