# pipeline/solver.py
import io
import logging
import re
import signal
import sys
import time
from typing import Optional

from TextGrad import run_textgrad_sync, OllamaLLM, GoogleGenerativeAI
from rag_handler import query_rag, format_rag_context, is_rag_available

logger = logging.getLogger(__name__)


# ── LLM caller ────────────────────────────────────────────────────────────────

def call_llm(
    message: str,
    system_prompt: str,
    model: str,
    api_key: Optional[str] = None,
) -> str:
    """Call LLM directly without TextGrad."""
    if model.startswith("gemini-"):
        llm = GoogleGenerativeAI(model=model, api_key=api_key)
    elif (
        model.startswith("gemma3:")
        or model.startswith("gpt-oss:")
        or model.startswith("deepseek-")
    ):
        llm = OllamaLLM(model=model, api_key=api_key)
    else:
        raise ValueError(f"Unsupported model type: {model}")
    return llm(message, system_prompt=system_prompt)


# ── Code extraction ────────────────────────────────────────────────────────────

def extract_code(response: str) -> str:
    """
    Extract Python code block from LLM response.
    Falls back to raw response if no code block found.
    """
    match = re.search(r"```python\s*(.*?)```", response, re.DOTALL)
    if match:
        return match.group(1).strip()
    match = re.search(r"```\s*(.*?)```", response, re.DOTALL)
    if match:
        return match.group(1).strip()
    return response.strip()


# ── Test executor ──────────────────────────────────────────────────────────────

def execute_against_tests(
    response: str,
    test_cases: list,
    timeout_seconds: int = 5,
) -> tuple[bool, float, Optional[str]]:
    """
    Run extracted code against test cases in a sandboxed exec.

    Returns:
        passed_all  (bool)          — True if all test cases passed
        pass_rate   (float)         — fraction of test cases passed (0.0–1.0)
        error_type  (str | None)    — "WA" | "TLE" | "RE" | "CE" | None
    """
    if not test_cases:
        return True, 1.0, None

    code = extract_code(response)

    # ── compile check ──
    try:
        compiled = compile(code, "<string>", "exec")
    except SyntaxError as e:
        logger.warning(f"CE: {e}")
        return False, 0.0, "CE"

    passed = 0
    last_error: Optional[str] = None

    for tc in test_cases:
        input_data = tc.get("input", "")
        expected   = str(tc.get("output", "")).strip()

        old_stdin  = sys.stdin
        old_stdout = sys.stdout

        try:
            def _timeout_handler(signum, frame):
                raise TimeoutError()

            signal.signal(signal.SIGALRM, _timeout_handler)
            signal.alarm(timeout_seconds)

            stdout_capture = io.StringIO()
            sys.stdin  = io.StringIO(input_data)
            sys.stdout = stdout_capture

            try:
                exec(compiled, {})
            finally:
                signal.alarm(0)
                sys.stdin  = old_stdin
                sys.stdout = old_stdout

            actual = stdout_capture.getvalue().strip()

            if actual == expected:
                passed += 1
            else:
                logger.debug(f"WA — expected: {expected!r}, got: {actual!r}")
                last_error = "WA"

        except TimeoutError:
            sys.stdin  = old_stdin
            sys.stdout = old_stdout
            logger.warning("TLE: test case exceeded time limit")
            last_error = "TLE"

        except Exception as e:
            sys.stdin  = old_stdin
            sys.stdout = old_stdout
            logger.warning(f"RE: {e}")
            last_error = "RE"

    pass_rate  = passed / len(test_cases)
    passed_all = passed == len(test_cases)
    return passed_all, pass_rate, (None if passed_all else last_error)


# ── RAG prompt builder ─────────────────────────────────────────────────────────

def build_rag_prompt(problem: str, rag_context: str) -> str:
    return f"""## Problem
{problem}

## Relevant Context (may or may not be useful)
{rag_context}

## Task
Solve the problem. Use the context above only if it is relevant."""


# ── Main pipeline ──────────────────────────────────────────────────────────────

async def run_pipeline(
    problem: str,
    test_cases: list,
    rag: bool,
    textgrad: bool,
    system_prompt: str,
    model: str,
    textgrad_model: str,
    textgrad_loops: int,
    textgrad_loss_prompt: str,
    api_key: Optional[str] = None,
    textgrad_api_key: Optional[str] = None,
) -> dict:
    """
    Full CP solver pipeline.

    Steps:
        1. RAG   — optionally augment the prompt with retrieved context
        2. LLM   — generate a solution (with or without TextGrad)
        3. Eval  — run the solution against test cases

    Returns a result dict with pass metrics, latency, and the raw response.
    """
    start = time.time()

    # ── Step 1: RAG augmentation ───────────────────────────────────────────────
    formatted_prompt = problem
    rag_context      = ""

    if rag:
        if is_rag_available():
            try:
                logger.info("Querying RAG...")
                rag_results = query_rag(problem, n_results=5)
                rag_context = format_rag_context(rag_results, include_metadata=True)
                if rag_context:
                    formatted_prompt = build_rag_prompt(problem, rag_context)
                    logger.info(f"RAG context added ({len(rag_context)} chars)")
                else:
                    logger.warning("RAG returned no context")
            except Exception as e:
                logger.error(f"RAG query failed: {e}", exc_info=True)
        else:
            logger.warning("RAG enabled but unavailable")

    # ── Step 2: Generate solution ──────────────────────────────────────────────
    response            = ""
    first_gen_passed    = False
    first_gen_pass_rate = 0.0
    second_gen_passed: Optional[bool] = None

    if textgrad:
        logger.info("Running TextGrad (first pass)...")
        try:
            response_first = run_textgrad_sync(
                prompt_text=formatted_prompt,
                system_prompt=system_prompt,
                loops=textgrad_loops,
                model=model,
                textGradModel=textgrad_model,
                api_key=textgrad_api_key,
                loss_prompt=textgrad_loss_prompt,
            )
            first_gen_passed, first_gen_pass_rate, _ = execute_against_tests(
                response_first, test_cases
            )

            logger.info("Running TextGrad (second pass with updated prompt)...")
            response = run_textgrad_sync(
                prompt_text=formatted_prompt,
                system_prompt=system_prompt,
                loops=textgrad_loops,
                model=model,
                textGradModel=textgrad_model,
                api_key=textgrad_api_key,
                loss_prompt=textgrad_loss_prompt,
            )
            second_gen_passed, _, _ = execute_against_tests(response, test_cases)

        except Exception as e:
            logger.error(f"TextGrad failed: {e}", exc_info=True)
            raise

    else:
        logger.info(f"Calling LLM directly (model={model})...")
        try:
            response = call_llm(
                message=formatted_prompt,
                system_prompt=system_prompt,
                model=model,
                api_key=api_key,
            )
            first_gen_passed, first_gen_pass_rate, _ = execute_against_tests(
                response, test_cases
            )
        except Exception as e:
            logger.error(f"LLM call failed: {e}", exc_info=True)
            raise

    # ── Step 3: Final evaluation ───────────────────────────────────────────────
    passed, pass_rate, error_type = execute_against_tests(response, test_cases)
    latency_ms = int((time.time() - start) * 1000)

    logger.info(
        f"Pipeline done — passed={passed}, pass_rate={pass_rate:.2f}, "
        f"latency={latency_ms}ms, error={error_type}"
    )

    return {
        "response":             response,
        "passed":               passed,
        "pass_rate":            pass_rate,
        "first_gen_passed":     first_gen_passed,
        "first_gen_pass_rate":  first_gen_pass_rate,
        "second_gen_passed":    second_gen_passed,
        "error_type":           error_type,
        "rag_context_included": bool(rag_context),
        "latency_ms":           latency_ms,
    }