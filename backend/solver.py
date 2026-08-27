# pipeline/solver.py
import io
import json
import logging
import re
import sys
import time
from typing import Callable, MutableMapping, Optional
import threading

from TextGrad import run_textgrad_sync
from llm_clients import create_llm_client
from config.generation import (
    DEFAULT_MAX_OUTPUT_TOKENS,
    DEFAULT_TEXTGRAD_INTERNAL_MAX_OUTPUT_TOKENS,
)
from rag_handler import query_rag, format_rag_context, is_rag_available
from lcb_runner.evaluation.compute_code_generation_metrics import codegen_metrics

logger = logging.getLogger(__name__)

RESULT_CODE_LABELS = {
    -2: "WRONG_ANSWER",
    -3: "TIME_LIMIT_EXCEEDED",
    -4: "RUNTIME_ERROR",
}


def summarize_test_outcomes(test_results: list) -> tuple[int, dict[str, int], Optional[str]]:
    """Count evaluator outcomes and choose a primary problem-level error."""
    counts = {
        "PASSED": 0,
        "WRONG_ANSWER": 0,
        "TIME_LIMIT_EXCEEDED": 0,
        "RUNTIME_ERROR": 0,
    }
    for result in test_results:
        if result is True:
            counts["PASSED"] += 1
        elif result is False or result == -2:
            counts["WRONG_ANSWER"] += 1
        elif result in RESULT_CODE_LABELS:
            counts[RESULT_CODE_LABELS[result]] += 1
        else:
            counts["RUNTIME_ERROR"] += 1

    # Prefer the most operationally severe observed failure for the legacy
    # single error_type field. Full detail remains available in outcome counts.
    primary_error = next((
        label for label in (
            "RUNTIME_ERROR", "TIME_LIMIT_EXCEEDED", "WRONG_ANSWER"
        ) if counts[label] > 0
    ), None)
    return counts["PASSED"], counts, primary_error


# ── LLM caller ────────────────────────────────────────────────────────────────

def call_llm(
    message: str,
    system_prompt: str,
    model: str,
    api_key: Optional[str] = None,
    temperature: float = 0.0,
    max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
    generation_records: Optional[list[dict]] = None,
    mode: Optional[str] = None,
    call_type: str = "final_generation",
) -> str:
    """Call LLM directly without TextGrad."""
    llm = create_llm_client(
        model=model, api_key=api_key, temperature=temperature,
        max_output_tokens=max_output_tokens, metadata_sink=generation_records,
        mode=mode, call_type=call_type,
    )
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

def _run_in_thread(compiled, input_data: str, timeout_seconds: int) -> tuple[Optional[str], Optional[Exception]]:
    """
    Execute compiled code in an isolated thread with its own stdin/stdout.
    Returns (stdout_output, exception_or_None).
    """
    stdout_capture = io.StringIO()
    exc_holder: list[Optional[Exception]] = [None]

    def target():
        local_stdin  = io.StringIO(input_data)
        local_stdout = io.StringIO()

        old_stdin  = sys.stdin
        old_stdout = sys.stdout
        sys.stdin  = local_stdin
        sys.stdout = local_stdout

        try:
            exec(compiled, {"__builtins__": __builtins__, "__name__": "__main__"})
        except Exception as e:
            exc_holder[0] = e
        finally:
            sys.stdin  = old_stdin
            sys.stdout = old_stdout
            stdout_capture.write(local_stdout.getvalue())

    t = threading.Thread(target=target, daemon=True)
    t.start()
    t.join(timeout_seconds)

    if t.is_alive():
        return None, TimeoutError(f"Exceeded {timeout_seconds}s")

    return stdout_capture.getvalue(), exc_holder[0]


def execute_against_tests(
    response: str,
    test_cases: list,
    timeout_seconds: int = 5,
) -> tuple[bool, float, Optional[str]]:
    """
    Run extracted code against test cases in an isolated thread.

    Returns:
        passed_all  (bool)   — True if all test cases passed
        pass_rate   (float)  — fraction of test cases passed (0.0–1.0)
        error_type  (str)    — "WA" | "TLE" | "RE" | "CE" | None
    """
    if not test_cases:
        return True, 1.0, None

    code = extract_code(response)

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

        output, exc = _run_in_thread(compiled, input_data, timeout_seconds)

        if isinstance(exc, TimeoutError):
            logger.warning("TLE: test case exceeded time limit")
            last_error = "TLE"
            continue

        if exc is not None:
            logger.warning(f"RE: {exc}")
            last_error = "RE"
            continue

        actual = (output or "").strip()
        if actual == expected:
            passed += 1
        else:
            logger.warning(f"WA — expected: {expected!r}, got: {actual!r}")
            last_error = "WA"

    pass_rate  = passed / len(test_cases)
    passed_all = passed == len(test_cases)
    return passed_all, pass_rate, (None if passed_all else last_error)


# ── RAG prompt builder ─────────────────────────────────────────────────────────

def build_rag_prompt(task_prompt: str, rag_context: str) -> str:
    """Add retrieval context without dropping the problem's I/O contract."""
    return f"""{task_prompt}

## Relevant Context (may or may not be useful)
{rag_context}

## Task
Solve the problem. Use the context above only if it is relevant."""

def build_task_prompt(problem: str, starter_code: Optional[str]) -> str:
    """Frame the problem the way LCB does: give the model the exact
    function/class skeleton for call-based problems, or explicit
    stdin/stdout instructions otherwise."""
    if starter_code:
        return (
            f"## Problem\n{problem}\n\n"
            "## Starter Code\n"
            "You will use the following starter code to write your solution. "
            "Implement your logic inside it and do not change the signature.\n"
            f"```python\n{starter_code}\n```"
        )
    else:
        return (
            f"## Problem\n{problem}\n\n"
            "## I/O\n"
            "Read all input from stdin and write your final answer to stdout. "
            "Do not define a class or function — write a script that runs "
            "directly when executed."
        )


# ── Main pipeline ──────────────────────────────────────────────────────────────

def run_pipeline(
    problem: str,
    evaluation_sample: list,
    rag: bool,
    textgrad: bool,
    system_prompt: str,
    model: str,
    textgrad_model: str,
    textgrad_loops: int,
    textgrad_loss_prompt: str,
    api_key: Optional[str] = None,
    textgrad_api_key: Optional[str] = None,
    temperature: float = 0.0,
    starter_code: Optional[str] = None,
    rag_context_cache: Optional[MutableMapping[str, dict]] = None,
    initial_response: Optional[str] = None,
    progress_callback: Optional[Callable[[str, str], None]] = None,
    max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
    textgrad_internal_max_output_tokens: int = DEFAULT_TEXTGRAD_INTERNAL_MAX_OUTPUT_TOKENS,
    mode: Optional[str] = None,
) -> dict:
    """
    Full CP solver pipeline.

    Steps:
        1. RAG   — optionally augment the prompt with retrieved context
        2. LLM   — generate a solution (with or without TextGrad)
        3. Eval  — run the solution against test cases

    Returns a result dict with pass metrics, latency, and the raw response.
    """
    pipeline_started = time.perf_counter()
    generation_records: list[dict] = []
    experiment_mode = mode or (
        "full" if rag and textgrad else
        "rag_only" if rag else
        "textgrad_only" if textgrad else
        "baseline"
    )

    def report(state: str, detail: str) -> None:
        if progress_callback:
            progress_callback(state, detail)

    formatted_prompt = build_task_prompt(problem, starter_code)

    # ── Step 1: RAG augmentation ───────────────────────────────────────────────
    rag_context = ""
    rag_results = []
    retrieval_duration_ms = 0.0
    retrieval_cache_hit = False

    if rag:
        retrieval_started = time.perf_counter()
        report("retrieving", "Retrieving relevant RAG context")
        if is_rag_available():
            try:
                if rag_context_cache is not None and problem in rag_context_cache:
                    retrieval_cache_hit = True
                    cached_rag = rag_context_cache[problem]
                    rag_context = cached_rag.get("context", "")
                    rag_results = cached_rag.get("results", [])
                    logger.info("Using cached RAG context for benchmark problem")
                else:
                    logger.info("Querying RAG...")
                    rag_results = query_rag(problem, n_results=5)
                    rag_context = format_rag_context(rag_results, include_metadata=True)
                    # Cache empty results too, otherwise an unsuccessful query
                    # would be repeated by the next RAG benchmark mode.
                    if rag_context_cache is not None:
                        rag_context_cache[problem] = {
                            "context": rag_context,
                            "results": rag_results,
                        }
                if rag_context:
                    formatted_prompt = build_rag_prompt(formatted_prompt, rag_context)
                    logger.info(f"RAG context added ({len(rag_context)} chars)")
                else:
                    logger.warning("RAG returned no context")
            except Exception as e:
                logger.error(f"RAG query failed: {e}", exc_info=True)
        else:
            logger.warning("RAG enabled but unavailable")
        retrieval_duration_ms = (time.perf_counter() - retrieval_started) * 1000

    # ── Step 2: Generate solution ──────────────────────────────────────────────
    response = ""
    improved_system_prompt = None
    generation_started = time.perf_counter()

    if textgrad:
        logger.info("Running TextGrad...")
        try:
            response, improved_system_prompt = run_textgrad_sync(
                prompt_text=formatted_prompt,
                system_prompt=system_prompt,
                loops=textgrad_loops,
                model=model,
                textGradModel=textgrad_model,
                api_key=api_key,
                textGrad_api_key=textgrad_api_key,
                loss_prompt=textgrad_loss_prompt,
                temperature=temperature,
                initial_answer=initial_response,
                progress_callback=progress_callback,
                return_details=True,
                max_output_tokens=max_output_tokens,
                internal_max_output_tokens=textgrad_internal_max_output_tokens,
                generation_records=generation_records,
                mode=experiment_mode,
            )
        except Exception as e:
            logger.error(f"TextGrad failed: {e}", exc_info=True)
            raise

    else:
        logger.info(f"Calling LLM directly (model={model})...")
        report("generating", "Generating solution")
        try:
            response = call_llm(
                message=formatted_prompt,
                system_prompt=system_prompt,
                model=model,
                api_key=api_key,
                temperature=temperature,
                max_output_tokens=max_output_tokens,
                generation_records=generation_records,
                mode=experiment_mode,
                call_type="final_generation",
            )
        except Exception as e:
            logger.error(f"LLM call failed: {e}", exc_info=True)
            raise

    generation_duration_ms = (time.perf_counter() - generation_started) * 1000

    # ── Step 3: Final evaluation ───────────────────────────────────────────────
    # test_data = {
    # "inputs": [t[0] for t in test_cases] if isinstance(test_cases[0], (tuple, list)) else test_cases.get("inputs", []),
    # "outputs": [t[1] for t in test_cases] if isinstance(test_cases[0], (tuple, list)) else test_cases.get("outputs", []),
    # "fn_name": test_cases.get("fn_name", None)  # Competitive programming problems use standard I/O streams, so fn_name is None
    # }
    evaluation_sample = [evaluation_sample] # evaluation_sample passed into run_pipeline
    generated_code_snippets = [[extract_code(response)]]

    report("judging", "Running generated code against test cases")
    judging_started = time.perf_counter()
    metrics, results, final_metadata = codegen_metrics(
        evaluation_sample,
        generated_code_snippets,
        k_list=[1],
        num_process_evaluate=1,  # Windows-safe
        timeout=10,
    )
    parsed_metadata = []
    for item in final_metadata:
        cleaned_list = []
        for x in item:
            if isinstance(x, str):
                try:
                    cleaned_list.append(json.loads(x))
                except Exception:
                    cleaned_list.append(x)
            elif isinstance(x, dict):
                # If it's already a dictionary (like the WA error frame), 
                # keep it as a dict so it passes through safely!
                cleaned_list.append(x)
            else:
                cleaned_list.append(x)
        parsed_metadata.append(cleaned_list)
    logger.debug("Evaluation complete: metrics=%s, result_groups=%d", metrics, len(results))
    test_results = results[0][0]
    pass_count, test_outcome_counts, error_type = summarize_test_outcomes(test_results)
    pass_rate = pass_count / len(test_results) if test_results else 0.0
    passed_all = all(r is True for r in test_results) if test_results else False
    graded     = results[0] if results else []  # results is a dict keyed by problem index
    judging_duration_ms = (time.perf_counter() - judging_started) * 1000
    latency_ms = (time.perf_counter() - pipeline_started) * 1000

    logger.info(
        f"Pipeline done — passed={passed_all}, pass_rate={pass_rate:.2f}, "
        f"latency={latency_ms}ms, error={error_type}"
    )

    return {
        "response":             response,
        "generation_records":    generation_records,
        "generated_code":       generated_code_snippets,    
        "passed":               passed_all,
        "pass_rate":            pass_rate,
        "passed_tests":         pass_count,
        "total_tests":          len(test_results),
        "graded_list":          graded,
        "error_type":           error_type,
        "test_outcome_counts":  test_outcome_counts,
        "rag_context_included": bool(rag_context),
        "rag_retrieved_data":  rag_results,
        "textgrad_improved_system_prompt": improved_system_prompt,
        "latency_ms":           latency_ms,
        "timings": {
            "end_to_end_ms": latency_ms,
            "retrieval_ms": retrieval_duration_ms,
            "retrieval_cache_hit": retrieval_cache_hit,
            "generation_workflow_ms": generation_duration_ms,
            "judging_ms": judging_duration_ms,
        },
        "metrics":               metrics,
        "metadata":              parsed_metadata,
    }
