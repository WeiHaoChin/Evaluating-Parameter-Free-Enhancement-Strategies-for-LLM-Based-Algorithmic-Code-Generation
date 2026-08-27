# benchmark/runner.py
import asyncio
import time
from typing import Callable, Optional
from .fetch_lcb import (
    load_lcb_problem,
    load_lcb_problems,
    select_lcb_problem_ids,
)
from .prefetch_lcb import is_parquet_prefetched
from .metrics import compute_metrics
from schemas import Settings
from solver import run_pipeline


MODES = [
    {"rag": False, "textgrad": False, "label": "baseline"},
    {"rag": True, "textgrad": False, "label": "rag_only"},
    {"rag": False, "textgrad": True, "label": "textgrad_only"},
    {"rag": True, "textgrad": True, "label": "full"},
]

benchmark_status = {
    "running": False,
    "progress": 0,
    "total": 0,
    "current_problem": "",
    "stop_requested": False,
    "modes": {},
}


def get_status() -> dict:
    """Get current benchmark status."""
    status = benchmark_status.copy()
    status["modes"] = {
        label: mode.copy() for label, mode in benchmark_status.get("modes", {}).items()
    }
    return status


def _fresh_mode_statuses() -> dict:
    return {
        mode["label"]: {
            "label": mode["label"], "state": "pending", "detail": "Waiting to start",
        }
        for mode in MODES
    }


def request_stop() -> None:
    """Request the benchmark to stop."""
    global benchmark_status
    benchmark_status["stop_requested"] = True


def reset_stop_flag() -> None:
    """Reset the stop flag."""
    global benchmark_status
    benchmark_status["stop_requested"] = False


async def run_benchmark(
    version: str = "release_v6",
    n: int = 30,
    difficulty: Optional[str] = None,
    seed: int = 42,
    settings: Optional[Settings] = None,
    start_question: int = 1,
    checkpoint_callback: Optional[
        Callable[[list[dict], dict, dict], None]
    ] = None,
) -> dict:
    """
    Run benchmark across all 4 modes.

    Args:
        version: LiveCodeBench dataset version
        n: Number of problems to benchmark
        difficulty: Filter by difficulty
        settings: Settings instance with model and prompt config (from settings.js)

    Returns:
        Dict with results for all problems and all modes
    """
    global benchmark_status
    
    # Use default settings if none provided
    if settings is None:
        settings = Settings()
    print(f"Running benchmark with settings: {settings.dict()} in runner")
    reset_stop_flag()
    benchmark_status["running"] = True
    benchmark_status["progress"] = 0
    benchmark_status["current_problem"] = "Loading dataset..."
    benchmark_status["modes"] = _fresh_mode_statuses()

    # Loading a multi-gigabyte Arrow split is blocking work. Keep it off the
    # FastAPI event loop so status/readiness endpoints remain responsive.
    parquet_mode = is_parquet_prefetched(version)
    if parquet_mode:
        selected = await asyncio.to_thread(
            select_lcb_problem_ids, version, n, difficulty, seed
        )
    else:
        selected = await asyncio.to_thread(
            load_lcb_problems,
            version=version,
            n=n,
            difficulty=difficulty,
            seed=seed,
        )
    # The seed defines one stable ordered sample. Starting at question N skips
    # the first N-1 entries from that same sample, which supports manual resume.
    selected = selected[max(0, start_question - 1):]
    benchmark_status["total"] = len(selected)

    all_results = []
    # Both RAG modes use the same problem statement. Keep the retrieval cache
    # local to this run so a later run always reflects the current knowledge
    # base, while each statement is embedded and searched at most once here.
    rag_context_cache: dict[str, dict] = {}

    for problem_idx, selected_problem in enumerate(selected):
        # Check if stop was requested
        if benchmark_status["stop_requested"]:
            print("Benchmark stop requested. Halting execution.")
            benchmark_status["running"] = False
            benchmark_status["stop_requested"] = False
            return {
                "results": all_results,
                "summary": compute_metrics(all_results) if all_results else {},
            }

        problem = (
            await asyncio.to_thread(load_lcb_problem, version, selected_problem)
            if parquet_mode else selected_problem
        )
        benchmark_status["current_problem"] = problem["title"]
        benchmark_status["progress"] = problem_idx
        benchmark_status["modes"] = _fresh_mode_statuses()

        # Evaluation samples and decoded tests are intentionally not retained
        # in accumulated results, allowing their memory to be reclaimed after
        # each problem finishes.
        result_problem = {
            key: value for key, value in problem.items()
            if key not in {"evaluation_sample", "private_tests", "test_cases"}
        }
        problem_results = {
            "problem": result_problem,
            "modes": {},
        }

        async def run_mode(mode: dict, initial_response: Optional[str] = None) -> tuple[str, dict]:
            """Run one blocking solver invocation in a worker thread."""
            mode_label = mode["label"]
            def report(state: str, detail: str) -> None:
                benchmark_status["modes"][mode_label].update({
                    "state": state, "detail": detail,
                })
            try:
                start_time = time.perf_counter()
                response = await asyncio.to_thread(
                    run_pipeline,
                    problem=problem["statement"],
                    evaluation_sample=problem["evaluation_sample"],
                    rag=mode["rag"],
                    textgrad=mode["textgrad"],
                    system_prompt=settings.systemPrompt,
                    model=settings.model,
                    temperature=settings.temperature,
                    max_output_tokens=settings.maxOutputTokens,
                    textgrad_internal_max_output_tokens=settings.textGradInternalMaxOutputTokens,
                    mode=mode_label,
                    textgrad_model=settings.textGradModel,
                    textgrad_loops=settings.textGradLoops,
                    textgrad_loss_prompt=settings.textGradLossPrompt,
                    api_key=settings.apiKey,
                    textgrad_api_key=settings.textGradApiKey,
                    starter_code=problem.get("starter_code"),
                    rag_context_cache=rag_context_cache,
                    initial_response=initial_response,
                    progress_callback=report,
                )
                elapsed_ms = (time.perf_counter() - start_time) * 1000

                mode_result = {**response, "textgrad_included": mode["textgrad"]}
                mode_result.setdefault("latency_ms", elapsed_ms)
                report("complete", "Completed")
            except Exception as e:
                mode_result = {
                    "response": None,
                    "passed": False,
                    "pass_rate": 0.0,
                    "error_type": "EXCEPTION",
                    "system_prompt_used": None,
                    "rag_retrieved_data": [],
                    "textgrad_improved_system_prompt": None,
                    "latency_ms": (time.perf_counter() - start_time) * 1000,
                    "textgrad_included": mode["textgrad"],
                    "exception": str(e),
                }
                report("error", str(e))
            return mode_label, mode_result

        # The independent initial generations start together. They are then
        # reused as the first TextGrad answer for the matching prompt: baseline
        # for textgrad_only and RAG for full. This removes two main-model calls
        # per problem while retaining a fair RAG + TextGrad comparison.
        baseline_task = asyncio.create_task(run_mode(MODES[0]))
        rag_task = asyncio.create_task(run_mode(MODES[1]))
        initial_modes = await asyncio.gather(baseline_task, rag_task)
        problem_results["modes"].update(dict(initial_modes))

        textgrad_task = asyncio.create_task(
            run_mode(MODES[2], problem_results["modes"]["baseline"].get("response"))
        )
        full_task = asyncio.create_task(
            run_mode(MODES[3], problem_results["modes"]["rag_only"].get("response"))
        )
        refined_modes = await asyncio.gather(textgrad_task, full_task)
        problem_results["modes"].update(dict(refined_modes))

        all_results.append(problem_results)
        # The cache is only shared by this problem's RAG and combined modes.
        # Releasing it here prevents retrieved documents accumulating across a
        # long benchmark while preserving the one-retrieval-per-problem reuse.
        rag_context_cache.pop(problem["statement"], None)
        benchmark_status["progress"] = problem_idx + 1
        if checkpoint_callback is not None:
            # All four modes for this problem are now complete. Persist one
            # recoverable checkpoint before moving to the next problem.
            partial_summary = compute_metrics(all_results)
            await asyncio.to_thread(
                checkpoint_callback, all_results, partial_summary, problem
            )
        await asyncio.sleep(0)

    summary = compute_metrics(all_results)

    benchmark_status["running"] = False

    return {
        "results": all_results,
        "summary": summary,
    }
