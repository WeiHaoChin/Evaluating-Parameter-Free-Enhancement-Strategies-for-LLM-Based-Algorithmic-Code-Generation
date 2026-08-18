# benchmark/runner.py
import asyncio
import time
from typing import Optional
from .fetch_lcb import load_lcb_problems
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
}


def get_status() -> dict:
    """Get current benchmark status."""
    return benchmark_status.copy()


def request_stop() -> None:
    """Request the benchmark to stop."""
    global benchmark_status
    benchmark_status["stop_requested"] = True


def reset_stop_flag() -> None:
    """Reset the stop flag."""
    global benchmark_status
    benchmark_status["stop_requested"] = False


async def run_benchmark(
    version: str = "release_v5", 
    n: int = 30, 
    difficulty: Optional[str] = None,
    settings: Optional[Settings] = None
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

    problems = load_lcb_problems(version=version, n=n, difficulty=difficulty)
    benchmark_status["total"] = len(problems)

    all_results = []
    # Both RAG modes use the same problem statement. Keep the retrieval cache
    # local to this run so a later run always reflects the current knowledge
    # base, while each statement is embedded and searched at most once here.
    rag_context_cache: dict[str, str] = {}

    for problem_idx, problem in enumerate(problems):
        # Check if stop was requested
        if benchmark_status["stop_requested"]:
            print("Benchmark stop requested. Halting execution.")
            benchmark_status["running"] = False
            benchmark_status["stop_requested"] = False
            return {
                "results": all_results,
                "summary": compute_metrics(all_results) if all_results else {},
            }

        benchmark_status["current_problem"] = problem["title"]
        benchmark_status["progress"] = problem_idx + 1

        problem_results = {
            "problem": problem,
            "modes": {},
        }

        for mode in MODES:
            mode_label = mode["label"]
            try:
                start_time = time.time()
                response = await run_pipeline(
                    problem=problem["statement"],
                    evaluation_sample=problem["evaluation_sample"],
                    rag=mode["rag"],
                    textgrad=mode["textgrad"],
                    system_prompt=settings.systemPrompt,
                    model=settings.model,
                    textgrad_model=settings.textGradModel,
                    textgrad_loops=settings.textGradLoops,
                    textgrad_loss_prompt=settings.textGradLossPrompt,
                    api_key=settings.apiKey,
                    textgrad_api_key=settings.textGradApiKey,
                    starter_code=problem.get("starter_code"),
                    rag_context_cache=rag_context_cache,
                )
                latency = (time.time() - start_time) * 1000

                mode_result = {
                    **response,
                    "latency_ms": latency,
                    "textgrad_included": mode["textgrad"],
                }

                problem_results["modes"][mode_label] = mode_result

            except Exception as e:
                problem_results["modes"][mode_label] = {
                    "response": None,
                    "passed": False,
                    "pass_rate": 0.0,
                    "error_type": "EXCEPTION",
                    "system_prompt_used": None,
                    "latency_ms": 0,
                    "textgrad_included": mode["textgrad"],
                    "exception": str(e),
                }

        all_results.append(problem_results)
        await asyncio.sleep(0)

    summary = compute_metrics(all_results)

    benchmark_status["running"] = False

    return {
        "results": all_results,
        "summary": summary,
    }
