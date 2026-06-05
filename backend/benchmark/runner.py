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
}


def get_status() -> dict:
    """Get current benchmark status."""
    return benchmark_status.copy()


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

    benchmark_status["running"] = True
    benchmark_status["progress"] = 0

    problems = load_lcb_problems(version=version, n=n, difficulty=difficulty)
    benchmark_status["total"] = len(problems)

    all_results = []

    for problem_idx, problem in enumerate(problems):
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
                    test_cases=problem["test_cases"],
                    rag=mode["rag"],
                    textgrad=mode["textgrad"],
                    system_prompt=settings.systemPrompt,
                    model=settings.model,
                    textgrad_model=settings.textGradModel,
                    textgrad_loops=settings.textGradLoops,
                    textgrad_loss_prompt=settings.textGradLossPrompt,
                    api_key=settings.apiKey,
                    textgrad_api_key=settings.textGradApiKey,
                )
                latency = (time.time() - start_time) * 1000

                mode_result = {
                    **response,
                    "latency_ms": latency,
                }

                problem_results["modes"][mode_label] = mode_result

            except Exception as e:
                problem_results["modes"][mode_label] = {
                    "response": None,
                    "passed": False,
                    "pass_rate": 0.0,
                    "first_gen_passed": False,
                    "second_gen_passed": False,
                    "error_type": "EXCEPTION",
                    "system_prompt_used": None,
                    "latency_ms": 0,
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
