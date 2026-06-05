# routes/benchmark.py
from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel, Field
from typing import Optional
from benchmark.runner import run_benchmark, get_status
from benchmark.logger import save_results, load_latest_results, load_all_results
from schemas import Settings

router = APIRouter(prefix="/benchmark", tags=["benchmark"])


class BenchmarkRequest(BaseModel):
    version: str = "release_v5"
    n: int = 30
    difficulty: Optional[str] = None
    settings: Optional[Settings] = Field(default_factory=Settings)


@router.post("/run")
async def start_benchmark(
    request: BenchmarkRequest, background_tasks: BackgroundTasks
) -> dict:
    """
    Start a benchmark run in the background.
    """
    if get_status()["running"]:
        raise HTTPException(status_code=409, detail="Benchmark already running")

    async def run_and_save() -> None:
        result = await run_benchmark(
            version=request.version,
            n=request.n,
            difficulty=request.difficulty,
            settings=request.settings,
        )
        save_results(result["results"], result["summary"])

    background_tasks.add_task(run_and_save)

    return {
        "status": "started",
        "n_problems": request.n,
        "version": request.version,
        "settings": request.settings.dict() if request.settings else Settings().dict(),
    }


@router.get("/status")
async def get_benchmark_status() -> dict:
    """
    Get current benchmark status.
    """
    return get_status()

@router.get("/results")
async def get_results() -> dict:
    """
    Get latest benchmark results.
    """
    latest = load_latest_results()
    if latest is None:
        raise HTTPException(status_code=404, detail="No results found")
    return latest


@router.get("/results/all")
async def get_all_results() -> list[dict]:
    """
    Get all historical benchmark results.
    """
    return load_all_results()
