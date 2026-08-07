# routes/benchmark.py
import asyncio

from fastapi import APIRouter, BackgroundTasks, HTTPException
from benchmark.runner import run_benchmark, get_status, request_stop
from benchmark.logger import save_results, load_latest_results, load_all_results
from benchmark.prefetch_lcb import is_prefetched, prefetch
from schemas import Settings, BenchmarkRequest

router = APIRouter(prefix="/benchmark", tags=["benchmark"])

_download_status = {"running": False, "version": None, "error": None}


def dataset_status(version: str = "release_v5") -> dict:
    downloading_this_version = (
        _download_status["running"] and _download_status["version"] == version
    )
    return {
        "version": version,
        # Avoid competing with Hugging Face's cache lock while a download runs.
        "available": False if downloading_this_version else is_prefetched(version),
        "downloading": downloading_this_version,
        "download_version": _download_status["version"],
        "error": _download_status["error"],
    }


async def _prefetch_dataset(version: str) -> None:
    try:
        await asyncio.to_thread(prefetch, version)
        _download_status["error"] = None
    except Exception as exc:
        _download_status["error"] = str(exc)
    finally:
        _download_status["running"] = False


@router.get("/dataset/status")
async def get_dataset_status(version: str = "release_v5") -> dict:
    return dataset_status(version)


@router.post("/dataset/download")
async def download_dataset(version: str = "release_v5") -> dict:
    if _download_status["running"] or is_prefetched(version):
        return dataset_status(version)

    _download_status.update({"running": True, "version": version, "error": None})
    asyncio.create_task(_prefetch_dataset(version))
    return dataset_status(version)

@router.post("/run")
async def start_benchmark(
    request: BenchmarkRequest, background_tasks: BackgroundTasks
) -> dict:
    """
    Start a benchmark run in the background.
    """
    if get_status()["running"]:
        raise HTTPException(status_code=409, detail="Benchmark already running")
    if not is_prefetched(request.version):
        raise HTTPException(
            status_code=409,
            detail=(f"LiveCodeBench {request.version} has not been downloaded. "
                    "Download it from Settings before starting a benchmark."),
        )
    # print(f"Starting benchmark with version={request.version}, n={request.n}, difficulty={request.difficulty}, settings={request.settings.dict()}")
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

@router.post("/stop")
async def stop_benchmark() -> dict:
    """
    Request the benchmark to stop.
    """
    status = get_status()
    if not status["running"]:
        raise HTTPException(status_code=400, detail="No benchmark is currently running")
    
    request_stop()
    return {
        "status": "stop_requested",
        "message": "Benchmark stop requested"
    }

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


@router.get("/defaults")
async def get_default_settings() -> dict:
    """
    Get default settings from schemas.Settings
    """
    return Settings().dict()
