# routes/benchmark.py
import asyncio

from fastapi import APIRouter, BackgroundTasks, HTTPException
from benchmark.runner import run_benchmark, get_status, request_stop
from benchmark.logger import save_results, load_latest_results, load_all_results
from benchmark.prefetch_lcb import is_prefetched, prefetch
from schemas import Settings, BenchmarkRequest, settings_defaults
from rag_handler import is_rag_available

router = APIRouter(prefix="/benchmark", tags=["benchmark"])

_download_status = {
    "running": False, "version": None, "error": None, "percent": 0,
    "completed_files": 0, "total_files": 0, "downloaded_bytes": 0,
    "total_bytes": 0, "message": "",
}


def dataset_status(version: str = "release_v6") -> dict:
    downloading_this_version = (
        _download_status["running"] and _download_status["version"] == version
    )
    available = False if downloading_this_version else is_prefetched(version)
    progress = {key: _download_status[key] for key in (
        "percent", "completed_files", "total_files", "downloaded_bytes",
        "total_bytes", "message",
    )}
    if available:
        progress.update({"percent": 100, "message": "Dataset downloaded and ready."})
    return {
        "version": version,
        # Avoid competing with Hugging Face's cache lock while a download runs.
        "available": available,
        "downloading": downloading_this_version,
        "download_version": _download_status["version"],
        "error": _download_status["error"],
        **progress,
    }


async def _prefetch_dataset(version: str) -> None:
    def report(completed: int, total: int, downloaded: int, total_bytes: int) -> None:
        percent = round((completed / total) * 100) if total else 0
        _download_status.update({
            "completed_files": completed, "total_files": total,
            "downloaded_bytes": downloaded, "total_bytes": total_bytes,
            "percent": percent,
            "message": f"Downloaded {completed} of {total} Parquet shards.",
        })
    try:
        await asyncio.to_thread(prefetch, version, report)
        _download_status["error"] = None
        _download_status.update({"percent": 100, "message": "Dataset downloaded and ready."})
    except Exception as exc:
        _download_status["error"] = str(exc)
    finally:
        _download_status["running"] = False


@router.get("/dataset/status")
async def get_dataset_status(version: str = "release_v6") -> dict:
    return dataset_status(version)


def benchmark_readiness(settings: Settings, version: str) -> dict:
    """Return the prerequisites needed for a meaningful full-pipeline run."""
    def api_key_ready(model: str, api_key: str | None) -> bool:
        """Keep benchmark validation in step with Settings' API-key behaviour."""
        return bool(model and (model == "mock-chat:1.0" or (api_key and api_key.strip())))

    checks = {
        "dataset": {
            "ready": is_prefetched(version),
            "label": "LiveCodeBench dataset",
            "detail": "Download the selected dataset from Settings.",
        },
        "rag": {
            "ready": settings.includeRag and is_rag_available(),
            "label": "RAG knowledge base",
            "detail": "Enable RAG in Settings and make sure the knowledge base is available.",
        },
        "llm_api": {
            "ready": api_key_ready(settings.model, settings.apiKey),
            "label": "Initial LLM model & API key",
            "detail": "Select an initial LLM and add its API key in Settings.",
        },
        "textgrad_api": {
            "ready": bool(
                settings.includeTextGrad
                and settings.textGradModel
                and api_key_ready(settings.textGradModel, settings.textGradApiKey)
            ),
            "label": "TextGrad model & API key",
            "detail": "Enable TextGrad, select its model, and add its API key in Settings.",
        },
    }
    return {"ready": all(check["ready"] for check in checks.values()), "checks": checks}


@router.post("/readiness")
async def get_benchmark_readiness(request: BenchmarkRequest) -> dict:
    return benchmark_readiness(request.settings or Settings(), request.version)


@router.post("/dataset/download")
async def download_dataset(version: str = "release_v6") -> dict:
    if _download_status["running"] or is_prefetched(version):
        return dataset_status(version)

    _download_status.update({
        "running": True, "version": version, "error": None, "percent": 0,
        "completed_files": 0, "total_files": 0, "downloaded_bytes": 0,
        "total_bytes": 0, "message": "Discovering Parquet shards...",
    })
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
    readiness = benchmark_readiness(request.settings or Settings(), request.version)
    if not readiness["ready"]:
        missing = [check["label"] for check in readiness["checks"].values() if not check["ready"]]
        raise HTTPException(
            status_code=422,
            detail=f"Benchmark setup is incomplete: {', '.join(missing)}.",
        )
    # print(f"Starting benchmark with version={request.version}, n={request.n}, difficulty={request.difficulty}, settings={request.settings.dict()}")
    async def run_and_save() -> None:
        result = await run_benchmark(
            version=request.version,
            n=request.n,
            difficulty=request.difficulty,
            seed=request.seed,
            settings=request.settings,
        )
        settings = request.settings or Settings()
        save_results(result["results"], result["summary"], settings.dict())

    background_tasks.add_task(run_and_save)

    return {
        "status": "started",
        "n_problems": request.n,
        "version": request.version,
        "seed": request.seed,
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
    return settings_defaults()
