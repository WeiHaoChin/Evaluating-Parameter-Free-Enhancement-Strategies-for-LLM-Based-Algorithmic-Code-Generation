# benchmark/logger.py
import json
from pathlib import Path
from datetime import datetime
from typing import Any, Optional

RESULTS_DIR = Path(__file__).parent / "results"


def _ensure_results_dir() -> None:
    """Ensure results directory exists."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def save_results(
    results: list[dict], summary: dict, settings: Optional[dict[str, Any]] = None
) -> None:
    """
    Save benchmark results to JSON file.

    Args:
        results: List of problem results
        summary: Summary metrics
    """
    _ensure_results_dir()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = RESULTS_DIR / f"{timestamp}.json"

    data: dict[str, Any] = {
        "schema_version": 2,
        "timestamp": datetime.now().isoformat(),
        "results": results,
        "summary": summary,
    }
    if settings is not None:
        # Keep the configuration that produced the run, including the
        # includeTextGrad setting from the UI.
        data["settings"] = settings

    with open(filename, "w") as f:
        json.dump(data, f, indent=2, default=str)


def load_latest_results() -> Optional[dict]:
    """
    Load the most recent results file.

    Returns:
        Dict with results and summary, or None if no results exist
    """
    _ensure_results_dir()

    result_files = sorted(RESULTS_DIR.glob("*.json"), reverse=True)
    if not result_files:
        return None

    latest = result_files[0]
    with open(latest, "r") as f:
        return _normalise_loaded_results(json.load(f))


def load_all_results() -> list[dict]:
    """
    Load all historical results files.

    Returns:
        List of result dicts from all files, newest first
    """
    _ensure_results_dir()

    result_files = sorted(RESULTS_DIR.glob("*.json"), reverse=True)
    all_results = []

    for file in result_files:
        with open(file, "r") as f:
            all_results.append(_normalise_loaded_results(json.load(f)))

    return all_results


def _normalise_loaded_results(data: Any) -> dict:
    """Return legacy and current result files in the current API shape."""
    if not isinstance(data, dict):
        return {"schema_version": 1, "results": [], "summary": {}, "settings": None}

    data.setdefault("schema_version", 1)
    if not isinstance(data.get("results"), list):
        data["results"] = []
    if not isinstance(data.get("summary"), dict):
        data["summary"] = {}

    # Older files have no settings snapshot. It cannot be inferred from the
    # modes because every benchmark includes TextGrad and non-TextGrad modes.
    if not isinstance(data.get("settings"), dict):
        data["settings"] = None

    for problem_result in data["results"]:
        if not isinstance(problem_result, dict):
            continue
        modes = problem_result.get("modes", {})
        if not isinstance(modes, dict):
            continue
        for mode_name, mode_result in modes.items():
            if isinstance(mode_result, dict):
                # Legacy records predate this explicit field; the mode name is
                # a safe source for its default value.
                mode_result.setdefault(
                    "textgrad_included", mode_name in {"textgrad_only", "full"}
                )

    return data
