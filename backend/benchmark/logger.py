# benchmark/logger.py
import json
from pathlib import Path
from datetime import datetime
from typing import Optional

RESULTS_DIR = Path(__file__).parent / "results"


def _ensure_results_dir() -> None:
    """Ensure results directory exists."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def save_results(results: list[dict], summary: dict) -> None:
    """
    Save benchmark results to JSON file.

    Args:
        results: List of problem results
        summary: Summary metrics
    """
    _ensure_results_dir()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = RESULTS_DIR / f"{timestamp}.json"

    data = {
        "timestamp": datetime.now().isoformat(),
        "results": results,
        "summary": summary,
    }

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
        return json.load(f)


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
            all_results.append(json.load(f))

    return all_results
