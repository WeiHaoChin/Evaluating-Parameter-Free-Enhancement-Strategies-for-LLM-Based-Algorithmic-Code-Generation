# benchmark/logger.py
import json
from pathlib import Path
from datetime import datetime
from typing import Any, Optional

RESULTS_DIR = Path(__file__).parent / "results"
SENSITIVE_SETTING_KEYS = {"apiKey", "textGradApiKey"}
LARGE_PROBLEM_FIELDS = {"evaluation_sample", "private_tests", "test_cases"}
TEST_CASE_SUFFIX = "_testcases.json"


def _safe_settings(settings: dict[str, Any]) -> dict[str, Any]:
    """Return a settings snapshot without credentials."""
    return {
        key: value for key, value in settings.items()
        if key not in SENSITIVE_SETTING_KEYS
    }


def _ensure_results_dir() -> None:
    """Ensure results directory exists."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def _compact_results(results: list[dict]) -> list[dict]:
    """Remove evaluation-only payloads from persisted and returned results."""
    compacted = []
    for problem_result in results:
        if not isinstance(problem_result, dict):
            compacted.append(problem_result)
            continue

        compact_result = dict(problem_result)
        problem = compact_result.get("problem")
        if isinstance(problem, dict):
            compact_result["problem"] = {
                key: value for key, value in problem.items()
                if key not in LARGE_PROBLEM_FIELDS
            }

        modes = compact_result.get("modes")
        if isinstance(modes, dict):
            compact_result["modes"] = {
                mode_name: {
                    key: value for key, value in mode_result.items()
                    if key != "results"
                } if isinstance(mode_result, dict) else mode_result
                for mode_name, mode_result in modes.items()
            }
        compacted.append(compact_result)
    return compacted
def new_results_path() -> Path:
    """Reserve one stable result path for a benchmark run."""
    _ensure_results_dir()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    return RESULTS_DIR / f"{timestamp}.json"


def test_cases_path(results_path: Path) -> Path:
    """Return the companion path containing evaluation-only problem data."""
    return results_path.with_name(f"{results_path.stem}{TEST_CASE_SUFFIX}")


def append_test_case(
    filename: Path,
    problem: Optional[dict],
    benchmark: Optional[dict[str, Any]] = None,
) -> Path:
    """Append one problem to a valid companion JSON without rereading it."""
    suffix = b"\n  ]\n}\n"
    if not filename.exists():
        header = (
            "{\n  \"schema_version\": 1,\n  \"benchmark\": "
            + json.dumps(benchmark or {}, default=str)
            + ",\n  \"problems\": ["
        ).encode("utf-8")
        with open(filename, "wb") as file:
            file.write(header)
            file.write(suffix)

    if problem is None:
        return filename

    record = {
        "id": problem.get("id"),
        "title": problem.get("title"),
        "test_cases": problem.get("test_cases", []),
        "private_tests": problem.get("private_tests", []),
        "evaluation_sample": problem.get("evaluation_sample"),
    }
    with open(filename, "r+b") as file:
        file.seek(-len(suffix), 2)
        suffix_start = file.tell()
        file.seek(suffix_start - 1)
        has_existing_record = file.read(1) != b"["
        file.seek(suffix_start)
        file.write(b",\n" if has_existing_record else b"\n")
        encoder = json.JSONEncoder(indent=2, default=str)
        for chunk in encoder.iterencode(record):
            file.write(chunk.encode("utf-8"))
        file.write(suffix)
        file.truncate()
        file.flush()
    return filename


def save_results(
    results: list[dict],
    summary: dict,
    settings: Optional[dict[str, Any]] = None,
    filename: Optional[Path] = None,
    benchmark: Optional[dict[str, Any]] = None,
) -> Path:
    """
    Save benchmark results to JSON file.

    Args:
        results: List of problem results
        summary: Summary metrics
    """
    _ensure_results_dir()

    filename = filename or new_results_path()

    data: dict[str, Any] = {
        "schema_version": 6,
        "timestamp": datetime.now().isoformat(),
        "results": _compact_results(results),
        "summary": summary,
    }
    if settings is not None:
        # Keep reproducibility settings, but never persist API credentials.
        data["settings"] = _safe_settings(settings)
    if benchmark is not None:
        data["benchmark"] = benchmark

    temporary = filename.with_suffix(filename.suffix + ".tmp")
    with open(temporary, "w") as f:
        json.dump(data, f, indent=2, default=str)
    temporary.replace(filename)
    return filename


def load_latest_results() -> Optional[dict]:
    """
    Load the most recent results file.

    Returns:
        Dict with results and summary, or None if no results exist
    """
    _ensure_results_dir()

    result_files = _benchmark_result_files()
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

    result_files = _benchmark_result_files()
    all_results = []

    for file in result_files:
        with open(file, "r") as f:
            all_results.append(_normalise_loaded_results(json.load(f)))

    return all_results


def _benchmark_result_files() -> list[Path]:
    """List main result files without their test-case companions."""
    return sorted(
        (
            path for path in RESULTS_DIR.glob("*.json")
            if not path.name.endswith(TEST_CASE_SUFFIX)
        ),
        reverse=True,
    )


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
    else:
        # Prevent legacy result files from exposing keys through the API.
        data["settings"] = _safe_settings(data["settings"])

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
                mode_result.setdefault("rag_retrieved_data", [])
                mode_result.setdefault("textgrad_improved_system_prompt", None)

    # Legacy files may contain huge private test inputs which are irrelevant
    # to result display and metric calculations.
    data["results"] = _compact_results(data["results"])

    return data
