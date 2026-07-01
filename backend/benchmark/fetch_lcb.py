# benchmark/fetch_lcb.py
import os
import json
import random
import logging
from pathlib import Path
from typing import Optional
import json, zlib, pickle, base64

logger = logging.getLogger(__name__)

# Must match prefetch_lcb.py — points HF at the same project-local cache
# folder that was baked into the Docker image, so this never hits network.
HF_CACHE_DIR = Path(__file__).resolve().parent.parent / "data" / "hf_cache"
os.environ.setdefault("HF_HOME", str(HF_CACHE_DIR))
os.environ.setdefault("HF_DATASETS_CACHE", str(HF_CACHE_DIR))

from datasets import load_dataset  # noqa: E402

def decode_private_tests(raw):
    try:
        return json.loads(raw)
    except Exception:
        return json.loads(
            pickle.loads(
                zlib.decompress(
                    base64.b64decode(raw.encode("utf-8"))
                )
            )
        )
    
def load_lcb_problems(
    version: str = "release_v5", n: int = 30, difficulty: Optional[str] = None
) -> list[dict]:
    """
    Load problems from the LiveCodeBench dataset.

    Reads from the local HF cache (populated ahead of time by running
    `python -m benchmark.prefetch_lcb`) — no network call if that cache is
    present. Filters by difficulty and randomly samples n problems fresh on
    every call.

    Args:
        version: Dataset version (e.g., "release_v5")
        n: Number of problems to load
        difficulty: Filter by difficulty ("easy", "medium", "hard", or None for all)

    Returns:
        List of problem dicts with: id, title, statement, difficulty, platform,
        release_date, test_cases, private_tests
    """
    dataset = load_dataset(
        "livecodebench/code_generation_lite", version, trust_remote_code=True, split="test"
    )
    candidate_indices = [
        i for i, item in enumerate(dataset)
        if not difficulty or item.get("difficulty") == difficulty
    ]
    chosen_indices = random.sample(candidate_indices, min(n, len(candidate_indices)))

    all_problems = []
    for idx in chosen_indices:
        item = dataset[idx]
        if difficulty and item.get("difficulty") != difficulty:
            continue

        test_cases = item.get("public_test_cases", [])
        if isinstance(test_cases, str):
            try:
                test_cases = json.loads(test_cases)
            except json.JSONDecodeError:
                test_cases = []

        private_tests = item.get("private_test_cases", "")
        private_tests = decode_private_tests(private_tests)
        all_tests = test_cases + private_tests
        metadata_raw = item.get("metadata", "{}")
        if isinstance(metadata_raw, str):
            try:
                metadata = json.loads(metadata_raw)
            except json.JSONDecodeError:
                metadata = {}
        else:
            metadata = metadata_raw or {}

        fn_name = metadata.get("func_name")
        evaluation_sample = {
            "input_output": json.dumps({
                "inputs": [t["input"] for t in all_tests],
                "outputs": [t["output"] for t in all_tests],
                "fn_name": fn_name,
            })
        }
        all_problems.append({
            "id": item.get("question_id"),
            "title": item.get("question_title"),
            "statement": item.get("question_content"),
            "difficulty": item.get("difficulty"),
            "platform": item.get("platform"),
            "release_date": item.get("contest_date"),
            "test_cases": test_cases,
            "private_tests": private_tests,
            "evaluation_sample": evaluation_sample
        })

    return all_problems