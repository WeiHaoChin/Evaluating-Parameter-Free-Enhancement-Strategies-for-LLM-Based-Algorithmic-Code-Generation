import base64
import json
import logging
import os
import pickle
import random
import zlib
from typing import Optional

from benchmark.prefetch_lcb import (
    HF_CACHE_DIR,
    is_parquet_prefetched,
    parquet_files,
)

logger = logging.getLogger(__name__)

os.environ.setdefault("HF_HOME", str(HF_CACHE_DIR))
os.environ.setdefault("HF_DATASETS_CACHE", str(HF_CACHE_DIR))

from datasets import load_dataset  # noqa: E402


def decode_private_tests(raw):
    try:
        return json.loads(raw)
    except Exception:
        return json.loads(
            pickle.loads(
                zlib.decompress(base64.b64decode(raw.encode("utf-8")))
            )
        )


def _stream_sample(
    dataset,
    n: int,
    difficulty: Optional[str],
    rng,
) -> tuple[list[dict], int]:
    """Reservoir-sample a stream while retaining at most ``n`` records."""
    selected: list[dict] = []
    eligible_count = 0
    for item in dataset:
        if difficulty and item.get("difficulty") != difficulty:
            continue
        eligible_count += 1
        if len(selected) < n:
            selected.append(item)
            continue
        if n:
            replacement_index = rng.randrange(eligible_count)
            if replacement_index < n:
                selected[replacement_index] = item
    return selected, eligible_count


def load_lcb_problems(
    version: str = "release_v6",
    n: int = 30,
    difficulty: Optional[str] = None,
    seed: Optional[int] = None,
) -> list[dict]:
    """Load and sample LiveCodeBench problems from the local cache.

    New downloads use official Parquet shards and streaming reservoir
    sampling, so at most ``n`` raw records are retained in memory. Completed
    legacy Arrow caches remain supported as a fallback.
    """
    requested = max(0, int(n))
    rng = random.Random(seed) if seed is not None else random

    if is_parquet_prefetched(version):
        paths = [str(path) for path in parquet_files(version)]
        dataset = load_dataset(
            "parquet",
            data_files={"test": paths},
            split="test",
            streaming=True,
        )
        chosen_items, eligible_count = _stream_sample(
            dataset, requested, difficulty, rng
        )
        logger.info(
            "Stream-selected %s of %s eligible %s problems from %s",
            len(chosen_items),
            eligible_count,
            difficulty or "all",
            version,
        )
    else:
        dataset = load_dataset(
            "livecodebench/code_generation_lite",
            version,
            trust_remote_code=True,
            split="test",
        )
        candidate_indices = [
            index for index, item in enumerate(dataset)
            if not difficulty or item.get("difficulty") == difficulty
        ]
        chosen_indices = rng.sample(
            candidate_indices, min(requested, len(candidate_indices))
        )
        chosen_items = [dataset[index] for index in chosen_indices]

    all_problems = []
    for item in chosen_items:
        test_cases = item.get("public_test_cases", [])
        if isinstance(test_cases, str):
            try:
                test_cases = json.loads(test_cases)
            except json.JSONDecodeError:
                test_cases = []

        private_tests = decode_private_tests(item.get("private_test_cases", ""))
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
                "inputs": [test["input"] for test in all_tests],
                "outputs": [test["output"] for test in all_tests],
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
            "starter_code": item.get("starter_code") or None,
            "evaluation_sample": evaluation_sample,
        })

    return all_problems
