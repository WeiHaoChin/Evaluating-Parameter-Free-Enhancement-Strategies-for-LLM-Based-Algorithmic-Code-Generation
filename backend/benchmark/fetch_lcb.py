import base64
import json
import logging
import os
import pickle
import random
import zlib
from pathlib import Path
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


def _parquet_sql(paths: list[Path]) -> str:
    """Return a DuckDB read_parquet expression for trusted local paths."""
    quoted = ", ".join(
        "'" + str(path).replace("'", "''") + "'" for path in paths
    )
    return f"read_parquet([{quoted}], union_by_name=true)"


def _duckdb_connection():
    """Create a deliberately memory-bounded connection for Parquet reads."""
    import duckdb

    temp_dir = HF_CACHE_DIR / "duckdb_tmp"
    temp_dir.mkdir(parents=True, exist_ok=True)
    connection = duckdb.connect()
    connection.execute("SET memory_limit = '1GB'")
    connection.execute("SET threads = 1")
    escaped_temp_dir = str(temp_dir).replace("'", "''")
    connection.execute(f"SET temp_directory = '{escaped_temp_dir}'")
    return connection


def select_lcb_problem_ids(
    version: str,
    n: int,
    difficulty: Optional[str],
    seed: int,
) -> list[str]:
    """Select deterministic IDs without reading statements or test cases."""
    paths = parquet_files(version)
    if not paths:
        return []
    source = _parquet_sql(paths)
    where = "WHERE difficulty = ?" if difficulty else ""
    params: list[object] = [difficulty] if difficulty else []
    # Ordering by a seeded hash gives a stable sample without retaining full
    # records or decoding the very large private_test_cases column.
    params.extend([str(seed), max(0, int(n))])
    query = f"""
        SELECT question_id
        FROM {source}
        {where}
        ORDER BY hash(question_id || ?)
        LIMIT ?
    """
    with _duckdb_connection() as connection:
        rows = connection.execute(query, params).fetchall()
    return [row[0] for row in rows]


def _prepare_problem(item: dict) -> dict:
    """Decode one selected dataset record into the benchmark representation."""
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

    return {
        "id": item.get("question_id"),
        "title": item.get("question_title"),
        "statement": item.get("question_content"),
        "difficulty": item.get("difficulty"),
        "platform": item.get("platform"),
        "release_date": item.get("contest_date"),
        "test_cases": test_cases,
        "private_tests": private_tests,
        "starter_code": item.get("starter_code") or None,
        "evaluation_sample": {
            "input_output": json.dumps({
                "inputs": [test["input"] for test in all_tests],
                "outputs": [test["output"] for test in all_tests],
                "fn_name": metadata.get("func_name"),
            })
        },
    }


def load_lcb_problem(version: str, question_id: str) -> dict:
    """Load and decode one selected Parquet record."""
    source = _parquet_sql(parquet_files(version))
    with _duckdb_connection() as connection:
        cursor = connection.execute(
            f"SELECT * FROM {source} WHERE question_id = ? LIMIT 1",
            [question_id],
        )
        row = cursor.fetchone()
        columns = [description[0] for description in cursor.description]
    if row is None:
        raise KeyError(f"LiveCodeBench question not found: {question_id}")
    return _prepare_problem(dict(zip(columns, row)))


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
        chosen_ids = select_lcb_problem_ids(
            version, requested, difficulty, seed if seed is not None else 0
        )
        chosen_items = []
        for question_id in chosen_ids:
            problem = load_lcb_problem(version, question_id)
            chosen_items.append(problem)
        logger.info(
            "Selected %s %s problems from %s",
            len(chosen_items),
            difficulty or "all",
            version,
        )
        return chosen_items
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

    return [_prepare_problem(item) for item in chosen_items]
