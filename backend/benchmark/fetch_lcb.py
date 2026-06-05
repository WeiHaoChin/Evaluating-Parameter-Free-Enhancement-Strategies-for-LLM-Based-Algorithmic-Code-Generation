# benchmark/fetch_lcb.py
from typing import Optional
from datasets import load_dataset
import json


def load_lcb_problems(
    version: str = "release_v5", n: int = 30, difficulty: Optional[str] = None
) -> list[dict]:
    """
    Load problems from HuggingFace LiveCodeBench dataset.

    Args:
        version: Dataset version (e.g., "release_v5")
        n: Number of problems to load
        difficulty: Filter by difficulty ("easy", "medium", "hard", or None for all)

    Returns:
        List of problem dicts with: id, title, statement, difficulty, platform,
        release_date, test_cases, private_tests
    """
    dataset = load_dataset("livecodebench/code_generation_lite", version, trust_remote_code=True)

    problems = []
    count = 0

    for item in dataset["test"]:
        if count >= n:
            break

        if difficulty and item.get("difficulty") != difficulty:
            continue

        test_cases = item.get("test_cases", [])
        if isinstance(test_cases, str):
            try:
                test_cases = json.loads(test_cases)
            except json.JSONDecodeError:
                test_cases = []

        private_tests = item.get("private_test_cases", [])
        if isinstance(private_tests, str):
            try:
                private_tests = json.loads(private_tests)
            except json.JSONDecodeError:
                private_tests = []

        problem = {
            "id": item.get("question_id"),
            "title": item.get("title"),
            "statement": item.get("description"),
            "difficulty": item.get("difficulty"),
            "platform": item.get("source"),
            "release_date": item.get("date_created"),
            "test_cases": test_cases,
            "private_tests": private_tests,
        }

        problems.append(problem)
        count += 1

    return problems
