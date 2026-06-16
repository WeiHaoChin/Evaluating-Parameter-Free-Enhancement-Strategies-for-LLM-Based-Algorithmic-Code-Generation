# benchmark/fetch_lcb.py
from typing import Optional
from datasets import load_dataset
import random
import zlib, base64, json,logging
import pickle

logger = logging.getLogger(__name__)
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

    all_problems = []

    for item in dataset["test"]:
        if difficulty and item.get("difficulty") != difficulty:
            continue

        test_cases = item.get("public_test_cases", [])
        if isinstance(test_cases, str):
            try:
                test_cases = json.loads(test_cases)
            except json.JSONDecodeError:
                test_cases = []

        private_tests = item.get("private_test_cases", "")
        # if isinstance(private_tests, str):
        #     try:
        #         private_tests = decode_private_tests(private_tests) 
        #     except json.JSONDecodeError:
        #         private_tests = json.loads(private_tests)

        all_problems.append({
            "id": item.get("question_id"),
            "title": item.get("question_title"),
            "statement": item.get("question_content"),
            "difficulty": item.get("difficulty"),
            "platform": item.get("platform"),
            "release_date": item.get("contest_date"),
            "test_cases": test_cases,
            "private_tests": private_tests,
        })

    # Randomly sample n problems instead of always taking the first n
    return random.sample(all_problems, min(n, len(all_problems)))
# def decode_private_tests(raw):
#     try:
#         # Step 1: base64 decode
#         compressed = base64.b64decode(raw)
#         print(f"base64 decoded length: {len(compressed)}")
#         # Step 2: zlib decompress
#         decompressed = zlib.decompress(compressed)
#         unpickled = pickle.loads(decompressed)
#         print(f"decompressed content: {repr(decompressed[:200])}")
#         # Step 3: JSON parse
#         if isinstance(unpickled, str):
#             return json.loads(unpickled)
#         elif isinstance(unpickled, list):
#             return unpickled
#         return []
#     except Exception as e:
#         logging.warning(f"Failed to decode private tests: {e}")
#         return []
