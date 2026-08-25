# benchmark/prefetch_lcb.py
"""
Run this manually before `docker build` to download the LCB dataset from
HuggingFace into a project-local cache folder (data/hf_cache/), so the
container never needs network access to load it.

This script does ONLY the download. No filtering, no sampling — that
happens at runtime in fetch_lcb.py, every time load_lcb_problems() is called.

Usage:
    python -m benchmark.prefetch_lcb
    python -m benchmark.prefetch_lcb --version release_v6
"""
import os
import argparse
from pathlib import Path

# Point HF's dataset cache at a project-local folder so it can be baked into
# the Docker image. Must be set BEFORE `datasets` is imported.
HF_CACHE_DIR = Path(__file__).resolve().parent.parent / "data" / "hf_cache"
HF_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("HF_HOME", str(HF_CACHE_DIR))
os.environ.setdefault("HF_DATASETS_CACHE", str(HF_CACHE_DIR))

from datasets import load_dataset  # noqa: E402


def prefetch(version: str = "release_v6") -> None:
    print(f"[fetch] downloading {version} into {HF_CACHE_DIR} ...")
    load_dataset("livecodebench/code_generation_lite", version, trust_remote_code=True)
    print(f"[done] {version} cached at {HF_CACHE_DIR}")


def is_prefetched(version: str = "release_v6") -> bool:
    """Check cache completeness without loading or rebuilding the dataset.

    Calling ``load_dataset`` here used to generate the full multi-gigabyte
    Arrow split during every readiness poll. Besides being slow, that work ran
    on FastAPI's event loop and made every endpoint appear offline.
    """
    version_cache = HF_CACHE_DIR / "livecodebench___code_generation_lite" / version
    if not version_cache.is_dir():
        return False

    for dataset_info in version_cache.glob("**/dataset_info.json"):
        build_dir = dataset_info.parent
        if build_dir.name.endswith(".incomplete"):
            continue
        arrow_files = list(build_dir.glob("*.arrow"))
        if arrow_files and all(path.stat().st_size > 0 for path in arrow_files):
            return True
    return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download and cache the raw LCB dataset.")
    parser.add_argument("--version", default="release_v6", help="Dataset version")
    args = parser.parse_args()

    prefetch(args.version)
