# benchmark/prefetch_lcb.py
"""
Download LiveCodeBench's pre-generated Parquet shards into the project-local
cache. Parquet avoids the legacy multi-gigabyte JSON-to-Arrow conversion that
could exhaust the backend container's memory.

This script does ONLY the download. No filtering, no sampling — that
happens at runtime in fetch_lcb.py, every time load_lcb_problems() is called.

Usage:
    python -m benchmark.prefetch_lcb
    python -m benchmark.prefetch_lcb --version release_v6
"""
import os
import argparse
import json
from pathlib import Path

# Point HF's dataset cache at a project-local folder so it can be baked into
# the Docker image. Must be set BEFORE `datasets` is imported.
HF_CACHE_DIR = Path(__file__).resolve().parent.parent / "data" / "hf_cache"
HF_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("HF_HOME", str(HF_CACHE_DIR))
os.environ.setdefault("HF_DATASETS_CACHE", str(HF_CACHE_DIR))

from huggingface_hub import HfApi, hf_hub_download  # noqa: E402

DATASET_REPO = "livecodebench/code_generation_lite"
# Pin the official materialized-Parquet snapshot. The dataset's main branch
# also contains a legacy loading script and cumulative JSON sources.
PARQUET_REVISION = "48d36ed304dca42cf8ab20e941262ccd096518a3"
PARQUET_CACHE_DIR = HF_CACHE_DIR / "lcb_parquet"


def parquet_version_dir(version: str) -> Path:
    return PARQUET_CACHE_DIR / version


def parquet_files(version: str) -> list[Path]:
    return sorted(parquet_version_dir(version).glob("*.parquet"))


def is_parquet_prefetched(version: str) -> bool:
    version_dir = parquet_version_dir(version)
    marker = version_dir / ".complete.json"
    files = parquet_files(version)
    if not marker.is_file() or not files:
        return False
    try:
        manifest = json.loads(marker.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    expected = manifest.get("files", {})
    return bool(expected) and all(
        path.name in expected and path.stat().st_size == expected[path.name]
        for path in files
    ) and len(files) == len(expected)


def prefetch(version: str = "release_v6", progress_callback=None) -> None:
    print(f"[fetch] downloading Parquet shards for {version} into {PARQUET_CACHE_DIR} ...")
    entries = HfApi().list_repo_tree(
        repo_id=DATASET_REPO,
        repo_type="dataset",
        revision=PARQUET_REVISION,
        path_in_repo=version,
    )
    remote_files = sorted(
        (entry for entry in entries if getattr(entry, "path", "").endswith(".parquet")),
        key=lambda entry: entry.path,
    )
    if not remote_files:
        raise RuntimeError(f"No remote Parquet shards were found for {version}.")

    total_bytes = sum(int(getattr(entry, "size", 0) or 0) for entry in remote_files)
    downloaded_bytes = 0
    if progress_callback:
        progress_callback(0, len(remote_files), 0, total_bytes)
    # Download one shard at a time. This keeps peak resource use low and gives
    # the UI a reliable completed-shard progress signal.
    for completed, entry in enumerate(remote_files, start=1):
        hf_hub_download(
            repo_id=DATASET_REPO,
            repo_type="dataset",
            revision=PARQUET_REVISION,
            filename=entry.path,
            local_dir=str(PARQUET_CACHE_DIR),
        )
        downloaded_bytes += int(getattr(entry, "size", 0) or 0)
        if progress_callback:
            progress_callback(completed, len(remote_files), downloaded_bytes, total_bytes)
    files = parquet_files(version)
    if not files:
        raise RuntimeError(f"No Parquet shards were found for {version}.")
    if any(path.stat().st_size == 0 for path in files):
        raise RuntimeError(f"One or more Parquet shards for {version} are empty.")

    manifest = {
        "version": version,
        "files": {path.name: path.stat().st_size for path in files},
    }
    marker = parquet_version_dir(version) / ".complete.json"
    marker.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    cached_bytes = sum(manifest["files"].values())
    print(f"[done] {len(files)} shards ({cached_bytes / 1_000_000_000:.2f} GB) cached")


def is_prefetched(version: str = "release_v6") -> bool:
    """Check cache completeness without loading or rebuilding the dataset.

    Calling ``load_dataset`` here used to generate the full multi-gigabyte
    Arrow split during every readiness poll. Besides being slow, that work ran
    on FastAPI's event loop and made every endpoint appear offline.
    """
    if is_parquet_prefetched(version):
        return True

    # Backward compatibility for already completed legacy Arrow caches (for
    # example an existing release_v5 cache). New downloads use Parquet.
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
