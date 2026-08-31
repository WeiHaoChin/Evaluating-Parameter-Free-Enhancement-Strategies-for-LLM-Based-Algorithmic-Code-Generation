"""
Main Orchestrator
Runs all scrapers sequentially then processes into RAG chunks.

Usage:
    python main.py                                      # all sources, local embedder, JSONL
    python main.py --sources codeforces,cph             # subset of sources
    python main.py --embedder local --vector-db chroma  # free local setup
    python main.py --embedder openai --vector-db qdrant # OpenAI + Qdrant cloud
    python main.py --pipeline-only                      # re-chunk existing data only
"""

import asyncio
import argparse
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from backend.RAG.scrapers.codeforces import CodeforcesScraper
from backend.RAG.scrapers.usaco import USACOScraper
from backend.RAG.scrapers.atcoder import AtCoderScraper
from backend.RAG.scrapers.cp_algorithms import CPAlgorithmsScraper
from backend.RAG.scrapers.cph_book import CPHScraper
from backend.RAG.pipeline.rag_pipeline import RAGPipeline, LocalEmbedder

logger = logging.getLogger(__name__)

DATA_ROOT = Path("data")
ALL_SOURCES = ["codeforces", "usaco", "atcoder", "cp_algorithms", "cph"]


async def run_scrapers(sources: list[str], args: argparse.Namespace):
    if "codeforces" in sources:
        async with CodeforcesScraper(
            output_dir=str(DATA_ROOT / "codeforces"),
            max_problems=args.max_cf,
        ) as s:
            logger.info("=" * 60)
            logger.info("SCRAPER: Codeforces")
            logger.info("=" * 60)
            await s.scrape()

    if "usaco" in sources:
        async with USACOScraper(
            output_dir=str(DATA_ROOT / "usaco"),
            max_contests=args.max_usaco_contests,
            start_contest=args.start_usaco_contest,
        ) as s:
            logger.info("=" * 60)
            logger.info("SCRAPER: USACO")
            logger.info("=" * 60)
            await s.scrape()

    if "atcoder" in sources:
        excluded_ids = set()
        max_atcoder = args.max_ac
        if args.additional_ac:
            for path in (DATA_ROOT / "atcoder").glob("*.json"):
                try:
                    import json
                    excluded_ids.add(json.loads(path.read_text(encoding="utf-8"))["problem_id"])
                except (OSError, ValueError, KeyError):
                    logger.warning("Could not read existing AtCoder ID from %s", path)
            max_atcoder = args.additional_ac
            logger.info("Excluding %d existing AtCoder IDs; requesting %d additional problems",
                        len(excluded_ids), max_atcoder)
        async with AtCoderScraper(
            output_dir=str(DATA_ROOT / "atcoder"),
            max_problems=max_atcoder,
            exclude_problem_ids=excluded_ids,
        ) as s:
            logger.info("=" * 60)
            logger.info("SCRAPER: AtCoder")
            logger.info("=" * 60)
            await s.scrape()

    if "cp_algorithms" in sources:
        async with CPAlgorithmsScraper(output_dir=str(DATA_ROOT / "cp_algorithms")) as s:
            logger.info("=" * 60)
            logger.info("SCRAPER: CP-Algorithms")
            logger.info("=" * 60)
            await s.scrape()

    if "cph" in sources:
        async with CPHScraper(output_dir=str(DATA_ROOT / "cph")) as s:
            logger.info("=" * 60)
            logger.info("SCRAPER: CPH Book")
            logger.info("=" * 60)
            await s.scrape()


def run_pipeline(args: argparse.Namespace):
    logger.info("=" * 60)
    logger.info("PIPELINE: Chunking & Indexing")
    logger.info(f"  embedder   : {args.embedder}")
    if args.embedder == "local":
        logger.info(f"  local model: {args.local_model}")
    logger.info(f"  vector db  : {args.vector_db}")
    logger.info("=" * 60)

    pipeline = RAGPipeline(
        data_dirs={
            "codeforces":    str(DATA_ROOT / "codeforces"),
            "usaco":         str(DATA_ROOT / "usaco"),
            "atcoder":       str(DATA_ROOT / "atcoder"),
            "cp-algorithms": str(DATA_ROOT / "cp_algorithms"),
            "cph_book":      str(DATA_ROOT / "cph"),
        },
        output_dir=str(DATA_ROOT / "rag_chunks"),
        vector_db=args.vector_db,
        embedder=args.embedder,
        local_model=args.local_model,
        qdrant_url=args.qdrant_url,
        qdrant_api_key=args.qdrant_api_key,
        collection_name=args.collection,
    )
    chunks = pipeline.run()

    logger.info("")
    logger.info("Pipeline complete!")
    logger.info(f"  Total chunks  : {len(chunks)}")
    logger.info(f"  Token estimate: {sum(c.token_estimate for c in chunks):,}")
    logger.info(f"  JSONL output  : {DATA_ROOT / 'rag_chunks' / 'chunks.jsonl'}")
    if args.vector_db == "chroma":
        logger.info(f"  ChromaDB      : {DATA_ROOT / 'rag_chunks' / 'chroma_db'}")
    elif args.vector_db == "qdrant":
        logger.info(f"  Qdrant        : {args.qdrant_url} / collection '{args.collection}'")


def parse_args():
    parser = argparse.ArgumentParser(
        description="CP RAG Database Builder",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Embedder options (--embedder):
  local   Free, no API key. Downloads ~90MB model on first run, cached after.
          Default model: BAAI/bge-small-en-v1.5 (good quality, fast on CPU)
          Override with: --local-model BAAI/bge-base-en-v1.5
  openai  Requires OPENAI_API_KEY env var. ~$0.02 for the full corpus.

Vector DB options (--vector-db):
  jsonl   Default. No setup needed — outputs chunks.jsonl.
  chroma  Local persistent DB. pip install chromadb
  qdrant  Local Docker or Qdrant Cloud. pip install qdrant-client

Examples:
  python main.py                                         # free local, JSONL
  python main.py --embedder local --vector-db chroma     # free, queryable
  python main.py --embedder openai --vector-db chroma    # OpenAI + ChromaDB
  python main.py --vector-db qdrant                      # local Docker Qdrant
  python main.py --vector-db qdrant \\
      --qdrant-url https://xyz.qdrant.io \\
      --qdrant-api-key YOUR_KEY                           # Qdrant Cloud
  python main.py --pipeline-only --embedder local \\
      --vector-db chroma                                  # re-chunk only
        """,
    )
    parser.add_argument(
        "--sources",
        type=lambda s: [x.strip() for x in s.split(",")],
        default=["all"],
        help=f"Comma-separated list or 'all'. Options: {', '.join(ALL_SOURCES)}",
    )
    parser.add_argument("--pipeline-only", action="store_true",
                        help="Skip scraping, only run the RAG pipeline")
    parser.add_argument("--scrape-only", action="store_true",
                        help="Skip pipeline, only scrape")

    # Embedding
    parser.add_argument("--embedder", choices=["local", "openai"], default="local",
                        help="Embedding backend (default: local — no API key needed)")
    parser.add_argument("--local-model", default=LocalEmbedder.DEFAULT_MODEL,
                        help=f"sentence-transformers model (default: {LocalEmbedder.DEFAULT_MODEL})")

    # Vector DB
    parser.add_argument("--vector-db", choices=["jsonl", "chroma", "qdrant"], default="jsonl",
                        help="Vector DB backend (default: jsonl)")
    parser.add_argument("--collection", default="cp_rag",
                        help="Collection/index name (default: cp_rag)")
    parser.add_argument("--qdrant-url", default="http://localhost:6333",
                        help="Qdrant URL (default: http://localhost:6333)")
    parser.add_argument("--qdrant-api-key", default=os.environ.get("QDRANT_API_KEY"),
                        help="Qdrant API key (or set QDRANT_API_KEY env var)")

    # Scraper limits
    parser.add_argument("--max-cf", type=int, default=500)
    parser.add_argument("--max-ac", type=int, default=300)
    parser.add_argument("--additional-ac", type=int, default=0,
                        help="Scrape this many additional AtCoder problems, excluding existing IDs")
    parser.add_argument("--max-usaco-contests", type=int, default=20)
    parser.add_argument("--start-usaco-contest", type=int, default=1,
                        help="1-based archive position to resume USACO scraping from")
    parser.add_argument("--data-root", type=str, default="data")
    return parser.parse_args()


async def main():
    args = parse_args()

    global DATA_ROOT
    DATA_ROOT = Path(args.data_root)
    DATA_ROOT.mkdir(parents=True, exist_ok=True)

    sources = ALL_SOURCES if "all" in args.sources else args.sources
    invalid = set(args.sources) - set(ALL_SOURCES) - {"all"}
    if invalid:
        logger.error(f"Unknown sources: {invalid}. Valid: {ALL_SOURCES}")
        sys.exit(1)

    logger.info("CP RAG Database Builder")
    logger.info(f"  Sources   : {sources}")
    logger.info(f"  Embedder  : {args.embedder}")
    logger.info(f"  Vector DB : {args.vector_db}")
    logger.info(f"  Data root : {DATA_ROOT.absolute()}")
    logger.info("")

    if not args.pipeline_only:
        await run_scrapers(sources, args)

    if not args.scrape_only:
        run_pipeline(args)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("scraper.log"),
        ],
    )
    asyncio.run(main())
