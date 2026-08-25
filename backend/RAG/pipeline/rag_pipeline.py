"""
RAG Pipeline
Transforms raw scraped documents into chunked, embedded records ready for vector DB ingestion.

Supports: ChromaDB (local), Qdrant, Pinecone, and raw JSONL output.
Embedding: OpenAI text-embedding-3-small (best cost/quality ratio) or local via sentence-transformers.
"""

import json
import hashlib
import logging
import re
import uuid
from pathlib import Path
from dataclasses import dataclass, asdict, field
from typing import Optional, Literal
from html.parser import HTMLParser

logger = logging.getLogger(__name__)

# ─── HTML → Text ─────────────────────────────────────────────────────────────

class HTMLTextExtractor(HTMLParser):
    """Minimal HTML to text converter that preserves structure."""
    SKIP_TAGS = {"script", "style", "nav", "header", "footer"}
    BLOCK_TAGS = {"p", "div", "h1", "h2", "h3", "h4", "li", "br", "tr"}

    def __init__(self):
        super().__init__()
        self._skip = 0
        self._parts = []
        self._current_tag = ""

    def handle_starttag(self, tag, attrs):
        self._current_tag = tag
        if tag in self.SKIP_TAGS:
            self._skip += 1
        if tag in self.BLOCK_TAGS and self._parts and self._parts[-1] != "\n":
            self._parts.append("\n")

    def handle_endtag(self, tag):
        if tag in self.SKIP_TAGS:
            self._skip = max(0, self._skip - 1)
        if tag in self.BLOCK_TAGS:
            self._parts.append("\n")

    def handle_data(self, data):
        if not self._skip:
            cleaned = data.replace("\u00a0", " ")
            self._parts.append(cleaned)

    def get_text(self) -> str:
        text = "".join(self._parts)
        text = re.sub(r" {2,}", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()


def html_to_text(html: str) -> str:
    if not html:
        return ""
    extractor = HTMLTextExtractor()
    extractor.feed(html)
    return extractor.get_text()


def latex_to_text(latex: str) -> str:
    """Simple LaTeX cleaning (good enough for RAG; not full rendering)."""
    text = latex
    text = re.sub(r"\\begin\{tikzpicture\}.*?\\end\{tikzpicture\}", "", text, flags=re.DOTALL)
    text = re.sub(r"\\begin\{center\}.*?\\end\{center\}", "", text, flags=re.DOTALL)
    # Keep math inline but mark it
    text = re.sub(r"\$\$(.+?)\$\$", r"[MATH: \1]", text, flags=re.DOTALL)
    text = re.sub(r"\$(.+?)\$", r"[math: \1]", text)
    # Common LaTeX commands
    text = re.sub(r"\\(?:textbf|emph|textit|texttt)\{([^}]+)\}", r"\1", text)
    text = re.sub(r"\\(?:section|subsection|subsubsection|chapter)\{([^}]+)\}", r"\n\n## \1\n\n", text)
    text = re.sub(r"\\(?:item)\b", "\n- ", text)
    text = re.sub(r"\\[a-zA-Z]+\{[^}]*\}", "", text)  # Remove other commands with args
    text = re.sub(r"\\[a-zA-Z]+", "", text)            # Remove bare commands
    text = re.sub(r"[{}]", "", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# ─── Chunking strategies ──────────────────────────────────────────────────────

@dataclass
class Chunk:
    chunk_id: str = ""
    source_id: str = ""          # Original document ID
    source: str = ""             # e.g., "codeforces", "cp-algorithms"
    chunk_type: str = ""         # "problem", "editorial", "theory", "code"
    title: str = ""
    text: str = ""
    metadata: dict = field(default_factory=dict)
    token_estimate: int = 0


def estimate_tokens(text: str) -> int:
    """Rough token estimate: ~0.75 words per token for English code/text mix."""
    return int(len(text.split()) * 1.33)


def chunk_text(
    text: str,
    max_tokens: int = 512,
    overlap_tokens: int = 64,
    min_tokens: int = 50,
) -> list[str]:
    """
    Semantic chunking: splits on paragraph boundaries first,
    then falls back to sentence/token boundaries.
    """
    if not text or estimate_tokens(text) <= max_tokens:
        return [text] if text.strip() else []

    # Split on double newlines (paragraphs)
    paragraphs = [p.strip() for p in re.split(r"\n\n+", text) if p.strip()]

    chunks = []
    current = []
    current_tokens = 0

    for para in paragraphs:
        para_tokens = estimate_tokens(para)

        if para_tokens > max_tokens:
            # Para too big — split by sentences
            sentences = re.split(r"(?<=[.!?])\s+", para)
            for sent in sentences:
                sent_tokens = estimate_tokens(sent)
                if current_tokens + sent_tokens > max_tokens and current:
                    chunks.append("\n\n".join(current))
                    # Overlap: keep last sentence for context
                    current = current[-1:] if overlap_tokens > 0 else []
                    current_tokens = estimate_tokens(current[0]) if current else 0
                current.append(sent)
                current_tokens += sent_tokens
        else:
            if current_tokens + para_tokens > max_tokens and current:
                chunks.append("\n\n".join(current))
                current = current[-1:] if overlap_tokens > 0 else []
                current_tokens = estimate_tokens(current[0]) if current else 0
            current.append(para)
            current_tokens += para_tokens

    if current:
        remainder = "\n\n".join(current)
        if estimate_tokens(remainder) >= min_tokens:
            chunks.append(remainder)

    return chunks


# ─── Document → Chunks converters ─────────────────────────────────────────────

def process_codeforces(doc: dict) -> list[Chunk]:
    chunks = []
    pid = doc.get("problem_id", "")
    tags = doc.get("tags", [])
    rating = doc.get("rating")

    base_meta = {
        "problem_id": pid,
        "contest_id": doc.get("contest_id"),
        "rating": rating,
        "tags": tags,
        "source_url": f"https://codeforces.com/problemset/problem/{doc.get('contest_id')}/{doc.get('index')}",
    }

    # Chunk 1: Problem statement
    stmt_text = html_to_text(doc.get("statement_html", ""))
    if stmt_text:
        for i, chunk_text_piece in enumerate(chunk_text(stmt_text, max_tokens=400)):
            chunks.append(Chunk(
                chunk_id=str(uuid.uuid4()),
                source_id=pid,
                source="codeforces",
                chunk_type="problem_statement",
                title=f"CF {pid}: {doc.get('name', '')} (part {i+1})",
                text=chunk_text_piece,
                metadata={**base_meta, "part": i + 1},
                token_estimate=estimate_tokens(chunk_text_piece),
            ))

    # Chunk 2: Examples (keep together for few-shot use)
    examples = doc.get("examples", [])
    if examples:
        ex_text = "\n\n".join(
            f"Input:\n{ex['input']}\nOutput:\n{ex['output']}"
            for ex in examples[:3]  # Cap at 3 examples
        )
        chunks.append(Chunk(
            chunk_id=str(uuid.uuid4()),
            source_id=pid,
            source="codeforces",
            chunk_type="examples",
            title=f"CF {pid} Examples",
            text=ex_text,
            metadata={**base_meta, "example_count": len(examples)},
            token_estimate=estimate_tokens(ex_text),
        ))

    # Chunk 3: Editorial
    ed_text = html_to_text(doc.get("editorial_html", ""))
    if ed_text:
        for i, chunk_text_piece in enumerate(chunk_text(ed_text, max_tokens=512)):
            chunks.append(Chunk(
                chunk_id=str(uuid.uuid4()),
                source_id=pid,
                source="codeforces",
                chunk_type="editorial",
                title=f"CF {pid} Editorial (part {i+1})",
                text=chunk_text_piece,
                metadata={**base_meta, "editorial_url": doc.get("editorial_url", ""), "part": i + 1},
                token_estimate=estimate_tokens(chunk_text_piece),
            ))

    return chunks


def process_usaco(doc: dict) -> list[Chunk]:
    chunks = []
    pid = doc.get("problem_id", "")
    base_meta = {
        "problem_id": pid,
        "contest": doc.get("contest"),
        "division": doc.get("division"),
        "topics": doc.get("topics", []),
    }

    stmt_text = html_to_text(doc.get("statement_html", ""))
    if stmt_text:
        for i, t in enumerate(chunk_text(stmt_text)):
            chunks.append(Chunk(
                chunk_id=str(uuid.uuid4()),
                source_id=pid,
                source="usaco",
                chunk_type="problem_statement",
                title=f"USACO {doc.get('problem_name', pid)} (part {i+1})",
                text=t,
                metadata={**base_meta, "part": i + 1},
                token_estimate=estimate_tokens(t),
            ))

    # Editorial (USACO.guide MDX)
    ed_raw = doc.get("editorial_html", "")
    if ed_raw:
        # Strip MDX-specific syntax
        ed_text = re.sub(r"<[A-Z][a-zA-Z]+[^>]*>.*?</[A-Z][a-zA-Z]+>", "", ed_raw, flags=re.DOTALL)
        ed_text = html_to_text(ed_text) if "<" in ed_text else ed_text
        for i, t in enumerate(chunk_text(ed_text, max_tokens=600)):
            chunks.append(Chunk(
                chunk_id=str(uuid.uuid4()),
                source_id=pid,
                source="usaco",
                chunk_type="editorial",
                title=f"USACO {doc.get('problem_name', pid)} Editorial (part {i+1})",
                text=t,
                metadata={**base_meta, "editorial_source": doc.get("editorial_source", ""), "part": i + 1},
                token_estimate=estimate_tokens(t),
            ))

    return chunks


def process_atcoder(doc: dict) -> list[Chunk]:
    chunks = []
    pid = doc.get("problem_id", "")
    base_meta = {
        "problem_id": pid,
        "contest_id": doc.get("contest_id"),
        "contest_type": doc.get("contest_type"),
        "difficulty": doc.get("difficulty"),
    }

    stmt = html_to_text(doc.get("statement_html", ""))
    if stmt:
        for i, t in enumerate(chunk_text(stmt)):
            chunks.append(Chunk(
                chunk_id=str(uuid.uuid4()),
                source_id=pid,
                source="atcoder",
                chunk_type="problem_statement",
                title=f"AtCoder {doc.get('title', pid)} (part {i+1})",
                text=t,
                metadata={**base_meta, "part": i + 1},
                token_estimate=estimate_tokens(t),
            ))

    ed = html_to_text(doc.get("editorial_html", ""))
    if ed:
        for i, t in enumerate(chunk_text(ed, max_tokens=600)):
            chunks.append(Chunk(
                chunk_id=str(uuid.uuid4()),
                source_id=pid,
                source="atcoder",
                chunk_type="editorial",
                title=f"AtCoder {doc.get('title', pid)} Editorial (part {i+1})",
                text=t,
                metadata={**base_meta, "editorial_url": doc.get("editorial_url", ""), "part": i + 1},
                token_estimate=estimate_tokens(t),
            ))

    return chunks


def process_cp_algorithms(doc: dict) -> list[Chunk]:
    chunks = []
    aid = doc.get("article_id", "")
    base_meta = {
        "article_id": aid,
        "category": doc.get("category"),
        "url": doc.get("url"),
        "complexity": doc.get("complexity"),
    }

    # Prefer markdown source (cleaner for RAG)
    raw = doc.get("content_markdown") or html_to_text(doc.get("content_html", ""))
    if raw:
        for i, t in enumerate(chunk_text(raw, max_tokens=600)):
            chunks.append(Chunk(
                chunk_id=str(uuid.uuid4()),
                source_id=aid,
                source="cp-algorithms",
                chunk_type="theory",
                title=f"{doc.get('title', aid)} (part {i+1})",
                text=t,
                metadata={**base_meta, "part": i + 1},
                token_estimate=estimate_tokens(t),
            ))

    # Code snippets as separate chunks (for code-specific retrieval)
    for j, snippet in enumerate(doc.get("code_snippets", [])[:5]):
        chunks.append(Chunk(
            chunk_id=str(uuid.uuid4()),
            source_id=aid,
            source="cp-algorithms",
            chunk_type="code",
            title=f"{doc.get('title', aid)} — Code {j+1}",
            text=f"```cpp\n{snippet}\n```",
            metadata={**base_meta, "code_index": j},
            token_estimate=estimate_tokens(snippet),
        ))

    return chunks


def process_cph(doc: dict) -> list[Chunk]:
    chunks = []
    cid = doc.get("chapter_id", "")
    base_meta = {
        "chapter_id": cid,
        "chapter_num": doc.get("chapter_num"),
        "part": doc.get("part"),
        "topics": doc.get("topics", []),
        "algorithms": doc.get("algorithms", []),
    }

    # Prefer cleaned text
    raw = doc.get("content_text") or latex_to_text(doc.get("content_latex", ""))
    clean = latex_to_text(raw)
    if not clean:
        return []
    if raw:
        for i, t in enumerate(chunk_text(clean, max_tokens=700)):
            chunks.append(Chunk(
                chunk_id=str(uuid.uuid4()),
                source_id=cid,
                source="cph-book",
                chunk_type="theory",
                title=f"CPH Ch.{doc.get('chapter_num')}: {doc.get('title', '')} (part {i+1})",
                text=t,
                metadata={**base_meta, "chunk_idx": i},
                token_estimate=estimate_tokens(t),
            ))

    for j, code in enumerate(doc.get("code_examples", [])[:3]):
        chunks.append(Chunk(
            chunk_id=str(uuid.uuid4()),
            source_id=cid,
            source="cph-book",
            chunk_type="code",
            title=f"CPH Ch.{doc.get('chapter_num')} Code Example {j+1}",
            text=f"```cpp\n{code}\n```",
            metadata={**base_meta, "code_index": j},
            token_estimate=estimate_tokens(code),
        ))

    return chunks


# ─── Source routing ───────────────────────────────────────────────────────────

PROCESSORS = {
    "codeforces": process_codeforces,
    "usaco": process_usaco,
    "atcoder": process_atcoder,
    "cp-algorithms": process_cp_algorithms,
    "cph_book": process_cph,
}


# ─── Pipeline orchestrator ────────────────────────────────────────────────────

# ─── Embedder backends ───────────────────────────────────────────────────────

class BaseEmbedder:
    """Common interface for all embedding backends."""
    def embed(self, texts: list[str]) -> list[list[float]]:
        raise NotImplementedError

    @property
    def dim(self) -> int:
        raise NotImplementedError

    @property
    def name(self) -> str:
        raise NotImplementedError


class OpenAIEmbedder(BaseEmbedder):
    """
    OpenAI text-embedding-3-small.
    Requires: pip install openai
    Env:       OPENAI_API_KEY
    Cost:      ~$0.02 per million tokens (~$0.05 for the full 2.3M-token corpus)
    """
    MODEL = "text-embedding-3-small"
    DIM = 1536
    BATCH = 512  # OpenAI max batch size

    def __init__(self):
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("Run: pip install openai")
        import os
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "OPENAI_API_KEY not set.\n"
                "Either export OPENAI_API_KEY=sk-... or use --embedder local for free local embeddings."
            )
        self._client = OpenAI(api_key=api_key)
        logger.info(f"OpenAI embedder ready (model={self.MODEL})")

    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors = []
        for i in range(0, len(texts), self.BATCH):
            batch = texts[i: i + self.BATCH]
            resp = self._client.embeddings.create(model=self.MODEL, input=batch)
            vectors.extend([r.embedding for r in resp.data])
            logger.debug(f"  OpenAI embedded {min(i + self.BATCH, len(texts))}/{len(texts)}")
        return vectors

    @property
    def dim(self) -> int:
        return self.DIM

    @property
    def name(self) -> str:
        return f"openai/{self.MODEL}"


class LocalEmbedder(BaseEmbedder):
    """
    Local sentence-transformers — runs on CPU or GPU, completely free.
    Requires: pip install sentence-transformers
    No API key needed.

    Default model: all-MiniLM-L6-v2
      - 384-dim, very fast on CPU (~1000 chunks/min)
      - Good quality for retrieval tasks

    For higher quality (slower):
      - BAAI/bge-small-en-v1.5  (384-dim, best small model for retrieval)
      - BAAI/bge-base-en-v1.5   (768-dim, excellent quality)
      - thenlper/gte-large       (1024-dim, near OpenAI quality)
    """
    DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"
    BATCH = 64

    def __init__(self, model_name: str = DEFAULT_MODEL):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError("Run: pip install sentence-transformers")

        logger.info(f"Loading local embedding model: {model_name}")
        logger.info("  (first run downloads the model ~90MB — cached after that)")
        self._model = SentenceTransformer(model_name)
        self._model_name = model_name
        self._dim = self._model.get_sentence_embedding_dimension()
        logger.info(f"Local embedder ready (dim={self._dim})")

    def embed(self, texts: list[str]) -> list[list[float]]:
        # BGE models benefit from a query prefix — for doc embedding use no prefix
        all_vecs = []
        for i in range(0, len(texts), self.BATCH):
            batch = texts[i: i + self.BATCH]
            vecs = self._model.encode(
                batch,
                normalize_embeddings=True,   # cosine similarity via dot product
                show_progress_bar=False,
            )
            all_vecs.extend(vecs.tolist())
            logger.debug(f"  Local embedded {min(i + self.BATCH, len(texts))}/{len(texts)}")
        return all_vecs

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def name(self) -> str:
        return f"local/{self._model_name}"


def make_embedder(embedder_type: str, local_model: str = LocalEmbedder.DEFAULT_MODEL) -> BaseEmbedder:
    """Factory — returns the right embedder based on CLI flag."""
    if embedder_type == "openai":
        return OpenAIEmbedder()
    elif embedder_type == "local":
        return LocalEmbedder(model_name=local_model)
    else:
        raise ValueError(f"Unknown embedder: {embedder_type}. Choose 'local' or 'openai'.")


# ─── RAG Pipeline ─────────────────────────────────────────────────────────────

class RAGPipeline:
    def __init__(
        self,
        data_dirs: dict[str, str],
        output_dir: str = "data/rag_chunks",
        vector_db: Literal["jsonl", "chroma", "qdrant"] = "jsonl",
        embedder: str = "local",                        # "local" or "openai"
        local_model: str = LocalEmbedder.DEFAULT_MODEL, # only used when embedder="local"
        qdrant_url: str = "http://localhost:6333",       # only used when vector_db="qdrant"
        qdrant_api_key: Optional[str] = None,
        collection_name: str = "cp_rag",
    ):
        self.data_dirs = {k: Path(v) for k, v in data_dirs.items()}
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.vector_db = vector_db
        self.embedder_type = embedder
        self.local_model = local_model
        self.qdrant_url = qdrant_url
        self.qdrant_api_key = qdrant_api_key
        self.collection_name = collection_name
        self._embedder: Optional[BaseEmbedder] = None

    def _get_embedder(self) -> BaseEmbedder:
        """Lazy-init embedder (only when actually needed for chroma/qdrant)."""
        if self._embedder is None:
            self._embedder = make_embedder(self.embedder_type, self.local_model)
        return self._embedder

    def load_documents(self, source: str) -> list[dict]:
        data_dir = self.data_dirs.get(source)
        if not data_dir or not data_dir.exists():
            logger.warning(f"Data directory not found for {source}: {data_dir}")
            return []
        docs = []
        for f in data_dir.glob("*.json"):
            if f.name == "index.json":
                continue
            try:
                docs.append(json.loads(f.read_text(encoding="utf-8")))
            except json.JSONDecodeError:
                logger.warning(f"Failed to parse {f}")
        logger.info(f"Loaded {len(docs)} documents from {source}")
        return docs

    def process_all(self) -> list[Chunk]:
        all_chunks = []
        for source, processor in PROCESSORS.items():
            docs = self.load_documents(source)
            for doc in docs:
                try:
                    chunks = processor(doc)
                    all_chunks.extend(chunks)
                except Exception as e:
                    logger.error(
                        f"Error processing {source} doc "
                        f"{doc.get('problem_id', doc.get('article_id', '?'))}: {e}"
                    )
        generated_count = len(all_chunks)
        unique_chunks: dict[str, Chunk] = {}
        for chunk in all_chunks:
            # Preserve indentation while normalizing line endings and trailing
            # whitespace for stable exact-duplicate detection.
            normalized_text = "\n".join(
                line.rstrip() for line in chunk.text.replace("\r\n", "\n").split("\n")
            ).strip()
            if not normalized_text:
                continue
            text_hash = hashlib.sha256(normalized_text.encode("utf-8")).hexdigest()
            if text_hash in unique_chunks:
                continue
            chunk.text = normalized_text
            chunk.chunk_id = text_hash
            unique_chunks[text_hash] = chunk

        deduplicated_chunks = list(unique_chunks.values())
        logger.info(
            "Generated %s chunks; retained %s unique chunks (%s duplicates removed)",
            generated_count,
            len(deduplicated_chunks),
            generated_count - len(deduplicated_chunks),
        )
        logger.info(f"  Breakdown: {self._chunk_stats(deduplicated_chunks)}")
        return deduplicated_chunks

    def _chunk_stats(self, chunks: list[Chunk]) -> str:
        from collections import Counter
        by_source = Counter(c.source for c in chunks)
        by_type = Counter(c.chunk_type for c in chunks)
        return f"by_source={dict(by_source)}, by_type={dict(by_type)}"

    def save_jsonl(self, chunks: list[Chunk]):
        """Saves all chunks as JSONL — no embedding needed, portable to any tool."""
        out_file = self.output_dir / "chunks.jsonl"
        with out_file.open("w", encoding="utf-8") as f:
            for chunk in chunks:
                f.write(json.dumps(asdict(chunk), ensure_ascii=False) + "\n")
        logger.info(f"Saved {len(chunks)} chunks to {out_file}")

        from collections import Counter
        stats = {
            "total_chunks": len(chunks),
            "total_tokens_estimate": sum(c.token_estimate for c in chunks),
            "embedder": self.embedder_type,
            "by_source": dict(Counter(c.source for c in chunks)),
            "by_type": dict(Counter(c.chunk_type for c in chunks)),
        }
        (self.output_dir / "stats.json").write_text(json.dumps(stats, indent=2))
        return out_file

    def ingest_to_chroma(self, chunks: list[Chunk]):
        """
        Ingest into ChromaDB with the selected embedder.
        - local:  sentence-transformers via chromadb's SentenceTransformerEmbeddingFunction
        - openai: OpenAI API via chromadb's OpenAIEmbeddingFunction
        Both paths let ChromaDB call the embedder internally per batch.
        """
        try:
            import chromadb
            from chromadb.utils import embedding_functions as ef
        except ImportError:
            logger.error("Run: pip install chromadb")
            return

        client = chromadb.PersistentClient(path=str(self.output_dir / "chroma_db"))

        if self.embedder_type == "local":
            embedding_fn = ef.SentenceTransformerEmbeddingFunction(
                model_name=self.local_model,
                normalize_embeddings=True,
            )
            logger.info(f"ChromaDB using local embedder: {self.local_model}")
        else:
            import os
            api_key = os.environ.get("OPENAI_API_KEY")
            if not api_key:
                raise EnvironmentError(
                    "OPENAI_API_KEY not set. Use --embedder local for free embeddings."
                )
            embedding_fn = ef.OpenAIEmbeddingFunction(
                api_key=api_key,
                model_name="text-embedding-3-small",
            )
            logger.info("ChromaDB using OpenAI embedder: text-embedding-3-small")

        # Rebuilding replaces the old index instead of appending to it.
        try:
            client.delete_collection(name=self.collection_name)
            logger.info("Deleted existing ChromaDB collection '%s'", self.collection_name)
        except chromadb.errors.NotFoundError:
            pass

        collection = client.create_collection(
            name=self.collection_name,
            embedding_function=embedding_fn,
            metadata={"hnsw:space": "cosine"},
        )

        BATCH = 64 if self.embedder_type == "local" else 256
        for i in range(0, len(chunks), BATCH):
            batch = chunks[i: i + BATCH]
            collection.add(
                ids=[c.chunk_id for c in batch],
                documents=[c.text for c in batch],
                metadatas=[
                    {**{k: v for k, v in c.metadata.items() if isinstance(v, (str, int, float, bool))},
                     "source": c.source, "chunk_type": c.chunk_type, "title": c.title}
                    for c in batch
                ],
            )
            logger.info(f"  ChromaDB: {min(i + BATCH, len(chunks))}/{len(chunks)} chunks ingested")

        logger.info(f"ChromaDB ingestion complete. Collection '{self.collection_name}' has {collection.count()} records.")
        logger.info(f"  DB path: {self.output_dir / 'chroma_db'}")
        logger.info("")
        logger.info("  Query example:")
        logger.info("    import chromadb")
        logger.info(f"    client = chromadb.PersistentClient('{self.output_dir / 'chroma_db'}')")
        logger.info(f"    col = client.get_collection('{self.collection_name}')")
        logger.info("    results = col.query(query_texts=['segment tree lazy propagation'], n_results=5)")

    def ingest_to_qdrant(self, chunks: list[Chunk]):
        """
        Ingest into Qdrant — works with both local Docker and Qdrant Cloud.

        Local (free, no key):
            docker run -p 6333:6333 qdrant/qdrant
            python main.py --vector-db qdrant

        Qdrant Cloud (free tier available):
            python main.py --vector-db qdrant \\
                --qdrant-url https://<cluster>.qdrant.io \\
                --qdrant-api-key <key>
        """
        try:
            from qdrant_client import QdrantClient
            from qdrant_client.models import Distance, VectorParams, PointStruct, PayloadSchemaType
        except ImportError:
            logger.error("Run: pip install qdrant-client")
            return

        embedder = self._get_embedder()

        # Connect
        client_kwargs: dict = {"url": self.qdrant_url}
        if self.qdrant_api_key:
            client_kwargs["api_key"] = self.qdrant_api_key
        client = QdrantClient(**client_kwargs)
        logger.info(f"Qdrant connected: {self.qdrant_url}")

        # Create collection if it doesn't exist
        existing = [c.name for c in client.get_collections().collections]
        if self.collection_name not in existing:
            client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=embedder.dim, distance=Distance.COSINE),
            )
            # Index metadata fields for filtered search
            client.create_payload_index(self.collection_name, "source", PayloadSchemaType.KEYWORD)
            client.create_payload_index(self.collection_name, "chunk_type", PayloadSchemaType.KEYWORD)
            client.create_payload_index(self.collection_name, "rating", PayloadSchemaType.INTEGER)
            logger.info(f"Created Qdrant collection '{self.collection_name}' (dim={embedder.dim})")
        else:
            logger.info(f"Qdrant collection '{self.collection_name}' already exists — upserting")

        # Embed and upsert in batches
        BATCH = 64
        for i in range(0, len(chunks), BATCH):
            batch = chunks[i: i + BATCH]
            texts = [c.text for c in batch]
            vectors = embedder.embed(texts)

            points = [
                PointStruct(
                    id=str(c.chunk_id),
                    vector=vec,
                    payload={
                        "source": c.source,
                        "source_id": c.source_id,
                        "chunk_type": c.chunk_type,
                        "title": c.title,
                        "text": c.text,          # store text in payload for retrieval
                        **{k: v for k, v in c.metadata.items()
                           if isinstance(v, (str, int, float, bool))},
                    },
                )
                for c, vec in zip(batch, vectors)
            ]
            client.upsert(collection_name=self.collection_name, points=points)
            logger.info(f"  Qdrant: {min(i + BATCH, len(chunks))}/{len(chunks)} chunks upserted")

        count = client.get_collection(self.collection_name).points_count
        logger.info(f"Qdrant ingestion complete. Collection '{self.collection_name}' has {count} points.")
        logger.info("")
        logger.info("  Query example:")
        logger.info("    from qdrant_client import QdrantClient")
        logger.info(f"    client = QdrantClient(url='{self.qdrant_url}')")
        logger.info(f"    results = client.search('{self.collection_name}', query_vector=<embed your query>, limit=5)")

    def run(self):
        """Full pipeline: load → chunk → save JSONL → embed → ingest."""
        chunks = self.process_all()
        self.save_jsonl(chunks)

        if self.vector_db == "chroma":
            self.ingest_to_chroma(chunks)
        elif self.vector_db == "qdrant":
            self.ingest_to_qdrant(chunks)
        else:
            logger.info("JSONL output only — skipping vector DB ingestion.")
            logger.info("Re-run with --vector-db chroma or --vector-db qdrant to ingest.")

        return chunks


# ─── CLI entry ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    import os
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser(
        description="CP RAG Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Embedder options:
  --embedder local    Free, no API key. Uses sentence-transformers (BAAI/bge-small-en-v1.5).
                      First run downloads ~90MB model; cached after that.
  --embedder openai   Requires OPENAI_API_KEY env var.
                      Cost: ~$0.02 for the full 2.3M-token corpus.

Local model alternatives (pass via --local-model):
  BAAI/bge-small-en-v1.5   384-dim  fast, good quality  (default)
  BAAI/bge-base-en-v1.5    768-dim  better quality, 2x slower
  thenlper/gte-large        1024-dim near-OpenAI quality, needs GPU

Examples:
  python backend/RAG/pipeline/rag_pipeline.py --data-root backend/data --embedder local --vector-db chroma
  python backend/RAG/pipeline/rag_pipeline.py --embedder openai --vector-db qdrant --qdrant-url http://localhost:6333
  python backend/RAG/pipeline/rag_pipeline.py --embedder local --vector-db qdrant \\
      --qdrant-url https://xyz.qdrant.io --qdrant-api-key YOUR_KEY
        """
    )
    parser.add_argument("--vector-db", choices=["jsonl", "chroma", "qdrant"], default="jsonl",
                        help="Vector DB backend (default: jsonl)")
    parser.add_argument("--embedder", choices=["local", "openai"], default="local",
                        help="Embedding backend (default: local — no API key needed)")
    parser.add_argument("--local-model", default=LocalEmbedder.DEFAULT_MODEL,
                        help=f"sentence-transformers model name (default: {LocalEmbedder.DEFAULT_MODEL})")
    parser.add_argument("--qdrant-url", default="http://localhost:6333",
                        help="Qdrant server URL (default: http://localhost:6333)")
    parser.add_argument("--qdrant-api-key", default=os.environ.get("QDRANT_API_KEY"),
                        help="Qdrant API key (or set QDRANT_API_KEY env var; leave blank for local Docker)")
    parser.add_argument("--collection", default="cp_rag",
                        help="Vector DB collection name (default: cp_rag)")
    parser.add_argument("--data-root", default="data")
    args = parser.parse_args()

    pipeline = RAGPipeline(
        data_dirs={
            "codeforces":    f"{args.data_root}/codeforces",
            "usaco":         f"{args.data_root}/usaco",
            "atcoder":       f"{args.data_root}/atcoder",
            "cp-algorithms": f"{args.data_root}/cp_algorithms",
            "cph_book":      f"{args.data_root}/cph",
        },
        output_dir=f"{args.data_root}/rag_chunks",
        vector_db=args.vector_db,
        embedder=args.embedder,
        local_model=args.local_model,
        qdrant_url=args.qdrant_url,
        qdrant_api_key=args.qdrant_api_key,
        collection_name=args.collection,
    )
    pipeline.run()
