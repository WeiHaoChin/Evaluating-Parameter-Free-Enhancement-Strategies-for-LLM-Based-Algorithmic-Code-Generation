"""
Competitive Programmer's Handbook Scraper
Downloads and parses the CPH by Antti Laaksonen (freely available under CC BY-NC-SA 4.0).
Source: https://cses.fi/book/book.pdf and LaTeX source on GitHub.

Since the book is openly licensed, we can include structured excerpts.
We prefer the LaTeX source for cleaner structured extraction.
"""

import asyncio
import json
import logging
import re
from pathlib import Path
from dataclasses import dataclass, asdict, field
from typing import Optional

import httpx

logger = logging.getLogger(__name__)

# GitHub repo for the book's LaTeX source (official)
CPH_GITHUB = "https://api.github.com/repos/pllk/cphb/contents"
CPH_RAW = "https://raw.githubusercontent.com/pllk/cphb/master"
CPH_PDF_URL = "https://cses.fi/book/book.pdf"

RATE_LIMIT_DELAY = 1.0


@dataclass
class CPHChapter:
    source: str = "cph_book"
    chapter_id: str = ""
    chapter_num: int = 0
    title: str = ""
    part: str = ""             # "Basic Techniques", "Graph Algorithms", etc.
    content_latex: str = ""    # Raw LaTeX
    content_text: str = ""     # Cleaned plain text
    topics: list = field(default_factory=list)
    algorithms: list = field(default_factory=list)   # Named algorithms in chapter
    data_structures: list = field(default_factory=list)
    code_examples: list = field(default_factory=list)  # C++ code blocks
    complexity_notes: list = field(default_factory=list)
    page_start: int = 0
    page_end: int = 0


# Book structure (from CPH table of contents)
CPH_STRUCTURE = {
    "I Basic Techniques": [
        (1, "Introduction"),
        (2, "Time Complexity"),
        (3, "Sorting"),
        (4, "Data Structures"),
        (5, "Complete Search"),
        (6, "Greedy Algorithms"),
        (7, "Dynamic Programming"),
        (8, "Amortized Analysis"),
        (9, "Range Queries"),
        (10, "Bit Manipulation"),
    ],
    "II Graph Algorithms": [
        (11, "Basics of Graphs"),
        (12, "Graph Traversal"),
        (13, "Shortest Paths"),
        (14, "Tree Algorithms"),
        (15, "Spanning Trees"),
        (16, "Directed Graphs"),
        (17, "Strong Connectivity"),
        (18, "Tree Queries"),
        (19, "Paths and Circuits"),
        (20, "Flows and Cuts"),
    ],
    "III Advanced Topics": [
        (21, "Number Theory"),
        (22, "Combinatorics"),
        (23, "Matrices"),
        (24, "Probability"),
        (25, "Game Theory"),
        (26, "String Algorithms"),
        (27, "Square Root Algorithms"),
        (28, "Segment Trees Revisited"),
        (29, "Geometry"),
        (30, "Sweep Line Algorithms"),
    ],
}


class CPHScraper:
    def __init__(self, output_dir: str = "data/cph"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.client = None

    async def __aenter__(self):
        self.client = httpx.AsyncClient(
            timeout=60,
            headers={
                "User-Agent": "CP-RAG-Scraper/1.0 (educational research)",
                "Accept": "application/vnd.github.v3+json",
            },
            follow_redirects=True,
        )
        return self

    async def __aexit__(self, *args):
        await self.client.aclose()

    async def _get(self, url: str, retries: int = 3) -> Optional[str]:
        for attempt in range(retries):
            try:
                resp = await self.client.get(url)
                resp.raise_for_status()
                await asyncio.sleep(RATE_LIMIT_DELAY)
                return resp.text
            except Exception as e:
                logger.warning(f"Attempt {attempt+1} for {url}: {e}")
                await asyncio.sleep(2 ** attempt)
        return None

    # ── Fetch chapter LaTeX source ────────────────────────────────────────
    async def fetch_chapter_source(self, chapter_num: int, title: str) -> str:
        """
        CPH GitHub repo structure:
        - chapter files are typically named chapter{N}.tex or {slug}.tex
        """
        slug = title.lower().replace(" ", "").replace("/", "")

        # Try common filename patterns
        for filename in [
            f"chapter{chapter_num:02d}.tex",
            f"chapter{chapter_num}.tex",
            f"{slug}.tex",
        ]:
            url = f"{CPH_RAW}/{filename}"
            content = await self._get(url)
            if content and len(content) > 100:
                return content

        # Fallback: list repo contents to find the right file
        repo_contents = await self._get(CPH_GITHUB)
        if repo_contents:
            import json as _json
            try:
                files = _json.loads(repo_contents)
                for f in files:
                    name = f.get("name", "")
                    if name.endswith(".tex") and (
                        str(chapter_num) in name or slug[:6] in name.lower()
                    ):
                        raw_url = f.get("download_url", "")
                        if raw_url:
                            return await self._get(raw_url) or ""
            except Exception:
                pass

        return ""

    # ── Parse LaTeX content ───────────────────────────────────────────────
    def parse_latex(self, latex: str, chapter_num: int, title: str, part: str) -> CPHChapter:
        chapter = CPHChapter(
            chapter_id=f"cph_ch{chapter_num:02d}",
            chapter_num=chapter_num,
            title=title,
            part=part,
            content_latex=latex,
        )

        # Clean LaTeX to readable text
        text = latex
        # Remove LaTeX commands while keeping content
        text = re.sub(r"\\chapter\{([^}]+)\}", r"\1\n" + "="*50, text)
        text = re.sub(r"\\section\{([^}]+)\}", r"\n## \1\n", text)
        text = re.sub(r"\\subsection\{([^}]+)\}", r"\n### \1\n", text)
        text = re.sub(r"\\textbf\{([^}]+)\}", r"**\1**", text)
        text = re.sub(r"\\emph\{([^}]+)\}", r"*\1*", text)
        text = re.sub(r"\\textit\{([^}]+)\}", r"*\1*", text)
        text = re.sub(r"\\url\{([^}]+)\}", r"\1", text)
        text = re.sub(r"\\label\{[^}]+\}", "", text)
        text = re.sub(r"\\ref\{[^}]+\}", "[ref]", text)
        text = re.sub(r"\\index\{[^}]+\}", "", text)
        text = re.sub(r"\\footnote\{([^}]+)\}", r" (\1)", text)
        text = re.sub(r"%.*$", "", text, flags=re.MULTILINE)  # Remove LaTeX comments
        text = re.sub(r"\n{3,}", "\n\n", text)

        chapter.content_text = text.strip()

        # Extract C++ code blocks
        code_blocks = re.findall(
            r"\\begin\{lstlisting\}(.*?)\\end\{lstlisting\}",
            latex, re.DOTALL
        )
        # Also look for verbatim and minted
        code_blocks += re.findall(
            r"\\begin\{(?:verbatim|minted)\{cpp\}\}(.*?)\\end\{(?:verbatim|minted)\}",
            latex, re.DOTALL
        )
        chapter.code_examples = [c.strip() for c in code_blocks if len(c.strip()) > 20]

        # Extract section titles as topics
        sections = re.findall(r"\\(?:sub)?section\{([^}]+)\}", latex)
        chapter.topics = sections

        # Extract complexity annotations
        complexities = re.findall(r"O\([^)]+\)", text)
        chapter.complexity_notes = list(dict.fromkeys(complexities))

        # Named algorithms/data structures
        alg_patterns = [
            r"(?:algorithm|method|approach|technique):\s*([A-Z][a-zA-Z\s]+?)(?:\n|\.)",
            r"\\emph\{([A-Z][a-zA-Z\s]+?)\}",
        ]
        algorithms = []
        for pattern in alg_patterns:
            algorithms.extend(re.findall(pattern, latex))
        chapter.algorithms = list(dict.fromkeys(algorithms[:20]))  # Cap at 20

        return chapter

    # ── Generate synthetic chapter from structure if source unavailable ───
    def create_stub_chapter(self, chapter_num: int, title: str, part: str) -> CPHChapter:
        """Creates a metadata-only record when LaTeX source isn't available."""
        return CPHChapter(
            chapter_id=f"cph_ch{chapter_num:02d}",
            chapter_num=chapter_num,
            title=title,
            part=part,
            content_latex="",
            content_text=f"Chapter {chapter_num}: {title} (Part: {part})\n[LaTeX source not available — use PDF extraction]",
        )

    # ── Download the PDF as fallback ──────────────────────────────────────
    async def download_pdf(self) -> Optional[bytes]:
        """Downloads CPH PDF for offline extraction if needed."""
        try:
            resp = await self.client.get(CPH_PDF_URL)
            resp.raise_for_status()
            pdf_path = self.output_dir / "cph_book.pdf"
            pdf_path.write_bytes(resp.content)
            logger.info(f"Downloaded CPH PDF ({len(resp.content)//1024}KB) to {pdf_path}")
            return resp.content
        except Exception as e:
            logger.error(f"PDF download failed: {e}")
            return None

    # ── Main entry ────────────────────────────────────────────────────────
    async def scrape(self) -> list[dict]:
        logger.info("Starting CPH scrape...")
        results = []

        # First, download the PDF for completeness
        await self.download_pdf()

        # Scrape each chapter
        for part, chapters in CPH_STRUCTURE.items():
            for chapter_num, title in chapters:
                out_file = self.output_dir / f"cph_ch{chapter_num:02d}.json"
                if out_file.exists():
                    logger.debug(f"Skipping Ch.{chapter_num} (cached)")
                    results.append(json.loads(out_file.read_text(encoding="utf-8")))
                    continue

                logger.info(f"Fetching Chapter {chapter_num}: {title} ({part})")
                latex_source = await self.fetch_chapter_source(chapter_num, title)

                if latex_source:
                    chapter = self.parse_latex(latex_source, chapter_num, title, part)
                else:
                    logger.warning(f"  No LaTeX source for Ch.{chapter_num} — creating stub")
                    chapter = self.create_stub_chapter(chapter_num, title, part)

                doc = asdict(chapter)
                out_file.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")
                results.append(doc)

        # Save the full book index
        index = {
            "title": "Competitive Programmer's Handbook",
            "author": "Antti Laaksonen",
            "license": "CC BY-NC-SA 4.0",
            "source": "https://cses.fi/book/",
            "chapters": [
                {"id": r["chapter_id"], "num": r["chapter_num"],
                 "title": r["title"], "part": r["part"]}
                for r in results
            ],
        }
        (self.output_dir / "index.json").write_text(json.dumps(index, indent=2))

        logger.info(f"CPH scrape complete. {len(results)} chapters saved.")
        return results


async def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    async with CPHScraper() as scraper:
        await scraper.scrape()


if __name__ == "__main__":
    asyncio.run(main())
