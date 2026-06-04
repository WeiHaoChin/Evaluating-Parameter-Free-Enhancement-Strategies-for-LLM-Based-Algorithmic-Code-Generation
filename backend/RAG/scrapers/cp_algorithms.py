"""
CP-Algorithms Scraper
Scrapes https://cp-algorithms.com — a comprehensive reference for competitive programming algorithms.
Content is structured by topic/algorithm, making it ideal for RAG knowledge chunks.
"""

import asyncio
import json
import logging
import re
from pathlib import Path
from dataclasses import dataclass, asdict, field
from typing import Optional
from urllib.parse import urljoin, urlparse

import httpx
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

CP_ALG_BASE = "https://cp-algorithms.com"
GITHUB_SOURCE = "https://raw.githubusercontent.com/e-maxx-eng/e-maxx-eng/master/src"
RATE_LIMIT_DELAY = 1.0


@dataclass
class AlgorithmArticle:
    source: str = "cp-algorithms"
    article_id: str = ""
    url: str = ""
    title: str = ""
    category: str = ""          # e.g., "graph", "string", "algebra"
    subcategory: str = ""
    content_html: str = ""
    content_markdown: str = ""  # Raw markdown from GitHub source
    concepts: list = field(default_factory=list)  # Extracted key concepts
    complexity: str = ""        # Time/space complexity if mentioned
    prerequisites: list = field(default_factory=list)
    related_topics: list = field(default_factory=list)
    code_snippets: list = field(default_factory=list)  # Extracted C++ code blocks


# The site's navigation structure — scrape these categories
CATEGORIES = {
    "algebra": ["fundamentals", "prime_numbers", "number_theory", "modular_arithmetic", "matrix"],
    "data_structures": ["stack_queue", "trees", "segment_trees", "sqrt_decomposition", "misc"],
    "dynamic_programming": ["introduction", "advanced"],
    "string_processing": ["fundamentals", "string_search", "suffix_structures"],
    "linear_algebra": ["gauss_elimination", "linear_recurrence"],
    "combinatorics": ["fundamentals", "catalan", "inclusion_exclusion", "burnside"],
    "numerical_methods": ["binary_search", "ternary_search", "newton"],
    "geometry": ["elementary", "polygon", "convex_hull", "sweep_line", "misc"],
    "graph_theory": ["graph_traversal", "connected_components", "shortest_paths",
                     "spanning_trees", "cycles", "topological_sort", "matchings",
                     "flows", "misc"],
    "misc": ["misc"],
}


class CPAlgorithmsScraper:
    def __init__(self, output_dir: str = "data/cp_algorithms"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.visited: set[str] = set()
        self.client = None

    async def __aenter__(self):
        self.client = httpx.AsyncClient(
            timeout=30,
            headers={"User-Agent": "CP-RAG-Scraper/1.0 (educational research)"},
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
            except httpx.HTTPStatusError as e:
                if e.response.status_code == 404:
                    return None
                logger.warning(f"HTTP {e.response.status_code}: {url}")
            except Exception as e:
                logger.warning(f"Attempt {attempt+1} for {url}: {e}")
                await asyncio.sleep(2 ** attempt)
        return None

    # ── Crawl navigation to find all article URLs ─────────────────────────
    async def discover_all_articles(self) -> list[dict]:
        """Fetches the main page and crawls the navigation sidebar."""
        html = await self._get(CP_ALG_BASE)
        if not html:
            return []

        soup = BeautifulSoup(html, "html.parser")
        articles = []

        # CP-algorithms has a nav with nested sections
        nav = soup.find("nav", class_="md-nav") or soup.find("nav")
        if nav:
            current_category = "general"
            for element in nav.find_all(["span", "a"]):
                if element.name == "span" and "md-nav__link" in element.get("class", []):
                    current_category = element.get_text(strip=True).lower().replace(" ", "_")
                elif element.name == "a" and element.get("href"):
                    href = element["href"]
                    if href.startswith("/") or href.startswith("http"):
                        full_url = urljoin(CP_ALG_BASE, href)
                        parsed = urlparse(full_url)
                        if parsed.netloc == "cp-algorithms.com" and parsed.path.endswith(".html"):
                            articles.append({
                                "url": full_url,
                                "title": element.get_text(strip=True),
                                "category": current_category,
                            })

        # Fallback: discover from sitemap or known URL patterns
        if len(articles) < 10:
            articles = await self._discover_from_sitemap()

        # Deduplicate
        seen = set()
        unique = []
        for a in articles:
            if a["url"] not in seen:
                seen.add(a["url"])
                unique.append(a)

        logger.info(f"Discovered {len(unique)} articles.")
        return unique

    async def _discover_from_sitemap(self) -> list[dict]:
        """Fallback: try sitemap.xml."""
        sitemap = await self._get(f"{CP_ALG_BASE}/sitemap.xml")
        if not sitemap:
            return []

        soup = BeautifulSoup(sitemap, "xml")
        articles = []
        for loc in soup.find_all("loc"):
            url = loc.get_text(strip=True)
            if url.endswith(".html"):
                path_parts = urlparse(url).path.strip("/").split("/")
                category = path_parts[0] if len(path_parts) > 1 else "general"
                articles.append({"url": url, "title": "", "category": category})

        return articles

    # ── Fetch markdown source from GitHub ────────────────────────────────
    async def fetch_markdown_source(self, article_url: str) -> str:
        """
        CP-algorithms source lives on GitHub as .md files.
        Maps URL like /graph/dfs.html -> GitHub src/graph/dfs.md
        """
        path = urlparse(article_url).path  # e.g., /graph/dfs.html
        md_path = path.replace(".html", ".md").lstrip("/")
        github_url = f"{GITHUB_SOURCE}/{md_path}"
        content = await self._get(github_url)
        return content or ""

    # ── Extract structured data from article ─────────────────────────────
    def extract_article_data(self, html: str, meta: dict) -> dict:
        soup = BeautifulSoup(html, "html.parser")
        result = {
            "content_html": "",
            "concepts": [],
            "complexity": "",
            "prerequisites": [],
            "related_topics": [],
            "code_snippets": [],
        }

        # Main content
        main = (soup.find("article") or soup.find("div", class_="md-content") or
                soup.find("div", {"role": "main"}))
        if main:
            result["content_html"] = str(main)

            # Extract C++ code blocks
            for code in main.find_all("code", class_=re.compile("cpp|c\\+\\+")):
                snippet = code.get_text(strip=True)
                if len(snippet) > 50:  # Skip tiny snippets
                    result["code_snippets"].append(snippet)

            # Extract complexity mentions (O(n log n) etc.)
            text = main.get_text()
            complexities = re.findall(r"O\([^)]+\)", text)
            if complexities:
                result["complexity"] = ", ".join(dict.fromkeys(complexities))  # unique, ordered

            # Extract headings as concepts
            for h in main.find_all(["h2", "h3"]):
                concept = h.get_text(strip=True)
                if concept and len(concept) < 100:
                    result["concepts"].append(concept)

            # Look for prerequisites sections
            prereq_section = main.find(string=re.compile(r"prerequisite|before reading", re.IGNORECASE))
            if prereq_section:
                parent = prereq_section.parent
                if parent:
                    ul = parent.find_next("ul")
                    for a in (ul.find_all("a") if ul else []):
                        result["prerequisites"].append(a.get_text(strip=True))

        return result

    # ── Scrape a single article ───────────────────────────────────────────
    async def scrape_article(self, meta: dict) -> Optional[dict]:
        url = meta["url"]
        article_id = urlparse(url).path.strip("/").replace("/", "_").replace(".html", "")

        out_file = self.output_dir / f"{article_id}.json"
        if out_file.exists():
            return json.loads(out_file.read_text(encoding="utf-8"))

        html = await self._get(url)
        if not html:
            return None

        # Get title from HTML if not already known
        title = meta.get("title", "")
        if not title:
            soup = BeautifulSoup(html, "html.parser")
            h1 = soup.find("h1")
            title = h1.get_text(strip=True) if h1 else article_id

        # Fetch markdown source
        markdown = await self.fetch_markdown_source(url)

        extracted = self.extract_article_data(html, meta)

        article = AlgorithmArticle(
            article_id=article_id,
            url=url,
            title=title,
            category=meta.get("category", "general"),
            content_html=extracted["content_html"],
            content_markdown=markdown,
            concepts=extracted["concepts"],
            complexity=extracted["complexity"],
            prerequisites=extracted["prerequisites"],
            related_topics=extracted["related_topics"],
            code_snippets=extracted["code_snippets"],
        )

        doc = asdict(article)
        out_file.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")
        return doc

    # ── Main entry ────────────────────────────────────────────────────────
    async def scrape(self) -> list[dict]:
        logger.info("Starting CP-Algorithms scrape...")
        articles_meta = await self.discover_all_articles()
        results = []

        for i, meta in enumerate(articles_meta):
            logger.info(f"[{i+1}/{len(articles_meta)}] {meta.get('title', meta['url'])}")
            article = await self.scrape_article(meta)
            if article:
                results.append(article)

        logger.info(f"CP-Algorithms scrape complete. {len(results)} articles saved.")
        return results


async def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    async with CPAlgorithmsScraper() as scraper:
        await scraper.scrape()


if __name__ == "__main__":
    asyncio.run(main())
