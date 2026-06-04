"""
AtCoder Scraper
Scrapes AtCoder contest problems and editorial blog posts.
Uses the AtCoder Problems API (unofficial) for metadata and direct scraping for editorials.
"""

import asyncio
import json
import logging
import re
from pathlib import Path
from dataclasses import dataclass, asdict, field
from typing import Optional

import httpx
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

# Unofficial AtCoder Problems API (kenkoooo.com) — widely used, respects robots.txt
ATCODER_PROBLEMS_API = "https://kenkoooo.com/atcoder/resources"
ATCODER_BASE = "https://atcoder.jp"
RATE_LIMIT_DELAY = 2.0  # AtCoder is strict; be respectful


@dataclass
class AtCoderProblem:
    source: str = "atcoder"
    problem_id: str = ""        # e.g. "abc300_a"
    contest_id: str = ""        # e.g. "abc300"
    contest_type: str = ""      # ABC / ARC / AGC
    title: str = ""
    difficulty: Optional[int] = None   # Elo-style rating from AtCoder Problems
    statement_html: str = ""
    examples: list = field(default_factory=list)
    editorial_url: str = ""
    editorial_html: str = ""
    editorial_text: str = ""
    tags: list = field(default_factory=list)


class AtCoderScraper:
    def __init__(self, output_dir: str = "data/atcoder", max_problems: int = 300,
                 contest_types: tuple = ("abc", "arc", "agc")):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.max_problems = max_problems
        self.contest_types = contest_types
        self.client = None

    async def __aenter__(self):
        self.client = httpx.AsyncClient(
            timeout=30,
            headers={
                "User-Agent": "CP-RAG-Scraper/1.0 (educational research; contact: rag@example.com)",
                "Accept-Language": "en,ja;q=0.9",
            },
            follow_redirects=True,
        )
        return self

    async def __aexit__(self, *args):
        await self.client.aclose()

    async def _get(self, url: str, retries: int = 3, as_json: bool = False):
        for attempt in range(retries):
            try:
                resp = await self.client.get(url)
                resp.raise_for_status()
                await asyncio.sleep(RATE_LIMIT_DELAY)
                return resp.json() if as_json else resp.text
            except httpx.HTTPStatusError as e:
                if e.response.status_code in (403, 404):
                    return None
                logger.warning(f"HTTP {e.response.status_code} for {url}")
            except Exception as e:
                logger.warning(f"Attempt {attempt+1} failed: {e}")
                await asyncio.sleep(2 ** attempt)
        return None

    # ── Fetch problem list from AtCoder Problems API ──────────────────────
    async def fetch_problem_list(self) -> list[dict]:
        """Uses kenkoooo API which provides difficulty ratings and metadata."""
        problems_url = f"{ATCODER_PROBLEMS_API}/problems.json"
        difficulties_url = f"{ATCODER_PROBLEMS_API}/problem-models.json"

        problems = await self._get(problems_url, as_json=True)
        difficulties = await self._get(difficulties_url, as_json=True)

        if not problems:
            logger.error("Failed to fetch AtCoder problem list")
            return []

        # Merge difficulty info
        diff_map = difficulties or {}
        enriched = []
        for p in problems:
            pid = p.get("id", "")
            contest = p.get("contest_id", "")
            if not any(contest.startswith(ct) for ct in self.contest_types):
                continue

            diff_info = diff_map.get(pid, {})
            enriched.append({
                "problem_id": pid,
                "contest_id": contest,
                "title": p.get("title", ""),
                "difficulty": diff_info.get("difficulty"),
            })

        # Sort by difficulty, pick a spread across difficulty levels
        enriched = [p for p in enriched if p["difficulty"] is not None]
        enriched.sort(key=lambda p: p["difficulty"])

        # Sample across the difficulty spectrum for diverse training data
        step = max(1, len(enriched) // self.max_problems)
        sampled = enriched[::step][: self.max_problems]
        logger.info(f"Sampled {len(sampled)} AtCoder problems across difficulty range.")
        return sampled

    # ── Scrape problem statement ──────────────────────────────────────────
    async def scrape_problem_statement(self, contest_id: str, problem_id: str) -> dict:
        # AtCoder problem URL format
        # English task pages: /contests/{contest}/tasks/{problem}
        task_name = problem_id  # AtCoder Problems IDs are already in task format

        for lang in ["en", ""]:  # Try English first, fall back
            url = f"{ATCODER_BASE}/contests/{contest_id}/tasks/{task_name}"
            if lang:
                url += f"?lang={lang}"
            html = await self._get(url)
            if not html:
                continue

            soup = BeautifulSoup(html, "html.parser")
            result = {"statement_html": "", "examples": []}

            # AtCoder statement div
            stmt = soup.find("div", id="task-statement")
            if stmt:
                result["statement_html"] = str(stmt)

                # Extract examples (AtCoder uses "入力例" / "Sample Input")
                pre_tags = stmt.find_all("pre")
                examples = []
                for j in range(0, len(pre_tags) - 1, 2):
                    examples.append({
                        "input": pre_tags[j].get_text("\n", strip=True),
                        "output": pre_tags[j+1].get_text("\n", strip=True),
                    })
                result["examples"] = examples
                return result

        return {}

    # ── Find editorial URL ────────────────────────────────────────────────
    async def find_editorial_url(self, contest_id: str) -> str:
        """AtCoder editorials are linked from the contest top page."""
        url = f"{ATCODER_BASE}/contests/{contest_id}"
        html = await self._get(url)
        if not html:
            return ""

        soup = BeautifulSoup(html, "html.parser")
        for a in soup.find_all("a", href=True):
            text = a.get_text(strip=True).lower()
            href = a["href"]
            if "editorial" in text or "解説" in text:  # 解説 = editorial in Japanese
                if href.startswith("http"):
                    return href
                return ATCODER_BASE + href

        return ""

    # ── Scrape editorial ──────────────────────────────────────────────────
    async def scrape_editorial(self, editorial_url: str) -> tuple[str, str]:
        if not editorial_url:
            return "", ""

        html = await self._get(editorial_url)
        if not html:
            return editorial_url, ""

        soup = BeautifulSoup(html, "html.parser")

        # AtCoder editorial pages (official) — content in task-statement or editorial divs
        for selector in ["div#editorial", "div.editorial", "div#task-statement",
                         "div.col-sm-12", "article"]:
            content = soup.select_one(selector)
            if content:
                return editorial_url, str(content)

        # Fallback: main content area
        body = soup.find("body")
        return editorial_url, str(body) if body else ""

    # ── Orchestrate ───────────────────────────────────────────────────────
    async def scrape(self) -> list[dict]:
        logger.info("Starting AtCoder scrape...")
        problems_meta = await self.fetch_problem_list()
        results = []

        # Cache editorial URLs per contest
        editorial_cache: dict[str, str] = {}

        for i, meta in enumerate(problems_meta):
            pid = meta["problem_id"]
            cid = meta["contest_id"]

            out_file = self.output_dir / f"{pid}.json"
            if out_file.exists():
                logger.debug(f"[{i+1}/{len(problems_meta)}] Skipping {pid} (cached)")
                results.append(json.loads(out_file.read_text(encoding="utf-8")))
                continue

            logger.info(f"[{i+1}/{len(problems_meta)}] Scraping {pid} (difficulty ~{meta['difficulty']:.0f})...")

            problem = AtCoderProblem(
                problem_id=pid,
                contest_id=cid,
                contest_type=cid[:3].upper(),
                title=meta["title"],
                difficulty=int(meta["difficulty"]) if meta["difficulty"] else None,
            )

            # Statement
            stmt = await self.scrape_problem_statement(cid, pid)
            problem.statement_html = stmt.get("statement_html", "")
            problem.examples = stmt.get("examples", [])

            # Editorial (cached per contest)
            if cid not in editorial_cache:
                editorial_cache[cid] = await self.find_editorial_url(cid)

            ed_url = editorial_cache[cid]
            ed_url, ed_html = await self.scrape_editorial(ed_url)
            problem.editorial_url = ed_url
            problem.editorial_html = ed_html

            doc = asdict(problem)
            out_file.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8") 
            results.append(doc)

        logger.info(f"AtCoder scrape complete. {len(results)} problems saved.")
        return results


async def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    async with AtCoderScraper(max_problems=150) as scraper:
        await scraper.scrape()


if __name__ == "__main__":
    asyncio.run(main())
