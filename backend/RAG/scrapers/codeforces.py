"""
Codeforces Scraper
Scrapes problems, editorials, tags, and contest data via Codeforces public API + HTML parsing.
"""

import asyncio
import json
import time
import re
import logging
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, asdict

import httpx
from bs4 import BeautifulSoup
from backend.RAG.pipeline.rag_pipeline import (
    extract_codeforces_problem_editorial,
    html_to_text,
    is_useful_editorial,
)
try:
    import cloudscraper
except ImportError:
    cloudscraper = None

logger = logging.getLogger(__name__)

CF_API = "https://codeforces.com/api"
CF_BASE = "https://codeforces.com"

RATE_LIMIT_DELAY = 1.5  # seconds between requests (CF TOS: be polite)


@dataclass
class CFProblem:
    source: str = "codeforces"
    problem_id: str = ""       # e.g. "1234A"
    contest_id: int = 0
    index: str = ""
    name: str = ""
    rating: Optional[int] = None
    tags: list = None
    statement_html: str = ""
    editorial_html: str = ""
    editorial_url: str = ""
    input_spec: str = ""
    output_spec: str = ""
    examples: list = None      # [{"input": ..., "output": ...}]
    time_limit_ms: int = 0
    memory_limit_mb: int = 0

    def __post_init__(self):
        if self.tags is None:
            self.tags = []
        if self.examples is None:
            self.examples = []


class CodeforcesScraper:
    def __init__(self, output_dir: str = "data/codeforces", max_problems: int = 500):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.max_problems = max_problems
        self.client = None
        self.cf_scraper = None

    async def __aenter__(self):
        self.client = httpx.AsyncClient(
            timeout=30,
            headers={"User-Agent": "CP-RAG-Scraper/1.0 (educational research)"},
            follow_redirects=True,
        )
        # Initialize cloudscraper for HTML scraping (bypasses Cloudflare)
        if cloudscraper:
            self.cf_scraper = cloudscraper.create_scraper(
                browser={'browser': 'chrome', 'platform': 'windows', 'desktop': True}
            )
        else:
            logger.warning("cloudscraper not installed. HTML scraping will fail. Install with: pip install cloudscraper")
        return self

    async def __aexit__(self, *args):
        await self.client.aclose()
        if self.cf_scraper:
            self.cf_scraper.close()

    async def _get(self, url: str, params: dict = None, retries: int = 3) -> dict | str | None:
        for attempt in range(retries):
            try:
                resp = await self.client.get(url, params=params)
                resp.raise_for_status()
                await asyncio.sleep(RATE_LIMIT_DELAY)
                ct = resp.headers.get("content-type", "")
                if "json" in ct:
                    return resp.json()
                return resp.text
            except httpx.HTTPStatusError as e:
                if e.response.status_code == 429:
                    wait = 2 ** (attempt + 2)
                    logger.warning(f"Rate limited. Waiting {wait}s...")
                    await asyncio.sleep(wait)
                else:
                    logger.error(f"HTTP {e.response.status_code} for {url}")
                    return None
            except Exception as e:
                logger.error(f"Request failed ({attempt+1}/{retries}): {e}")
                await asyncio.sleep(2 ** attempt)
        return None

    # ── Step 1: Fetch problem list ──────────────────────────────────────────
    async def fetch_problem_list(self) -> list[dict]:
        """Returns raw problem dicts from CF API, sorted by rating."""
        data = await self._get(f"{CF_API}/problemset.problems")
        if not data or data.get("status") != "OK":
            logger.error("Failed to fetch problemset")
            return []

        problems = data["result"]["problems"]
        stats = {p["contestId"] * 100 + ord(p["index"][0]): p
                 for p in data["result"]["problemStatistics"]}

        # Filter: only rated problems with editorials more likely (div1/2 rounds)
        rated = [p for p in problems if p.get("rating") and p.get("contestId", 0) < 2000]
        rated.sort(key=lambda p: (p.get("rating", 0), p.get("contestId", 0)))

        logger.info(f"Found {len(rated)} rated problems. Capping at {self.max_problems}.")
        return rated[: self.max_problems]

    # ── Step 2: Scrape problem statement ───────────────────────────────────
    async def scrape_problem_statement(self, contest_id: int, index: str) -> dict:
        url = f"{CF_BASE}/contest/{contest_id}/problem/{index}"
        # Use cloudscraper to bypass Cloudflare
        if not self.cf_scraper:
            logger.warning(f"Skipping HTML scrape for {contest_id}{index}: cloudscraper not available")
            return {}
        
        try:
            html = self.cf_scraper.get(url, timeout=30).text
            await asyncio.sleep(RATE_LIMIT_DELAY)
        except Exception as e:
            logger.warning(f"Failed to scrape statement for {contest_id}{index}: {e}")
            return {}
        
        if not html or len(html) < 500:
            return {}

        soup = BeautifulSoup(html, "html.parser")
        result = {"statement_html": "", "input_spec": "", "output_spec": "", "examples": [],
                  "time_limit_ms": 0, "memory_limit_mb": 0}

        # Main problem statement
        stmt = soup.find("div", class_="problem-statement")
        if stmt:
            result["statement_html"] = str(stmt)

            # Extract time limit and memory limit from header
            time_limit_div = stmt.find("div", class_="time-limit")
            if time_limit_div:
                time_text = time_limit_div.get_text(strip=True)
                # Extract number and unit (e.g., "time limit per test1 second")
                match = re.search(r'(\d+(?:\.\d+)?)\s*(second|minute|hour)', time_text, re.IGNORECASE)
                if match:
                    value = float(match.group(1))
                    unit = match.group(2).lower()
                    if unit == 'second':
                        result["time_limit_ms"] = int(value * 1000)
                    elif unit == 'minute':
                        result["time_limit_ms"] = int(value * 60000)
                    elif unit == 'hour':
                        result["time_limit_ms"] = int(value * 3600000)
            
            memory_limit_div = stmt.find("div", class_="memory-limit")
            if memory_limit_div:
                memory_text = memory_limit_div.get_text(strip=True)
                # Extract number and unit (e.g., "memory limit per test256 megabytes")
                match = re.search(r'(\d+(?:\.\d+)?)\s*(byte|kilobyte|megabyte|gigabyte)', memory_text, re.IGNORECASE)
                if match:
                    value = float(match.group(1))
                    unit = match.group(2).lower()
                    if unit == 'byte':
                        result["memory_limit_mb"] = int(value / (1024 * 1024))
                    elif unit == 'kilobyte':
                        result["memory_limit_mb"] = int(value / 1024)
                    elif unit == 'megabyte':
                        result["memory_limit_mb"] = int(value)
                    elif unit == 'gigabyte':
                        result["memory_limit_mb"] = int(value * 1024)

            # Input/output specs
            for div in stmt.find_all("div", class_="section-title"):
                title = div.get_text(strip=True).lower()
                section_text = div.parent.get_text(separator="\n", strip=True)
                if "input" in title and "output" not in title:
                    result["input_spec"] = section_text
                elif "output" in title:
                    result["output_spec"] = section_text

            # Examples
            inputs = stmt.find_all("div", class_="input")
            outputs = stmt.find_all("div", class_="output")
            for inp, out in zip(inputs, outputs):
                pre_in = inp.find("pre")
                pre_out = out.find("pre")
                if pre_in and pre_out:
                    result["examples"].append({
                        "input": pre_in.get_text("\n", strip=True),
                        "output": pre_out.get_text("\n", strip=True),
                    })

        return result

    # ── Step 3: Find and scrape editorial ─────────────────────────────────
    async def find_editorial(self, contest_id: int, problem_index: str) -> tuple[str, str]:
        """Returns (editorial_url, editorial_html). Searches contest blog posts."""
        # CF editorials are usually posted by the problem setter as blog entries
        if not self.cf_scraper:
            return "", ""
        
        url = f"{CF_BASE}/contest/{contest_id}"
        try:
            html = self.cf_scraper.get(url, timeout=30).text
            await asyncio.sleep(RATE_LIMIT_DELAY)
        except Exception as e:
            logger.warning(f"Failed to scrape contest page for {contest_id}: {e}")
            return "", ""
        
        if not html:
            return "", ""

        soup = BeautifulSoup(html, "html.parser")
        editorial_url = ""

        # Look for editorial link in sidebar/announcements
        for a in soup.find_all("a", href=True):
            href = a["href"]
            text = a.get_text(strip=True).lower()
            if "editorial" in text or "tutorial" in text:
                if "/blog/entry/" in href:
                    editorial_url = CF_BASE + href if href.startswith("/") else href
                    break

        if not editorial_url:
            return "", ""

        try:
            ed_html = self.cf_scraper.get(editorial_url, timeout=30).text
            await asyncio.sleep(RATE_LIMIT_DELAY)
        except Exception as e:
            logger.warning(f"Failed to scrape editorial {editorial_url}: {e}")
            return editorial_url, ""
        
        if not ed_html:
            return editorial_url, ""

        ed_soup = BeautifulSoup(ed_html, "html.parser")
        # Extract the blog content
        content = ed_soup.find("div", class_="ttypography")
        if content:
            problem_html = extract_codeforces_problem_editorial(
                str(content), f"{contest_id}{problem_index}"
            )
            if is_useful_editorial(html_to_text(problem_html)):
                return editorial_url, problem_html
            return editorial_url, ""
        return editorial_url, ""

    # ── Step 4: Orchestrate scraping ───────────────────────────────────────
    async def scrape(self):
        logger.info("Starting Codeforces scrape...")
        problems_meta = await self.fetch_problem_list()
        results = []

        for i, meta in enumerate(problems_meta):
            contest_id = meta["contestId"]
            index = meta["index"]
            problem_id = f"{contest_id}{index}"

            out_file = self.output_dir / f"{problem_id}.json"
            if out_file.exists():
                cached = json.loads(out_file.read_text(encoding="utf-8"))
                ed_soup = BeautifulSoup(cached.get("editorial_html", ""), "html.parser")
                tutorial_codes = {
                    node.get("problemcode")
                    for node in ed_soup.select(".problemTutorial[problemcode]")
                }
                contest_wide_editorial = len(tutorial_codes) > 1
                if cached.get("statement_html") and not contest_wide_editorial:
                    logger.debug(f"[{i+1}/{len(problems_meta)}] Skipping {problem_id} (valid cache)")
                    results.append(cached)
                    continue
                logger.info(f"[{i+1}/{len(problems_meta)}] Refreshing invalid cache for {problem_id}")

            logger.info(f"[{i+1}/{len(problems_meta)}] Scraping {problem_id}...")

            problem = CFProblem(
                problem_id=problem_id,
                contest_id=contest_id,
                index=index,
                name=meta.get("name", ""),
                rating=meta.get("rating"),
                tags=meta.get("tags", []),
            )

            # Statement
            stmt_data = await self.scrape_problem_statement(contest_id, index)
            problem.statement_html = stmt_data.get("statement_html", "")
            problem.input_spec = stmt_data.get("input_spec", "")
            problem.output_spec = stmt_data.get("output_spec", "")
            problem.examples = stmt_data.get("examples", [])

            # Editorial (cached per contest to avoid re-fetching)
            ed_url, ed_html = await self.find_editorial(contest_id, index)
            problem.editorial_url = ed_url
            problem.editorial_html = ed_html

            doc = asdict(problem)
            out_file.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")
            results.append(doc)

        logger.info(f"Codeforces scrape complete. {len(results)} problems saved to {self.output_dir}")
        return results


async def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    async with CodeforcesScraper(max_problems=200) as scraper:
        await scraper.scrape()


if __name__ == "__main__":
    asyncio.run(main())
