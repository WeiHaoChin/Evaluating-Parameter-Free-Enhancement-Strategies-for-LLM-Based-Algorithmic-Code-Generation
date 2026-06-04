"""
USACO Scraper
Scrapes USACO problems and high-quality editorial writeups from usaco.org and usaco.guide.
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

USACO_BASE = "https://usaco.org"
USACO_GUIDE = "https://usaco.guide"
RATE_LIMIT_DELAY = 1.5


@dataclass
class USACOProblem:
    source: str = "usaco"
    problem_id: str = ""
    contest: str = ""        # e.g. "2023 January"
    division: str = ""       # Bronze / Silver / Gold / Platinum
    problem_name: str = ""
    statement_html: str = ""
    input_format: str = ""
    output_format: str = ""
    examples: list = field(default_factory=list)
    constraints: str = ""
    editorial_html: str = ""
    editorial_source: str = ""
    topics: list = field(default_factory=list)
    difficulty: str = ""


DIVISIONS = ["bronze", "silver", "gold", "platinum"]

# USACO contests are stored at /index.php?page=contests
CONTEST_LIST_URL = f"{USACO_BASE}/index.php?page=contests"


class USACOScraper:
    def __init__(self, output_dir: str = "data/usaco", max_contests: int = 20):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.max_contests = max_contests
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
                logger.error(f"HTTP {e.response.status_code}: {url}")
                if e.response.status_code in (403, 404):
                    return None
            except Exception as e:
                logger.warning(f"Attempt {attempt+1} failed for {url}: {e}")
                await asyncio.sleep(2 ** attempt)
        return None

    # ── Contest discovery ─────────────────────────────────────────────────
    async def fetch_contest_links(self) -> list[dict]:
        """Scrapes usaco.org to find all contest pages."""
        html = await self._get(CONTEST_LIST_URL)
        if not html:
            return []

        soup = BeautifulSoup(html, "html.parser")
        contests = []

        # USACO lists contests with links like ?page=jan24results, ?page=dec23results, etc.
        for a in soup.find_all("a", href=True):
            href = a["href"]
            text = a.get_text(strip=True)
            if "results" in href and "page=" in href:
                contests.append({
                    "name": text,
                    "url": f"{USACO_BASE}/{href}" if not href.startswith("http") else href,
                    "page_key": re.search(r"page=(\w+)", href).group(1) if re.search(r"page=(\w+)", href) else "",
                })

        # Deduplicate and limit
        seen = set()
        unique = []
        for c in contests:
            if c["page_key"] not in seen:
                seen.add(c["page_key"])
                unique.append(c)

        logger.info(f"Found {len(unique)} contests. Using first {self.max_contests}.")
        return unique[: self.max_contests]

    # ── Problem page scraping ─────────────────────────────────────────────
    async def scrape_problem_page(self, problem_url: str) -> dict:
        html = await self._get(problem_url)
        if not html:
            return {}

        soup = BeautifulSoup(html, "html.parser")
        result = {
            "statement_html": "",
            "input_format": "",
            "output_format": "",
            "examples": [],
            "constraints": "",
        }

        # Main problem body
        body = soup.find("div", id="content-inner") or soup.find("div", class_="problem-text")
        if body:
            result["statement_html"] = str(body)
            text = body.get_text("\n", strip=True)

            # Extract input/output sections
            inp_match = re.search(r"INPUT FORMAT.*?\n(.*?)(?=OUTPUT FORMAT|$)", text, re.DOTALL | re.IGNORECASE)
            out_match = re.search(r"OUTPUT FORMAT.*?\n(.*?)(?=SCORING|SAMPLE|$)", text, re.DOTALL | re.IGNORECASE)
            if inp_match:
                result["input_format"] = inp_match.group(1).strip()
            if out_match:
                result["output_format"] = out_match.group(1).strip()

            # Sample cases (USACO uses "SAMPLE INPUT/OUTPUT" headers)
            samples = re.findall(
                r"SAMPLE INPUT.*?\n(.*?)SAMPLE OUTPUT.*?\n(.*?)(?=SAMPLE|$)",
                text, re.DOTALL | re.IGNORECASE
            )
            result["examples"] = [{"input": s[0].strip(), "output": s[1].strip()} for s in samples]

        return result

    # ── Editorial scraping from USACO.guide ──────────────────────────────
    async def scrape_usaco_guide_editorial(self, problem_name: str, division: str) -> str:
        """
        USACO.guide hosts community editorials in MDX format.
        We search their GitHub raw content for the editorial.
        """
        # USACO Guide organizes by module/topic. Try a GitHub search approach.
        # Their content lives at: https://raw.githubusercontent.com/cpinitiative/usaco-guide/main/solutions/
        div_map = {"bronze": "Bronze", "silver": "Silver", "gold": "Gold", "platinum": "Platinum"}
        div_str = div_map.get(division.lower(), "Silver")

        # Normalize name for filename lookup (USACO Guide uses slug-style names)
        slug = re.sub(r"[^a-z0-9]+", "-", problem_name.lower()).strip("-")
        base_url = f"https://raw.githubusercontent.com/cpinitiative/usaco-guide/main/solutions/{div_str}"

        # Try common filename patterns
        for suffix in [f"{slug}.mdx", f"usaco-{slug}.mdx"]:
            url = f"{base_url}/{suffix}"
            content = await self._get(url)
            if content and len(content) > 200:
                # Strip MDX frontmatter
                content = re.sub(r"^---.*?---\n", "", content, flags=re.DOTALL)
                return content

        return ""

    # ── Contest results page ──────────────────────────────────────────────
    async def scrape_contest(self, contest: dict) -> list[dict]:
        html = await self._get(contest["url"])
        if not html:
            return []

        soup = BeautifulSoup(html, "html.parser")
        problems = []

        # Extract year/month from contest name
        name = contest.get("name", "")
        year_match = re.search(r"20\d\d", name)
        contest_year = year_match.group() if year_match else "unknown"

        # Find problem links per division
        for div in DIVISIONS:
            # Look for section headers and problem links
            headers = soup.find_all(["h2", "h3", "h4"], string=re.compile(div, re.IGNORECASE))
            for header in headers:
                # Find links following this header
                for sibling in header.find_next_siblings(["a", "p", "ul", "li"]):
                    for a in sibling.find_all("a", href=True) if sibling.name != "a" else [sibling]:
                        href = a.get("href", "")
                        if "prob" in href.lower() or "problem" in href.lower():
                            problem_url = f"{USACO_BASE}/{href}" if not href.startswith("http") else href
                            problem_name = a.get_text(strip=True)

                            logger.info(f"  Scraping problem: {problem_name} ({div})")
                            stmt = await self.scrape_problem_page(problem_url)
                            editorial = await self.scrape_usaco_guide_editorial(problem_name, div)

                            problem_id = f"usaco_{contest_year}_{div}_{re.sub(r'[^a-z0-9]', '_', problem_name.lower())}"
                            prob = USACOProblem(
                                problem_id=problem_id,
                                contest=name,
                                division=div,
                                problem_name=problem_name,
                                statement_html=stmt.get("statement_html", ""),
                                input_format=stmt.get("input_format", ""),
                                output_format=stmt.get("output_format", ""),
                                examples=stmt.get("examples", []),
                                editorial_html=editorial,
                                editorial_source="usaco.guide" if editorial else "",
                            )
                            problems.append(asdict(prob))

        return problems

    # ── Main entry ────────────────────────────────────────────────────────
    async def scrape(self) -> list[dict]:
        logger.info("Starting USACO scrape...")
        contests = await self.fetch_contest_links()
        all_problems = []

        for i, contest in enumerate(contests):
            logger.info(f"[{i+1}/{len(contests)}] Contest: {contest['name']}")
            problems = await self.scrape_contest(contest)
            all_problems.extend(problems)

            for p in problems:
                out_file = self.output_dir / f"{p['problem_id']}.json"
                out_file.write_text(json.dumps(p, ensure_ascii=False, indent=2))

        logger.info(f"USACO scrape complete. {len(all_problems)} problems saved.")
        return all_problems


async def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    async with USACOScraper(max_contests=10) as scraper:
        await scraper.scrape()


if __name__ == "__main__":
    asyncio.run(main())
