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
from urllib.parse import parse_qs, urljoin, urlparse

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

# The training page contains the historical contest-results archive.
CONTEST_LIST_URL = f"{USACO_BASE}/?page=training"


class USACOScraper:
    def __init__(
        self, output_dir: str = "backend/data/usaco", max_contests: int = 20,
        start_contest: int = 1,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.max_contests = max_contests
        self.start_contest = max(1, start_contest)
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
                await asyncio.sleep(2 ** attempt)
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
            page_match = re.search(r"[?&]page=([a-z0-9_]+results)\b", href, re.IGNORECASE)
            if page_match:
                contests.append({
                    "name": text,
                    "url": urljoin(f"{USACO_BASE}/", href),
                    "page_key": page_match.group(1).lower(),
                })

        # Deduplicate and limit
        seen = set()
        unique = []
        for c in contests:
            if c["page_key"] not in seen:
                seen.add(c["page_key"])
                unique.append(c)

        start = self.start_contest - 1
        selected = unique[start:start + self.max_contests]
        logger.info(
            "Found %d contests. Using archive positions %d-%d.",
            len(unique), self.start_contest, start + len(selected),
        )
        return selected

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
            "problem_name": "",
            "contest": "",
            "division": "",
        }

        # Main problem body
        body = soup.find("div", id="content-inner") or soup.find("div", class_="problem-text")
        if body is None:
            # Newer USACO pages no longer consistently expose either legacy
            # container. Locate the problem heading and climb to the smallest
            # ancestor containing the complete statement instead.
            heading = soup.find(
                re.compile(r"^h[1-6]$"),
                string=re.compile(r"Problem\s+\d+\.", re.IGNORECASE),
            )
            candidate = heading
            while candidate is not None:
                candidate_text = candidate.get_text(" ", strip=True)
                if ("INPUT FORMAT" in candidate_text.upper() and
                        "OUTPUT FORMAT" in candidate_text.upper()):
                    body = candidate
                    break
                candidate = candidate.parent
        if body:
            result["statement_html"] = str(body)
            text = body.get_text("\n", strip=True)
            page_text = soup.get_text("\n", strip=True)

            title_match = re.search(
                r"Problem\s+\d+\.\s*([^\n]+)", page_text, re.IGNORECASE
            )
            if title_match:
                result["problem_name"] = title_match.group(1).strip()

            contest_match = re.search(
                r"USACO\s+(.+?(?:Contest|US Open)),\s*"
                r"(Bronze|Silver|Gold|Platinum)\b",
                page_text, re.IGNORECASE,
            )
            if contest_match:
                result["contest"] = contest_match.group(1).strip()
                result["division"] = contest_match.group(2).lower()

            # Extract input/output sections
            inp_match = re.search(r"INPUT FORMAT.*?\n(.*?)(?=OUTPUT FORMAT|$)", text, re.DOTALL | re.IGNORECASE)
            out_match = re.search(r"OUTPUT FORMAT.*?\n(.*?)(?=SCORING|SAMPLE|$)", text, re.DOTALL | re.IGNORECASE)
            if inp_match:
                result["input_format"] = inp_match.group(1).strip()
            if out_match:
                result["output_format"] = out_match.group(1).strip()

            # Sample data is held in <pre> blocks. Reading those directly avoids
            # swallowing the prose explanation that often follows sample output.
            pending_inputs = []
            examples = []
            for heading in body.find_all(re.compile(r"^h[1-6]$")):
                label = heading.get_text(" ", strip=True).upper()
                if "SAMPLE INPUT" not in label and "SAMPLE OUTPUT" not in label:
                    continue
                pre = heading.find_next("pre")
                next_heading = heading.find_next(re.compile(r"^h[1-6]$"))
                if pre is None or (next_heading is not None and pre.sourceline and
                                   next_heading.sourceline and pre.sourceline > next_heading.sourceline):
                    continue
                value = pre.get_text("\n", strip=True)
                if "SAMPLE INPUT" in label:
                    pending_inputs.append(value)
                elif pending_inputs:
                    examples.append({"input": pending_inputs.pop(0), "output": value})
            result["examples"] = examples

        return result

    # ── Editorial scraping from USACO.guide ──────────────────────────────
    async def scrape_usaco_guide_editorial(
        self, problem_id: str, division: str, problem_name: str = "",
    ) -> str:
        """
        USACO.guide hosts community editorials in MDX format.
        We search their GitHub raw content for the editorial.
        """
        # USACO Guide organizes by module/topic. Try a GitHub search approach.
        # Their content lives at: https://raw.githubusercontent.com/cpinitiative/usaco-guide/main/solutions/
        division = division.lower()
        directories = [division]
        if division == "platinum":
            directories.append("plat")

        # USACO Guide keys USACO solutions by the canonical numeric cpid.
        for directory in directories:
            url = (
                "https://raw.githubusercontent.com/cpinitiative/usaco-guide/"
                f"master/solutions/{directory}/usaco-{problem_id}.mdx"
            )
            content = await self._get(url)
            if content and len(content) > 200:
                return re.sub(r"\A---\s*\n.*?\n---\s*\n", "", content, flags=re.DOTALL)

        return ""

    # ── Contest results page ──────────────────────────────────────────────
    async def scrape_contest(self, contest: dict) -> list[dict]:
        html = await self._get(contest["url"])
        if not html:
            return []

        soup = BeautifulSoup(html, "html.parser")
        problems = []

        seen_cpids = set()
        for a in soup.find_all("a", href=True):
            problem_url = urljoin(contest["url"], a["href"])
            query = parse_qs(urlparse(problem_url).query)
            cpid = (query.get("cpid") or [""])[0]
            page = (query.get("page") or [""])[0].lower()
            if not cpid.isdigit() or "viewproblem" not in page or cpid in seen_cpids:
                continue
            seen_cpids.add(cpid)

            stmt = await self.scrape_problem_page(problem_url)
            if not stmt.get("statement_html") or not stmt.get("problem_name"):
                logger.warning("Skipping malformed USACO problem page: %s", problem_url)
                continue

            problem_name = stmt["problem_name"]
            division = stmt.get("division", "")
            if division not in DIVISIONS:
                logger.warning("Skipping problem with unknown division: %s", problem_url)
                continue

            logger.info("  Scraping problem: %s (%s, cpid=%s)", problem_name, division, cpid)
            editorial = await self.scrape_usaco_guide_editorial(cpid, division, problem_name)
            prob = USACOProblem(
                problem_id=f"usaco_{cpid}",
                contest=stmt.get("contest") or contest.get("name", ""),
                division=division,
                problem_name=problem_name,
                statement_html=stmt["statement_html"],
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
                out_file.write_text(
                    json.dumps(p, ensure_ascii=False, indent=2), encoding="utf-8"
                )

        logger.info(f"USACO scrape complete. {len(all_problems)} problems saved.")
        return all_problems


async def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    async with USACOScraper(max_contests=10) as scraper:
        await scraper.scrape()


if __name__ == "__main__":
    asyncio.run(main())
