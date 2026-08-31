import asyncio

from RAG.scrapers.usaco import USACOScraper


class StubUSACOScraper(USACOScraper):
    def __init__(self, responses):
        self.responses = responses
        self.requested_urls = []
        self.max_contests = 20
        self.start_contest = 1

    async def _get(self, url, retries=3):
        self.requested_urls.append(url)
        return self.responses.get(url)


def run(coro):
    return asyncio.run(coro)


def test_fetch_contest_links_uses_results_pages_and_deduplicates():
    archive = """
        <a href="index.php?page=jan24results">2024 January Contest Results</a>
        <a href="/index.php?page=jan24results">duplicate</a>
        <a href="index.php?page=open24results">2024 US Open Contest Results</a>
        <a href="index.php?page=training">not a result</a>
    """
    scraper = StubUSACOScraper({"https://usaco.org/?page=training": archive})

    contests = run(scraper.fetch_contest_links())

    assert [item["page_key"] for item in contests] == ["jan24results", "open24results"]
    assert contests[0]["url"] == "https://usaco.org/index.php?page=jan24results"


def test_fetch_contest_links_supports_resume_position():
    archive = "".join(
        f'<a href="index.php?page=c{i}results">Contest {i}</a>' for i in range(1, 5)
    )
    scraper = StubUSACOScraper({"https://usaco.org/?page=training": archive})
    scraper.start_contest = 3
    scraper.max_contests = 1

    contests = run(scraper.fetch_contest_links())

    assert [item["page_key"] for item in contests] == ["c3results"]


def test_problem_page_extracts_identity_and_clean_sample_blocks():
    page = """
        <div id="content-inner">
          <h2>USACO 2024 US Open Contest, Gold</h2>
          <h2>Problem 1. Cowreography</h2>
          <h4>INPUT FORMAT (input arrives from the terminal / stdin):</h4>
          <p>The first line contains N and K.</p>
          <h4>OUTPUT FORMAT (print output to the terminal / stdout):</h4>
          <p>The minimum number of moves.</p>
          <h4>SAMPLE INPUT:</h4><pre>4 1\n0111\n1110</pre>
          <h4>SAMPLE OUTPUT:</h4><pre>3</pre>
          <p>One possible dance: 0111 -&gt; 1110</p>
          <h4>SCORING:</h4><p>No additional constraints.</p>
        </div>
    """
    url = "https://usaco.org/index.php?cpid=1425&page=viewproblem2"
    scraper = StubUSACOScraper({url: page})

    result = run(scraper.scrape_problem_page(url))

    assert result["problem_name"] == "Cowreography"
    assert result["contest"] == "2024 US Open Contest"
    assert result["division"] == "gold"
    assert result["examples"] == [{"input": "4 1\n0111\n1110", "output": "3"}]


def test_problem_page_falls_back_when_legacy_container_is_absent():
    page = """
        <html><body><main><section>
          <h2>USACO 2026 Third Contest, Platinum</h2>
          <h2>Problem 1. All Pairs Shortest Paths</h2>
          <p>Statement text.</p>
          <h4>INPUT FORMAT:</h4><p>N</p>
          <h4>OUTPUT FORMAT:</h4><p>answer</p>
        </section></main></body></html>
    """
    url = "https://usaco.org/index.php?page=viewproblem2&cpid=1596"
    scraper = StubUSACOScraper({url: page})

    result = run(scraper.scrape_problem_page(url))

    assert result["problem_name"] == "All Pairs Shortest Paths"
    assert result["contest"] == "2026 Third Contest"
    assert result["division"] == "platinum"
    assert "Statement text" in result["statement_html"]


def test_problem_page_accepts_legacy_us_open_header():
    page = """
        <div class="problem-text">
          <h4>INPUT FORMAT:</h4><p>N</p>
          <h4>OUTPUT FORMAT:</h4><p>answer</p>
        </div>
        <h2>USACO 2021 US Open, Platinum</h2>
        <h2>Problem 2. Routing Schemes</h2>
    """
    url = "https://usaco.org/index.php?page=viewproblem2&cpid=1141"
    scraper = StubUSACOScraper({url: page})

    result = run(scraper.scrape_problem_page(url))

    assert result["contest"] == "2021 US Open"
    assert result["division"] == "platinum"


def test_contest_uses_cpid_not_view_problem_anchor_text():
    contest_url = "https://usaco.org/index.php?page=open24results"
    problem_url = "https://usaco.org/index.php?cpid=1425&page=viewproblem2"
    contest_page = """
        <a href="index.php?cpid=1425&amp;page=viewproblem2">View problem</a>
        <a href="index.php?cpid=1425&amp;page=viewproblem2">Duplicate link</a>
    """
    problem_page = """
        <div id="content-inner">
          <h2>USACO 2024 US Open Contest, Gold</h2>
          <h2>Problem 1. Cowreography</h2>
          <h4>INPUT FORMAT:</h4><p>N</p>
          <h4>OUTPUT FORMAT:</h4><p>answer</p>
        </div>
    """
    scraper = StubUSACOScraper({contest_url: contest_page, problem_url: problem_page})
    scraper.scrape_usaco_guide_editorial = lambda *args, **kwargs: _empty_editorial()

    problems = run(scraper.scrape_contest({"name": "2024 Open", "url": contest_url}))

    assert len(problems) == 1
    assert problems[0]["problem_id"] == "usaco_1425"
    assert problems[0]["problem_name"] == "Cowreography"
    assert problems[0]["division"] == "gold"


async def _empty_editorial():
    return ""


def test_editorial_lookup_uses_master_lowercase_and_cpid():
    expected_url = (
        "https://raw.githubusercontent.com/cpinitiative/usaco-guide/"
        "master/solutions/bronze/usaco-1011.mdx"
    )
    content = "---\ntitle: Test\n---\n" + ("Editorial text. " * 20)
    scraper = StubUSACOScraper({expected_url: content})

    result = run(scraper.scrape_usaco_guide_editorial("1011", "bronze"))

    assert result.startswith("Editorial text.")
    assert scraper.requested_urls == [expected_url]
