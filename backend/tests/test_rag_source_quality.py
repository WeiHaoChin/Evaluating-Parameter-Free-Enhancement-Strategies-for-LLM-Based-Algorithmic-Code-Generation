from backend.RAG.pipeline.rag_pipeline import (
    extract_codeforces_problem_editorial,
    is_useful_editorial,
    process_atcoder,
    process_codeforces,
    process_cp_algorithms,
    process_cph,
)


def test_atcoder_navigation_is_not_an_editorial():
    navigation = """
    <div>Contest Duration: 100 minutes Back to Home Top Tasks
    Clarifications Results All Submissions Standings Virtual Standings
    Editorial Discussion</div>
    """
    chunks = process_atcoder({
        "problem_id": "abc001_a",
        "title": "A",
        "editorial_html": navigation,
    })
    assert not [chunk for chunk in chunks if chunk.chunk_type == "editorial"]


def test_codeforces_extracts_only_matching_problem_widget():
    html = """
    <div class="spoiler"><b class="spoiler-title">Tutorial</b>
      <div class="problemTutorial" problemcode="123A">Tutorial is loading...</div>
    </div>
    <div class="spoiler"><b class="spoiler-title">Tutorial</b>
      <div class="problemTutorial" problemcode="123B">A useful explanation with enough words """ + "x " * 60 + """</div>
    </div>
    """
    extracted = extract_codeforces_problem_editorial(html, "123B")
    assert "123B" in extracted
    assert "123A" not in extracted


def test_codeforces_unloaded_tutorial_is_rejected():
    chunks = process_codeforces({
        "problem_id": "123A",
        "contest_id": 123,
        "index": "A",
        "editorial_html": (
            '<div class="spoiler"><div class="problemTutorial" '
            'problemcode="123A">Tutorial is loading...</div></div>'
        ),
    })
    assert not [chunk for chunk in chunks if chunk.chunk_type == "editorial"]


def test_cp_algorithms_navigation_and_redirects_are_rejected():
    assert process_cp_algorithms({"article_id": "tags", "title": "Tags¶", "content_markdown": "# Tags"}) == []
    assert process_cp_algorithms({
        "article_id": "old-lis",
        "title": "Longest increasing subsequence¶",
        "content_markdown": '<meta http-equiv="refresh" content="0; url=new">',
    }) == []


def test_cph_tiny_code_is_not_a_standalone_chunk():
    chunks = process_cph({
        "chapter_id": "c1",
        "chapter_num": 1,
        "title": "Intro",
        "content_text": "Useful theory " * 100,
        "code_examples": ["sort(v.begin(), v.end());"],
    })
    assert not [chunk for chunk in chunks if chunk.chunk_type == "code"]


def test_substantive_editorial_is_useful():
    assert is_useful_editorial("Explanation " + "algorithm " * 60)
