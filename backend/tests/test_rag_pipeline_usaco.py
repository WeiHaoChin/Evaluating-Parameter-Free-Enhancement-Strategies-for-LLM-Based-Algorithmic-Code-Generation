from backend.RAG.pipeline.rag_pipeline import (
    clean_usaco_editorial_mdx,
    process_usaco,
)


def test_mdx_cleaner_keeps_prose_and_python_but_drops_other_languages():
    mdx = """
import LanguageSection from '@components/LanguageSection';

## Analysis

Use a greedy invariant.

<LanguageSection>
<PySection>
```py
print("python kept")
```
</PySection>
<CPPSection>
```cpp
cout << "cpp removed";
```
</CPPSection>
<JavaSection>
```java
System.out.println("java removed");
```
</JavaSection>
</LanguageSection>
"""

    cleaned = clean_usaco_editorial_mdx(mdx)

    assert "Use a greedy invariant" in cleaned
    assert 'print("python kept")' in cleaned
    assert "cpp removed" not in cleaned
    assert "java removed" not in cleaned
    assert "LanguageSection" not in cleaned
    assert "PySection" not in cleaned


def test_mdx_cleaner_unwraps_explanatory_components():
    mdx = """
<Info title="Invariant">Maintain the smallest reachable endpoint.</Info>
<Spoiler title="Proof">The exchange argument proves optimality.</Spoiler>
<YouTube id="unused" />
"""

    cleaned = clean_usaco_editorial_mdx(mdx)

    assert "Maintain the smallest reachable endpoint" in cleaned
    assert "exchange argument proves optimality" in cleaned
    assert "YouTube" not in cleaned


def test_process_usaco_uses_division_as_difficulty_fallback():
    doc = {
        "problem_id": "usaco_1",
        "problem_name": "Example",
        "contest": "Example Contest",
        "division": "gold",
        "topics": [],
        "statement_html": "<p>Example statement with enough useful text.</p>",
        "editorial_html": "<PySection>Use prefix sums.</PySection>",
        "editorial_source": "usaco.guide",
    }

    chunks = process_usaco(doc)

    assert chunks
    assert all(chunk.metadata["difficulty"] == "gold" for chunk in chunks)
    assert any("Use prefix sums" in chunk.text for chunk in chunks)
