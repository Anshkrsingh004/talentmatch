"""Tests for CP3 retrieval. Uses a throwaway temp collection."""

import tempfile
import uuid

from app.retrieval import extract_queries, format_context, retrieve_for_jd
from app.vectorstore import VectorStore

RESUME_CHUNKS = [
    "Built internal REST APIs in Python using Flask and wrote pytest unit tests.",
    "Skills: Python, SQL, PostgreSQL, Git, Docker basics and data structures.",
    "Hobbies include hiking, photography and playing the guitar on weekends.",
]

JD_TEXT = """Junior Backend Engineer
Responsibilities:
- Design and develop RESTful APIs using Python.
- Work with relational databases such as PostgreSQL.
"""


def _store_with_resume() -> VectorStore:
    store = VectorStore(
        persist_dir=tempfile.mkdtemp(),
        collection_name=f"test_{uuid.uuid4().hex[:8]}",
    )
    store.add_chunks(RESUME_CHUNKS, source="resume.txt", doc_type="resume")
    return store


def test_extract_queries_splits_requirements():
    queries = extract_queries(JD_TEXT)
    # Short header/blank lines are dropped; requirement lines are kept.
    assert any("RESTful APIs" in q for q in queries)
    assert all(len(q) >= 25 for q in queries)


def test_retrieve_for_jd_ranks_relevant_first():
    store = _store_with_resume()
    hits = retrieve_for_jd(store, JD_TEXT, top_k=3)
    assert len(hits) >= 1
    # The REST API chunk should be the strongest match; the hobbies chunk not top.
    assert "REST APIs in Python" in hits[0]["text"]
    scores = [h["score"] for h in hits]
    assert scores == sorted(scores, reverse=True)


def test_format_context_has_citations():
    store = _store_with_resume()
    hits = retrieve_for_jd(store, JD_TEXT, top_k=2)
    context = format_context(hits)
    assert "[1]" in context
    assert "source:" in context
    assert "relevance" in context


def _run_all():
    tests = [v for k, v in globals().items() if k.startswith("test_")]
    passed = 0
    for t in tests:
        t()
        print(f"  PASS  {t.__name__}")
        passed += 1
    print(f"\n{passed}/{len(tests)} tests passed.")


if __name__ == "__main__":
    _run_all()
