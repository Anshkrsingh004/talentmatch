"""Tests for CP2 embeddings + vector store. Uses a throwaway temp collection
so it never touches the real data/chroma_db."""

import tempfile
import uuid

from app.vectorstore import VectorStore

CHUNKS = [
    "Built internal REST APIs in Python using Flask and wrote pytest unit tests.",
    "Studied data structures, algorithms, operating systems and DBMS at university.",
    "Enjoys hiking, photography and playing the guitar on weekends.",
]


def _fresh_store() -> VectorStore:
    tmp = tempfile.mkdtemp()
    return VectorStore(persist_dir=tmp, collection_name=f"test_{uuid.uuid4().hex[:8]}")


def test_add_and_count():
    store = _fresh_store()
    n = store.add_chunks(CHUNKS, source="unit_test.txt", doc_type="resume")
    assert n == 3
    assert store.count() == 3


def test_query_returns_most_relevant_first():
    store = _fresh_store()
    store.add_chunks(CHUNKS, source="unit_test.txt", doc_type="resume")

    results = store.query("Python REST API backend experience", top_k=3)
    assert len(results) == 3
    # The API/Python chunk should rank first.
    assert "REST APIs in Python" in results[0]["text"]
    # Scores are cosine similarities in [-1, 1], best first (descending).
    scores = [r["score"] for r in results]
    assert scores == sorted(scores, reverse=True)


def test_upsert_does_not_duplicate():
    store = _fresh_store()
    store.add_chunks(CHUNKS, source="unit_test.txt")
    store.add_chunks(CHUNKS, source="unit_test.txt")  # same source -> overwrite
    assert store.count() == 3


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
