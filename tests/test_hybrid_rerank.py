"""Tests for CP7 hybrid retrieval, reranking, and eval metrics."""

import tempfile
import uuid

from app.evaluation import _first_hit_rank
from app.hybrid import HybridRetriever
from app.rerank import Reranker
from app.vectorstore import VectorStore

CORPUS = [
    {"text": "Migrated microservices to Kubernetes and wrote Terraform modules.",
     "metadata": {"source": "devops.txt", "chunk_index": 0}},
    {"text": "Built interactive Tableau dashboards and complex SQL queries.",
     "metadata": {"source": "analyst.txt", "chunk_index": 0}},
    {"text": "Developed REST APIs in Python with Flask and pytest tests.",
     "metadata": {"source": "backend.txt", "chunk_index": 0}},
]


def _retriever() -> HybridRetriever:
    store = VectorStore(
        persist_dir=tempfile.mkdtemp(), collection_name=f"t_{uuid.uuid4().hex[:8]}"
    )
    for c in CORPUS:
        store.add_chunks([c["text"]], source=c["metadata"]["source"])
    return HybridRetriever(store, CORPUS)


def test_bm25_finds_exact_rare_term():
    """BM25 should rank the Terraform chunk first for an exact-term query
    (pure keyword match, no embedding model needed)."""
    r = _retriever()
    top = r.bm25_search("Terraform", top_k=1)
    assert top[0]["metadata"]["source"] == "devops.txt"


def test_hybrid_search_returns_requested_count():
    r = _retriever()
    hits = r.hybrid_search("container orchestration", top_k=2)
    assert len(hits) == 2
    assert all("text" in h and "metadata" in h for h in hits)


def test_reranker_orders_relevant_first():
    reranker = Reranker()
    query = "data visualization dashboards"
    ranked = reranker.rerank(query, CORPUS, top_k=3)
    assert ranked[0]["metadata"]["source"] == "analyst.txt"


def test_rerank_scores_are_normalised():
    reranker = Reranker()
    scored = reranker.rerank_with_scores("SQL reporting", CORPUS, top_k=3)
    assert all(0.0 <= s <= 1.0 for _, s in scored)


def test_first_hit_rank_metric():
    chunks = [{"text": "no match here"}, {"text": "mentions Kubernetes clearly"}]
    assert _first_hit_rank(chunks, "Kubernetes") == 2
    assert _first_hit_rank(chunks, "Terraform") is None


def _run_all():
    tests = [v for k, v in globals().items() if k.startswith("test_")]
    for t in tests:
        t()
        print(f"  PASS  {t.__name__}")
    print(f"\n{len(tests)}/{len(tests)} tests passed.")


if __name__ == "__main__":
    _run_all()
