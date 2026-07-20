"""CP7 — Hybrid retrieval (BM25 + vector) with Reciprocal Rank Fusion.

Vector search understands *meaning* ("container orchestration" -> Kubernetes) but
can miss rare exact terms. BM25 nails exact keywords ("Terraform", "pytest") but
misses paraphrases. Fusing both with RRF gets the best of each.

The retriever works over an in-memory `corpus` (a list of chunk dicts, each with
"text" and "metadata" holding source + chunk_index) that is index-aligned with a
ChromaDB `VectorStore` holding the same chunks.
"""

from __future__ import annotations

import re
from collections import defaultdict

from rank_bm25 import BM25Okapi

from app.config import TOP_K
from app.vectorstore import VectorStore

RRF_K = 60          # standard RRF damping constant
CANDIDATE_K = 10    # candidates pulled from each retriever before fusion
PER_QUERY_K = 5     # hybrid hits kept per JD requirement when pooling candidates


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


class HybridRetriever:
    def __init__(self, store: VectorStore, corpus: list[dict]):
        self.store = store
        self.corpus = corpus
        self.bm25 = BM25Okapi([_tokenize(c["text"]) for c in corpus])
        # Map (source, chunk_index) -> position in `corpus`, to align vector hits.
        self.key_to_idx = {
            (c["metadata"]["source"], c["metadata"]["chunk_index"]): i
            for i, c in enumerate(corpus)
        }

    # --- individual retrievers (return corpus indices, best-first) ---
    def _vector_ranking(self, query: str, k: int) -> list[int]:
        idxs = []
        for hit in self.store.query(query, top_k=k):
            key = (hit["metadata"]["source"], hit["metadata"]["chunk_index"])
            if key in self.key_to_idx:
                idxs.append(self.key_to_idx[key])
        return idxs

    def _bm25_ranking(self, query: str, k: int) -> list[int]:
        scores = self.bm25.get_scores(_tokenize(query))
        order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        return order[:k]

    # --- public search methods (return chunk dicts, best-first) ---
    def vector_search(self, query: str, top_k: int = TOP_K) -> list[dict]:
        return [self.corpus[i] for i in self._vector_ranking(query, top_k)]

    def bm25_search(self, query: str, top_k: int = TOP_K) -> list[dict]:
        return [self.corpus[i] for i in self._bm25_ranking(query, top_k)]

    def hybrid_search(
        self,
        query: str,
        top_k: int = TOP_K,
        candidate_k: int = CANDIDATE_K,
        rrf_k: int = RRF_K,
    ) -> list[dict]:
        """Fuse vector + BM25 rankings with Reciprocal Rank Fusion."""
        fused: dict[int, float] = defaultdict(float)
        for ranking in (
            self._vector_ranking(query, candidate_k),
            self._bm25_ranking(query, candidate_k),
        ):
            for rank, idx in enumerate(ranking):
                fused[idx] += 1.0 / (rrf_k + rank)
        ordered = sorted(fused, key=lambda i: fused[i], reverse=True)
        return [self.corpus[i] for i in ordered[:top_k]]


def retrieve_for_jd_hybrid(
    store: VectorStore,
    corpus: list[dict],
    jd_text: str,
    top_k: int = TOP_K,
    reranker=None,
    per_query_k: int = PER_QUERY_K,
) -> list[dict]:
    """JD-level retrieval using hybrid search, optionally reranked.

    Pools hybrid candidates across each JD requirement, dedupes them, then (if a
    reranker is given) reranks the pool against the full JD. Returns hits in the
    standard shape ({"text", "metadata", "score"}) so `format_context` works.
    """
    from app.retrieval import extract_queries

    retriever = HybridRetriever(store, corpus)
    queries = extract_queries(jd_text)

    # Pool hybrid candidates across each JD requirement.
    pool: dict[tuple, dict] = {}
    for query in queries:
        for chunk in retriever.hybrid_search(query, top_k=per_query_k):
            key = (chunk["metadata"]["source"], chunk["metadata"]["chunk_index"])
            pool.setdefault(key, chunk)
    candidates = list(pool.values())

    # Order the final chunks: rerank if available, else a hybrid pass over the JD.
    if reranker is not None and candidates:
        # A cross-encoder prefers a focused query, so join the requirement lines
        # rather than feeding the whole JD (company blurb, perks, etc.).
        rerank_query = " ".join(queries)
        ordered = [
            c for c, _ in reranker.rerank_with_scores(rerank_query, candidates, top_k=top_k)
        ]
    else:
        ordered = retriever.hybrid_search(jd_text, top_k=top_k)

    # Attach a cosine relevance (chunk vs. whole JD) for a friendly display score.
    cos_map = {}
    for h in store.query(jd_text, top_k=max(len(corpus), 1)):
        key = (h["metadata"]["source"], h["metadata"]["chunk_index"])
        cos_map[key] = h["score"]

    hits = []
    for c in ordered:
        key = (c["metadata"]["source"], c["metadata"]["chunk_index"])
        hits.append(
            {"text": c["text"], "metadata": c["metadata"], "score": round(cos_map.get(key, 0.0), 4)}
        )
    return hits


if __name__ == "__main__":
    # Demo on a tiny inline corpus to show BM25 catching an exact term.
    corpus = [
        {"text": "Migrated microservices to Kubernetes and wrote Terraform modules.",
         "metadata": {"source": "devops.txt", "chunk_index": 0}},
        {"text": "Built interactive Tableau dashboards and complex SQL queries.",
         "metadata": {"source": "analyst.txt", "chunk_index": 0}},
        {"text": "Developed REST APIs in Python with Flask and pytest tests.",
         "metadata": {"source": "backend.txt", "chunk_index": 0}},
    ]
    store = VectorStore(collection_name="hybrid_demo")
    store.reset()
    for c in corpus:
        store.add_chunks([c["text"]], source=c["metadata"]["source"])

    r = HybridRetriever(store, corpus)
    q = "infrastructure as code tooling"
    print(f"Query: {q!r}")
    print("vector:", [h["metadata"]["source"] for h in r.vector_search(q, 2)])
    print("bm25  :", [h["metadata"]["source"] for h in r.bm25_search(q, 2)])
    print("hybrid:", [h["metadata"]["source"] for h in r.hybrid_search(q, 2)])
