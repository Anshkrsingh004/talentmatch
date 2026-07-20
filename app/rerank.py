"""CP7 — Cross-encoder reranking.

Bi-encoders (our embedding model) encode the query and each chunk *separately*,
which is fast but approximate. A cross-encoder feeds (query, chunk) together
through the model and scores their relevance directly — slower, but much more
precise. We use it to reorder a small candidate set from the hybrid retriever.

Model: cross-encoder/ms-marco-MiniLM-L-6-v2 (~80 MB, local, free).
"""

from __future__ import annotations

from app.config import TOP_K

DEFAULT_RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class Reranker:
    def __init__(self, model_name: str = DEFAULT_RERANK_MODEL):
        self.model_name = model_name
        self._model = None

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(self.model_name)
        return self._model

    def rerank(
        self, query: str, candidates: list[dict], top_k: int = TOP_K
    ) -> list[dict]:
        """Reorder candidate chunk dicts by cross-encoder relevance to the query."""
        return [c for c, _ in self.rerank_with_scores(query, candidates, top_k)]

    def rerank_with_scores(
        self, query: str, candidates: list[dict], top_k: int = TOP_K
    ) -> list[tuple[dict, float]]:
        """Like `rerank`, but also return a 0..1 relevance score per chunk.

        Cross-encoder logits are unbounded; a sigmoid maps them to a friendly
        0..1 relevance for display.
        """
        import math

        if not candidates:
            return []
        pairs = [(query, c["text"]) for c in candidates]
        scores = self.model.predict(pairs)
        ranked = sorted(
            zip(candidates, scores), key=lambda pair: pair[1], reverse=True
        )
        return [
            (c, 1.0 / (1.0 + math.exp(-float(s)))) for c, s in ranked[:top_k]
        ]


if __name__ == "__main__":
    reranker = Reranker()
    query = "infrastructure as code tooling"
    candidates = [
        {"text": "Built interactive Tableau dashboards and SQL reports."},
        {"text": "Wrote Terraform modules to provision AWS infrastructure as code."},
        {"text": "Developed React components with TypeScript."},
    ]
    print(f"Query: {query!r}\n")
    for c in reranker.rerank(query, candidates, top_k=3):
        print(" ->", c["text"])
