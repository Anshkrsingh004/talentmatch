"""CP3 — Retrieval.

Given a job description, find the most relevant resume chunks. Instead of one
big query, we split the JD into individual requirement lines and query each,
then aggregate the best-scoring chunks. This improves coverage: a resume
section that matches *one* specific requirement still surfaces.
"""

from __future__ import annotations

from app.config import TOP_K
from app.vectorstore import VectorStore

MIN_QUERY_LEN = 25  # ignore short header/blank lines when splitting a JD


def extract_queries(jd_text: str) -> list[str]:
    """Break a job description into meaningful requirement-style queries."""
    queries = []
    for line in jd_text.splitlines():
        line = line.strip(" -•\t")
        if len(line) >= MIN_QUERY_LEN:
            queries.append(line)
    # Fall back to the whole JD if it had no substantial lines.
    return queries or [jd_text.strip()]


def retrieve_for_jd(
    store: VectorStore,
    jd_text: str,
    top_k: int = TOP_K,
    per_query_k: int = 3,
    doc_type: str | None = "resume",
) -> list[dict]:
    """Multi-query retrieval: query each JD requirement, keep the best hits.

    Returns up to `top_k` unique resume chunks, ranked by their best similarity
    score across all sub-queries.
    """
    where = {"doc_type": doc_type} if doc_type else None
    best: dict[str, dict] = {}

    for query in extract_queries(jd_text):
        for hit in store.query(query, top_k=per_query_k, where=where):
            meta = hit["metadata"]
            key = f"{meta['source']}:{meta['chunk_index']}"
            if key not in best or hit["score"] > best[key]["score"]:
                best[key] = hit

    ranked = sorted(best.values(), key=lambda h: h["score"], reverse=True)
    return ranked[:top_k]


def format_context(hits: list[dict]) -> str:
    """Render retrieved chunks as a numbered, citation-friendly context block."""
    blocks = []
    for i, hit in enumerate(hits, start=1):
        meta = hit["metadata"]
        header = (
            f"[{i}] (source: {meta['source']}, chunk {meta['chunk_index']}, "
            f"relevance {hit['score']:.2f})"
        )
        blocks.append(f"{header}\n{hit['text']}")
    return "\n\n".join(blocks)


if __name__ == "__main__":
    from app.config import BASE_DIR
    from app.ingestion import prepare_document

    samples = BASE_DIR / "data" / "samples"
    resume = samples / "sample_resume.txt"
    jd = samples / "sample_jd.txt"

    # Build a fresh index from the sample resume.
    store = VectorStore()
    store.reset()
    store.add_chunks(prepare_document(resume), source=resume.name, doc_type="resume")

    jd_text = jd.read_text(encoding="utf-8")
    hits = retrieve_for_jd(store, jd_text, top_k=TOP_K)

    print(f"Retrieved {len(hits)} resume chunks for the job description:\n")
    print(format_context(hits))
