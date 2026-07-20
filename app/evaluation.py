"""CP7 — Retrieval evaluation harness.

Builds a corpus from the sample resumes, then measures retrieval quality for four
methods on a labeled eval set:

    Vector only  ·  BM25 only  ·  Hybrid (RRF)  ·  Hybrid + Rerank

Metrics:
    Hit@1  — gold phrase appears in the top-1 retrieved chunk
    Hit@3  — gold phrase appears in the top-3 retrieved chunks
    MRR    — mean reciprocal rank of the first chunk containing the gold phrase

Run:  python -m app.evaluation
"""

from __future__ import annotations

import json
from pathlib import Path

from app.config import BASE_DIR
from app.hybrid import HybridRetriever
from app.ingestion import prepare_document
from app.rerank import Reranker
from app.vectorstore import VectorStore

RESUME_FILES = [
    "sample_resume.txt",
    "resume_devops.txt",
    "resume_frontend.txt",
    "resume_data_analyst.txt",
]
EVAL_PATH = BASE_DIR / "data" / "eval" / "eval_set.json"
RESULTS_MD = BASE_DIR / "data" / "eval" / "results.md"

RETRIEVE_K = 5          # how many chunks each method returns
RERANK_CANDIDATES = 8   # candidates fed to the reranker before trimming to K

# Finer-grained chunks -> more distractors -> a realistic retrieval challenge
# (with a handful of big chunks, plain vector search trivially scores 100%).
EVAL_CHUNK_SIZE = 180
EVAL_CHUNK_OVERLAP = 30


def build_corpus() -> tuple[VectorStore, list[dict]]:
    """Ingest all resumes into a vector store + an index-aligned corpus list."""
    samples = BASE_DIR / "data" / "samples"
    store = VectorStore(collection_name="eval_corpus")
    store.reset()
    corpus: list[dict] = []
    for name in RESUME_FILES:
        chunks = prepare_document(
            samples / name, chunk_size=EVAL_CHUNK_SIZE, overlap=EVAL_CHUNK_OVERLAP
        )
        store.add_chunks(chunks, source=name, doc_type="resume")
        for i, text in enumerate(chunks):
            corpus.append({"text": text, "metadata": {"source": name, "chunk_index": i}})
    return store, corpus


def _first_hit_rank(chunks: list[dict], gold: str) -> int | None:
    """1-based rank of the first chunk whose text contains `gold` (case-insensitive)."""
    gold_l = gold.lower()
    for rank, chunk in enumerate(chunks, start=1):
        if gold_l in chunk["text"].lower():
            return rank
    return None


def evaluate_method(name: str, retrieve_fn, eval_set: list[dict]) -> dict:
    """Run one retrieval function over the eval set and aggregate metrics."""
    hit1 = hit3 = 0
    rr_sum = 0.0
    for item in eval_set:
        chunks = retrieve_fn(item["query"])
        rank = _first_hit_rank(chunks, item["gold"])
        if rank is not None:
            if rank == 1:
                hit1 += 1
            if rank <= 3:
                hit3 += 1
            rr_sum += 1.0 / rank
    n = len(eval_set)
    return {
        "method": name,
        "hit@1": hit1 / n,
        "hit@3": hit3 / n,
        "mrr": rr_sum / n,
    }


def run_evaluation() -> list[dict]:
    store, corpus = build_corpus()
    retriever = HybridRetriever(store, corpus)
    reranker = Reranker()

    def hybrid_rerank(query: str) -> list[dict]:
        candidates = retriever.hybrid_search(query, top_k=RERANK_CANDIDATES)
        return reranker.rerank(query, candidates, top_k=RETRIEVE_K)

    methods = [
        ("Vector only", lambda q: retriever.vector_search(q, RETRIEVE_K)),
        ("BM25 only", lambda q: retriever.bm25_search(q, RETRIEVE_K)),
        ("Hybrid (RRF)", lambda q: retriever.hybrid_search(q, RETRIEVE_K)),
        ("Hybrid + Rerank", hybrid_rerank),
    ]

    eval_set = json.loads(EVAL_PATH.read_text(encoding="utf-8"))
    print(f"Corpus: {len(corpus)} chunks from {len(RESUME_FILES)} resumes")
    print(f"Eval set: {len(eval_set)} labeled queries\n")

    results = [evaluate_method(name, fn, eval_set) for name, fn in methods]
    _print_table(results)
    _save_markdown(results, len(corpus), len(eval_set))
    return results


def _print_table(results: list[dict]) -> None:
    header = f"{'Method':<18}{'Hit@1':>8}{'Hit@3':>8}{'MRR':>8}"
    print(header)
    print("-" * len(header))
    for r in results:
        print(
            f"{r['method']:<18}{r['hit@1']*100:>7.1f}%{r['hit@3']*100:>7.1f}%"
            f"{r['mrr']:>8.3f}"
        )


def _save_markdown(results: list[dict], n_chunks: int, n_queries: int) -> None:
    lines = [
        "# Retrieval Evaluation Results",
        "",
        f"Corpus: **{n_chunks} chunks** from {len(RESUME_FILES)} resumes · "
        f"Eval set: **{n_queries} labeled queries**",
        "",
        "| Method | Hit@1 | Hit@3 | MRR |",
        "|--------|------:|------:|----:|",
    ]
    for r in results:
        lines.append(
            f"| {r['method']} | {r['hit@1']*100:.1f}% | {r['hit@3']*100:.1f}% "
            f"| {r['mrr']:.3f} |"
        )
    RESULTS_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nSaved results table to {RESULTS_MD.relative_to(BASE_DIR)}")


if __name__ == "__main__":
    run_evaluation()
