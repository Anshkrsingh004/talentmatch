"""Shared orchestration: the full RAG flow in one place.

Both the CLI (CP5) and the Streamlit UI (CP6) call `analyze_resume_against_jd`
so the pipeline logic lives in exactly one spot.

Retrieval modes:
    "vector"        — semantic multi-query retrieval (the CP3 baseline)
    "hybrid_rerank" — BM25 + vector fusion, then cross-encoder reranking (CP7, best)
"""

from __future__ import annotations

from pathlib import Path

from app.config import GROQ_MODEL, TOP_K
from app.generation import generate_match_analysis
from app.hybrid import retrieve_for_jd_hybrid
from app.ingestion import prepare_document
from app.retrieval import format_context, retrieve_for_jd
from app.vectorstore import VectorStore


def analyze_resume_against_jd(
    resume_path: str | Path,
    jd_text: str,
    top_k: int = TOP_K,
    model: str = GROQ_MODEL,
    store: VectorStore | None = None,
    retrieval_mode: str = "hybrid_rerank",
    reranker=None,
) -> dict:
    """Run the end-to-end pipeline for one resume vs one job description.

    Returns a dict with the structured analysis plus the retrieval details, so
    callers can also show *why* the model said what it did (citations).
    """
    resume_path = Path(resume_path)

    # 1. Ingest: load + clean + chunk the resume.
    chunks = prepare_document(resume_path)
    if not chunks:
        raise ValueError(f"No text could be extracted from {resume_path.name}")

    # 2. Index: build a fresh vector store + an aligned corpus list for this resume.
    store = store or VectorStore()
    store.reset()
    store.add_chunks(chunks, source=resume_path.name, doc_type="resume")
    corpus = [
        {"text": t, "metadata": {"source": resume_path.name, "chunk_index": i}}
        for i, t in enumerate(chunks)
    ]

    # 3. Retrieve: pull the resume chunks most relevant to the JD.
    if retrieval_mode == "hybrid_rerank":
        if reranker is None:
            from app.rerank import Reranker

            reranker = Reranker()
        hits = retrieve_for_jd_hybrid(
            store, corpus, jd_text, top_k=top_k, reranker=reranker
        )
    else:  # "vector"
        hits = retrieve_for_jd(store, jd_text, top_k=top_k)
    context = format_context(hits)

    # 4. Generate: ask Groq for a grounded, structured match analysis.
    analysis = generate_match_analysis(jd_text, context, model=model)

    return {
        "analysis": analysis,
        "hits": hits,
        "context": context,
        "num_chunks": len(chunks),
        "retrieval_mode": retrieval_mode,
    }
