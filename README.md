# TalentMatch 🎯

A **Retrieval-Augmented Generation (RAG)** system that matches a resume against a job
description — retrieving the most relevant experience, scoring the fit, and explaining
skill gaps.

Built to be **100% free**: local embeddings + local vector store + a free LLM API.

## 🏗️ Architecture

```
Resume + Job Description
   │
   ▼
[ Ingestion ]               →  load (PDF/DOCX/TXT), clean, chunk with overlap
   │
   ▼
[ sentence-transformers ]   →  local embeddings (free, offline)
   │
   ▼
[ ChromaDB  +  BM25 ]       →  hybrid retrieval: semantic + keyword (RRF fusion)
   │
   ▼
[ Cross-encoder reranker ]  →  precision reranking of top candidates
   │
   ▼
[ Groq + Llama 3.3 ]        →  grounded match score + gap analysis (free API)
   │
   ▼
Result: score, matched skills, gaps, citations
```

## 🧰 Tech stack

| Layer | Tool | Why |
|-------|------|-----|
| Embeddings | `sentence-transformers` (all-MiniLM-L6-v2) | Free, local, no API needed |
| Vector store | ChromaDB (cosine) | Simple, local, persistent |
| Keyword search | BM25 (`rank-bm25`) | Catches exact terms embeddings miss |
| Reranker | Cross-encoder (`ms-marco-MiniLM-L-6-v2`) | Precision reranking, local |
| Generation | Groq (Llama 3.3 70B) | Free, very fast inference |
| UI | Streamlit | Quick, clean web demo |

## 🚀 Getting started

```bash
# 1. Create & activate a virtual environment
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Add your free Groq API key
cp .env.example .env      # then edit .env and paste your key
```

## 📍 Project status

Built checkpoint by checkpoint:

- [x] **CP0** — Scaffolding & setup
- [x] **CP1** — Document ingestion
- [x] **CP2** — Embeddings + vector store
- [x] **CP3** — Retrieval
- [x] **CP4** — Generation (Groq)
- [x] **CP5** — End-to-end CLI MVP
- [x] **CP6** — Streamlit UI
- [x] **CP7** — Hybrid search + reranking + evaluation
- [ ] **CP8** — Deploy + polish

## 📊 Evaluation results

Retrieval quality on a labeled benchmark (**35 chunks** from 4 resumes, **16 queries**).
Run it yourself: `python -m app.evaluation`.

| Method | Hit@1 | Hit@3 | MRR |
|--------|------:|------:|----:|
| Vector only (baseline) | 62.5% | 93.8% | 0.776 |
| BM25 only | 68.8% | 100% | 0.844 |
| Hybrid (RRF) | 68.8% | 100% | 0.844 |
| **Hybrid + Rerank** | **81.2%** | **100%** | **0.896** |

➡️ Adding hybrid search + cross-encoder reranking improved **Hit@1 from 62.5% → 81.2%**
and **MRR from 0.78 → 0.90** over a plain vector-search baseline.

## 📁 Structure

```
RAGProject/
├── app/
│   ├── config.py          # central configuration
│   ├── ingestion.py       # (CP1) load & chunk documents
│   ├── vectorstore.py     # (CP2) embeddings + ChromaDB
│   ├── retrieval.py       # (CP3) semantic multi-query search
│   ├── generation.py      # (CP4) Groq LLM calls
│   ├── pipeline.py        # (CP5) end-to-end orchestration
│   ├── hybrid.py          # (CP7) BM25 + vector hybrid retrieval (RRF)
│   ├── rerank.py          # (CP7) cross-encoder reranker
│   └── evaluation.py      # (CP7) Hit@k / MRR benchmark
├── data/
│   ├── samples/           # sample resumes + JD
│   ├── eval/              # eval_set.json + results.md
│   └── chroma_db/         # vector store (gitignored)
├── tests/                 # standalone test scripts
├── main.py                # (CP5) CLI entry point
├── streamlit_app.py       # (CP6) web UI
└── requirements.txt
```

## 🧑‍💼 Resume bullets

- Built a **RAG pipeline** matching resumes to job descriptions (local embeddings +
  ChromaDB + Groq/Llama 3.3), with grounded, citation-backed output.
- Implemented **hybrid retrieval (BM25 + vector, RRF)** and **cross-encoder reranking**;
  improved retrieval **Hit@1 from 62.5% → 81.2%** on a labeled benchmark.
- Built an **evaluation harness** (Hit@k, MRR) and a **Streamlit** web app; kept the
  entire stack **100% free**.
