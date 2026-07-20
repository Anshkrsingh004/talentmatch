"""CP2 — Embeddings + vector store.

Wraps a local sentence-transformers model (for embeddings) and ChromaDB (for
persistent vector storage + similarity search). Nothing here calls a paid API.
"""

from __future__ import annotations

import re
from pathlib import Path

import chromadb

from app.config import (
    CHROMA_COLLECTION,
    CHROMA_DIR,
    EMBEDDING_MODEL,
    TOP_K,
)


# --------------------------------------------------------------------------- #
# Embeddings (local, free)
# --------------------------------------------------------------------------- #
class Embedder:
    """Lazy wrapper around a sentence-transformers model.

    The model (~90 MB) is downloaded once on first use and cached locally, then
    loaded lazily so importing this module stays cheap.
    """

    def __init__(self, model_name: str = EMBEDDING_MODEL):
        self.model_name = model_name
        self._model = None

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)
        return self._model

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of texts into L2-normalised vectors (cosine-ready)."""
        vectors = self.model.encode(
            texts,
            normalize_embeddings=True,   # so cosine similarity == dot product
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return vectors.tolist()

    def embed_one(self, text: str) -> list[float]:
        return self.embed([text])[0]


# --------------------------------------------------------------------------- #
# Vector store (ChromaDB)
# --------------------------------------------------------------------------- #
def _slug(text: str) -> str:
    """Turn a source name into a safe id prefix."""
    return re.sub(r"[^a-zA-Z0-9]+", "_", text).strip("_").lower() or "doc"


class VectorStore:
    """Persistent ChromaDB collection using cosine similarity."""

    def __init__(
        self,
        persist_dir: str | Path = CHROMA_DIR,
        collection_name: str = CHROMA_COLLECTION,
        embedder: Embedder | None = None,
    ):
        Path(persist_dir).mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=str(persist_dir))
        self.collection_name = collection_name
        self.embedder = embedder or Embedder()
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},  # distance = 1 - cosine similarity
        )

    def add_chunks(
        self,
        chunks: list[str],
        source: str,
        doc_type: str = "resume",
    ) -> int:
        """Embed and store chunks. Re-adding the same source overwrites (upsert)."""
        if not chunks:
            return 0

        prefix = _slug(source)
        ids = [f"{prefix}-{i}" for i in range(len(chunks))]
        metadatas = [
            {"source": source, "doc_type": doc_type, "chunk_index": i}
            for i in range(len(chunks))
        ]
        embeddings = self.embedder.embed(chunks)
        self.collection.upsert(
            ids=ids,
            documents=chunks,
            embeddings=embeddings,
            metadatas=metadatas,
        )
        return len(chunks)

    def query(
        self,
        text: str,
        top_k: int = TOP_K,
        where: dict | None = None,
    ) -> list[dict]:
        """Return the top-k most similar chunks with a 0..1 similarity score."""
        q_emb = self.embedder.embed_one(text)
        res = self.collection.query(
            query_embeddings=[q_emb],
            n_results=top_k,
            where=where,
        )
        docs = res["documents"][0]
        metas = res["metadatas"][0]
        dists = res["distances"][0]
        results = []
        for doc, meta, dist in zip(docs, metas, dists):
            results.append(
                {
                    "text": doc,
                    "metadata": meta,
                    "distance": dist,
                    "score": round(1.0 - dist, 4),  # cosine similarity
                }
            )
        return results

    def count(self) -> int:
        return self.collection.count()

    def reset(self) -> None:
        """Drop and recreate the collection (useful before a fresh ingest)."""
        self.client.delete_collection(self.collection_name)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )


if __name__ == "__main__":
    # Demo: ingest the sample resume, then run a JD-style query.
    from app.config import BASE_DIR
    from app.ingestion import prepare_document

    resume = BASE_DIR / "data" / "samples" / "sample_resume.txt"

    store = VectorStore()
    store.reset()  # start clean for the demo
    chunks = prepare_document(resume)
    n = store.add_chunks(chunks, source=resume.name, doc_type="resume")
    print(f"Stored {n} chunks. Collection now holds {store.count()} vectors.\n")

    query = "Python backend developer with REST APIs, SQL and testing experience"
    print(f"Query: {query!r}\n")
    for rank, hit in enumerate(store.query(query, top_k=3), start=1):
        snippet = hit["text"][:140].replace("\n", " ")
        print(f"#{rank}  score={hit['score']:.3f}  [{hit['metadata']['source']}]")
        print(f"     {snippet}...\n")
