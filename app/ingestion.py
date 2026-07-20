"""CP1 — Document ingestion.

Load a resume or job description from PDF / DOCX / TXT, clean the text, and
split it into overlapping chunks ready for embedding.
"""

from __future__ import annotations

import re
from pathlib import Path

from app.config import CHUNK_OVERLAP, CHUNK_SIZE

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt"}


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def load_document(path: str | Path) -> str:
    """Extract raw text from a document, dispatching on file extension."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    ext = path.suffix.lower()
    if ext == ".pdf":
        return _load_pdf(path)
    if ext == ".docx":
        return _load_docx(path)
    if ext == ".txt":
        return path.read_text(encoding="utf-8", errors="ignore")

    raise ValueError(
        f"Unsupported file type '{ext}'. Supported: {sorted(SUPPORTED_EXTENSIONS)}"
    )


def _load_pdf(path: Path) -> str:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n".join(pages)


def _load_docx(path: Path) -> str:
    import docx  # python-docx

    document = docx.Document(str(path))
    return "\n".join(p.text for p in document.paragraphs)


# --------------------------------------------------------------------------- #
# Cleaning
# --------------------------------------------------------------------------- #
def clean_text(text: str) -> str:
    """Normalise whitespace while preserving paragraph breaks."""
    # Collapse runs of spaces/tabs.
    text = re.sub(r"[ \t]+", " ", text)
    # Collapse 3+ newlines into a clean paragraph break.
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Strip trailing spaces on each line.
    text = "\n".join(line.strip() for line in text.splitlines())
    return text.strip()


# --------------------------------------------------------------------------- #
# Chunking
# --------------------------------------------------------------------------- #
def chunk_text(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> list[str]:
    """Split text into overlapping chunks, snapping to word boundaries.

    Overlap keeps context continuous so a sentence split across two chunks is
    still retrievable from either one.
    """
    text = text.strip()
    if not text:
        return []
    if len(text) <= chunk_size:
        return [text]

    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))

        # Snap the end back to the nearest space so we don't cut mid-word.
        if end < len(text):
            boundary = text.rfind(" ", start, end)
            if boundary > start:
                end = boundary

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        if end >= len(text):
            break

        # Step forward, keeping `overlap` chars of context. Guard against
        # a degenerate step that would loop forever.
        next_start = end - overlap
        start = next_start if next_start > start else end

    return chunks


def prepare_document(
    path: str | Path,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> list[str]:
    """Load -> clean -> chunk a document in one call."""
    raw = load_document(path)
    cleaned = clean_text(raw)
    return chunk_text(cleaned, chunk_size, overlap)


if __name__ == "__main__":
    # Quick manual check against the bundled sample resume.
    from app.config import BASE_DIR

    sample = BASE_DIR / "data" / "samples" / "sample_resume.txt"
    chunks = prepare_document(sample)
    print(f"Loaded '{sample.name}' -> {len(chunks)} chunks\n")
    for i, c in enumerate(chunks):
        print(f"--- chunk {i} ({len(c)} chars) ---")
        print(c[:200] + ("..." if len(c) > 200 else ""))
        print()
