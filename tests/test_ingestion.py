"""Tests for CP1 ingestion. Runs standalone (python -m tests.test_ingestion)
or under pytest."""

from app.config import BASE_DIR
from app.ingestion import chunk_text, clean_text, load_document, prepare_document

SAMPLE_RESUME = BASE_DIR / "data" / "samples" / "sample_resume.txt"


def test_clean_text_collapses_whitespace():
    messy = "Hello    world\n\n\n\nGoodbye   \n"
    cleaned = clean_text(messy)
    assert "    " not in cleaned
    assert "\n\n\n" not in cleaned
    assert cleaned.startswith("Hello world")


def test_chunk_text_respects_size_and_overlap():
    text = " ".join(f"word{i}" for i in range(300))  # long text
    chunks = chunk_text(text, chunk_size=200, overlap=50)
    assert len(chunks) > 1
    # No chunk wildly exceeds the target size (allow a little slack).
    assert all(len(c) <= 210 for c in chunks)
    # Chunks should overlap: the end of one appears near the start of the next.
    assert chunks[0].split()[-1] in chunks[1]


def test_chunk_text_short_input_single_chunk():
    assert chunk_text("short text") == ["short text"]
    assert chunk_text("") == []


def test_load_and_prepare_sample_resume():
    text = load_document(SAMPLE_RESUME)
    assert "ANANYA SHARMA" in text
    chunks = prepare_document(SAMPLE_RESUME)
    assert len(chunks) >= 1
    # The whole document should be represented across the chunks.
    joined = " ".join(chunks)
    assert "Freshworks" in joined
    assert "PostgreSQL" in joined


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
