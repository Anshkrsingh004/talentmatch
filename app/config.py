"""Central configuration for TalentMatch.

All tunable settings live here so the rest of the code stays clean.
Values can be overridden via environment variables (see .env.example).
"""

import os
from pathlib import Path

from dotenv import load_dotenv

# Load variables from .env into the environment (if the file exists).
load_dotenv()

# --- Paths ---
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RESUME_DIR = DATA_DIR / "resumes"
JD_DIR = DATA_DIR / "job_descriptions"
CHROMA_DIR = DATA_DIR / "chroma_db"

# --- Embeddings (local, free, runs offline) ---
EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
)

# --- Vector store ---
CHROMA_COLLECTION = "resume_chunks"

# --- Chunking ---
CHUNK_SIZE = 500        # characters per chunk (approx)
CHUNK_OVERLAP = 100     # overlap between consecutive chunks

# --- Retrieval ---
TOP_K = 5               # number of chunks to retrieve per query

# --- Generation (Groq) ---
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")


def require_groq_key() -> str:
    """Return the Groq API key or raise a helpful error if it's missing."""
    if not GROQ_API_KEY:
        raise RuntimeError(
            "GROQ_API_KEY is not set. Copy .env.example to .env and add your "
            "free key from https://console.groq.com"
        )
    return GROQ_API_KEY
