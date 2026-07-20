"""TalentMatch — a RAG system that matches resumes against job descriptions."""

# ChromaDB requires sqlite >= 3.35. Some cloud hosts (e.g. Streamlit Cloud) ship an
# older system sqlite; pysqlite3-binary (Linux wheels) provides a modern build.
# Swap it in for the stdlib sqlite3 if available; no-op locally on Windows/macOS.
try:
    __import__("pysqlite3")
    import sys

    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
except ImportError:
    pass

__version__ = "0.1.0"
