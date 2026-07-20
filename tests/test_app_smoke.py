"""Smoke test for the Streamlit UI (CP6).

Runs the app script headlessly with Streamlit's AppTest and asserts it renders
without raising. Catches runtime rendering errors that a syntax check misses.
The initial run does NOT click Analyze, so it never calls Groq or loads the
embedding model — it's fast and free.
"""

from pathlib import Path

from streamlit.testing.v1 import AppTest

APP_PATH = str(Path(__file__).resolve().parent.parent / "streamlit_app.py")


FAKE_RESULT = {
    "analysis": {
        "match_score": 82,
        "verdict": "Strong match",
        "summary": "Solid overlap on Python and REST APIs; minor gaps in DevOps.",
        "matched_skills": ["Python", "REST APIs", "SQL", "Git"],
        "missing_skills": ["Docker & Kubernetes", "Cloud (AWS/GCP)"],
        "recommendations": ["Add a Dockerized project", "Try a small AWS deploy"],
    },
    "hits": [
        {
            "text": "Built REST APIs in Python with Flask and pytest.",
            "metadata": {"source": "resume.txt", "chunk_index": 2},
            "score": 0.59,
        }
    ],
    "num_chunks": 4,
}


def test_app_renders_without_exception():
    at = AppTest.from_file(APP_PATH, default_timeout=60)
    at.run()
    assert not at.exception, f"App raised an exception on load: {at.exception}"
    # Sanity: the hero title and the primary button should be present.
    assert any("TalentMatch" in (m.value or "") for m in at.markdown)
    assert any(b.label == "🔍 Analyze Match" for b in at.button)


FAKE_RANKED = [
    ("Ananya Sharma", FAKE_RESULT),
    ("Rahul Verma", {
        "analysis": {
            "match_score": 55, "verdict": "Moderate match",
            "summary": "Some overlap but missing core backend skills.",
            "matched_skills": ["Git", "Linux"],
            "missing_skills": ["Python", "REST APIs", "SQL"],
            "recommendations": ["Build a Python API project"],
        },
        "hits": [], "num_chunks": 3, "retrieval_mode": "hybrid_rerank",
    }),
]


def test_leaderboard_renders_without_exception():
    """Seed a fake ranked list so the recruiter-mode leaderboard renders without Groq."""
    at = AppTest.from_file(APP_PATH, default_timeout=60)
    at.session_state["ranked"] = FAKE_RANKED
    at.run()
    assert not at.exception, f"Leaderboard rendering raised: {at.exception}"
    all_markdown = " ".join(m.value or "" for m in at.markdown)
    assert "Ananya Sharma" in all_markdown


def test_results_render_without_exception():
    """Seed a fake result so render_single_result (gauge, donut, chips) runs without Groq."""
    at = AppTest.from_file(APP_PATH, default_timeout=60)
    at.session_state["result"] = FAKE_RESULT
    at.run()
    assert not at.exception, f"Results rendering raised: {at.exception}"
    # A skill chip (rendered as HTML markdown) should appear. The score now lives
    # in a Plotly gauge, so we don't assert on it as text.
    all_markdown = " ".join(m.value or "" for m in at.markdown)
    assert "Python" in all_markdown


def _run_all():
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"  PASS  {name}")
    print("\nAll app smoke tests passed.")


if __name__ == "__main__":
    _run_all()
