"""CP4 — Generation with Groq.

Takes the job description + retrieved resume context and asks a Groq-hosted LLM
(Llama 3.3 70B) for a structured match analysis. Uses JSON mode so the result is
easy to render, and instructs the model to use ONLY the provided context to keep
it grounded (no hallucinated skills).
"""

from __future__ import annotations

import json

from app.config import GROQ_MODEL, require_groq_key

SYSTEM_PROMPT = """You are TalentMatch, an expert technical recruiter assistant.
You compare a candidate's resume against a job description and produce an honest,
evidence-based assessment.

Rules:
- Use ONLY the information in the provided resume context. Never invent skills,
  tools, or experience that are not present in the context.
- If the resume shows no evidence for a requirement, treat it as a gap.
- Be specific and concise; refer to concrete evidence from the resume.
- Respond with a single valid JSON object matching the requested schema."""

RESPONSE_SCHEMA = """Return a JSON object with EXACTLY these keys:
{
  "match_score": integer from 0 to 100 (overall fit for the role),
  "verdict": one of "Strong match", "Moderate match", "Weak match",
  "summary": string of 2-3 sentences explaining the score,
  "matched_skills": array of strings (requirements the resume clearly satisfies),
  "missing_skills": array of strings (requirements with no evidence in the resume),
  "recommendations": array of strings (concrete things the candidate should add or improve)
}"""

# Keys we guarantee downstream code (CLI / UI) can rely on.
_DEFAULTS = {
    "match_score": None,
    "verdict": "Unknown",
    "summary": "",
    "matched_skills": [],
    "missing_skills": [],
    "recommendations": [],
}


def build_user_prompt(jd_text: str, context: str) -> str:
    return (
        f"JOB DESCRIPTION:\n{jd_text.strip()}\n\n"
        f"CANDIDATE RESUME CONTEXT (retrieved excerpts):\n{context.strip()}\n\n"
        f"{RESPONSE_SCHEMA}"
    )


def generate_match_analysis(
    jd_text: str,
    context: str,
    model: str = GROQ_MODEL,
    temperature: float = 0.2,
) -> dict:
    """Call Groq and return a structured match analysis dict.

    Always returns a dict containing the keys in `_DEFAULTS`, even if the model
    output is malformed.
    """
    from groq import Groq

    client = Groq(api_key=require_groq_key())

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_user_prompt(jd_text, context)},
            ],
            temperature=temperature,
            response_format={"type": "json_object"},
        )
    except Exception as exc:  # noqa: BLE001 - surface a friendly, actionable error
        raise RuntimeError(f"Groq API call failed: {exc}") from exc

    raw = response.choices[0].message.content or "{}"
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        parsed = {"summary": raw}

    # Merge onto defaults so every expected key is always present.
    result = {**_DEFAULTS, **parsed}
    result["model"] = model
    return result


def format_analysis_text(data: dict) -> str:
    """Render the analysis dict as readable text for the CLI."""
    lines = []
    score = data.get("match_score")
    score_str = f"{score}/100" if score is not None else "N/A"
    lines.append(f"MATCH SCORE : {score_str}   ({data.get('verdict', 'Unknown')})")
    lines.append("")
    lines.append("SUMMARY")
    lines.append(f"  {data.get('summary', '').strip()}")

    def _section(title: str, items: list[str]) -> None:
        lines.append("")
        lines.append(title)
        if items:
            for item in items:
                lines.append(f"  - {item}")
        else:
            lines.append("  (none)")

    _section("MATCHED SKILLS", data.get("matched_skills", []))
    _section("GAPS / MISSING", data.get("missing_skills", []))
    _section("RECOMMENDATIONS", data.get("recommendations", []))
    return "\n".join(lines)


if __name__ == "__main__":
    # End-to-end demo: retrieve context from the sample resume, then ask Groq.
    from app.config import BASE_DIR, GROQ_API_KEY
    from app.ingestion import prepare_document
    from app.retrieval import format_context, retrieve_for_jd
    from app.vectorstore import VectorStore

    if not GROQ_API_KEY:
        print(
            "GROQ_API_KEY not set.\n"
            "1. Get a free key at https://console.groq.com\n"
            "2. Copy .env.example to .env and paste your key\n"
            "3. Re-run:  python -m app.generation"
        )
        raise SystemExit(0)

    samples = BASE_DIR / "data" / "samples"
    store = VectorStore()
    store.reset()
    store.add_chunks(
        prepare_document(samples / "sample_resume.txt"),
        source="sample_resume.txt",
        doc_type="resume",
    )

    jd_text = (samples / "sample_jd.txt").read_text(encoding="utf-8")
    context = format_context(retrieve_for_jd(store, jd_text))

    print("Calling Groq for match analysis...\n")
    analysis = generate_match_analysis(jd_text, context)
    print(format_analysis_text(analysis))
