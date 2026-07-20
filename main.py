"""CP5 — TalentMatch CLI.

Match a resume against a job description from the terminal.

Examples:
    python main.py --demo
    python main.py --resume data/resumes/my_cv.pdf --jd data/job_descriptions/role.txt
    python main.py --resume my_cv.pdf --jd-text "We need a Python backend engineer..."
"""

from __future__ import annotations

import argparse
import json
import sys

# Windows consoles default to cp1252, which can't encode characters that may
# appear in model output (em dashes, arrows, etc.). Force UTF-8 so printing
# never crashes.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass

from app.config import BASE_DIR, GROQ_MODEL, TOP_K
from app.generation import format_analysis_text


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="talentmatch",
        description="Match a resume against a job description (RAG + Groq).",
    )
    p.add_argument("--resume", help="Path to resume file (PDF/DOCX/TXT)")
    p.add_argument("--jd", help="Path to job description file (PDF/DOCX/TXT)")
    p.add_argument("--jd-text", help="Job description as raw text (instead of --jd)")
    p.add_argument("--top-k", type=int, default=TOP_K, help="Chunks to retrieve")
    p.add_argument("--model", default=GROQ_MODEL, help="Groq model id")
    p.add_argument("--json", action="store_true", help="Print raw JSON analysis")
    p.add_argument(
        "--demo", action="store_true", help="Run on the bundled sample resume + JD"
    )
    return p.parse_args()


def _resolve_inputs(args: argparse.Namespace):
    """Figure out the resume path and JD text from the CLI arguments."""
    if args.demo:
        samples = BASE_DIR / "data" / "samples"
        return samples / "sample_resume.txt", (
            samples / "sample_jd.txt"
        ).read_text(encoding="utf-8")

    if not args.resume:
        sys.exit("Error: --resume is required (or use --demo). See --help.")

    if args.jd_text:
        jd_text = args.jd_text
    elif args.jd:
        from app.ingestion import load_document

        jd_text = load_document(args.jd)
    else:
        sys.exit("Error: provide --jd <file> or --jd-text <text> (or use --demo).")

    return args.resume, jd_text


def main() -> None:
    args = _parse_args()
    resume_path, jd_text = _resolve_inputs(args)

    # Import here so --help works even before dependencies/model load.
    from app.pipeline import analyze_resume_against_jd

    print("=" * 60)
    print("  TalentMatch - Resume vs Job Description Analyzer")
    print("=" * 60)
    print(f"Resume : {resume_path}")
    print(f"Model  : {args.model}")
    print("Analyzing (embedding, retrieving, generating)...\n")

    try:
        result = analyze_resume_against_jd(
            resume_path, jd_text, top_k=args.top_k, model=args.model
        )
    except Exception as exc:  # noqa: BLE001 - CLI should fail gracefully
        sys.exit(f"\nFailed: {exc}")

    if args.json:
        print(json.dumps(result["analysis"], indent=2))
        return

    print(format_analysis_text(result["analysis"]))
    print("\n" + "-" * 60)
    print(f"Grounded on {len(result['hits'])} retrieved resume chunk(s):")
    for i, hit in enumerate(result["hits"], start=1):
        meta = hit["metadata"]
        print(f"  [{i}] {meta['source']} · chunk {meta['chunk_index']} "
              f"· relevance {hit['score']:.2f}")


if __name__ == "__main__":
    main()
