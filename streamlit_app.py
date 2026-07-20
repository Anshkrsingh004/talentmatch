"""CP6 / CP6.5 — TalentMatch Streamlit UI (Tier-2, vibrant).

Two modes:
  • Match a resume   — one resume vs one JD: KPI tiles, Plotly gauge, coverage
    donut, skill chips, citations, and a downloadable report.
  • Rank candidates  — score many resumes against one JD into a leaderboard.

Color rules (from the dataviz method):
  • Decorative surfaces (hero, KPI tiles) may be vibrant/gradient freely.
  • Data-encoding marks (gauge, donut, chips) use the validated status palette
    (good/warning/critical) AND carry labels + icons, so meaning is never
    color-alone (safe for red-green color-vision deficiency).

Run with:  streamlit run streamlit_app.py
"""

from __future__ import annotations

import html
import tempfile
from pathlib import Path

import plotly.graph_objects as go
import streamlit as st

from app.config import BASE_DIR, GROQ_API_KEY, GROQ_MODEL, TOP_K
from app.pipeline import analyze_resume_against_jd
from app.vectorstore import VectorStore

st.set_page_config(page_title="TalentMatch", page_icon="🎯", layout="wide")

SAMPLES = BASE_DIR / "data" / "samples"
SAMPLE_RESUMES = [
    "sample_resume.txt",
    "resume_devops.txt",
    "resume_frontend.txt",
    "resume_data_analyst.txt",
]


# --------------------------------------------------------------------------- #
# Styling
# --------------------------------------------------------------------------- #
def inject_css() -> None:
    st.markdown(
        """
        <style>
          @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
          html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
          .block-container { padding-top: 1.6rem; max-width: 1200px; }

          /* Vibrant hero */
          .tm-hero {
            background: linear-gradient(120deg, #4f46e5 0%, #7c3aed 45%, #db2777 100%);
            padding: 30px 36px; border-radius: 20px; color: #fff; position: relative;
            box-shadow: 0 16px 40px rgba(124,58,237,0.35); margin-bottom: 14px; overflow: hidden;
          }
          .tm-hero::after {
            content: ""; position: absolute; top: -40%; right: -8%; width: 320px; height: 320px;
            background: radial-gradient(circle, rgba(255,255,255,0.22), transparent 70%);
          }
          .tm-hero h1 { margin: 0; font-size: 2.25rem; font-weight: 800; letter-spacing: -0.02em; }
          .tm-hero p  { margin: 8px 0 0; opacity: 0.94; font-size: 1.02rem; max-width: 760px; }

          /* Section headers */
          .tm-h { font-size: 1.15rem; font-weight: 700; color: #111827; margin: 6px 0 12px;
            display: flex; align-items: center; gap: 8px; }
          .tm-h::before { content: ""; width: 5px; height: 20px; border-radius: 4px;
            background: linear-gradient(#6366f1, #db2777); display: inline-block; }

          /* Vibrant KPI tiles */
          .tm-kpi { border-radius: 16px; padding: 18px 20px; color: #fff;
            box-shadow: 0 10px 22px rgba(0,0,0,0.14); }
          .tm-kpi .n { font-size: 2.05rem; font-weight: 800; line-height: 1; }
          .tm-kpi .l { font-size: 0.78rem; opacity: 0.96; text-transform: uppercase;
            letter-spacing: 0.05em; margin-top: 8px; font-weight: 600; }
          .kpi-green  { background: linear-gradient(135deg, #059669, #34d399); }
          .kpi-rose   { background: linear-gradient(135deg, #e11d48, #fb7185); }
          .kpi-indigo { background: linear-gradient(135deg, #4f46e5, #818cf8); }
          .kpi-sky    { background: linear-gradient(135deg, #0284c7, #38bdf8); }

          /* Cards */
          .tm-card { background: #fff; border: 1px solid #ececf3; border-radius: 16px;
            padding: 20px 22px; box-shadow: 0 2px 10px rgba(17,17,26,0.04);
            height: 100%; transition: box-shadow .2s, transform .2s; }
          .tm-card:hover { box-shadow: 0 8px 24px rgba(99,102,241,0.12); transform: translateY(-2px); }
          .tm-card h4 { margin: 0 0 10px; font-size: 1.05rem; color: #111827; }

          .tm-badge { display: inline-block; padding: 6px 16px; border-radius: 999px;
            font-weight: 700; font-size: 0.9rem; }

          /* Skill chips (icon + text = secondary encoding, CVD-safe) */
          .tm-chip { display: inline-block; padding: 6px 13px; margin: 4px 4px 0 0;
            border-radius: 999px; font-size: 0.85rem; font-weight: 500; }
          .tm-chip-match { background: #dcfce7; color: #15803d; border: 1px solid #86efac; }
          .tm-chip-gap   { background: #fef2f2; color: #b91c1c; border: 1px solid #fca5a5; }

          .tm-rec { background: linear-gradient(90deg, #f5f3ff, #fdf2f8); border-left: 4px solid #8b5cf6;
            border-radius: 8px; padding: 12px 16px; margin-bottom: 10px; font-size: 0.95rem; }

          /* Leaderboard */
          .tm-rank { display:flex; align-items:center; gap:16px; background:#fff;
            border:1px solid #ececf3; border-left-width:6px; border-radius:14px;
            padding:14px 18px; margin-bottom:10px; box-shadow:0 2px 10px rgba(17,17,26,0.04); }
          .tm-rank .pos { font-size:1.6rem; font-weight:800; width:46px; text-align:center; }
          .tm-rank .name { font-weight:700; font-size:1.08rem; }
          .tm-rank .sub  { color:#6b7280; font-size:0.85rem; }
        </style>
        """,
        unsafe_allow_html=True,
    )


# --------------------------------------------------------------------------- #
# Cached models
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner="Loading embedding model (first run only)...")
def get_store() -> VectorStore:
    return VectorStore()


@st.cache_resource(show_spinner="Loading reranker model (first run only)...")
def get_reranker():
    from app.rerank import Reranker

    return Reranker()


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _save_upload(uploaded) -> Path:
    suffix = Path(uploaded.name).suffix
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(uploaded.getbuffer())
    tmp.close()
    return Path(tmp.name)


def _score_palette(score):
    """(accent, badge_bg, badge_fg) from the validated status palette, readable."""
    if score is None:
        return "#9ca3af", "#f3f4f6", "#374151"
    if score >= 75:
        return "#0ca30c", "#dcfce7", "#15803d"   # good
    if score >= 50:
        return "#d97706", "#fef3c7", "#b45309"   # warning
    return "#dc2626", "#fee2e2", "#b91c1c"        # critical


def _retrieval_kwargs(use_rerank: bool) -> dict:
    return {
        "retrieval_mode": "hybrid_rerank" if use_rerank else "vector",
        "reranker": get_reranker() if use_rerank else None,
    }


def _pretty_name(filename: str) -> str:
    stem = Path(filename).stem.replace("resume_", "").replace("sample_", "")
    return stem.replace("_", " ").title() or filename


def section(title: str) -> None:
    st.markdown(f'<div class="tm-h">{title}</div>', unsafe_allow_html=True)


def score_gauge(score, accent: str):
    value = score if score is not None else 0
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=value,
            number={"suffix": "/100", "font": {"size": 34, "color": accent}},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#cbd5e1"},
                "bar": {"color": accent, "thickness": 0.3},
                "borderwidth": 0,
                "steps": [
                    {"range": [0, 50], "color": "#fde8e8"},
                    {"range": [50, 75], "color": "#fef3c7"},
                    {"range": [75, 100], "color": "#dcfce7"},
                ],
            },
        )
    )
    fig.update_layout(
        height=240, margin=dict(l=24, r=24, t=24, b=8),
        paper_bgcolor="rgba(0,0,0,0)", font={"family": "Inter"},
    )
    return fig


def coverage_donut(matched: int, missing: int):
    fig = go.Figure(
        go.Pie(
            labels=["Matched", "Gaps"], values=[matched, missing], hole=0.62,
            marker_colors=["#0ca30c", "#dc2626"], sort=False,
            textinfo="value", textfont={"size": 16, "color": "#fff"},
        )
    )
    fig.update_layout(
        height=240, margin=dict(l=10, r=10, t=24, b=10),
        paper_bgcolor="rgba(0,0,0,0)", font={"family": "Inter"},
        showlegend=True, legend={"orientation": "h", "y": -0.1},
        annotations=[{"text": "Skill<br>coverage", "showarrow": False, "font": {"size": 13}}],
    )
    return fig


def chips_html(items, kind: str) -> str:
    if not items:
        return '<span style="color:#9ca3af">None identified.</span>'
    cls, icon = ("tm-chip-match", "✓") if kind == "match" else ("tm-chip-gap", "✗")
    return "".join(
        f'<span class="tm-chip {cls}">{icon} {html.escape(str(i))}</span>' for i in items
    )


def build_report_markdown(analysis: dict, title: str) -> str:
    lines = [
        f"# {title}", "",
        f"**Match score:** {analysis.get('match_score')}/100 "
        f"({analysis.get('verdict', 'Unknown')})", "",
        "## Summary", analysis.get("summary", ""), "",
        "## Matched skills",
    ]
    lines += [f"- {s}" for s in analysis.get("matched_skills", [])] or ["- (none)"]
    lines += ["", "## Gaps / missing"]
    lines += [f"- {s}" for s in analysis.get("missing_skills", [])] or ["- (none)"]
    lines += ["", "## Recommendations"]
    lines += [f"- {s}" for s in analysis.get("recommendations", [])] or ["- (none)"]
    return "\n".join(lines)


def render_kpis(matched: int, missing: int, chunks: int) -> None:
    total = matched + missing
    coverage = round(100 * matched / total) if total else 0
    tiles = [
        ("kpi-green", matched, "Skills matched"),
        ("kpi-rose", missing, "Gaps found"),
        ("kpi-indigo", f"{coverage}%", "Coverage"),
        ("kpi-sky", chunks, "Evidence chunks"),
    ]
    for col, (cls, n, label) in zip(st.columns(4, gap="medium"), tiles):
        col.markdown(
            f'<div class="tm-kpi {cls}"><div class="n">{n}</div><div class="l">{label}</div></div>',
            unsafe_allow_html=True,
        )


def render_single_result(result: dict, title: str = "resume") -> None:
    analysis = result["analysis"]
    score = analysis.get("match_score")
    matched = analysis.get("matched_skills", [])
    missing = analysis.get("missing_skills", [])
    recs = analysis.get("recommendations", [])
    accent, badge_bg, badge_fg = _score_palette(score)

    section("📊 Overview")
    render_kpis(len(matched), len(missing), len(result["hits"]))

    st.write("")
    g, d, s = st.columns([1.1, 1, 1.4], gap="large")
    with g:
        st.plotly_chart(score_gauge(score, accent), use_container_width=True,
                        config={"displayModeBar": False})
        st.markdown(
            f"<div style='text-align:center'><span class='tm-badge' "
            f"style='background:{badge_bg};color:{badge_fg}'>"
            f"{html.escape(analysis.get('verdict', 'Unknown'))}</span></div>",
            unsafe_allow_html=True,
        )
    with d:
        st.plotly_chart(coverage_donut(len(matched), len(missing)),
                        use_container_width=True, config={"displayModeBar": False})
    with s:
        st.markdown(
            f'<div class="tm-card"><h4>📝 Summary</h4>{html.escape(analysis.get("summary", ""))}</div>',
            unsafe_allow_html=True,
        )

    st.write("")
    section("🧩 Skills breakdown")
    c1, c2 = st.columns(2, gap="large")
    with c1:
        st.markdown(
            f'<div class="tm-card"><h4>✅ Matched skills</h4>{chips_html(matched, "match")}</div>',
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            f'<div class="tm-card"><h4>⚠️ Gaps / missing</h4>{chips_html(missing, "gap")}</div>',
            unsafe_allow_html=True,
        )

    if recs:
        st.write("")
        section("💡 Recommendations")
        for r in recs:
            st.markdown(f'<div class="tm-rec">{html.escape(str(r))}</div>', unsafe_allow_html=True)

    st.write("")
    dl1, dl2 = st.columns([1, 3])
    with dl1:
        st.download_button(
            "📥 Download report",
            data=build_report_markdown(analysis, f"TalentMatch — {title}"),
            file_name="talentmatch_report.md",
            mime="text/markdown",
            use_container_width=True,
        )
    with dl2:
        st.caption(f"Retrieval: **{result.get('retrieval_mode', 'n/a')}** · "
                   f"grounded on {len(result['hits'])} chunks")

    with st.expander(f"🔎 Retrieved resume evidence ({len(result['hits'])} chunks)"):
        for i, hit in enumerate(result["hits"], start=1):
            meta = hit["metadata"]
            st.markdown(
                f"**[{i}]** `{meta['source']}` · chunk {meta['chunk_index']} · "
                f"relevance **{hit['score']:.2f}**"
            )
            st.text(hit["text"])
            st.divider()


# --------------------------------------------------------------------------- #
# App shell
# --------------------------------------------------------------------------- #
inject_css()
st.session_state.setdefault("jd_text", "")
st.session_state.setdefault("jd_text_rank", "")

st.markdown(
    """
    <div class="tm-hero">
      <h1>🎯 TalentMatch</h1>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("⚙️ Settings")
    top_k = st.slider("Chunks to retrieve (top-k)", 2, 10, TOP_K)
    model = st.text_input("Groq model", value=GROQ_MODEL)
    use_rerank = st.toggle(
        "Advanced retrieval", value=True,
        help="Hybrid (BM25 + vector) search with cross-encoder reranking. "
        "Off = plain semantic search.",
    )
    st.divider()
    if GROQ_API_KEY:
        st.success("Groq API key loaded ✅")
    else:
        st.error("No Groq API key. Add GROQ_API_KEY to `.env`.")
    st.divider()
    st.markdown(
        "**How it works**\n\n"
        "1. Resume chunked & embedded locally\n"
        "2. Hybrid search (BM25 + vectors) in ChromaDB\n"
        "3. Cross-encoder reranks the best chunks\n"
        "4. Groq scores the match — grounded only in the resume"
    )

tab_match, tab_rank = st.tabs(["🎯 Match a resume", "🏆 Rank candidates"])


# --------------------------------------------------------------------------- #
# Tab 1 — single match
# --------------------------------------------------------------------------- #
with tab_match:
    section("1 · Provide the documents")
    use_sample_resume = st.toggle("Use the built-in sample resume", value=False)

    col_resume, col_jd = st.columns(2, gap="large")
    with col_resume:
        if use_sample_resume:
            st.info("Using sample resume: **sample_resume.txt**")
            resume_file = None
        else:
            resume_file = st.file_uploader(
                "📄 Upload a resume (PDF, DOCX, or TXT)", type=["pdf", "docx", "txt"]
            )
    with col_jd:
        if st.button("📋 Load sample job description", key="load_jd_single"):
            st.session_state.jd_text = (SAMPLES / "sample_jd.txt").read_text(encoding="utf-8")
        st.text_area("💼 Paste the job description", height=230, key="jd_text")

    section("2 · Run the match")
    if st.button("🔍 Analyze Match", type="primary", use_container_width=True):
        jd_text = st.session_state.jd_text
        if not GROQ_API_KEY:
            st.error("Cannot run: no Groq API key.")
        elif not use_sample_resume and resume_file is None:
            st.warning("Please upload a resume (or switch on the sample resume).")
        elif not jd_text.strip():
            st.warning("Please paste a job description (or load the sample).")
        else:
            if use_sample_resume:
                resume_path, cleanup = SAMPLES / "sample_resume.txt", False
            else:
                resume_path, cleanup = _save_upload(resume_file), True
            try:
                with st.spinner("Analyzing (embedding → retrieving → generating)..."):
                    st.session_state.result = analyze_resume_against_jd(
                        resume_path, jd_text, top_k=top_k, model=model,
                        store=get_store(), **_retrieval_kwargs(use_rerank),
                    )
            except Exception as exc:  # noqa: BLE001
                st.error(f"Something went wrong: {exc}")
                st.session_state.pop("result", None)
            finally:
                if cleanup:
                    Path(resume_path).unlink(missing_ok=True)

    if st.session_state.get("result"):
        st.divider()
        render_single_result(st.session_state.result)


# --------------------------------------------------------------------------- #
# Tab 2 — recruiter mode: rank many resumes against one JD
# --------------------------------------------------------------------------- #
with tab_rank:
    section("Rank multiple candidates against one job description")
    use_sample_pool = st.toggle("Use the 4 built-in sample resumes", value=True)

    if use_sample_pool:
        st.info("Candidate pool: " + ", ".join(f"`{r}`" for r in SAMPLE_RESUMES))
        uploaded_pool = None
    else:
        uploaded_pool = st.file_uploader(
            "📄 Upload multiple resumes", type=["pdf", "docx", "txt"],
            accept_multiple_files=True,
        )

    if st.button("📋 Load sample job description", key="load_jd_rank"):
        st.session_state.jd_text_rank = (SAMPLES / "sample_jd.txt").read_text(encoding="utf-8")
    st.text_area("💼 Paste the job description", height=180, key="jd_text_rank")

    if st.button("🏆 Rank Candidates", type="primary", use_container_width=True):
        jd_text = st.session_state.jd_text_rank
        if not GROQ_API_KEY:
            st.error("Cannot run: no Groq API key.")
        elif not jd_text.strip():
            st.warning("Please paste a job description (or load the sample).")
        else:
            pool: list[tuple[str, Path, bool]] = []
            if use_sample_pool:
                pool = [(_pretty_name(r), SAMPLES / r, False) for r in SAMPLE_RESUMES]
            elif uploaded_pool:
                pool = [(_pretty_name(f.name), _save_upload(f), True) for f in uploaded_pool]

            if not pool:
                st.warning("Please add at least one resume to the pool.")
            else:
                ranked = []
                progress = st.progress(0.0, text="Scoring candidates...")
                try:
                    for i, (name, path, _) in enumerate(pool):
                        res = analyze_resume_against_jd(
                            path, jd_text, top_k=top_k, model=model,
                            store=get_store(), **_retrieval_kwargs(use_rerank),
                        )
                        ranked.append((name, res))
                        progress.progress((i + 1) / len(pool),
                                          text=f"Scored {i + 1}/{len(pool)}")
                except Exception as exc:  # noqa: BLE001
                    st.error(f"Something went wrong: {exc}")
                finally:
                    for _, path, cleanup in pool:
                        if cleanup:
                            Path(path).unlink(missing_ok=True)
                progress.empty()

                ranked.sort(key=lambda r: r[1]["analysis"].get("match_score") or 0, reverse=True)
                st.session_state.ranked = ranked

    if st.session_state.get("ranked"):
        st.divider()
        section("🏆 Leaderboard")
        for pos, (name, res) in enumerate(st.session_state.ranked, start=1):
            a = res["analysis"]
            score = a.get("match_score")
            accent, badge_bg, badge_fg = _score_palette(score)
            medal = {1: "🥇", 2: "🥈", 3: "🥉"}.get(pos, f"#{pos}")
            st.markdown(
                f"""<div class="tm-rank" style="border-left-color:{accent}">
                  <div class="pos">{medal}</div>
                  <div style="flex:1">
                    <div class="name">{html.escape(name)}</div>
                    <div class="sub">{len(a.get('matched_skills', []))} matched ·
                      {len(a.get('missing_skills', []))} gaps</div>
                  </div>
                  <div class="tm-badge" style="background:{badge_bg};color:{badge_fg}">
                    {score if score is not None else '—'}/100</div>
                </div>""",
                unsafe_allow_html=True,
            )
            with st.expander(f"Details — {name}"):
                st.write(a.get("summary", ""))
                st.markdown("**Matched:** " + (", ".join(a.get("matched_skills", [])) or "—"))
                st.markdown("**Gaps:** " + (", ".join(a.get("missing_skills", [])) or "—"))
