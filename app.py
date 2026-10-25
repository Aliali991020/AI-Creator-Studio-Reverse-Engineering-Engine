"""AI Creator Studio — local Streamlit control room. Run: streamlit run app.py"""
from __future__ import annotations

import asyncio
import importlib.util
import os
import re
import uuid
from pathlib import Path

import streamlit as st

from core.analytics import AnalyticsAggregator
from core.github_learner import GitHubLearner
from core.models import LANGUAGE_NAMES, GenerationRequest
from core.ollama import OllamaClient
from core.orchestrator import StudioOrchestrator
from core.sales_agent import SalesAgent
from core.utils import ROOT, UPLOAD_DIR, ensure_directories, ffmpeg_binary

ensure_directories()
st.set_page_config(page_title="AI Creator Studio", page_icon="◆", layout="wide", initial_sidebar_state="expanded")

st.markdown(
    """
<style>
    :root { --ink:#edf2ff; --muted:#98a2b8; --line:#293142; --panel:#111722; --accent:#c7f36b; }
    .stApp { background: #0b0f16; color: var(--ink); }
    [data-testid="stHeader"] { background: rgba(11,15,22,.8); }
    [data-testid="stSidebar"] { background: #0e141e; border-right: 1px solid var(--line); }
    [data-testid="stSidebar"] * { color: #dce4f2; }
    h1,h2,h3 { letter-spacing: -.035em; color: #f5f7fb; }
    p, label, .stMarkdown { color: #c5cddc; }
    .hero { padding: 2.3rem 0 1.3rem; border-bottom: 1px solid var(--line); margin-bottom: 1.4rem; }
    .eyebrow { color: var(--accent); text-transform: uppercase; letter-spacing: .18em; font-size: .72rem; font-weight: 700; }
    .hero h1 { font-size: clamp(2.3rem, 5vw, 4.2rem); line-height: .98; margin: .55rem 0 .8rem; max-width: 850px; }
    .hero p { max-width: 720px; font-size: 1.03rem; color: #9da9bb; }
    .local-badge { display:inline-block; padding:.32rem .65rem; border:1px solid #3c4a36; border-radius:99px;
                   background:#172014; color:#c7f36b; font-size:.72rem; font-weight:700; margin-right:.4rem; }
    .studio-note { background:#111722; border:1px solid #293142; border-radius:12px; padding:.9rem 1rem; margin:.7rem 0 1rem; }
    .studio-note strong { color:#edf2ff; }
    .stButton > button[kind="primary"], .stButton > button[data-testid="baseButton-primary"] {
        background:#c7f36b; color:#10150b; border:0; border-radius:9px; font-weight:800; padding:.8rem 1rem;
    }
    .stButton > button[kind="primary"]:hover { background:#d8ff8b; color:#10150b; }
    .stTextInput input, .stTextArea textarea, .stSelectbox div[data-baseweb="select"] > div {
        background:#111722; border-color:#303a4c; color:#eef2f8;
    }
    .stTabs [data-baseweb="tab-list"] { gap:1.2rem; border-bottom:1px solid #293142; }
    .stTabs [data-baseweb="tab"] { color:#9ba8bb; }
    .stTabs [aria-selected="true"] { color:#c7f36b !important; }
    [data-testid="stMetric"] { background:#111722; border:1px solid #293142; border-radius:10px; padding:.8rem; }
    [data-testid="stMetricValue"] { color:#f1f5fb; }
    .result-path { font-family:monospace; font-size:.8rem; color:#96a3b8; background:#0e141e; padding:.7rem; border-radius:7px; }
    #MainMenu {visibility:hidden;} footer {visibility:hidden;}
</style>
<div class="hero">
  <div class="eyebrow">Local autonomous production system</div>
  <h1>One idea.<br/>An entire studio.</h1>
  <p>Script, narrate, visualize, caption, dub, and qualify leads from one local workspace.
     No SaaS media APIs. No automatic publishing.</p>
  <span class="local-badge">100% manual upload</span>
  <span class="local-badge">FOSS-first engines</span>
</div>
""",
    unsafe_allow_html=True,
)


@st.cache_resource
def sales_agent() -> SalesAgent:
    return SalesAgent()


@st.cache_resource
def analytics_engine() -> AnalyticsAggregator:
    return AnalyticsAggregator()


@st.cache_data(ttl=15, show_spinner=False)
def backend_status() -> dict[str, bool | str]:
    return {
        "FFmpeg": bool(ffmpeg_binary()),
        "Ollama": OllamaClient().available(),
        "XTTS": importlib.util.find_spec("TTS") is not None,
        "Whisper": importlib.util.find_spec("faster_whisper") is not None,
        "Diffusion": bool(os.getenv("STUDIO_DIFFUSION_MODEL")),
    }


def save_upload(upload: object, category: str) -> Path:
    name = Path(getattr(upload, "name", "upload.bin")).name
    safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "-", name).strip(".-") or "upload.bin"
    folder = UPLOAD_DIR / f"{category}-{uuid.uuid4().hex[:12]}"
    folder.mkdir(parents=True, exist_ok=False)
    path = folder / safe_name
    data = upload.getvalue()  # type: ignore[attr-defined]
    path.write_bytes(data)
    return path


def display_result(result: dict) -> None:
    run_dir = Path(result["run_dir"])
    st.divider()
    st.subheader("Your review package")
    st.markdown(f"<div class='result-path'>{run_dir}</div>", unsafe_allow_html=True)
    files = result["files"]
    if "video" in files:
        video_path = run_dir / files["video"]
        st.video(str(video_path))
    for warning in result.get("warnings", []):
        st.warning(warning)
    download_specs = [
        ("script", "Script · Markdown", "text/markdown"),
        ("audio", "Narration · WAV", "audio/wav"),
        ("video", "Vertical short · MP4", "video/mp4"),
        ("captions", "Captions · SRT", "application/x-subrip"),
        ("manifest", "Run manifest · JSON", "application/json"),
    ]
    columns = st.columns(3)
    for index, (key, label, mime) in enumerate(download_specs):
        if key not in files:
            continue
        path = run_dir / files[key]
        if path.is_file():
            with columns[index % 3]:
                st.download_button(
                    f"↓ {label}", path.read_bytes(), file_name=path.name, mime=mime, use_container_width=True
                )
    dub_keys = sorted(key for key in files if key.startswith("dub_") and key.endswith("_video"))
    if dub_keys:
        with st.expander(f"Multilingual dubs · {len(dub_keys)}"):
            for key in dub_keys:
                language = key.removeprefix("dub_").removesuffix("_video")
                path = run_dir / files[key]
                st.write(f"**{LANGUAGE_NAMES.get(language, language)}**")
                st.download_button(
                    f"↓ {language} MP4", path.read_bytes(), file_name=f"{run_dir.name}-{language}.mp4", mime="video/mp4"
                )
    st.success("Ready for human review. Nothing has been uploaded or posted.")


with st.sidebar:
    st.markdown("### Studio control")
    st.caption("Local execution · manual publishing")
    st.markdown("---")
    st.markdown("**Engine availability**")
    statuses = backend_status()
    for name, available in statuses.items():
        st.write(f"{'●' if available else '○'} {name}  {'ready' if available else 'optional / offline'}")
    st.caption("The procedural writer, visuals, and estimated captions work without model services. AI backends activate when installed.")
    st.markdown("---")
    st.markdown("**Safety rails**")
    st.write("✓ No social posting code")
    st.write("✓ Local SQLite data")
    st.write("✓ Consent-gated references")
    st.write("✓ Cloned repos never executed")
    st.caption(f"Workspace\n{ROOT}")

studio_tab, maya_tab, analytics_tab, discovery_tab = st.tabs(
    ["Create", "Maya · Sales", "Analytics · Brief", "FOSS Discovery"]
)

with studio_tab:
    st.markdown(
        "<div class='studio-note'><strong>One-click run.</strong> Add an idea or voice memo. "
        "Optional references and dubs are below; the main trigger always stops at local files for manual review.</div>",
        unsafe_allow_html=True,
    )
    topic = st.text_input("Topic / idea", placeholder="Example: How freelancers can price their first strategy call")
    hook = st.text_area(
        "Optional psychological hook",
        placeholder="Most freelancers are pricing the wrong thing…",
        max_chars=240,
        height=80,
    )
    voice_memo = st.file_uploader(
        "Or add a voice memo (maximum 2 minutes)",
        type=["wav", "mp3", "m4a", "ogg", "flac"],
        help="Requires faster-whisper. If a topic is also present, the transcript becomes source notes.",
    )
    first, second, third = st.columns([1, 1, 2])
    with first:
        duration = st.slider("Target seconds", min_value=15, max_value=90, value=45, step=5)
    with second:
        language = st.selectbox(
            "Narration language", list(LANGUAGE_NAMES), format_func=lambda code: LANGUAGE_NAMES[code]
        )
    with third:
        cta = st.text_input("Call to action (optional)", placeholder="Save this and follow for the next breakdown.")

    with st.expander("Voice, visuals & multilingual production"):
        llm_label = st.selectbox(
            "Script engine",
            ["Auto · Ollama with offline fallback", "Offline · deterministic writer", "Ollama only · no fallback"],
        )
        llm_mode = {"Auto · Ollama with offline fallback": "auto", "Offline · deterministic writer": "offline", "Ollama only · no fallback": "ollama"}[llm_label]
        dub_languages = st.multiselect(
            "Additional local dubs",
            [code for code in LANGUAGE_NAMES if code != language],
            format_func=lambda code: LANGUAGE_NAMES[code],
            help="Uses local Argos Translate or Ollama, then XTTS/a speech fallback. Failed dubs are reported, never faked.",
        )
        voice_reference = st.file_uploader(
            "Voice-clone reference (WAV/MP3/M4A)",
            type=["wav", "mp3", "m4a", "flac"],
            help="XTTS zero-shot voice conditioning. A reference is not used unless you confirm consent below.",
        )
        visual_anchors = st.file_uploader(
            "Image anchors / selfies (up to 3)",
            type=["png", "jpg", "jpeg", "webp"],
            accept_multiple_files=True,
        )
        consent = st.checkbox("I own these references or have explicit permission to process them.")
        remove_dead_air = st.checkbox("Tighten dead air with FFmpeg", value=True)

    st.write("")
    generate = st.button("🚀 Generate Content (1-Click Studio Process)", type="primary", use_container_width=True)
    log_placeholder = st.empty()
    progress_placeholder = st.empty()

    if generate:
        if not topic.strip() and not voice_memo:
            st.error("Add a topic or a voice memo first.")
        elif (voice_reference or visual_anchors) and not consent:
            st.error("Confirm that you own or have permission to use the uploaded references.")
        else:
            events: list[str] = []
            progress_bar = progress_placeholder.progress(0.0, text="Preparing the studio…")
            stage_progress = {
                "intake": 0.08,
                "discovery": 0.12,
                "script": 0.25,
                "voice": 0.43,
                "visuals": 0.57,
                "edit": 0.78,
                "dubbing": 0.88,
                "complete": 1.0,
                "failed": 1.0,
            }

            def on_progress(event: dict[str, str]) -> None:
                icon = "✓" if event["level"] == "info" else "!" if event["level"] == "warning" else "×"
                events.append(f"{icon} [{event['stage']}] {event['message']}")
                log_placeholder.code("\n".join(events), language="text")
                progress_bar.progress(
                    stage_progress.get(event["stage"], 0.1), text=event["message"][:110]
                )

            try:
                request = GenerationRequest(
                    topic=topic,
                    hook=hook,
                    target_duration=float(duration),
                    language=language,
                    cta=cta,
                    dub_languages=dub_languages,
                    voice_memo=save_upload(voice_memo, "memo") if voice_memo else None,
                    voice_reference=save_upload(voice_reference, "voice") if voice_reference else None,
                    visual_anchors=[save_upload(item, "anchor") for item in (visual_anchors or [])[:3]],
                    reference_consent=consent,
                    remove_dead_air=remove_dead_air,
                    llm_mode=llm_mode,
                    auto_discover=True,
                )
                result = asyncio.run(StudioOrchestrator().generate(request, progress=on_progress))
                st.session_state.last_result = result.to_dict()
                st.session_state.last_logs = events
                progress_bar.progress(1.0, text="Local production complete")
            except Exception as exc:
                st.session_state.last_logs = events
                st.error(f"The run stopped safely: {exc}")
                st.info("Partial logs and any completed assets remain in the run's manifest. Nothing was posted.")
    elif st.session_state.get("last_logs"):
        with st.expander("Latest run log"):
            st.code("\n".join(st.session_state.last_logs), language="text")

    if st.session_state.get("last_result"):
        display_result(st.session_state.last_result)

with maya_tab:
    st.subheader("Maya, your local sales concierge")
    st.caption("Knowledge-grounded replies, fit scoring, and checkout/booking intent in database/leads.db. No outbound messages.")
    agent = sales_agent()
    if "maya_session" not in st.session_state:
        st.session_state.maya_session = uuid.uuid4().hex
    if "maya_messages" not in st.session_state:
        st.session_state.maya_messages = []
    lead_col, chat_col = st.columns([1, 3])
    with lead_col:
        lead = agent.get_lead(st.session_state.maya_session)
        st.metric("Fit score", f"{lead['fit_score'] if lead else 0}/100")
        st.write(f"**Intent:** {lead['intent'] if lead else 'unknown'}")
        st.write(f"**State:** {lead['state'] if lead else 'new'}")
        st.write(f"**Budget:** {lead['budget_text'] if lead and lead['budget_text'] else 'not qualified'}")
    with chat_col:
        for message in st.session_state.maya_messages:
            with st.chat_message(message["role"]):
                st.write(message["content"])
        prompt = st.chat_input("Ask about a course, coaching, or a brand deal…")
        if prompt:
            st.session_state.maya_messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.write(prompt)
            with st.spinner("Maya is checking the local knowledge base…"):
                reply = agent.reply(st.session_state.maya_session, prompt)
            st.session_state.maya_messages.append({"role": "assistant", "content": reply.text})
            with st.chat_message("assistant"):
                st.write(reply.text)
                if reply.sources:
                    st.caption("Knowledge: " + ", ".join(reply.sources) + f" · {reply.engine}")
            st.rerun()
    with st.expander("Local lead database"):
        leads = agent.list_leads()
        st.dataframe(leads, use_container_width=True, hide_index=True) if leads else st.info("No qualified conversations yet.")

with analytics_tab:
    st.subheader("Manual analytics in. One clear morning brief out.")
    st.caption("Import platform CSV exports; this studio does not scrape, post, or claim live platform API data.")
    analytics = analytics_engine()
    import_col, brief_col = st.columns(2)
    with import_col:
        csv_upload = st.file_uploader("Platform metrics CSV", type=["csv"])
        if st.button("Import metrics", use_container_width=True):
            if not csv_upload:
                st.error("Choose a CSV export first.")
            else:
                try:
                    path = save_upload(csv_upload, "analytics")
                    count = analytics.import_csv(path)
                    st.success(f"Imported or updated {count} metric rows in studio.db.")
                except Exception as exc:
                    st.error(str(exc))
        sample = "date,platform,language,topic,views,follower_growth,conversions,revenue\n2026-01-01,youtube,en,pricing basics,1200,18,3,49.00\n"
        st.download_button("↓ CSV template", sample, file_name="studio-metrics-template.csv", mime="text/csv")
    with brief_col:
        if st.button("Generate morning brief", use_container_width=True):
            path, content = analytics.generate_morning_brief()
            st.session_state.morning_brief = {"path": str(path), "content": content}
        if st.session_state.get("morning_brief"):
            brief = st.session_state.morning_brief
            st.download_button(
                "↓ Morning brief",
                brief["content"],
                file_name=Path(brief["path"]).name,
                mime="text/markdown",
                use_container_width=True,
            )
    metrics = analytics.summary(24)
    metric_columns = st.columns(4)
    metric_columns[0].metric("24h views", f"{metrics['views']:,}")
    metric_columns[1].metric("24h revenue", f"${metrics['revenue_cents'] / 100:,.2f}")
    metric_columns[2].metric("Conversions", f"{metrics['conversions']:,}")
    metric_columns[3].metric("Follower velocity", f"{metrics['follower_growth_per_day']:+,}/day")
    if st.session_state.get("morning_brief"):
        st.markdown(st.session_state.morning_brief["content"])
    with st.expander("Recent imported metrics"):
        rows = analytics.recent_metrics()
        st.dataframe(rows, use_container_width=True, hide_index=True) if rows else st.info("No metrics imported yet.")

with discovery_tab:
    st.subheader("Zero-link FOSS reverse engineering")
    st.write(
        "Describe a missing capability. The learner searches public GitHub repositories, shallow-clones the best match, "
        "and extracts Python symbols and documentation into a bounded local index."
    )
    st.warning("Reference-only safety boundary: cloned repositories are never installed, imported, or executed automatically.")
    capability = st.text_input("Capability to learn", placeholder="Whisper word-level caption alignment")
    if st.button("🔎 Discover, clone & index", type="secondary"):
        if not capability.strip():
            st.error("Describe a capability first.")
        else:
            try:
                with st.spinner("Searching GitHub and building a non-executing local index…"):
                    report = GitHubLearner().learn(capability, limit=2)
                st.session_state.discovery_report = report
            except Exception as exc:
                st.error(str(exc))
    report = st.session_state.get("discovery_report")
    if report:
        st.success(f"Indexed {len(report['repositories'])} repositories for {report['capability']}.")
        for repository in report["repositories"]:
            metadata = repository.get("metadata", {})
            with st.expander(f"{repository['full_name']} · {metadata.get('stars', 0)} stars"):
                st.write(metadata.get("description") or "No repository description.")
                st.write(f"**License reported by GitHub:** {metadata.get('license', 'See repository')}")
                st.write(f"**Files indexed:** {repository['file_count']} · **Execution:** {repository['execution_policy']}")
                st.markdown(f"[Inspect public repository]({metadata.get('html_url', 'https://github.com')})")
                symbols = [
                    (file["path"], symbol)
                    for file in repository["files"]
                    for symbol in file.get("symbols", [])
                ][:30]
                if symbols:
                    st.code(
                        "\n".join(
                            f"{path}:{symbol['line']}  {symbol['kind']} {symbol['name']} — {symbol['summary']}"
                            for path, symbol in symbols
                        ),
                        language="text",
                    )
        for error in report.get("errors", []):
            st.warning(error)
