# AI Creator Studio & FOSS Reverse-Engineering Engine

A local-first creator business workspace that turns a topic or two-minute voice memo into a structured script, narration track, captioned 9:16 MP4, optional multilingual dubs, and a review manifest. It also includes a knowledge-grounded sales concierge, SQLite lead tracking, manual CSV analytics, morning briefs, and non-executing GitHub repository discovery.

**Everything stops at `ready_to_upload/` for human review. There is no automated external posting, scheduling, uploading, payment, booking, or DM code.**

## Quick start

Python 3.10–3.11 is recommended for the broadest AI-backend compatibility.

```bash
python -m venv .venv
source .venv/bin/activate             # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

Open the local Streamlit URL, enter a topic, and press **🚀 Generate Content (1-Click Studio Process)**.

The base install includes Streamlit and a bundled FFmpeg executable. It can generate scripts, procedural visuals, estimated captions, and rendered MP4s. **If eSpeak/system speech is unavailable, `voice.wav` is deliberately silent and the UI/manifest warns that narration must be replaced.** It is never presented as successful speech synthesis.

For non-silent minimal narration, install `espeak-ng` with your operating system's package manager. For higher-quality and cloned voices, install the optional backends below.

## Optional local AI engines

```bash
pip install -r requirements-ai.txt
```

This optional file installs large packages. Install only the components you need, and use a PyTorch build appropriate for your hardware.

| Capability | Backend | Configuration / notes |
|---|---|---|
| LLM scripts and translation | Ollama | Start Ollama and pull `llama3.1:8b`, or set `OLLAMA_MODEL`. `OLLAMA_HOST` defaults to `http://127.0.0.1:11434`. |
| Voice-memo intake | faster-whisper | `WHISPER_MODEL=base` by default. Set it to a local model path if desired. Memo length is validated at 120 seconds. |
| Measured word-level captions | faster-whisper | Falls back to clearly labeled script-estimated timings if unavailable. |
| Multilingual voice cloning | Coqui XTTS v2 | Supply a consented voice reference. `XTTS_MODEL` can override the model. Initial model download may be required. Exact speaker identity is not guaranteed. |
| Offline translation | Argos Translate | Install the required Argos source→target language package. Ollama is the other local route. |
| Generated B-roll | diffusers + PyTorch | Set `STUDIO_DIFFUSION_MODEL` to a downloaded/local model path. Loading uses `local_files_only=True`; no surprise model download. |
| Video encoding | FFmpeg | Base `imageio-ffmpeg` package or set `FFMPEG_BINARY` / `FFPROBE_BINARY`. |

Useful environment variables:

```bash
export OLLAMA_MODEL=llama3.1:8b
export WHISPER_MODEL=base
export WHISPER_LOCAL_FILES_ONLY=false   # true prevents initial model downloads
export STUDIO_DIFFUSION_MODEL=/path/to/local/diffusion/model
export DIFFUSION_STEPS=20
```

The deterministic offline script writer is English-only. Other source languages require a real local LLM/translation route; the studio will not mislabel English output as another language.

## Output package

Each click creates an isolated directory:

```text
ready_to_upload/<timestamp-topic>/
├── script.md                 # Spoken hook/body/CTA + visual direction
├── script.json               # Structured sections and timings
├── voice.wav                 # Narration or explicitly flagged silent timing track
├── narration_clean.wav       # Optional dead-air-tightened version
├── short.mp4                 # 1080×1920 H.264/AAC with burned ASS captions
├── captions.ass              # Animated caption project
├── captions.srt              # Portable captions
├── word_timestamps.json      # Measured words or explicit estimated source
├── scenes/                   # Local diffusion, image anchors, or procedural art
├── dubs/<language>/          # Independent script/audio/video per successful language
└── manifest.json             # Engines, warnings, files, logs, review checklist
```

Generated media, uploaded references, local databases, model caches, and cloned repositories are ignored by Git.

## Maya and local lead tracking

Edit the example documents in `knowledge/` before handling real inquiries. Maya uses bounded local Markdown retrieval and optionally Ollama, then records:

- course, coaching, or brand-deal intent;
- budget and contact signals;
- a deterministic 0–100 fit score;
- new/nurturing/qualified/checkout-intent/booking-requested states;
- the local conversation history.

Records are stored in `database/leads.db`. Maya explicitly states when no payment or booking has occurred.

## Analytics and morning brief

Import human-exported CSV metrics in the Analytics tab. Required columns are `platform`, `views`, and either `date` or `recorded_at`.

```csv
date,platform,language,topic,views,follower_growth,conversions,revenue
2026-01-01,youtube,en,pricing basics,1200,18,3,49.00
2026-01-01,instagram,es,precios para freelancers,900,11,2,29.00
```

Rows are upserted into `database/studio.db` by timestamp/platform/language/topic. The one-page brief reports 24-hour revenue, views, conversions, follower growth velocity, language rankings, and hook hypotheses. Recommendations are labeled as experiments, not guaranteed outcomes.

## Zero-link GitHub discovery

Enter a missing capability in **FOSS Discovery**, or hit a specifically required missing backend during a run. The learner searches public GitHub without asking for URLs, shallow-clones a selected repository, extracts AST symbols/headings, and writes a local report to `.studio/github_index/`.

It **never installs, imports, or executes cloned code**. Indexing is bounded by file size, total bytes, and file count. Set an existing `GITHUB_TOKEN` environment variable only if your own GitHub setup requires a higher API rate limit; the app never prompts for credentials.

## Tests

The test suite uses only the standard library:

```bash
python -m unittest discover -s tests -v
```

An FFmpeg integration test runs when FFmpeg is available and otherwise skips explicitly.

## Project map

```text
core/
├── orchestrator.py       # Async end-to-end execution and manifests
├── script_generator.py   # Ollama + deterministic script structuring
├── voice_cloner.py       # XTTS/eSpeak/system speech/silent fallback
├── transcription.py      # Voice memos and faster-whisper word timings
├── visual_renderer.py    # Local diffusion, anchors, procedural scenes
├── video_editor.py       # Dead air, ASS/SRT, FFmpeg vertical rendering
├── dubbing.py            # Argos/Ollama multilingual translation
├── sales_agent.py        # Maya, knowledge retrieval, lead SQLite
├── analytics.py          # Metrics SQLite and morning briefs
└── github_learner.py     # Search, clone, bounded non-executing index
```

See `SKILL.md` for operating controls and `docs/FOSS_WORKFLOWS.md` for clean-room workflow substitutions.
