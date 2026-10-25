# AI Creator Studio — Master Operating Skill

## Mission

Operate a local-first, FOSS media production studio and digital business workspace. Convert one topic or voice memo into human-reviewable scripts, narration, vertical video, captions, multilingual tracks, lead records, analytics, and a morning brief.

## Non-negotiable controls

1. **Manual upload only.** Final media is compiled beneath `ready_to_upload/`. Never add social-network posting, scheduling, auto-upload, or auto-DM behavior.
2. **Local data.** Leads and metrics live in local SQLite files. Uploaded references, model caches, and discovery indexes stay beneath `.studio/` and are not source artifacts.
3. **Honest backend reporting.** Never label silence as speech, an unmodified voice as a clone, estimated timings as Whisper measurements, or untranslated text as a dub. Surface warnings in the UI and manifest.
4. **Consent and rights.** Process voice/image references only after ownership or permission is confirmed. A human verifies licensing, claims, disclosures, translations, and final creative.
5. **Reference-only discovery.** Public GitHub repositories may be searched, shallow-cloned, and indexed without user-supplied URLs. Never install, import, or execute cloned code automatically.
6. **Clean-room substitution.** Recreate high-level SaaS workflows with FOSS and native Python; do not copy proprietary code, bypass authentication, or evade paid-service controls.

## Production graph

1. **Intake:** validate a text topic or transcribe a voice memo of at most two minutes with local faster-whisper.
2. **Script:** use Ollama when available, with an explicit deterministic English fallback. Produce a three-second hook, paced practical body, CTA, and per-scene visual prompts.
3. **Voice:** prefer Coqui XTTS v2 when a consented reference is supplied. Otherwise use eSpeak or system speech. If no speech engine exists, write a clearly marked silent timing track.
4. **Visuals:** optionally render with a locally cached diffusion pipeline; otherwise use consented image anchors or procedural vertical backdrops.
5. **Edit:** optionally tighten dead air with FFmpeg, obtain faster-whisper word timings when available, write ASS/SRT captions, and render a 1080×1920 MP4 with burned captions.
6. **Dub:** translate locally with Argos Translate or Ollama, re-synthesize in the target language, and render independent review assets. Continue other languages if one fails.
7. **Review:** save `script.md`, `voice.wav`, `short.mp4`, captions, scene assets, and `manifest.json` locally. Stop before any external publishing.

## Business systems

- **Maya:** answer from `knowledge/` only, qualify course/coaching/brand-deal intent, extract budget and timeline signals, score fit, log checkout/booking states and messages in `database/leads.db`, and never claim an external action completed.
- **Analytics:** import human-supplied CSV exports into `database/studio.db`; aggregate views, follower velocity, conversions, and revenue without implying a live platform integration.
- **Morning brief:** summarize the last 24 hours, rank language tracks, and present clearly labeled hook experiments.

## Main operator command

```bash
streamlit run app.py
```

The main production control is **🚀 Generate Content (1-Click Studio Process)**. The run is complete only when local review assets and a manifest exist; it is never complete because something was posted.
