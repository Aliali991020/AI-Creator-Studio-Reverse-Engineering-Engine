# Clean-room FOSS workflow substitutions

This document maps observable, high-level creator workflows to local open-source components. It does not contain or derive from proprietary SaaS source code.

| Commercial workflow category | Local studio implementation | Important boundary |
|---|---|---|
| ElevenLabs-style scripting/voice workflow | Ollama + Coqui XTTS v2; eSpeak/system-speech fallback | XTTS reference conditioning is not a guarantee of exact identity. Consent and listening review are mandatory. |
| Descript-style transcription/editing/captions | faster-whisper word timestamps + native Python caption grouping + FFmpeg silence filters and ASS rendering | Without faster-whisper, timings are explicitly marked as estimates. |
| Higgsfield-style visual ideation | Local diffusers text-to-image/image-to-image + FFmpeg scene motion | A local cached model must be configured. The fallback is motion graphics or an image-anchor slideshow, not a fake talking head. |
| Metricool-style reporting | Manual CSV imports + SQLite aggregation + Markdown morning brief | There is no hidden scraper or live platform integration and no scheduler/posting. |
| Sales/chat CRM | Maya + local Markdown retrieval + SQLite lead/message state | Maya logs intent only; it never claims payment, booking, email, or DM completion. |
| Multilingual dubbing | Argos Translate or Ollama + XTTS + per-language FFmpeg renders | A missing translation route fails that language explicitly; the original audio is never passed off as a dub. |

## Autonomous repository learning

`core/github_learner.py` searches GitHub's public repository API, constructs only canonical `https://github.com/owner/repository.git` clone URLs, performs bounded shallow clones, and indexes source symbols with Python AST parsing.

Safety limits include repository-name validation, symlink rejection, ignored dependency/build directories, per-file and total-byte caps, file-count caps, and a strict no-install/no-import/no-execute policy. Generated indexes live under `.studio/github_index/` and are not used as trusted sales knowledge.
