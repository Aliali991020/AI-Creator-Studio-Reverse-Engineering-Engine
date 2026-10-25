"""Central asynchronous execution of the local, manual-upload studio pipeline."""
from __future__ import annotations

import asyncio
from typing import Any, Callable, TypeVar

from .dubbing import DubbingEngine
from .github_learner import GitHubLearner
from .models import GenerationRequest, PipelineResult
from .script_generator import ScriptGenerator
from .transcription import LocalTranscriber
from .utils import (
    READY_DIR,
    StudioError,
    ensure_directories,
    ffmpeg_binary,
    new_run_id,
    utc_now,
    write_json,
)
from .video_editor import VideoEditor
from .visual_renderer import VisualRenderer
from .voice_cloner import VoiceCloner

T = TypeVar("T")
ProgressCallback = Callable[[dict[str, str]], None]


class StudioOrchestrator:
    def __init__(
        self,
        *,
        script_generator: ScriptGenerator | None = None,
        voice_cloner: VoiceCloner | None = None,
        video_editor: VideoEditor | None = None,
        visual_renderer: VisualRenderer | None = None,
        dubbing_engine: DubbingEngine | None = None,
        github_learner: GitHubLearner | None = None,
    ) -> None:
        ensure_directories()
        self.script_generator = script_generator or ScriptGenerator()
        self.voice_cloner = voice_cloner or VoiceCloner()
        self.transcriber = LocalTranscriber()
        self.video_editor = video_editor or VideoEditor(self.transcriber)
        self.visual_renderer = visual_renderer or VisualRenderer()
        self.dubbing_engine = dubbing_engine or DubbingEngine()
        self.github_learner = github_learner or GitHubLearner()
        self._events: list[dict[str, str]] = []
        self._callback: ProgressCallback | None = None

    def _emit(self, stage: str, message: str, level: str = "info") -> None:
        event = {"time": utc_now(), "stage": stage, "level": level, "message": message}
        self._events.append(event)
        if self._callback:
            try:
                self._callback(event)
            except Exception:
                # A display callback must never be able to corrupt a media run.
                pass

    async def _run(self, stage: str, message: str, function: Callable[[], T]) -> T:
        self._emit(stage, message)
        result = await asyncio.to_thread(function)
        self._emit(stage, f"{message} — done")
        return result

    async def _discover(self, capability: str, query: str) -> None:
        self._emit("discovery", f"Missing {capability}; autonomously discovering public FOSS references…", "warning")
        try:
            report = await asyncio.to_thread(self.github_learner.learn, capability, query=query, limit=1)
            names = ", ".join(item["full_name"] for item in report["repositories"])
            self._emit(
                "discovery",
                f"Indexed {names}. Reference notes are in .studio/github_index; no cloned code was executed.",
            )
        except Exception as exc:
            self._emit("discovery", f"Discovery could not complete: {exc}", "warning")

    async def generate(
        self, request: GenerationRequest, progress: ProgressCallback | None = None
    ) -> PipelineResult:
        self._events = []
        self._callback = progress
        ensure_directories()
        run_id = new_run_id(request.topic or "voice-memo")
        run_dir = READY_DIR / run_id
        run_dir.mkdir(parents=True, exist_ok=False)
        manifest_path = run_dir / "manifest.json"
        warnings: list[str] = []
        files: dict[str, str] = {}
        self._emit("intake", f"Created local run folder: {run_dir.name}")
        try:
            if (request.voice_reference or request.visual_anchors) and not request.reference_consent:
                raise StudioError(
                    "Voice and image references require explicit ownership/permission confirmation before processing."
                )
            topic = request.topic.strip()
            source_notes = ""
            if request.voice_memo:
                if not self.transcriber.available() and request.auto_discover:
                    await self._discover(
                        "faster-whisper voice memo transcription",
                        "faster-whisper SYSTRAN language:Python",
                    )
                transcription = await self._run(
                    "intake",
                    "Transcribing the two-minute voice memo locally",
                    lambda: self.transcriber.transcribe_memo(request.voice_memo),
                )
                source_notes = transcription.text
                if not topic:
                    # Keep the run title concise while passing the complete memo as source notes.
                    topic = transcription.text.strip().split("\n")[0][:140]
                (run_dir / "voice_memo_transcript.md").write_text(
                    f"# Voice memo transcript\n\n{transcription.text}\n", encoding="utf-8"
                )
                files["voice_memo_transcript"] = "voice_memo_transcript.md"
            if not topic:
                raise StudioError("Enter a topic or upload a transcribable voice memo before starting.")
            if request.voice_reference and not self.voice_cloner.xtts_available() and request.auto_discover:
                await self._discover("Coqui XTTS voice cloning", "coqui TTS XTTS voice cloning language:Python")
            if (
                request.llm_mode == "ollama"
                and request.auto_discover
                and not self.script_generator.ollama.available()
            ):
                await self._discover("Ollama local LLM runtime", "Ollama local LLM runtime")

            script = await self._run(
                "script",
                "Structuring the hook, paced body, CTA, and visual directions",
                lambda: self.script_generator.generate(
                    topic,
                    hook=request.hook,
                    target_duration=request.target_duration,
                    language=request.language,
                    cta=request.cta,
                    source_notes=source_notes,
                    mode=request.llm_mode,
                ),
            )
            (run_dir / "script.md").write_text(script.to_markdown(), encoding="utf-8")
            write_json(run_dir / "script.json", script.to_dict())
            files.update({"script": "script.md", "script_data": "script.json"})
            warnings.extend(script.warnings)

            audio = await self._run(
                "voice",
                "Synthesizing narration with the available local FOSS backend",
                lambda: self.voice_cloner.synthesize(
                    script, run_dir / "voice.wav", voice_reference=request.voice_reference
                ),
            )
            files["audio"] = "voice.wav"
            warnings.extend(audio.warnings)

            visuals, visuals_source, visual_warnings = await self._run(
                "visuals",
                "Generating vertical B-roll scenes or image-anchor motion visuals",
                lambda: self.visual_renderer.create_visuals(
                    script, run_dir / "scenes", anchors=request.visual_anchors
                ),
            )
            warnings.extend(visual_warnings)
            if not ffmpeg_binary() and request.auto_discover:
                await self._discover("FFmpeg video editing", "FFmpeg video editing captions")
            video = await self._run(
                "edit",
                "Aligning captions, tightening pauses, and rendering the 9:16 MP4",
                lambda: self.video_editor.render(
                    audio,
                    script,
                    run_dir,
                    visuals,
                    visuals_source=visuals_source,
                    remove_dead_air=request.remove_dead_air,
                ),
            )
            files.update(
                {
                    "video": "short.mp4",
                    "captions": "captions.srt",
                    "caption_project": "captions.ass",
                }
            )
            if video.audio_path.name != "voice.wav":
                files["edited_audio"] = video.audio_path.name
            warnings.extend(video.warnings)

            dub_records: list[dict[str, Any]] = []
            for language in dict.fromkeys(request.dub_languages):
                if language == request.language:
                    continue
                dub_dir = run_dir / "dubs" / language
                dub_dir.mkdir(parents=True, exist_ok=True)
                self._emit("dubbing", f"Creating the {language} neural dub…")
                try:
                    translated = await asyncio.to_thread(self.dubbing_engine.translate, script, language)
                    (dub_dir / "script.md").write_text(translated.to_markdown(), encoding="utf-8")
                    write_json(dub_dir / "script.json", translated.to_dict())
                    dub_audio = await asyncio.to_thread(
                        self.voice_cloner.synthesize,
                        translated,
                        dub_dir / "voice.wav",
                        voice_reference=request.voice_reference,
                    )
                    dub_video = await asyncio.to_thread(
                        self.video_editor.render,
                        dub_audio,
                        translated,
                        dub_dir,
                        visuals,
                        visuals_source=visuals_source,
                        remove_dead_air=request.remove_dead_air,
                    )
                    prefix = f"dubs/{language}/"
                    files[f"dub_{language}_script"] = prefix + "script.md"
                    files[f"dub_{language}_audio"] = prefix + "voice.wav"
                    files[f"dub_{language}_video"] = prefix + "short.mp4"
                    dub_records.append(
                        {
                            "language": language,
                            "status": "complete",
                            "translation_engine": translated.engine,
                            "audio": dub_audio.to_dict(),
                            "video": dub_video.to_dict(),
                        }
                    )
                    warnings.extend(f"{language}: {item}" for item in dub_audio.warnings + dub_video.warnings)
                    self._emit("dubbing", f"{language} dub rendered — review translation and pronunciation manually.")
                except Exception as exc:
                    message = f"{language} dub was not produced: {exc}"
                    warnings.append(message)
                    dub_records.append({"language": language, "status": "failed", "error": str(exc)})
                    self._emit("dubbing", message, "warning")

            # De-duplicate while preserving the order in which reviewers need to see warnings.
            warnings = list(dict.fromkeys(warnings))
            status = "ready_for_manual_review"
            self._emit("complete", "Assets compiled locally. No external upload or posting was performed.")
            manifest = {
                "run_id": run_id,
                "created_at": utc_now(),
                "status": status,
                "publishing": "manual_only",
                "topic": topic,
                "narration_ready": not audio.is_silent,
                "voice_identity_cloned": audio.cloned_voice,
                "reference_consent_confirmed": request.reference_consent,
                "files": files,
                "script": script.to_dict(),
                "audio": audio.to_dict(),
                "video": video.to_dict(),
                "visuals_source": visuals_source,
                "dubs": dub_records,
                "warnings": warnings,
                "review_checklist": [
                    "Confirm narration is present and pronunciation is acceptable.",
                    "Verify every factual claim, translation, caption, and CTA.",
                    "Confirm consent for voice/image references and disclose synthetic media where required.",
                    "Upload manually to each chosen platform; this studio never posts automatically.",
                ],
                "events": self._events,
            }
            write_json(manifest_path, manifest)
            files["manifest"] = "manifest.json"
            return PipelineResult(run_id, run_dir, status, files, warnings, self._events.copy(), manifest_path)
        except Exception as exc:
            self._emit("failed", str(exc), "error")
            write_json(
                manifest_path,
                {
                    "run_id": run_id,
                    "created_at": utc_now(),
                    "status": "failed",
                    "publishing": "manual_only",
                    "error": str(exc),
                    "files": files,
                    "warnings": warnings,
                    "events": self._events,
                },
            )
            raise
        finally:
            self._callback = None
