"""Local faster-whisper transcription and word-level alignment."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .utils import StudioError, wav_duration


@dataclass(slots=True)
class TranscriptionResult:
    text: str
    language: str
    duration: float
    words: list[dict[str, Any]] = field(default_factory=list)


class LocalTranscriber:
    def __init__(self, model_name: str | None = None) -> None:
        self.model_name = model_name or os.getenv("WHISPER_MODEL", "base")
        self._model: Any = None

    @staticmethod
    def available() -> bool:
        try:
            import faster_whisper  # noqa: F401

            return True
        except ImportError:
            return False

    def _get_model(self) -> Any:
        if self._model is None:
            try:
                from faster_whisper import WhisperModel
            except ImportError as exc:
                raise StudioError(
                    "Voice memos and measured word timings require faster-whisper. "
                    "Install requirements-ai.txt or use the script's estimated timings."
                ) from exc
            local_only = os.getenv("WHISPER_LOCAL_FILES_ONLY", "false").lower() in {"1", "true", "yes"}
            try:
                self._model = WhisperModel(
                    self.model_name,
                    device=os.getenv("WHISPER_DEVICE", "auto"),
                    compute_type=os.getenv("WHISPER_COMPUTE_TYPE", "int8"),
                    local_files_only=local_only,
                )
            except Exception as exc:
                raise StudioError(
                    f"Could not load local Whisper model {self.model_name!r}: {exc}. "
                    "Set WHISPER_MODEL to a downloaded model or allow the initial model download."
                ) from exc
        return self._model

    def transcribe(self, audio_path: Path, *, word_timestamps: bool = False) -> TranscriptionResult:
        if not audio_path.is_file():
            raise StudioError(f"Audio file does not exist: {audio_path}")
        model = self._get_model()
        try:
            segments, info = model.transcribe(
                str(audio_path),
                word_timestamps=word_timestamps,
                vad_filter=True,
            )
            words: list[dict[str, Any]] = []
            text_parts: list[str] = []
            for segment in segments:
                text_parts.append(segment.text)
                if word_timestamps:
                    for word in segment.words or []:
                        if word.start is None or word.end is None:
                            continue
                        words.append(
                            {
                                "start": float(word.start),
                                "end": float(word.end),
                                "word": str(word.word).strip(),
                                "probability": float(getattr(word, "probability", 0.0) or 0.0),
                            }
                        )
        except Exception as exc:
            raise StudioError(f"Local Whisper transcription failed: {exc}") from exc
        return TranscriptionResult(
            text=" ".join("".join(text_parts).split()),
            language=str(getattr(info, "language", "unknown")),
            duration=float(getattr(info, "duration", 0.0) or wav_duration(audio_path)),
            words=[word for word in words if word["word"]],
        )

    def transcribe_memo(self, audio_path: Path, max_seconds: float = 120.0) -> TranscriptionResult:
        duration = wav_duration(audio_path)
        if duration > max_seconds + 0.25:
            raise StudioError(
                f"Voice memo is {duration:.1f}s long; the intake limit is {max_seconds:.0f}s. Trim it and retry."
            )
        result = self.transcribe(audio_path)
        if not result.text:
            raise StudioError("No speech was detected in the voice memo. Add a text topic instead.")
        return result
