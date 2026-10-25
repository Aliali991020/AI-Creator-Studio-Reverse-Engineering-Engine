"""Shared, dependency-free domain models for the creator studio."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


LANGUAGE_NAMES: dict[str, str] = {
    "en": "English",
    "es": "Spanish",
    "pt": "Portuguese",
    "de": "German",
    "hi": "Hindi",
    "ja": "Japanese",
    "fr": "French",
    "ar": "Arabic",
    "it": "Italian",
    "ko": "Korean",
    "nl": "Dutch",
    "pl": "Polish",
    "tr": "Turkish",
    "zh-cn": "Chinese (Simplified)",
}


@dataclass(slots=True)
class ScriptSection:
    """One spoken section. Visual prompts are metadata, never narration."""

    label: str
    text: str
    visual_prompt: str = ""
    start: float = 0.0
    end: float = 0.0

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


@dataclass(slots=True)
class Script:
    topic: str
    language: str
    target_duration: float
    sections: list[ScriptSection]
    engine: str
    cta: str = ""
    source_notes: str = ""
    warnings: list[str] = field(default_factory=list)

    @property
    def spoken_text(self) -> str:
        return "\n".join(section.text.strip() for section in self.sections if section.text.strip())

    @property
    def duration(self) -> float:
        return self.sections[-1].end if self.sections else self.target_duration

    def to_markdown(self) -> str:
        lines = [
            f"# {self.topic}",
            "",
            f"- **Language:** {LANGUAGE_NAMES.get(self.language, self.language)}",
            f"- **Target runtime:** {self.target_duration:g} seconds",
            f"- **Script engine:** {self.engine}",
            "- **Publishing:** Manual review and upload only",
            "",
        ]
        for section in self.sections:
            lines.extend(
                [
                    f"## {section.label} · {section.start:05.2f}–{section.end:05.2f}s",
                    "",
                    section.text,
                    "",
                ]
            )
            if section.visual_prompt:
                lines.extend([f"**Visual direction:** {section.visual_prompt}", ""])
        if self.warnings:
            lines.extend(["## Production warnings", ""])
            lines.extend(f"- {warning}" for warning in self.warnings)
            lines.append("")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class AudioResult:
    path: Path
    duration: float
    engine: str
    cloned_voice: bool = False
    is_silent: bool = False
    sample_rate: int = 22050
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["path"] = str(self.path)
        return data


@dataclass(slots=True)
class Caption:
    start: float
    end: float
    text: str


@dataclass(slots=True)
class VideoResult:
    path: Path
    audio_path: Path
    duration: float
    captions_path: Path
    srt_path: Path
    alignment_source: str
    visuals_source: str
    dead_air_removed: bool = False
    warnings: list[str] = field(default_factory=list)
    width: int = 1080
    height: int = 1920

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        for key in ("path", "audio_path", "captions_path", "srt_path"):
            data[key] = str(data[key])
        return data


@dataclass(slots=True)
class GenerationRequest:
    topic: str = ""
    hook: str = ""
    target_duration: float = 45.0
    language: str = "en"
    cta: str = ""
    dub_languages: list[str] = field(default_factory=list)
    voice_memo: Path | None = None
    voice_reference: Path | None = None
    visual_anchors: list[Path] = field(default_factory=list)
    reference_consent: bool = False
    remove_dead_air: bool = True
    llm_mode: str = "auto"  # auto, offline, ollama
    auto_discover: bool = True


@dataclass(slots=True)
class PipelineResult:
    run_id: str
    run_dir: Path
    status: str
    files: dict[str, str]
    warnings: list[str]
    events: list[dict[str, str]]
    manifest_path: Path

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["run_dir"] = str(self.run_dir)
        data["manifest_path"] = str(self.manifest_path)
        return data
