"""Small, safe operating-system helpers shared by studio subsystems."""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import wave
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
READY_DIR = ROOT / "ready_to_upload"
WORK_DIR = ROOT / ".studio" / "work"
UPLOAD_DIR = ROOT / ".studio" / "uploads"


class StudioError(RuntimeError):
    """A user-actionable studio error."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def ensure_directories() -> None:
    for path in (READY_DIR, WORK_DIR, UPLOAD_DIR, ROOT / "database", ROOT / "knowledge"):
        path.mkdir(parents=True, exist_ok=True)


def slugify(value: str, max_length: int = 42) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return (slug[:max_length].rstrip("-") or "studio-run")


def new_run_id(topic: str) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S-%f")
    return f"{stamp}-{slugify(topic)}"


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    temporary.replace(path)


def ffmpeg_binary() -> str | None:
    configured = os.getenv("FFMPEG_BINARY")
    if configured and Path(configured).is_file():
        return configured
    binary = shutil.which("ffmpeg")
    if binary:
        return binary
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except (ImportError, RuntimeError, OSError):
        return None


def ffprobe_binary() -> str | None:
    configured = os.getenv("FFPROBE_BINARY")
    if configured and Path(configured).is_file():
        return configured
    binary = shutil.which("ffprobe")
    if binary:
        return binary
    ffmpeg = ffmpeg_binary()
    if ffmpeg:
        sibling = Path(ffmpeg).with_name("ffprobe")
        if sibling.is_file():
            return str(sibling)
    return None


def run_command(command: Iterable[str], *, cwd: Path | None = None, timeout: int = 900) -> str:
    """Run a fixed argv command; never invoke a shell or downloaded code implicitly."""

    argv = [str(part) for part in command]
    try:
        process = subprocess.run(
            argv,
            cwd=str(cwd) if cwd else None,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except FileNotFoundError as exc:
        raise StudioError(f"Required executable was not found: {argv[0]}") from exc
    except subprocess.TimeoutExpired as exc:
        raise StudioError(f"Command timed out after {timeout} seconds: {argv[0]}") from exc
    if process.returncode:
        detail = (process.stderr or process.stdout or "Unknown process error").strip()
        # Keep errors useful without flooding a UI with an entire FFmpeg build banner.
        detail = "\n".join(detail.splitlines()[-30:])
        raise StudioError(f"{Path(argv[0]).name} failed ({process.returncode}):\n{detail}")
    return process.stdout


def wav_duration(path: Path) -> float:
    try:
        with wave.open(str(path), "rb") as audio:
            rate = audio.getframerate()
            return audio.getnframes() / rate if rate else 0.0
    except (wave.Error, OSError, EOFError):
        pass
    probe = ffprobe_binary()
    if not probe:
        raise StudioError("Cannot read audio duration. Install FFmpeg or provide a PCM WAV file.")
    output = run_command(
        [probe, "-v", "error", "-show_entries", "format=duration", "-of", "json", str(path)],
        timeout=60,
    )
    try:
        return float(json.loads(output)["format"]["duration"])
    except (KeyError, ValueError, TypeError, json.JSONDecodeError) as exc:
        raise StudioError(f"Could not determine the duration of {path.name}.") from exc


def convert_to_wav(source: Path, destination: Path, *, sample_rate: int = 22050) -> Path:
    ffmpeg = ffmpeg_binary()
    if not ffmpeg:
        raise StudioError("FFmpeg is required to convert this audio format to WAV.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    run_command(
        [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(source),
            "-ac",
            "1",
            "-ar",
            str(sample_rate),
            "-c:a",
            "pcm_s16le",
            str(destination),
        ],
        timeout=180,
    )
    return destination


def relative_or_absolute(path: Path, base: Path) -> str:
    try:
        return str(path.resolve().relative_to(base.resolve()))
    except ValueError:
        return str(path.resolve())
