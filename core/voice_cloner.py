"""FOSS speech synthesis: XTTS voice cloning, local speech fallbacks, honest timing tracks."""
from __future__ import annotations

import os
import shutil
import wave
from pathlib import Path
from typing import Any

from .models import AudioResult, Script
from .utils import StudioError, convert_to_wav, run_command, wav_duration


class VoiceCloner:
    """Synthesize locally and explicitly report when voice identity was not cloned."""

    def __init__(self) -> None:
        self._tts: Any = None

    @staticmethod
    def xtts_available() -> bool:
        try:
            import TTS  # noqa: F401

            return True
        except ImportError:
            return False

    @staticmethod
    def speech_fallback() -> str | None:
        if shutil.which("espeak-ng"):
            return "espeak-ng"
        if shutil.which("espeak"):
            return "espeak"
        try:
            import pyttsx3  # noqa: F401

            return "pyttsx3"
        except ImportError:
            return None

    def synthesize(
        self,
        script: Script,
        output_path: Path,
        *,
        voice_reference: Path | None = None,
    ) -> AudioResult:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        warnings: list[str] = []
        if voice_reference and not voice_reference.is_file():
            raise StudioError(f"Voice reference was not found: {voice_reference}")

        if voice_reference and self.xtts_available():
            try:
                self._synthesize_xtts(script, output_path, voice_reference)
                return AudioResult(
                    path=output_path,
                    duration=wav_duration(output_path),
                    engine="coqui-xtts-v2",
                    cloned_voice=True,
                    warnings=[
                        "XTTS conditions output on the supplied reference; verify pronunciation and consent before publishing."
                    ],
                )
            except Exception as exc:
                warnings.append(
                    f"XTTS voice cloning failed ({exc}). A non-cloned local fallback was used; review the audio carefully."
                )
        elif voice_reference:
            warnings.append(
                "Coqui XTTS is not installed, so the supplied voice reference was NOT cloned. "
                "The fallback voice does not preserve the speaker's identity."
            )

        fallback = self.speech_fallback()
        if fallback in {"espeak", "espeak-ng"}:
            try:
                self._synthesize_espeak(script, output_path, fallback)
                return AudioResult(
                    path=output_path,
                    duration=wav_duration(output_path),
                    engine=fallback,
                    cloned_voice=False,
                    warnings=warnings
                    + ["A local non-cloned speech voice was used. Review pronunciation before manual upload."],
                )
            except Exception as exc:
                warnings.append(f"{fallback} synthesis failed: {exc}")
        elif fallback == "pyttsx3":
            try:
                self._synthesize_pyttsx3(script, output_path)
                return AudioResult(
                    path=output_path,
                    duration=wav_duration(output_path),
                    engine="pyttsx3",
                    cloned_voice=False,
                    warnings=warnings + ["A system speech voice was used; speaker identity is not cloned."],
                )
            except Exception as exc:
                warnings.append(f"pyttsx3 synthesis failed: {exc}")

        self._write_silent_timing_track(output_path, max(script.target_duration, script.duration))
        warnings.append(
            "No local speech engine was available. voice.wav is a SILENT timing track, not narration. "
            "Install a FOSS speech backend or replace it with recorded narration before uploading."
        )
        return AudioResult(
            path=output_path,
            duration=wav_duration(output_path),
            engine="silent-timing-track",
            cloned_voice=False,
            is_silent=True,
            warnings=warnings,
        )

    def _get_xtts(self) -> Any:
        if self._tts is None:
            from TTS.api import TTS

            model_name = os.getenv("XTTS_MODEL", "tts_models/multilingual/multi-dataset/xtts_v2")
            try:
                import torch

                gpu = bool(torch.cuda.is_available())
            except ImportError:
                gpu = False
            self._tts = TTS(model_name=model_name, progress_bar=False, gpu=gpu)
        return self._tts

    def _synthesize_xtts(self, script: Script, output_path: Path, voice_reference: Path) -> None:
        reference = voice_reference
        if voice_reference.suffix.lower() != ".wav":
            reference = convert_to_wav(voice_reference, output_path.parent / "voice_reference_16bit.wav")
        tts = self._get_xtts()
        tts.tts_to_file(
            text=script.spoken_text,
            speaker_wav=str(reference),
            language=script.language,
            file_path=str(output_path),
            split_sentences=True,
        )
        if not output_path.is_file() or output_path.stat().st_size < 100:
            raise StudioError("XTTS did not create a usable audio file.")

    @staticmethod
    def _synthesize_espeak(script: Script, output_path: Path, binary: str) -> None:
        run_command(
            [
                shutil.which(binary) or binary,
                "-v",
                script.language,
                "-s",
                os.getenv("ESPEAK_WORDS_PER_MINUTE", "165"),
                "-w",
                str(output_path),
                script.spoken_text,
            ],
            timeout=180,
        )

    @staticmethod
    def _synthesize_pyttsx3(script: Script, output_path: Path) -> None:
        import pyttsx3

        engine = pyttsx3.init()
        try:
            engine.setProperty("rate", 165)
            engine.save_to_file(script.spoken_text, str(output_path))
            engine.runAndWait()
        finally:
            engine.stop()
        if not output_path.is_file():
            raise StudioError("The system speech engine did not create an audio file.")

    @staticmethod
    def _write_silent_timing_track(path: Path, duration: float, sample_rate: int = 22050) -> None:
        frames = max(1, int(duration * sample_rate))
        block = b"\x00\x00" * sample_rate
        with wave.open(str(path), "wb") as audio:
            audio.setnchannels(1)
            audio.setsampwidth(2)
            audio.setframerate(sample_rate)
            remaining = frames
            while remaining:
                count = min(sample_rate, remaining)
                audio.writeframesraw(block[: count * 2])
                remaining -= count
            audio.writeframes(b"")
