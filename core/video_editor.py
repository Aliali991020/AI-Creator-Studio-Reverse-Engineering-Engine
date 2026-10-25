"""Whisper-aligned captions, dead-air tightening, and local FFmpeg vertical-video rendering."""
from __future__ import annotations

import re
from pathlib import Path

from .models import AudioResult, Caption, Script, VideoResult
from .transcription import LocalTranscriber
from .utils import StudioError, ffmpeg_binary, run_command, wav_duration, write_json


class VideoEditor:
    def __init__(self, transcriber: LocalTranscriber | None = None) -> None:
        self.transcriber = transcriber or LocalTranscriber()

    def render(
        self,
        audio: AudioResult,
        script: Script,
        output_dir: Path,
        visuals: list[Path],
        *,
        visuals_source: str,
        remove_dead_air: bool = True,
    ) -> VideoResult:
        ffmpeg = ffmpeg_binary()
        if not ffmpeg:
            raise StudioError(
                "FFmpeg is required for MP4 rendering. Install the base requirements (imageio-ffmpeg) "
                "or set FFMPEG_BINARY to a local FFmpeg executable."
            )
        if not visuals:
            raise StudioError("At least one visual scene is required to render a video.")
        output_dir.mkdir(parents=True, exist_ok=True)
        warnings: list[str] = []
        narration = audio.path
        duration = audio.duration
        dead_air_removed = False
        if remove_dead_air and not audio.is_silent:
            try:
                narration = output_dir / "narration_clean.wav"
                self._remove_dead_air(ffmpeg, audio.path, narration)
                cleaned_duration = wav_duration(narration)
                if cleaned_duration < max(0.5, audio.duration * 0.35):
                    raise StudioError("dead-air removal discarded too much narration")
                duration = cleaned_duration
                dead_air_removed = True
            except Exception as exc:
                narration = audio.path
                duration = audio.duration
                warnings.append(f"Dead-air removal was skipped: {exc}")

        captions, alignment_source, raw_words, alignment_warnings = self._align(audio, script, narration, duration)
        warnings.extend(alignment_warnings)
        ass_path = output_dir / "captions.ass"
        srt_path = output_dir / "captions.srt"
        ass_path.write_text(self._to_ass(captions), encoding="utf-8")
        srt_path.write_text(self._to_srt(captions), encoding="utf-8")
        write_json(output_dir / "word_timestamps.json", {"source": alignment_source, "words": raw_words})
        concat_path = output_dir / "visuals.ffconcat"
        concat_path.write_text(self._ffconcat(visuals, script, duration), encoding="utf-8")

        temporary = output_dir / "short.rendering.mp4"
        final = output_dir / "short.mp4"
        temporary.unlink(missing_ok=True)
        # fps before zoompan turns each still into frames; pzoom provides restrained continuous movement.
        video_filter = (
            "scale=1200:2133:force_original_aspect_ratio=increase,"
            "crop=1200:2133,fps=30,"
            "zoompan=z='min(max(zoom,pzoom)+0.0002,1.06)':"
            "x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':d=1:s=1080x1920:fps=30,"
            "subtitles=captions.ass"
        )
        command = [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            concat_path.name,
            "-i",
            str(narration.resolve()),
            "-vf",
            video_filter,
            "-t",
            f"{duration:.3f}",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "21",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-movflags",
            "+faststart",
            "-metadata",
            "comment=Created locally for manual review; no automatic publishing",
            temporary.name,
        ]
        try:
            run_command(command, cwd=output_dir, timeout=1800)
            if not temporary.is_file() or temporary.stat().st_size < 1000:
                raise StudioError("FFmpeg completed without creating a usable MP4.")
            temporary.replace(final)
        except Exception:
            temporary.unlink(missing_ok=True)
            raise
        return VideoResult(
            path=final,
            audio_path=narration,
            duration=duration,
            captions_path=ass_path,
            srt_path=srt_path,
            alignment_source=alignment_source,
            visuals_source=visuals_source,
            dead_air_removed=dead_air_removed,
            warnings=warnings,
        )

    @staticmethod
    def _remove_dead_air(ffmpeg: str, source: Path, destination: Path) -> None:
        run_command(
            [
                ffmpeg,
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-i",
                str(source),
                "-af",
                (
                    "silenceremove=start_periods=1:start_duration=0.12:start_threshold=-45dB:"
                    "stop_periods=-1:stop_duration=0.55:stop_threshold=-45dB:stop_silence=0.20"
                ),
                "-ac",
                "1",
                "-ar",
                "22050",
                "-c:a",
                "pcm_s16le",
                str(destination),
            ],
            timeout=300,
        )

    def _align(
        self,
        original_audio: AudioResult,
        script: Script,
        narration: Path,
        duration: float,
    ) -> tuple[list[Caption], str, list[dict], list[str]]:
        if not original_audio.is_silent and self.transcriber.available():
            try:
                result = self.transcriber.transcribe(narration, word_timestamps=True)
                if result.words:
                    return self._captions_from_words(result.words), "faster-whisper-word-level", result.words, []
            except Exception as exc:
                return (
                    self.estimate_captions(script.spoken_text, duration),
                    "estimated-from-script",
                    [],
                    [f"Whisper word alignment failed ({exc}); caption timings are estimated from the script."],
                )
        source = "estimated-from-script"
        warnings: list[str] = []
        if not self.transcriber.available() and not original_audio.is_silent:
            warnings.append(
                "faster-whisper is not installed; captions use estimated word timings rather than measured alignment."
            )
        return self.estimate_captions(script.spoken_text, duration), source, [], warnings

    @staticmethod
    def estimate_captions(text: str, duration: float, max_words: int = 4) -> list[Caption]:
        words = re.findall(r"\S+", text)
        if not words:
            return []
        groups: list[list[str]] = []
        current: list[str] = []
        for word in words:
            current.append(word)
            if len(current) >= max_words or (word.endswith((".", "?", "!")) and len(current) >= 2):
                groups.append(current)
                current = []
        if current:
            groups.append(current)

        def weight(word: str) -> float:
            pause = 4.0 if word.endswith((".", "?", "!")) else 1.5 if word.endswith((",", ":", ";")) else 0.0
            return max(2.0, len(word) * 0.65) + pause

        weights = [sum(weight(word) for word in group) for group in groups]
        total = sum(weights) or 1.0
        captions: list[Caption] = []
        cursor = 0.0
        for index, (group, group_weight) in enumerate(zip(groups, weights)):
            end = duration if index == len(groups) - 1 else cursor + duration * group_weight / total
            captions.append(Caption(round(cursor, 3), round(max(cursor + 0.12, end), 3), " ".join(group)))
            cursor = end
        return captions

    @staticmethod
    def _captions_from_words(words: list[dict], max_words: int = 4) -> list[Caption]:
        groups: list[list[dict]] = []
        current: list[dict] = []
        for word in words:
            if current and float(word["start"]) - float(current[-1]["end"]) > 0.65:
                groups.append(current)
                current = []
            current.append(word)
            if len(current) >= max_words or (
                str(word["word"]).endswith((".", "?", "!")) and len(current) >= 2
            ):
                groups.append(current)
                current = []
        if current:
            groups.append(current)
        captions = []
        for group in groups:
            start = float(group[0]["start"])
            end = max(start + 0.12, float(group[-1]["end"]))
            text = " ".join(str(word["word"]).strip() for word in group if str(word["word"]).strip())
            if text:
                captions.append(Caption(start, end, text))
        return captions

    @staticmethod
    def _ffconcat(visuals: list[Path], script: Script, duration: float) -> str:
        script_duration = sum(max(0.0, section.duration) for section in script.sections)
        scale = duration / script_duration if script_duration > 0 else duration / len(visuals)
        lines = ["ffconcat version 1.0"]
        for index, visual in enumerate(visuals):
            escaped = str(visual.resolve()).replace("'", "'\\''")
            lines.append(f"file '{escaped}'")
            if index < len(script.sections):
                scene_duration = max(0.12, script.sections[index].duration * scale)
            else:
                scene_duration = max(0.12, duration / len(visuals))
            lines.append(f"duration {scene_duration:.3f}")
        # FFmpeg's concat demuxer needs the final image repeated to honor its last duration.
        escaped_last = str(visuals[-1].resolve()).replace("'", "'\\''")
        lines.append(f"file '{escaped_last}'")
        return "\n".join(lines) + "\n"

    @staticmethod
    def _srt_time(seconds: float) -> str:
        milliseconds = max(0, round(seconds * 1000))
        hours, milliseconds = divmod(milliseconds, 3_600_000)
        minutes, milliseconds = divmod(milliseconds, 60_000)
        secs, milliseconds = divmod(milliseconds, 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{milliseconds:03d}"

    @classmethod
    def _to_srt(cls, captions: list[Caption]) -> str:
        blocks = []
        for index, caption in enumerate(captions, 1):
            blocks.append(
                f"{index}\n{cls._srt_time(caption.start)} --> {cls._srt_time(caption.end)}\n{caption.text}\n"
            )
        return "\n".join(blocks)

    @staticmethod
    def _ass_time(seconds: float) -> str:
        centiseconds = max(0, round(seconds * 100))
        hours, centiseconds = divmod(centiseconds, 360_000)
        minutes, centiseconds = divmod(centiseconds, 6_000)
        secs, centiseconds = divmod(centiseconds, 100)
        return f"{hours}:{minutes:02d}:{secs:02d}.{centiseconds:02d}"

    @classmethod
    def _to_ass(cls, captions: list[Caption]) -> str:
        header = """[Script Info]
Title: Creator Studio Dynamic Captions
ScriptType: v4.00+
PlayResX: 1080
PlayResY: 1920
WrapStyle: 0
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,DejaVu Sans,74,&H00FFFFFF,&H0000FFFF,&H00101424,&H80000000,-1,0,0,0,100,100,0,0,1,6,0,2,88,88,235,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
        events = []
        for caption in captions:
            text = caption.text.replace("\\", r"\\").replace("{", r"\{").replace("}", r"\}")
            text = text.replace("\n", r"\N")
            # Restrained fade + scale settle gives motion without sacrificing readability.
            animation = r"{\fad(70,70)\fscx96\fscy96\t(0,130,\fscx100\fscy100)}"
            events.append(
                f"Dialogue: 0,{cls._ass_time(caption.start)},{cls._ass_time(caption.end)},Default,,0,0,0,,{animation}{text}"
            )
        return header + "\n".join(events) + "\n"
