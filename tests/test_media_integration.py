import tempfile
import unittest
from pathlib import Path

from core.models import AudioResult, Script, ScriptSection
from core.utils import ffmpeg_binary, wav_duration
from core.video_editor import VideoEditor
from core.visual_renderer import VisualRenderer
from core.voice_cloner import VoiceCloner


@unittest.skipUnless(ffmpeg_binary(), "FFmpeg is not installed")
class MediaIntegrationTests(unittest.TestCase):
    def test_real_vertical_mp4_with_burned_caption_project(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            script = Script(
                topic="Local studio test",
                language="en",
                target_duration=2.0,
                engine="test",
                sections=[
                    ScriptSection("Hook", "Test this local studio.", "Abstract light", 0.0, 1.0),
                    ScriptSection("CTA", "Review every asset.", "Clean final frame", 1.0, 2.0),
                ],
            )
            audio_path = root / "voice.wav"
            VoiceCloner._write_silent_timing_track(audio_path, 2.0)
            audio = AudioResult(
                audio_path, wav_duration(audio_path), "silent-timing-track", is_silent=True
            )
            visuals, source, _warnings = VisualRenderer().create_visuals(script, root / "scenes")
            result = VideoEditor().render(
                audio,
                script,
                root,
                visuals,
                visuals_source=source,
                remove_dead_air=False,
            )
            self.assertTrue(result.path.is_file())
            self.assertGreater(result.path.stat().st_size, 1000)
            self.assertEqual((result.width, result.height), (1080, 1920))
            self.assertIn("Dialogue:", result.captions_path.read_text(encoding="utf-8"))
            self.assertIn("-->", result.srt_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
