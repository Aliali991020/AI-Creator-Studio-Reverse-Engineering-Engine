import tempfile
import unittest
import wave
from pathlib import Path

from core.video_editor import VideoEditor
from core.voice_cloner import VoiceCloner


class VoiceAndCaptionTests(unittest.TestCase):
    def test_silent_timing_track_is_valid_wave(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "timing.wav"
            VoiceCloner._write_silent_timing_track(path, 1.25, sample_rate=8000)
            with wave.open(str(path), "rb") as audio:
                self.assertEqual(audio.getnchannels(), 1)
                self.assertEqual(audio.getframerate(), 8000)
                self.assertAlmostEqual(audio.getnframes() / 8000, 1.25, places=2)

    def test_estimated_captions_cover_duration(self):
        captions = VideoEditor.estimate_captions(
            "One small test can make the whole workflow easier to review.", 4.0, max_words=3
        )
        self.assertGreater(len(captions), 2)
        self.assertEqual(captions[0].start, 0)
        self.assertAlmostEqual(captions[-1].end, 4.0)
        self.assertTrue(all(caption.end > caption.start for caption in captions))
        self.assertTrue(all(len(caption.text.split()) <= 3 for caption in captions))

    def test_srt_and_ass_are_real_timed_formats(self):
        captions = VideoEditor.estimate_captions("A useful caption appears here.", 2.0)
        srt = VideoEditor._to_srt(captions)
        ass = VideoEditor._to_ass(captions)
        self.assertIn("00:00:00,000 -->", srt)
        self.assertIn("Dialogue: 0,", ass)
        self.assertIn(r"\fad(70,70)", ass)
        self.assertIn("PlayResX: 1080", ass)

    def test_measured_words_are_grouped_without_overlap(self):
        words = [
            {"word": "Hello", "start": 0.1, "end": 0.4},
            {"word": "local", "start": 0.45, "end": 0.7},
            {"word": "studio", "start": 0.75, "end": 1.1},
        ]
        captions = VideoEditor._captions_from_words(words, max_words=2)
        self.assertEqual(len(captions), 2)
        self.assertEqual(captions[0].text, "Hello local")
        self.assertEqual(captions[1].start, 0.75)


if __name__ == "__main__":
    unittest.main()
