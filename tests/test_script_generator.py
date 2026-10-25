import unittest

from core.script_generator import ScriptGenerator
from core.utils import StudioError


class ScriptGeneratorTests(unittest.TestCase):
    def test_offline_script_has_hook_body_cta_and_visuals(self):
        script = ScriptGenerator().generate(
            "building a calm morning routine",
            hook="Your morning does not need more hacks.",
            target_duration=45,
            mode="offline",
        )
        self.assertEqual(script.engine, "deterministic-offline")
        self.assertGreaterEqual(len(script.sections), 5)
        self.assertEqual(script.sections[0].text, "Your morning does not need more hacks.")
        self.assertEqual(script.sections[0].start, 0)
        self.assertEqual(script.sections[0].end, 3)
        self.assertAlmostEqual(script.duration, 45)
        self.assertTrue(all(section.visual_prompt for section in script.sections))
        self.assertIn("Manual review and upload only", script.to_markdown())

    def test_offline_writer_does_not_mislabel_english_as_translation(self):
        with self.assertRaises(StudioError):
            ScriptGenerator().generate("productividad", language="es", mode="offline")

    def test_rejects_empty_topic(self):
        with self.assertRaises(StudioError):
            ScriptGenerator().generate("   ", mode="offline")


if __name__ == "__main__":
    unittest.main()
