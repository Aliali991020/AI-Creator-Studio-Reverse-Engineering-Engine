import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from core.github_learner import GitHubLearner
from core.models import GenerationRequest
from core.orchestrator import StudioOrchestrator
from core.utils import StudioError


class OrchestratorSafetyTests(unittest.TestCase):
    def test_reference_processing_requires_consent_at_core_boundary(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            orchestrator = StudioOrchestrator(github_learner=GitHubLearner(root / "discovery"))
            request = GenerationRequest(
                topic="Consent check",
                voice_reference=root / "reference.wav",
                reference_consent=False,
                auto_discover=False,
                llm_mode="offline",
            )
            with patch("core.orchestrator.READY_DIR", root / "ready"):
                with self.assertRaises(StudioError):
                    asyncio.run(orchestrator.generate(request))
                runs = list((root / "ready").iterdir())
                self.assertEqual(len(runs), 1)
                manifest = json.loads((runs[0] / "manifest.json").read_text(encoding="utf-8"))
                self.assertEqual(manifest["status"], "failed")
                self.assertIn("permission", manifest["error"])
                self.assertFalse((runs[0] / "voice.wav").exists())


if __name__ == "__main__":
    unittest.main()
