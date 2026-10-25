import tempfile
import unittest
from pathlib import Path

from core.github_learner import GitHubLearner
from core.utils import StudioError


class GitHubLearnerTests(unittest.TestCase):
    def test_ast_index_extracts_details_without_executing(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            repository = root / "repository"
            repository.mkdir()
            (repository / ".git").mkdir()
            (repository / ".git" / "ignored.py").write_text("class ShouldNotAppear: pass\n")
            (repository / "pipeline.py").write_text(
                'class Parser:\n    """Parse local captions."""\n    pass\n\ndef align(words):\n    """Align words."""\n    return words\n',
                encoding="utf-8",
            )
            (repository / "README.md").write_text("# Example\n\n## Implementation\n", encoding="utf-8")
            learner = GitHubLearner(root / "storage")
            index = learner.index_repository(repository, "example/tool")
            paths = {file["path"] for file in index["files"]}
            self.assertIn("pipeline.py", paths)
            self.assertNotIn(".git/ignored.py", paths)
            symbols = [symbol for file in index["files"] for symbol in file.get("symbols", [])]
            self.assertIn("Parser", {symbol["name"] for symbol in symbols})
            self.assertIn("align", {symbol["name"] for symbol in symbols})
            self.assertEqual(index["execution_policy"], "not_executed")
            self.assertTrue((root / "storage" / "github_index" / "example--tool.md").is_file())

    def test_rejects_non_github_repository_identity(self):
        with tempfile.TemporaryDirectory() as temporary:
            learner = GitHubLearner(Path(temporary))
            with self.assertRaises(StudioError):
                learner.clone_repository("../../outside")


if __name__ == "__main__":
    unittest.main()
