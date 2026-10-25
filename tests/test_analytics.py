import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from core.analytics import AnalyticsAggregator


class AnalyticsTests(unittest.TestCase):
    def test_import_upsert_summary_and_brief(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            csv_path = root / "metrics.csv"
            csv_path.write_text(
                "date,platform,language,topic,views,follower_growth,conversions,revenue\n"
                "2026-01-02,youtube,en,local workflows,1000,20,4,99.50\n"
                "2026-01-02,instagram,es,local workflows,500,10,2,25.50\n"
                "2025-12-01,youtube,en,old data,99999,999,99,999.00\n",
                encoding="utf-8",
            )
            engine = AnalyticsAggregator(root / "studio.db")
            self.assertEqual(engine.import_csv(csv_path), 3)
            now = datetime(2026, 1, 2, 12, tzinfo=timezone.utc)
            metrics = engine.summary(24, now=now)
            self.assertEqual(metrics["views"], 1500)
            self.assertEqual(metrics["revenue_cents"], 12500)
            self.assertEqual(metrics["follower_growth"], 30)
            self.assertEqual(metrics["languages"][0]["language"], "en")
            path, content = engine.generate_morning_brief(output_dir=root / "briefs", now=now)
            self.assertTrue(path.is_file())
            self.assertIn("$125.00", content)
            self.assertIn("Hook experiments", content)
            # Re-importing updates the unique metric row rather than duplicating it.
            engine.import_csv(csv_path)
            self.assertEqual(engine.summary(24, now=now)["views"], 1500)


if __name__ == "__main__":
    unittest.main()
