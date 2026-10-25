import tempfile
import unittest
from pathlib import Path

from core.sales_agent import SalesAgent


class SalesAgentTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        root = Path(self.temporary.name)
        knowledge = root / "knowledge"
        knowledge.mkdir()
        (knowledge / "offers.md").write_text(
            "# Brand campaigns\n\nBrand campaign scopes include deliverables, usage rights, timeline, and budget.\n",
            encoding="utf-8",
        )
        self.agent = SalesAgent(root / "leads.db", knowledge, use_ollama=False)

    def tearDown(self):
        self.temporary.cleanup()

    def test_qualifies_and_persists_brand_lead(self):
        reply = self.agent.reply(
            "session-a",
            "We need a two-video brand campaign for our launch this month. Our budget is $5,000. Contact me at producer@example.com.",
        )
        self.assertEqual(reply.lead["intent"], "brand_deal")
        self.assertEqual(reply.lead["budget_text"], "$5,000")
        self.assertEqual(reply.lead["contact"], "producer@example.com")
        self.assertGreaterEqual(reply.lead["fit_score"], 75)
        self.assertEqual(reply.lead["state"], "qualified")
        self.assertTrue((Path(self.temporary.name) / "leads.db").is_file())

    def test_booking_is_logged_but_never_claimed_as_completed(self):
        reply = self.agent.reply("session-b", "Can you book a coaching call for next week?")
        self.assertEqual(reply.lead["state"], "booking_requested")
        self.assertIn("no external calendar event", reply.text.lower())
        self.assertEqual(len(self.agent.list_leads()), 1)

    def test_llm_action_claim_is_replaced_by_safe_local_reply(self):
        class UnsafeOllama:
            model = "test-model"

            def available(self):
                return True

            def chat(self, *_args, **_kwargs):
                return "I've booked your call and processed your payment."

        root = Path(self.temporary.name)
        agent = SalesAgent(
            root / "guarded.db",
            root / "knowledge",
            ollama=UnsafeOllama(),
            use_ollama=True,
        )
        reply = agent.reply("guarded-session", "Book a coaching call next week.")
        self.assertEqual(reply.engine, "deterministic-local-action-guard")
        self.assertNotIn("processed your payment", reply.text)
        self.assertIn("no external calendar event", reply.text.lower())

    def test_knowledge_retrieval_returns_source(self):
        sources = self.agent.retrieve("Tell me about brand campaign usage rights")
        self.assertTrue(sources)
        self.assertEqual(sources[0][0], "offers.md")


if __name__ == "__main__":
    unittest.main()
