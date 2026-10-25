"""Maya: local knowledge-grounded lead qualification with SQLite event logging."""
from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .ollama import OllamaClient
from .utils import ROOT, utc_now


@dataclass(slots=True)
class SalesReply:
    text: str
    lead: dict[str, Any]
    sources: list[str]
    engine: str


class SalesAgent:
    def __init__(
        self,
        database_path: Path | None = None,
        knowledge_dir: Path | None = None,
        ollama: OllamaClient | None = None,
        use_ollama: bool = True,
    ) -> None:
        self.database_path = database_path or ROOT / "database" / "leads.db"
        self.knowledge_dir = knowledge_dir or ROOT / "knowledge"
        self.ollama = ollama or OllamaClient()
        self.use_ollama = use_ollama
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self._migrate()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path, timeout=10)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL")
        return connection

    def _migrate(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS leads (
                    session_id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    intent TEXT NOT NULL DEFAULT 'unknown',
                    budget_text TEXT NOT NULL DEFAULT '',
                    contact TEXT NOT NULL DEFAULT '',
                    fit_score INTEGER NOT NULL DEFAULT 0,
                    fit_label TEXT NOT NULL DEFAULT 'cold',
                    state TEXT NOT NULL DEFAULT 'new',
                    last_message TEXT NOT NULL DEFAULT ''
                );
                CREATE TABLE IF NOT EXISTS lead_messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL CHECK(role IN ('user', 'assistant')),
                    content TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(session_id) REFERENCES leads(session_id)
                );
                CREATE INDEX IF NOT EXISTS idx_lead_messages_session
                    ON lead_messages(session_id, id);
                """
            )

    def reply(self, session_id: str, message: str) -> SalesReply:
        session_id = session_id.strip()[:120]
        message = message.strip()
        if not session_id:
            raise ValueError("A local session id is required.")
        if not message:
            raise ValueError("A message is required.")
        if len(message) > 12_000:
            raise ValueError("Message is too long (12,000 character limit).")
        now = utc_now()
        with self._connect() as connection:
            connection.execute(
                "INSERT OR IGNORE INTO leads(session_id, created_at, updated_at, last_message) VALUES(?,?,?,?)",
                (session_id, now, now, message),
            )
            connection.execute(
                "INSERT INTO lead_messages(session_id, role, content, created_at) VALUES(?,?,?,?)",
                (session_id, "user", message, now),
            )
            rows = connection.execute(
                "SELECT role, content FROM lead_messages WHERE session_id=? ORDER BY id DESC LIMIT 12",
                (session_id,),
            ).fetchall()
        history = list(reversed(rows))
        user_text = "\n".join(row["content"] for row in history if row["role"] == "user")
        qualification = self._qualify(user_text, message)
        sources = self.retrieve(message)
        response, engine = self._compose_response(qualification, message, history, sources)
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE leads SET updated_at=?, intent=?, budget_text=?, contact=?, fit_score=?,
                    fit_label=?, state=?, last_message=? WHERE session_id=?
                """,
                (
                    now,
                    qualification["intent"],
                    qualification["budget_text"],
                    qualification["contact"],
                    qualification["fit_score"],
                    qualification["fit_label"],
                    qualification["state"],
                    message,
                    session_id,
                ),
            )
            connection.execute(
                "INSERT INTO lead_messages(session_id, role, content, created_at) VALUES(?,?,?,?)",
                (session_id, "assistant", response, utc_now()),
            )
        return SalesReply(
            text=response,
            lead=self.get_lead(session_id) or {},
            sources=[source[0] for source in sources],
            engine=engine,
        )

    @staticmethod
    def _qualify(all_user_text: str, current_message: str) -> dict[str, Any]:
        text = all_user_text.lower()
        current = current_message.lower()
        intent_signals = {
            "course": ("course", "curriculum", "cohort", "learn", "training", "student"),
            "coaching": ("coaching", "coach", "1:1", "mentor", "consulting", "audit"),
            "brand_deal": ("brand", "sponsor", "campaign", "collaboration", "ambassador", "ugc"),
        }
        intent_scores = {
            intent: sum(1 for signal in signals if signal in text) for intent, signals in intent_signals.items()
        }
        intent = max(intent_scores, key=intent_scores.get)
        if intent_scores[intent] == 0:
            intent = "unknown"
        budget_patterns = [
            r"(?:\$|€|£)\s?\d[\d,]*(?:\.\d{1,2})?(?:\s?[kK])?",
            r"\b\d[\d,]*(?:\.\d{1,2})?\s?(?:usd|eur|gbp|dollars?|euros?)\b",
            r"\bbudget(?:\s+is|\s+of|:)?\s+\d[\d,]*(?:\s?[kK])?\b",
        ]
        budget = ""
        for pattern in budget_patterns:
            match = re.search(pattern, text, flags=re.IGNORECASE)
            if match:
                budget = match.group(0).strip()
                break
        contact_match = re.search(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", all_user_text, re.I)
        contact = contact_match.group(0) if contact_match else ""
        has_timeline = bool(
            re.search(r"\b(this week|this month|today|tomorrow|next week|next month|q[1-4]|asap|immediately)\b", text)
        )
        score = 10
        score += 25 if intent != "unknown" else 0
        score += 25 if budget else 0
        score += 15 if has_timeline else 0
        score += 10 if contact else 0
        score += 15 if len(all_user_text.split()) >= 20 else 5
        score = min(100, score)
        label = "hot" if score >= 75 else "warm" if score >= 45 else "cold"
        if re.search(r"\b(checkout|purchase|buy now|payment|pay now)\b", current):
            state = "checkout_intent"
        elif re.search(r"\b(book|schedule|calendar|discovery call|book a call)\b", current):
            state = "booking_requested"
        elif score >= 75:
            state = "qualified"
        elif intent != "unknown":
            state = "nurturing"
        else:
            state = "new"
        return {
            "intent": intent,
            "budget_text": budget,
            "contact": contact,
            "fit_score": score,
            "fit_label": label,
            "state": state,
        }

    def retrieve(self, query: str, limit: int = 3) -> list[tuple[str, str]]:
        if not self.knowledge_dir.is_dir():
            return []
        stopwords = {"the", "and", "for", "with", "that", "this", "you", "your", "are", "what", "how"}
        terms = {word for word in re.findall(r"[a-z0-9]+", query.lower()) if len(word) > 2} - stopwords
        candidates: list[tuple[int, str, str]] = []
        for path in sorted(self.knowledge_dir.rglob("*.md"))[:40]:
            if path.is_symlink() or path.stat().st_size > 200_000:
                continue
            try:
                content = path.read_text(encoding="utf-8")
            except OSError:
                continue
            relative = str(path.relative_to(self.knowledge_dir))
            for paragraph in re.split(r"\n\s*\n", content):
                paragraph = paragraph.strip()
                if len(paragraph) < 30:
                    continue
                words = set(re.findall(r"[a-z0-9]+", paragraph.lower()))
                score = len(terms & words)
                if score:
                    candidates.append((score, relative, paragraph[:1800]))
        candidates.sort(key=lambda item: item[0], reverse=True)
        return [(source, text) for _score, source, text in candidates[:limit]]

    def _compose_response(
        self,
        qualification: dict[str, Any],
        message: str,
        history: list[sqlite3.Row],
        sources: list[tuple[str, str]],
    ) -> tuple[str, str]:
        fallback = self._deterministic_response(qualification, sources)
        if not self.use_ollama or not self.ollama.available():
            return fallback, "deterministic-local"
        context = "\n\n".join(f"Source: {source}\n{text}" for source, text in sources) or "No matching knowledge documents."
        conversation = [
            {
                "role": "system",
                "content": (
                    "You are Maya, a concise local sales concierge. Answer only from the supplied knowledge base; "
                    "never invent prices, availability, results, links, discounts, or completed actions. Qualify the lead "
                    "with at most one useful follow-up question. Never claim to have emailed, booked, charged, or posted anything. "
                    f"Current qualification: {qualification}. Knowledge:\n{context}"
                ),
            }
        ]
        conversation.extend({"role": row["role"], "content": row["content"]} for row in history[-8:])
        try:
            response = self.ollama.chat(conversation, temperature=0.25)
            # A language model has no ability to perform these actions in this studio.
            # Reject accidental claims rather than displaying them as completed facts.
            action_claim = re.search(
                r"\b(i(?:'ve| have)?\s+(?:booked|scheduled|charged|emailed|sent|processed)|"
                r"payment (?:is |was )?(?:complete|processed)|booking (?:is |was )?confirmed)\b",
                response,
                flags=re.IGNORECASE,
            )
            if action_claim:
                return fallback, "deterministic-local-action-guard"
            return response, f"ollama:{self.ollama.model}"
        except Exception:
            return fallback, "deterministic-local"

    @staticmethod
    def _deterministic_response(qualification: dict[str, Any], sources: list[tuple[str, str]]) -> str:
        intent = qualification["intent"]
        state = qualification["state"]
        source_note = f" I found relevant details in `{sources[0][0]}`." if sources else ""
        if state == "checkout_intent":
            return (
                "I’ve logged your checkout intent. No payment has been processed in this local chat. "
                "Please use the official checkout instructions in the product guide after reviewing the offer."
                + source_note
            )
        if state == "booking_requested":
            return (
                "I’ve logged your booking request, but no external calendar event has been created. "
                "What time zone and preferred day should the human reviewer use when confirming your call?"
            )
        if intent == "brand_deal":
            budget = f" I noted your budget as {qualification['budget_text']}." if qualification["budget_text"] else ""
            return (
                "Thanks—this sounds like a brand collaboration."
                + budget
                + " What campaign outcome, deliverables, usage rights, and timeline do you have in mind?"
                + source_note
            )
        if intent == "coaching":
            return (
                "It sounds like 1:1 coaching may be the right path."
                + source_note
                + " What specific outcome would make coaching worthwhile for you over the next month?"
            )
        if intent == "course":
            return (
                "The course could be a fit if you prefer a structured, self-paced path."
                + source_note
                + " What are you trying to learn or improve right now?"
            )
        return (
            "Hi, I’m Maya, the local studio concierge. Are you asking about the course, 1:1 coaching, "
            "or a brand collaboration? A little context about your goal will help me qualify the right option."
        )

    def get_lead(self, session_id: str) -> dict[str, Any] | None:
        with self._connect() as connection:
            row = connection.execute("SELECT * FROM leads WHERE session_id=?", (session_id,)).fetchone()
        return dict(row) if row else None

    def list_leads(self, limit: int = 100) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM leads ORDER BY updated_at DESC LIMIT ?", (max(1, min(500, limit)),)
            ).fetchall()
        return [dict(row) for row in rows]
