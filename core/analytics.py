"""Manual multi-platform CSV analytics, local SQLite aggregation, and morning briefs."""
from __future__ import annotations

import csv
import sqlite3
from datetime import date, datetime, time, timedelta, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

from .utils import ROOT, READY_DIR, StudioError, utc_now


class AnalyticsAggregator:
    def __init__(self, database_path: Path | None = None) -> None:
        self.database_path = database_path or ROOT / "database" / "studio.db"
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
                CREATE TABLE IF NOT EXISTS platform_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    recorded_at TEXT NOT NULL,
                    platform TEXT NOT NULL,
                    language TEXT NOT NULL DEFAULT 'en',
                    topic TEXT NOT NULL DEFAULT '',
                    views INTEGER NOT NULL DEFAULT 0,
                    follower_growth INTEGER NOT NULL DEFAULT 0,
                    conversions INTEGER NOT NULL DEFAULT 0,
                    revenue_cents INTEGER NOT NULL DEFAULT 0,
                    imported_at TEXT NOT NULL,
                    UNIQUE(recorded_at, platform, language, topic)
                );
                CREATE TABLE IF NOT EXISTS morning_briefs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at TEXT NOT NULL,
                    path TEXT NOT NULL,
                    revenue_cents INTEGER NOT NULL,
                    views INTEGER NOT NULL
                );
                """
            )

    @staticmethod
    def _timestamp(value: str) -> str:
        value = value.strip()
        try:
            if re_full_date(value):
                parsed = datetime.combine(date.fromisoformat(value), time.min, tzinfo=timezone.utc)
            else:
                parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
                if parsed.tzinfo is None:
                    parsed = parsed.replace(tzinfo=timezone.utc)
                parsed = parsed.astimezone(timezone.utc)
            return parsed.isoformat(timespec="seconds")
        except ValueError as exc:
            raise StudioError(f"Invalid analytics timestamp {value!r}; use YYYY-MM-DD or ISO 8601.") from exc

    def import_csv(self, csv_path: Path) -> int:
        if not csv_path.is_file():
            raise StudioError(f"Analytics CSV was not found: {csv_path}")
        now = utc_now()
        count = 0
        with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            required = {"platform", "views"}
            if not reader.fieldnames or not required.issubset(set(reader.fieldnames)):
                raise StudioError("Analytics CSV needs at least: platform, views, and date or recorded_at.")
            if not ({"date", "recorded_at"} & set(reader.fieldnames)):
                raise StudioError("Analytics CSV needs a date or recorded_at column.")
            with self._connect() as connection:
                for line_number, row in enumerate(reader, 2):
                    try:
                        platform = (row.get("platform") or "").strip().lower()
                        if not platform:
                            raise StudioError("platform is empty")
                        recorded_at = self._timestamp(row.get("recorded_at") or row.get("date") or "")
                        views = self._integer(row.get("views"), "views", minimum=0)
                        followers = self._integer(
                            row.get("follower_growth") or row.get("followers") or "0", "follower_growth"
                        )
                        conversions = self._integer(row.get("conversions") or "0", "conversions", minimum=0)
                        revenue_cents = self._money(row.get("revenue") or "0")
                        language = (row.get("language") or "en").strip().lower()
                        topic = (row.get("topic") or "").strip()[:240]
                        connection.execute(
                            """
                            INSERT INTO platform_metrics(
                                recorded_at, platform, language, topic, views, follower_growth,
                                conversions, revenue_cents, imported_at
                            ) VALUES(?,?,?,?,?,?,?,?,?)
                            ON CONFLICT(recorded_at, platform, language, topic) DO UPDATE SET
                                views=excluded.views, follower_growth=excluded.follower_growth,
                                conversions=excluded.conversions, revenue_cents=excluded.revenue_cents,
                                imported_at=excluded.imported_at
                            """,
                            (
                                recorded_at,
                                platform,
                                language,
                                topic,
                                views,
                                followers,
                                conversions,
                                revenue_cents,
                                now,
                            ),
                        )
                        count += 1
                    except (StudioError, ValueError) as exc:
                        raise StudioError(f"CSV row {line_number}: {exc}") from exc
        return count

    @staticmethod
    def _integer(value: Any, field: str, minimum: int | None = None) -> int:
        try:
            number = int(str(value or "0").replace(",", "").strip())
        except ValueError as exc:
            raise StudioError(f"{field} must be an integer") from exc
        if minimum is not None and number < minimum:
            raise StudioError(f"{field} must be at least {minimum}")
        return number

    @staticmethod
    def _money(value: Any) -> int:
        try:
            amount = Decimal(str(value or "0").replace(",", "").replace("$", "").strip())
        except InvalidOperation as exc:
            raise StudioError("revenue must be a decimal amount") from exc
        if amount < 0:
            raise StudioError("revenue cannot be negative")
        return int(amount * 100)

    def summary(self, hours: int = 24, *, now: datetime | None = None) -> dict[str, Any]:
        hours = max(1, min(24 * 90, int(hours)))
        current = now or datetime.now(timezone.utc)
        if current.tzinfo is None:
            current = current.replace(tzinfo=timezone.utc)
        current = current.astimezone(timezone.utc)
        cutoff = (current - timedelta(hours=hours)).isoformat(timespec="seconds")
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM platform_metrics WHERE recorded_at >= ? AND recorded_at <= ? ORDER BY views DESC",
                (cutoff, current.isoformat(timespec="seconds")),
            ).fetchall()
        totals = {"views": 0, "follower_growth": 0, "conversions": 0, "revenue_cents": 0}
        languages: dict[str, dict[str, int]] = {}
        platforms: dict[str, dict[str, int]] = {}
        for row in rows:
            for key in totals:
                totals[key] += int(row[key])
            for group, name in ((languages, row["language"]), (platforms, row["platform"])):
                bucket = group.setdefault(
                    name, {"views": 0, "follower_growth": 0, "conversions": 0, "revenue_cents": 0}
                )
                for key in totals:
                    bucket[key] += int(row[key])
        ranked_languages = sorted(
            ({"language": key, **values} for key, values in languages.items()),
            key=lambda item: (item["views"], item["revenue_cents"]),
            reverse=True,
        )
        ranked_platforms = sorted(
            ({"platform": key, **values} for key, values in platforms.items()),
            key=lambda item: item["views"],
            reverse=True,
        )
        return {
            "hours": hours,
            "window_start": cutoff,
            "window_end": current.isoformat(timespec="seconds"),
            **totals,
            "follower_growth_per_day": round(totals["follower_growth"] * 24 / hours, 2),
            "languages": ranked_languages,
            "platforms": ranked_platforms,
            "row_count": len(rows),
            "topics": [row["topic"] for row in rows if row["topic"]][:5],
        }

    def generate_morning_brief(self, *, output_dir: Path | None = None, now: datetime | None = None) -> tuple[Path, str]:
        current = now or datetime.now(timezone.utc)
        if current.tzinfo is None:
            current = current.replace(tzinfo=timezone.utc)
        metrics = self.summary(24, now=current)
        topic = metrics["topics"][0] if metrics["topics"] else "your niche"
        money = metrics["revenue_cents"] / 100
        lines = [
            f"# Morning Brief — {current.date().isoformat()}",
            "",
            f"Window: `{metrics['window_start']}` to `{metrics['window_end']}` (UTC).",
            "",
            "## Last 24 hours",
            "",
            f"- **Revenue:** ${money:,.2f}",
            f"- **Views:** {metrics['views']:,}",
            f"- **Conversions:** {metrics['conversions']:,}",
            f"- **Follower growth:** {metrics['follower_growth']:+,} ({metrics['follower_growth_per_day']:+,}/day velocity)",
            f"- **Imported metric rows:** {metrics['row_count']}",
            "",
            "## Language tracks",
            "",
        ]
        if metrics["languages"]:
            lines.extend(["| Language | Views | Conversions | Revenue |", "|---|---:|---:|---:|"])
            for item in metrics["languages"]:
                lines.append(
                    f"| {item['language']} | {item['views']:,} | {item['conversions']:,} | ${item['revenue_cents'] / 100:,.2f} |"
                )
            top = metrics["languages"][0]
            lines.extend(["", f"**Top track by views:** {top['language']} ({top['views']:,} views)."])
        else:
            lines.append("No metrics imported for this window. Import platform exports before drawing conclusions.")
        lines.extend(
            [
                "",
                "## Hook experiments for today",
                "",
                "These are recommendations to test, not claims of guaranteed performance:",
                "",
                f"1. **Constraint:** “If {topic} feels harder than it should, watch this.”",
                f"2. **Contrarian:** “Most advice about {topic} starts in the wrong place.”",
                f"3. **Open loop:** “One small change can simplify {topic}—here it is.”",
                "",
                "## Operating notes",
                "",
                "- Verify imported figures against each platform before making spending decisions.",
                "- Review all generated media and captions manually. Nothing is posted automatically.",
                "- Respect voice-reference consent and each platform's synthetic-media disclosure rules.",
                "",
            ]
        )
        content = "\n".join(lines)
        destination_dir = output_dir or READY_DIR / "briefs"
        destination_dir.mkdir(parents=True, exist_ok=True)
        path = destination_dir / f"{current.date().isoformat()}-morning-brief.md"
        path.write_text(content, encoding="utf-8")
        with self._connect() as connection:
            connection.execute(
                "INSERT INTO morning_briefs(created_at, path, revenue_cents, views) VALUES(?,?,?,?)",
                (utc_now(), str(path), metrics["revenue_cents"], metrics["views"]),
            )
        return path, content

    def recent_metrics(self, limit: int = 100) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM platform_metrics ORDER BY recorded_at DESC LIMIT ?",
                (max(1, min(500, limit)),),
            ).fetchall()
        result = []
        for row in rows:
            item = dict(row)
            item["revenue"] = item.pop("revenue_cents") / 100
            result.append(item)
        return result


def re_full_date(value: str) -> bool:
    """Avoid treating arbitrary ISO strings as dates; kept separate for easy testing."""

    if len(value) != 10:
        return False
    try:
        date.fromisoformat(value)
        return True
    except ValueError:
        return False
