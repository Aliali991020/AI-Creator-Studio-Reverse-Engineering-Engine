"""High-retention script generation with Ollama and a deterministic offline mode."""
from __future__ import annotations

import json
import re
from typing import Any

from .models import LANGUAGE_NAMES, Script, ScriptSection
from .ollama import OllamaClient, OllamaUnavailable
from .utils import StudioError


SCRIPT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "sections": {
            "type": "array",
            "minItems": 4,
            "maxItems": 8,
            "items": {
                "type": "object",
                "properties": {
                    "label": {"type": "string"},
                    "text": {"type": "string"},
                    "visual_prompt": {"type": "string"},
                },
                "required": ["label", "text", "visual_prompt"],
            },
        }
    },
    "required": ["sections"],
}


def _allocate_timing(sections: list[ScriptSection], target_duration: float) -> None:
    if not sections:
        return
    target = max(3.0, float(target_duration))
    cursor = 0.0
    # The first section is intentionally limited to the three-second hook window.
    hook_duration = min(3.0, target / max(1, len(sections)))
    if len(sections) == 1:
        hook_duration = target
    sections[0].start = 0.0
    sections[0].end = hook_duration
    cursor = hook_duration
    remaining = target - hook_duration
    weights = [max(1, len(re.findall(r"\b[\w’'-]+\b", section.text))) for section in sections[1:]]
    total_weight = sum(weights) or 1
    for index, section in enumerate(sections[1:]):
        section.start = round(cursor, 3)
        if index == len(weights) - 1:
            section.end = target
        else:
            section.end = round(cursor + remaining * weights[index] / total_weight, 3)
        cursor = section.end


def _extract_json(content: str) -> dict[str, Any]:
    stripped = re.sub(r"^```(?:json)?\s*|\s*```$", "", content.strip(), flags=re.MULTILINE).strip()
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        start, end = stripped.find("{"), stripped.rfind("}")
        if start < 0 or end <= start:
            raise StudioError("The local LLM did not return a valid script object.")
        try:
            return json.loads(stripped[start : end + 1])
        except json.JSONDecodeError as exc:
            raise StudioError("The local LLM returned malformed script JSON.") from exc


class ScriptGenerator:
    def __init__(self, ollama: OllamaClient | None = None) -> None:
        self.ollama = ollama or OllamaClient()

    def generate(
        self,
        topic: str,
        *,
        hook: str = "",
        target_duration: float = 45.0,
        language: str = "en",
        cta: str = "",
        source_notes: str = "",
        mode: str = "auto",
    ) -> Script:
        topic = topic.strip()
        if not topic:
            raise StudioError("A topic or a transcribed voice memo is required.")
        if not 5 <= target_duration <= 180:
            raise StudioError("Target duration must be between 5 and 180 seconds.")
        if mode not in {"auto", "offline", "ollama"}:
            raise StudioError(f"Unknown script mode: {mode}")
        if mode != "offline":
            try:
                return self._generate_with_ollama(
                    topic,
                    hook=hook,
                    target_duration=target_duration,
                    language=language,
                    cta=cta,
                    source_notes=source_notes,
                )
            except (OllamaUnavailable, StudioError, ValueError) as exc:
                if mode == "ollama":
                    raise StudioError(
                        f"Ollama-only mode could not generate a script: {exc}. "
                        "Start Ollama and pull the configured model, or choose Auto."
                    ) from exc
                script = self._generate_offline(
                    topic,
                    hook=hook,
                    target_duration=target_duration,
                    language=language,
                    cta=cta,
                )
                script.warnings.append(f"Ollama was unavailable; used the deterministic offline writer instead ({exc}).")
                return script
        return self._generate_offline(
            topic, hook=hook, target_duration=target_duration, language=language, cta=cta
        )

    def _generate_with_ollama(
        self,
        topic: str,
        *,
        hook: str,
        target_duration: float,
        language: str,
        cta: str,
        source_notes: str,
    ) -> Script:
        language_name = LANGUAGE_NAMES.get(language, language)
        words = max(20, int(target_duration * 2.35))
        requested_hook = f"Use this hook verbatim: {hook!r}." if hook.strip() else "Write an original hook."
        requested_cta = f"Use this CTA: {cta!r}." if cta.strip() else "Write a natural, non-spammy CTA."
        notes = f"Source voice-memo notes:\n{source_notes[:8000]}" if source_notes.strip() else "No extra source notes."
        prompt = f"""
Create a factually responsible vertical short-video script about: {topic}

Hard requirements:
- Write every spoken word and visual direction in {language_name}.
- Approximately {words} spoken words, suitable for {target_duration:g} seconds.
- The first section is a psychological spoken hook of at most 8 words (three seconds).
- Build curiosity without fake claims, invented statistics, or misleading promises.
- Use 4–6 sections: hook, tension/reframe, practical value, payoff/example, CTA.
- Use short conversational sentences. Do not include speaker names, camera directions, or timestamps in text fields.
- {requested_hook}
- {requested_cta}
- Give each section a specific visual_prompt for a local image generator; no trademarks or real-person likenesses.
- Return only JSON matching the supplied schema.

{notes}
""".strip()
        content = self.ollama.chat(
            [
                {
                    "role": "system",
                    "content": "You are a rigorous short-form script editor. You return valid JSON and never fabricate evidence.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.75,
            json_schema=SCRIPT_SCHEMA,
        )
        data = _extract_json(content)
        raw_sections = data.get("sections")
        if not isinstance(raw_sections, list) or not raw_sections:
            raise StudioError("The local LLM response contained no script sections.")
        sections: list[ScriptSection] = []
        for index, raw in enumerate(raw_sections[:8]):
            if not isinstance(raw, dict) or not str(raw.get("text", "")).strip():
                continue
            sections.append(
                ScriptSection(
                    label=str(raw.get("label") or f"Section {index + 1}")[:80],
                    text=str(raw["text"]).strip(),
                    visual_prompt=str(raw.get("visual_prompt", "")).strip()[:1200],
                )
            )
        if len(sections) < 2:
            raise StudioError("The local LLM response did not contain enough usable sections.")
        _allocate_timing(sections, target_duration)
        return Script(
            topic=topic,
            language=language,
            target_duration=target_duration,
            sections=sections,
            engine=f"ollama:{self.ollama.model}",
            cta=sections[-1].text,
            source_notes=source_notes,
        )

    def _generate_offline(
        self,
        topic: str,
        *,
        hook: str,
        target_duration: float,
        language: str,
        cta: str,
    ) -> Script:
        if language != "en":
            raise StudioError(
                "The deterministic writer is English-only. For other source languages, start Ollama "
                "or configure a local Argos Translate model; no untranslated audio will be mislabeled."
            )
        clean_topic = topic.rstrip(".!?")
        sections = [
            ScriptSection(
                "3-second hook",
                hook.strip() or f"If {clean_topic} feels harder than it should, watch this.",
                f"High-contrast abstract visual metaphor for {clean_topic}, bold geometric shapes, no text",
            ),
            ScriptSection(
                "Open the loop",
                f"Most advice about {clean_topic} adds more steps. The useful move is the opposite: remove everything that does not serve one clear outcome.",
                "A cluttered arrangement becoming one clean, focused composition, cinematic light, vertical frame",
            ),
            ScriptSection(
                "Three-part payoff",
                "First, name the result in one sentence. Next, choose the smallest action that creates visible progress. Finally, review what changed and repeat only what worked.",
                "Three sequential visual beats showing focus, a small action, and a measurable result, no numbers or text",
            ),
            ScriptSection(
                "Concrete example",
                f"For {clean_topic}, that means testing one change today instead of redesigning everything. Small feedback beats a perfect plan you never start.",
                "Close-up of one deliberate action followed by a visible before-and-after transformation, photorealistic",
            ),
            ScriptSection(
                "Call to action",
                cta.strip() or "Save this for later, and follow for the next practical breakdown.",
                "Clean resolved final composition with generous negative space, warm highlight, vertical frame",
            ),
        ]
        _allocate_timing(sections, target_duration)
        return Script(
            topic=topic,
            language=language,
            target_duration=target_duration,
            sections=sections,
            engine="deterministic-offline",
            cta=sections[-1].text,
        )
