"""Local neural translation and multilingual voice-track preparation."""
from __future__ import annotations

from dataclasses import replace

from .models import LANGUAGE_NAMES, Script, ScriptSection
from .ollama import OllamaClient, OllamaUnavailable
from .utils import StudioError


class DubbingEngine:
    """Translate locally with Argos or Ollama; never label untranslated text as a dub."""

    def __init__(self, ollama: OllamaClient | None = None) -> None:
        self.ollama = ollama or OllamaClient()

    @staticmethod
    def argos_available() -> bool:
        try:
            import argostranslate  # noqa: F401

            return True
        except ImportError:
            return False

    def translate(self, script: Script, target_language: str) -> Script:
        if target_language not in LANGUAGE_NAMES:
            raise StudioError(f"Unsupported dub language: {target_language}")
        if target_language == script.language:
            return script
        errors: list[str] = []
        if self.argos_available():
            try:
                return self._translate_argos(script, target_language)
            except Exception as exc:
                errors.append(f"Argos: {exc}")
        try:
            return self._translate_ollama(script, target_language)
        except Exception as exc:
            errors.append(f"Ollama: {exc}")
        raise StudioError(
            f"No local translation route could produce {LANGUAGE_NAMES[target_language]}: "
            + " | ".join(errors)
            + ". Install an Argos language pair or start Ollama. The original script was not substituted."
        )

    @staticmethod
    def _argos_code(code: str) -> str:
        return {"zh-cn": "zh"}.get(code, code)

    def _translate_argos(self, script: Script, target_language: str) -> Script:
        from argostranslate import translate

        translation = translate.get_translation_from_codes(
            self._argos_code(script.language), self._argos_code(target_language)
        )
        if translation is None:
            raise StudioError(
                f"the {script.language}→{target_language} package is not installed in Argos Translate"
            )
        sections = []
        for section in script.sections:
            translated = str(translation.translate(section.text)).strip()
            if not translated:
                raise StudioError(f"empty translation for section {section.label!r}")
            sections.append(replace(section, text=translated))
        return Script(
            topic=script.topic,
            language=target_language,
            target_duration=script.target_duration,
            sections=sections,
            engine=f"{script.engine} + argos-translate",
            cta=sections[-1].text,
            source_notes=script.source_notes,
        )

    def _translate_ollama(self, script: Script, target_language: str) -> Script:
        target_name = LANGUAGE_NAMES[target_language]
        sections: list[ScriptSection] = []
        for section in script.sections:
            translated = self.ollama.chat(
                [
                    {
                        "role": "system",
                        "content": (
                            "You are a professional audiovisual translator. Preserve meaning, tone, and spoken brevity. "
                            "Return only the translated spoken text, with no notes or quotation marks."
                        ),
                    },
                    {
                        "role": "user",
                        "content": f"Translate this narration into {target_name}:\n\n{section.text}",
                    },
                ],
                temperature=0.2,
            )
            translated = translated.strip().strip('"')
            if not translated:
                raise OllamaUnavailable(f"empty {target_name} translation")
            sections.append(replace(section, text=translated))
        return Script(
            topic=script.topic,
            language=target_language,
            target_duration=script.target_duration,
            sections=sections,
            engine=f"{script.engine} + ollama:{self.ollama.model}",
            cta=sections[-1].text,
            source_notes=script.source_notes,
        )
