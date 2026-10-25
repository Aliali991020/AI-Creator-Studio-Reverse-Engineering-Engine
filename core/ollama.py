"""Minimal local Ollama client implemented with Python's standard library."""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Any


class OllamaUnavailable(RuntimeError):
    pass


class OllamaClient:
    def __init__(self, model: str | None = None, host: str | None = None, timeout: int = 180) -> None:
        self.model = model or os.getenv("OLLAMA_MODEL", "llama3.1:8b")
        self.host = (host or os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")).rstrip("/")
        if not self.host.startswith(("http://", "https://")):
            self.host = "http://" + self.host
        self.timeout = timeout

    def _request(self, endpoint: str, payload: dict[str, Any] | None = None, timeout: int | None = None) -> dict[str, Any]:
        data = json.dumps(payload).encode("utf-8") if payload is not None else None
        request = urllib.request.Request(
            self.host + endpoint,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST" if data is not None else "GET",
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout or self.timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError) as exc:
            raise OllamaUnavailable(f"Ollama is not available at {self.host}: {exc}") from exc

    def available(self) -> bool:
        try:
            self._request("/api/tags", timeout=1.0)
            return True
        except OllamaUnavailable:
            return False

    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.7,
        json_schema: dict[str, Any] | None = None,
    ) -> str:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": temperature},
        }
        if json_schema is not None:
            payload["format"] = json_schema
        response = self._request("/api/chat", payload)
        content = response.get("message", {}).get("content", "")
        if not content.strip():
            raise OllamaUnavailable("Ollama returned an empty response.")
        return content.strip()
