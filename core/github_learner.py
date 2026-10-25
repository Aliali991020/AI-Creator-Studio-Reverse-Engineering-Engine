"""Zero-link public GitHub discovery with bounded, non-executing repository indexing.

Repositories are treated as untrusted reference material: this module clones and parses
files, but never installs dependencies, imports modules, or runs repository code.
"""
from __future__ import annotations

import ast
import json
import os
import re
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

from .utils import ROOT, StudioError, run_command, utc_now, write_json


class GitHubLearner:
    API = "https://api.github.com"
    MAX_FILE_BYTES = 512_000
    MAX_TOTAL_BYTES = 5_000_000
    MAX_FILES = 400
    INDEXABLE = {".py", ".md", ".rst", ".toml", ".yaml", ".yml", ".json", ".txt"}
    IGNORED_PARTS = {".git", "node_modules", "vendor", "dist", "build", "__pycache__", ".venv"}

    def __init__(self, storage_dir: Path | None = None) -> None:
        self.storage_dir = storage_dir or ROOT / ".studio"
        self.repository_dir = self.storage_dir / "repositories"
        self.index_dir = self.storage_dir / "github_index"
        self.repository_dir.mkdir(parents=True, exist_ok=True)
        self.index_dir.mkdir(parents=True, exist_ok=True)

    def search_repositories(self, query: str, limit: int = 3) -> list[dict[str, Any]]:
        query = query.strip()[:240]
        if not query:
            raise StudioError("A GitHub discovery query is required.")
        limit = max(1, min(5, int(limit)))
        qualified = f"{query} in:name,description,readme archived:false"
        url = self.API + "/search/repositories?" + urllib.parse.urlencode(
            {"q": qualified, "sort": "stars", "order": "desc", "per_page": limit}
        )
        headers = {
            "Accept": "application/vnd.github+json",
            "User-Agent": "local-ai-creator-studio/1.0",
            "X-GitHub-Api-Version": "2022-11-28",
        }
        # Use an already-configured environment credential if present; never prompt for one.
        token = os.getenv("GITHUB_TOKEN")
        if token:
            headers["Authorization"] = f"Bearer {token}"
        request = urllib.request.Request(url, headers=headers)
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:500]
            raise StudioError(f"GitHub search returned HTTP {exc.code}: {detail}") from exc
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError) as exc:
            raise StudioError(f"GitHub discovery failed: {exc}") from exc
        repositories = []
        for item in payload.get("items", []):
            full_name = str(item.get("full_name", ""))
            if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", full_name):
                continue
            repositories.append(
                {
                    "full_name": full_name,
                    "description": item.get("description") or "",
                    "stars": int(item.get("stargazers_count") or 0),
                    "license": (item.get("license") or {}).get("spdx_id") or "See repository",
                    "default_branch": item.get("default_branch") or "main",
                    "html_url": f"https://github.com/{full_name}",
                }
            )
        return repositories

    def learn(self, capability: str, *, query: str | None = None, limit: int = 2) -> dict[str, Any]:
        """Search, shallow-clone, index, and summarize without asking for repository URLs."""

        search_query = query or f"{capability} FOSS implementation language:Python"
        repositories = self.search_repositories(search_query, limit=limit)
        if not repositories:
            raise StudioError(f"No suitable public GitHub repositories found for {capability!r}.")
        indexed: list[dict[str, Any]] = []
        errors: list[str] = []
        for repository in repositories:
            try:
                path = self.clone_repository(repository["full_name"])
                index = self.index_repository(path, repository["full_name"], metadata=repository)
                indexed.append(index)
            except Exception as exc:
                errors.append(f"{repository['full_name']}: {exc}")
        if not indexed:
            raise StudioError("Repositories were found but none could be indexed: " + " | ".join(errors))
        report = {
            "capability": capability,
            "query": search_query,
            "created_at": utc_now(),
            "repositories": indexed,
            "errors": errors,
            "safety": "Reference only. No cloned code was installed, imported, or executed.",
        }
        write_json(self.index_dir / f"{self._safe_name(capability)}-discovery.json", report)
        return report

    def clone_repository(self, full_name: str) -> Path:
        if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", full_name):
            raise StudioError("Invalid GitHub repository identity.")
        destination = self.repository_dir / self._safe_name(full_name)
        if destination.exists():
            if not (destination / ".git").exists():
                raise StudioError(f"Refusing to overwrite non-repository path: {destination}")
            return destination
        # Construct the URL ourselves rather than trusting an arbitrary clone_url field.
        clone_url = f"https://github.com/{full_name}.git"
        run_command(
            [
                "git",
                "clone",
                "--depth",
                "1",
                "--filter=blob:none",
                "--single-branch",
                "--",
                clone_url,
                str(destination),
            ],
            timeout=420,
        )
        return destination

    def index_repository(
        self, repository_path: Path, full_name: str | None = None, *, metadata: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        repository_path = repository_path.resolve()
        if not repository_path.is_dir():
            raise StudioError(f"Repository directory does not exist: {repository_path}")
        full_name = full_name or repository_path.name
        files: list[dict[str, Any]] = []
        total_bytes = 0
        truncated = False
        for path in sorted(repository_path.rglob("*")):
            if len(files) >= self.MAX_FILES or total_bytes >= self.MAX_TOTAL_BYTES:
                truncated = True
                break
            if not path.is_file() or path.is_symlink():
                continue
            try:
                relative = path.relative_to(repository_path)
            except ValueError:
                continue
            if any(part in self.IGNORED_PARTS for part in relative.parts):
                continue
            if path.suffix.lower() not in self.INDEXABLE and path.name not in {"Dockerfile", "Makefile"}:
                continue
            try:
                size = path.stat().st_size
            except OSError:
                continue
            if size > self.MAX_FILE_BYTES or total_bytes + size > self.MAX_TOTAL_BYTES:
                continue
            try:
                content = path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            total_bytes += size
            entry: dict[str, Any] = {"path": str(relative), "bytes": size}
            if path.suffix.lower() == ".py":
                entry["symbols"] = self._python_symbols(content)
            elif path.suffix.lower() in {".md", ".rst"}:
                entry["headings"] = re.findall(r"^#{1,3}\s+(.+)$", content, flags=re.MULTILINE)[:12]
            files.append(entry)
        index = {
            "full_name": full_name,
            "local_path": str(repository_path),
            "indexed_at": utc_now(),
            "metadata": metadata or {},
            "file_count": len(files),
            "bytes_read": total_bytes,
            "truncated": truncated,
            "files": files,
            "execution_policy": "not_executed",
        }
        safe_name = self._safe_name(full_name)
        write_json(self.index_dir / f"{safe_name}.json", index)
        self._write_summary(index)
        return index

    @staticmethod
    def _python_symbols(source: str) -> list[dict[str, Any]]:
        try:
            tree = ast.parse(source)
        except (SyntaxError, ValueError):
            return []
        symbols: list[dict[str, Any]] = []
        for node in ast.walk(tree):
            if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                symbols.append(
                    {
                        "kind": "class" if isinstance(node, ast.ClassDef) else "function",
                        "name": node.name,
                        "line": node.lineno,
                        "summary": (ast.get_docstring(node) or "").strip().split("\n")[0][:240],
                    }
                )
                if len(symbols) >= 80:
                    break
        return symbols

    def _write_summary(self, index: dict[str, Any]) -> None:
        lines = [
            f"# FOSS reference: {index['full_name']}",
            "",
            "> Automatically indexed reference material. No repository code was executed.",
            "",
            f"- Files indexed: {index['file_count']}",
            f"- Index truncated by safety limits: {index['truncated']}",
            f"- Local reference: `{index['local_path']}`",
            "",
            "## Extracted implementation symbols",
            "",
        ]
        count = 0
        for file in index["files"]:
            for symbol in file.get("symbols", []):
                lines.append(
                    f"- `{file['path']}:{symbol['line']}` — **{symbol['kind']} `{symbol['name']}`**"
                    + (f": {symbol['summary']}" if symbol["summary"] else "")
                )
                count += 1
                if count >= 120:
                    break
            if count >= 120:
                break
        if count == 0:
            lines.append("No Python symbols were found within the indexing limits.")
        (self.index_dir / f"{self._safe_name(index['full_name'])}.md").write_text(
            "\n".join(lines) + "\n", encoding="utf-8"
        )

    @staticmethod
    def _safe_name(value: str) -> str:
        safe = re.sub(r"[^A-Za-z0-9_.-]+", "--", value).strip(".-")
        if not safe:
            raise StudioError("Could not create a safe local storage name.")
        return safe[:120]
