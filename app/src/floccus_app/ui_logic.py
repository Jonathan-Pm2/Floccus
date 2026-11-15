"""Pure helpers behind the Streamlit interface (kept testable)."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from biofloc_review.protocol import (
    Protocol,
    ProtocolError,
    dump_protocol,
    load_protocol,
    protocol_from_dict,
)

COLUMNS = ("id", "type", "question", "justification", "positive_question", "check", "params")


def load_project_protocol(project: Path) -> Protocol:
    return load_protocol(project / "protocol.yaml")


def protocol_to_rows(p: Protocol) -> list[dict[str, str]]:
    rows = []
    for c in p.criteria:
        check = p.checks.get(c.id)
        params = ""
        if check and check.params:
            params = yaml.safe_dump(dict(check.params), default_flow_style=True, allow_unicode=True).strip()
        rows.append({
            "id": c.id,
            "type": c.type,
            "question": c.question,
            "justification": c.justification,
            "positive_question": c.positive_question or "",
            "check": check.name if check else "",
            "params": params,
        })
    return rows


def _blank(value: Any) -> bool:
    return value is None or (isinstance(value, float) and value != value) or str(value).strip() == ""


def rows_to_protocol(name: str, language: str, topic: str, rows: list[dict[str, Any]]) -> Protocol:
    """Build a protocol from table rows; raises ProtocolError with a clear message."""
    entries = []
    for row in rows:
        if all(_blank(row.get(k)) for k in COLUMNS):
            continue  # empty row added by the table editor
        entry: dict[str, Any] = {k: row.get(k) for k in ("id", "type", "question", "justification")}
        if not _blank(row.get("positive_question")):
            entry["positive_question"] = row["positive_question"]
        if not _blank(row.get("check")):
            entry["check"] = row["check"]
            if not _blank(row.get("params")):
                try:
                    entry["params"] = yaml.safe_load(str(row["params"]))
                except yaml.YAMLError as exc:
                    raise ProtocolError(f"{row.get('id')}: params is not valid YAML") from exc
        entries.append(entry)
    return protocol_from_dict({"name": name, "language": language, "topic": topic, "criteria": entries})


def save_rows(project: Path, name: str, language: str, topic: str, rows: list[dict[str, Any]]) -> str | None:
    """Write protocol.yaml when valid; return the error message otherwise (file untouched)."""
    try:
        p = rows_to_protocol(name, language, topic, rows)
    except ProtocolError as exc:
        return str(exc)
    tmp = project / "protocol.yaml.tmp"
    tmp.write_text(dump_protocol(p), encoding="utf-8")
    tmp.replace(project / "protocol.yaml")
    return None


def llm_settings() -> dict[str, str]:
    """Current LLM configuration from the environment; the API key is never returned."""
    from biofloc_review.dedup.llm_judge import llm_provider
    from biofloc_review.dedup.ollama_embed import _load_env_once

    _load_env_once()
    provider = llm_provider()
    if provider == "anthropic":
        return {
            "provider": "anthropic",
            "model": os.environ.get("ANTHROPIC_MODEL") or "(not set)",
            "url": "https://api.anthropic.com",
            "api_key": "set" if os.environ.get("ANTHROPIC_API_KEY") else "not set",
        }
    return {
        "provider": "ollama",
        "model": os.environ.get("OLLAMA_LLM_MODEL") or os.environ.get("OLLAMA_MODEL") or "qwen3:14b",
        "url": os.environ.get("OLLAMA_URL") or "http://localhost:11434",
        "api_key": "set" if os.environ.get("OLLAMA_TOKEN") else "not set",
    }


def step_command(home: Path, project: str, step: str, limit: int | None = None) -> list[str]:
    """Command line that runs one pipeline step for a project (used by the UI)."""
    import sys

    argv = [sys.executable, "-m", "floccus_app.cli", "--home", str(home), "run", project, step]
    if limit:
        argv += ["--limit", str(limit)]
    return argv


def launch_step(home: Path, project: str, step: str, limit: int | None = None) -> None:
    """Start the step in the background; output goes to runs/<step>.log."""
    import subprocess

    log_path = home / project / "runs" / f"{step}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w", encoding="utf-8") as log:
        subprocess.Popen(step_command(home, project, step, limit), stdout=log, stderr=subprocess.STDOUT,
                         start_new_session=True)


def log_tail(home: Path, project: str, step: str, lines: int = 15) -> str:
    path = home / project / "runs" / f"{step}.log"
    if not path.is_file():
        return ""
    return "\n".join(path.read_text(encoding="utf-8", errors="replace").splitlines()[-lines:])
