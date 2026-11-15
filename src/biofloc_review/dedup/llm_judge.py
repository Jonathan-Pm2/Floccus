from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass
from typing import Any

import requests
from tenacity import retry, stop_after_attempt, wait_exponential

from biofloc_review.dedup.ollama_embed import _load_env_once, ollama_headers, ollama_url

log = logging.getLogger(__name__)


DEFAULT_LLM_MODEL = "qwen3:14b"
ANTHROPIC_URL = "https://api.anthropic.com/v1/messages"
ANTHROPIC_VERSION = "2023-06-01"
MAX_OUTPUT_TOKENS = 400


def llm_provider() -> str:
    """Generation backend: ``ollama`` (default) or ``anthropic`` (LLM_PROVIDER).

    Embeddings always stay on Ollama; only text generation is switched.
    """
    _load_env_once()
    return (os.environ.get("LLM_PROVIDER") or "ollama").strip().lower()


def default_llm_model() -> str:
    _load_env_once()
    if llm_provider() == "anthropic":
        model = os.environ.get("ANTHROPIC_MODEL")
        if not model:
            raise RuntimeError("LLM_PROVIDER=anthropic requires ANTHROPIC_MODEL (or pass --model)")
        return model
    return os.environ.get("OLLAMA_LLM_MODEL") or os.environ.get("OLLAMA_MODEL") or DEFAULT_LLM_MODEL


SYSTEM_PROMPT = """Eres un experto en bibliometría que decide si dos registros bibliográficos describen el MISMO trabajo académico (duplicado entre bases) o son trabajos distintos.

Un par es DUPLICATE cuando: mismo DOI, o mismo título salvo formato/traducción/subtítulo, o claramente la misma investigación publicada en versiones distintas (preprint+publicado, traducción).

Un par es NOT_DUPLICATE cuando: mismo tema o área pero papers distintos, mismos autores pero estudios diferentes, mismo título genérico (ej. "Activated sludge") pero objetos distintos (capítulo de libro, entrada de enciclopedia, paper diferente con título corto).

Un par es UNCLEAR cuando la información disponible no permite decidir con confianza razonable.

Responde EXACTAMENTE con este formato:
VERDICT: DUPLICATE | NOT_DUPLICATE | UNCLEAR
CONFIDENCE: <0-10>
REASONING: <una frase breve explicando la decisión>
"""


JUDGMENT_TEMPLATE = """Compara estos dos registros y decide si son duplicados.

=== RECORD A ===
title: {a_title}
authors: {a_authors}
year: {a_year}
venue: {a_venue}
doi: {a_doi}
source: {a_source}
abstract: {a_abstract}

=== RECORD B ===
title: {b_title}
authors: {b_authors}
year: {b_year}
venue: {b_venue}
doi: {b_doi}
source: {b_source}
abstract: {b_abstract}

Similarity score (cosine of embeddings): {score:.4f}

Aplica el formato de respuesta indicado."""


@dataclass
class Verdict:
    verdict: str            # DUPLICATE | NOT_DUPLICATE | UNCLEAR
    confidence: int         # 0-10
    reasoning: str          # textual justification
    raw: str                # full LLM response


def _truncate(s: str | None, n: int) -> str:
    if not s:
        return "(none)"
    s = str(s).strip()
    return s if len(s) <= n else s[:n] + "..."


def _format_authors(authors: list[str] | None, max_n: int = 4) -> str:
    if not authors:
        return "(none)"
    a = list(authors)[:max_n]
    suffix = f", +{len(authors) - max_n} more" if len(authors) > max_n else ""
    return ", ".join(a) + suffix


def build_prompt(rec_a: dict, rec_b: dict, score: float) -> str:
    return JUDGMENT_TEMPLATE.format(
        a_title=_truncate(rec_a.get("title"), 200),
        a_authors=_format_authors(rec_a.get("authors")),
        a_year=rec_a.get("year") or "?",
        a_venue=_truncate(rec_a.get("venue"), 80),
        a_doi=rec_a.get("doi") or "(none)",
        a_source=rec_a.get("source") or "?",
        a_abstract=_truncate(rec_a.get("abstract"), 600),
        b_title=_truncate(rec_b.get("title"), 200),
        b_authors=_format_authors(rec_b.get("authors")),
        b_year=rec_b.get("year") or "?",
        b_venue=_truncate(rec_b.get("venue"), 80),
        b_doi=rec_b.get("doi") or "(none)",
        b_source=rec_b.get("source") or "?",
        b_abstract=_truncate(rec_b.get("abstract"), 600),
        score=score,
    )


_VERDICT_RE = re.compile(r"VERDICT\s*:\s*(DUPLICATE|NOT_DUPLICATE|UNCLEAR)", re.IGNORECASE)
_CONFIDENCE_RE = re.compile(r"CONFIDENCE\s*:\s*(\d+)", re.IGNORECASE)
_REASONING_RE = re.compile(r"REASONING\s*:\s*(.+?)(?:\n\n|\Z)", re.IGNORECASE | re.DOTALL)


def parse_verdict(text: str) -> Verdict:
    v_match = _VERDICT_RE.search(text)
    c_match = _CONFIDENCE_RE.search(text)
    r_match = _REASONING_RE.search(text)
    verdict = (v_match.group(1).upper() if v_match else "UNCLEAR")
    try:
        confidence = int(c_match.group(1)) if c_match else 0
    except ValueError:
        confidence = 0
    confidence = max(0, min(10, confidence))
    reasoning = (r_match.group(1).strip() if r_match else text.strip())[:1000]
    return Verdict(verdict=verdict, confidence=confidence, reasoning=reasoning, raw=text)


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=2, min=2, max=20),
    reraise=True,
)
def _ollama_chat(messages: list[dict], model: str, temperature: float = 0.0, timeout: float = 90.0) -> str:
    """Single chat entry point for every LLM call (name kept for compatibility).

    Dispatches to Anthropic when LLM_PROVIDER=anthropic, with the same
    prompts, temperature and output-token cap as the Ollama path.
    """
    if llm_provider() == "anthropic":
        return _anthropic_chat(messages, model=model, temperature=temperature, timeout=timeout)
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "think": False,
        "options": {"temperature": temperature, "num_predict": MAX_OUTPUT_TOKENS},
    }
    r = requests.post(
        f"{ollama_url()}/api/chat",
        json=payload,
        headers=ollama_headers(),
        timeout=timeout,
    )
    r.raise_for_status()
    data = r.json()
    return data.get("message", {}).get("content", "")


def _anthropic_chat(messages: list[dict], model: str, temperature: float, timeout: float) -> str:
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        raise RuntimeError("LLM_PROVIDER=anthropic requires ANTHROPIC_API_KEY")
    system = "\n\n".join(m["content"] for m in messages if m["role"] == "system")
    payload: dict[str, Any] = {
        "model": model,
        "messages": [m for m in messages if m["role"] != "system"],
        "max_tokens": MAX_OUTPUT_TOKENS,
    }
    # Some newer models reject `temperature`; ANTHROPIC_TEMPERATURE=none omits it
    # (the run is then not deterministic, unlike the temperature-0 Ollama runs).
    if (os.environ.get("ANTHROPIC_TEMPERATURE") or "").strip().lower() != "none":
        payload["temperature"] = temperature
    if system:
        payload["system"] = system
    r = requests.post(
        ANTHROPIC_URL,
        json=payload,
        headers={
            "x-api-key": key,
            "anthropic-version": ANTHROPIC_VERSION,
            "content-type": "application/json",
        },
        timeout=timeout,
    )
    r.raise_for_status()
    data = r.json()
    return "".join(b.get("text", "") for b in data.get("content", []) if b.get("type") == "text")


def judge_pair(rec_a: dict, rec_b: dict, score: float, model: str | None = None) -> Verdict:
    prompt = build_prompt(rec_a, rec_b, score)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]
    text = _ollama_chat(messages, model=model or default_llm_model())
    return parse_verdict(text)
