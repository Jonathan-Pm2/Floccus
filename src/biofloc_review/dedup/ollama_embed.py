from __future__ import annotations

import logging
import os
from collections.abc import Iterable, Iterator
from pathlib import Path

import requests
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_exponential

log = logging.getLogger(__name__)


DEFAULT_OLLAMA_URL = "http://localhost:11434"
DEFAULT_MODEL = "qwen3-embedding:8b"
EMBEDDING_DIM = 4096


def _load_env_once() -> None:
    if getattr(_load_env_once, "_loaded", False):
        return
    here = Path(__file__).resolve()
    for candidate in (here.parents[3] / ".env", here.parents[4] / ".env"):
        if candidate.exists():
            load_dotenv(candidate, override=False)
            log.debug("Loaded env from %s", candidate)
            break
    _load_env_once._loaded = True  # type: ignore[attr-defined]


def ollama_url() -> str:
    _load_env_once()
    return os.environ.get("OLLAMA_URL", DEFAULT_OLLAMA_URL)


def ollama_headers() -> dict[str, str]:
    _load_env_once()
    headers = {"Content-Type": "application/json"}
    token = os.environ.get("OLLAMA_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def default_embed_model() -> str:
    _load_env_once()
    return os.environ.get("OLLAMA_EMBED_MODEL") or DEFAULT_MODEL


@retry(
    stop=stop_after_attempt(2),
    wait=wait_exponential(multiplier=1, min=1, max=8),
    reraise=True,
)
def embed_batch(texts: list[str], model: str | None = None, timeout: float = 60.0) -> list[list[float]]:
    if not texts:
        return []
    payload = {"model": model or default_embed_model(), "input": texts}
    r = requests.post(
        f"{ollama_url()}/api/embed",
        json=payload,
        headers=ollama_headers(),
        timeout=timeout,
    )
    r.raise_for_status()
    data = r.json()
    return data["embeddings"]


def chunked(iterable: Iterable, n: int) -> Iterator[list]:
    buf: list = []
    for x in iterable:
        buf.append(x)
        if len(buf) >= n:
            yield buf
            buf = []
    if buf:
        yield buf
