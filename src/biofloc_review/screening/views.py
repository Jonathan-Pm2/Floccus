"""Build the three input views (A=título, B=abstract, C=keywords+venue) from a PaperRecord.

A view returns ``None`` when the underlying metadata is empty or insufficient,
so the classifier can short-circuit to ABSTAIN without an LLM call.
"""
from __future__ import annotations

from biofloc_review.schema import PaperRecord

# Minimum significant words in an abstract before we trust the LLM to judge it.
# Aligned with prompt B's ABSTAIN clause ("menos de 30 palabras significativas").
_MIN_ABSTRACT_WORDS = 30

# Truncation cap for the abstract view. Qwen3:14b has 32k context but we want
# bounded token budget per call.
_ABSTRACT_MAX_CHARS = 2000


def view_title(rec: PaperRecord) -> str:
    """Vista A: solo el título (siempre presente, PaperRecord lo valida)."""
    return rec.title.strip()


def view_abstract(rec: PaperRecord) -> str | None:
    """Vista B: solo el abstract. None si está ausente o demasiado corto."""
    if not rec.abstract:
        return None
    abstract = rec.abstract.strip()
    if len(abstract.split()) < _MIN_ABSTRACT_WORDS:
        return None
    if len(abstract) > _ABSTRACT_MAX_CHARS:
        return abstract[:_ABSTRACT_MAX_CHARS] + "..."
    return abstract


def view_kw_venue(rec: PaperRecord) -> str | None:
    """Vista C: keywords + venue concatenados. None si ambos vacíos."""
    parts: list[str] = []
    if rec.keywords:
        parts.append(f"keywords: {', '.join(rec.keywords)}")
    if rec.venue:
        parts.append(f"venue: {rec.venue}")
    if not parts:
        return None
    return " | ".join(parts)
