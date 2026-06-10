"""Deterministic evaluation of I4 (document_type) and D1 (abstract length).
These do not require an LLM call.

Each function returns a (answer, evidence) tuple where answer is one of
"YES" | "NO" and evidence is a short string for the audit trail.

Historical note: an earlier version also defined an E5 language-exclusion
criterion that filtered records whose ``language`` field was not English,
Spanish or Portuguese. The criterion was removed from the protocol after
a corpus review showed (i) zero records were excluded by it in practice,
because the source exports rarely tag non-Western papers explicitly, and
(ii) the Qwen3 judge handles Chinese, Korean, German, French and Arabic
abstracts natively, so an arbitrary language barrier was unjustified.
The active criterion set is I1-I4, E1-E4, D1-D3 (eleven total).
"""
from __future__ import annotations

from biofloc_review.schema import PaperRecord


# ---------- I4: research article or review ----------

_DOCTYPES_INCLUDED = {
    "article", "research article", "journal article", "journal-article",
    "review", "review article", "literature review",
}

_DOCTYPES_EXCLUDED = {
    "editorial", "editorial material", "editorial note",
    "letter", "note", "correspondence",
    "conference paper", "proceedings paper", "proceedings-article",
    "conference review",
    "book chapter", "book-chapter", "book", "book review",
    "erratum", "correction", "retraction", "retracted publication",
    "data paper", "short communication",
}


def is_eligible_doctype(rec: PaperRecord) -> tuple[str, str]:
    """I4 = YES means it's a research article or review.

    Returns ("YES" | "NO", evidence).
    """
    if not rec.document_type:
        # Conservative default: assume Article (most common). Don't filter out.
        return ("YES", "document_type=(none); assumed Article")
    dt = rec.document_type.strip().lower()
    if dt in _DOCTYPES_INCLUDED:
        return ("YES", f"document_type={rec.document_type!r}")
    if dt in _DOCTYPES_EXCLUDED:
        return ("NO", f"document_type={rec.document_type!r}")
    # Unknown type: conservative → preserve (route via D criteria if needed).
    return ("YES", f"document_type={rec.document_type!r} unrecognized; assumed eligible")


# ---------- D1: abstract too short ----------

_MIN_ABSTRACT_WORDS = 100


def is_abstract_too_short(rec: PaperRecord) -> tuple[str, str]:
    """D1 = YES means abstract is <100 words (route to human review).

    Returns ("YES" | "NO", evidence).
    """
    if not rec.abstract:
        return ("YES", "abstract=(none)")
    n_words = len(rec.abstract.split())
    if n_words < _MIN_ABSTRACT_WORDS:
        return ("YES", f"abstract={n_words} words (<{_MIN_ABSTRACT_WORDS})")
    return ("NO", f"abstract={n_words} words")
