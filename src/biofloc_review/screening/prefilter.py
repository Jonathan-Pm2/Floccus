"""Deterministic evaluation of E5 (language), I4 (document_type) and D1
(abstract length). These do not require an LLM call.

Each function returns a (answer, evidence) tuple where answer is one of
"YES" | "NO" and evidence is a short string for the audit trail.
"""
from __future__ import annotations

import re

from biofloc_review.schema import PaperRecord


# ---------- E5: language not in {EN, ES, PT} ----------

# Tokens that identify a language string. Lowercased before matching.
# Sources: Scopus "Language of Original Document", WoS "Language",
# SciELO "Language(s)", OpenAlex "language" (ISO 639-1 code).
_LANG_IN_SCOPE = {
    "english", "en", "eng",
    "spanish", "es", "spa", "castellano", "castilian", "español",
    "portuguese", "pt", "por", "português",
}

_LANG_OUT_OF_SCOPE_HINT = {
    # Common languages we want to flag as out of scope when present.
    "chinese", "zh", "chi", "zho", "mandarin", "中文",
    "german", "de", "ger", "deu", "deutsch",
    "french", "fr", "fra", "fre", "français",
    "italian", "it", "ita",
    "japanese", "ja", "jpn",
    "korean", "ko", "kor",
    "russian", "ru", "rus", "русский",
    "polish", "pl", "pol",
    "turkish", "tr", "tur",
    "arabic", "ar", "ara",
    "dutch", "nl", "nld", "nederlands",
    "czech", "cs", "ces", "cze",
    "ukrainian", "uk", "ukr",
    "persian", "fa", "fas", "per",
    "hebrew", "he", "heb",
    "vietnamese", "vi", "vie",
    "indonesian", "id", "ind",
    "thai", "th", "tha",
    "greek", "el", "ell", "gre",
}


def _normalize_language_field(raw: str | None) -> list[str]:
    """Split a language field into individual lowercase tokens."""
    if not raw:
        return []
    # Sources sometimes write "English; Spanish" or "English, Spanish".
    parts = re.split(r"[;,/]| and ", str(raw).strip().lower())
    return [p.strip() for p in parts if p.strip()]


def is_excluded_by_language(rec: PaperRecord) -> tuple[str, str]:
    """E5 = YES means the article is in a language out of scope (exclude).

    Returns (answer, evidence):
      - ("YES", "language=Chinese")     → out of scope, EXCLUDE via E5
      - ("NO",  "language=English")     → in scope (or unknown), keep
    """
    tokens = _normalize_language_field(rec.language)
    if not tokens:
        # Unknown language: assume English (most common in corpus), don't exclude.
        return ("NO", "language=(none); assumed in-scope")

    in_scope = any(t in _LANG_IN_SCOPE or any(s in t for s in _LANG_IN_SCOPE) for t in tokens)
    if in_scope:
        return ("NO", f"language={rec.language!r} in {{EN, ES, PT}}")

    out_of_scope = any(t in _LANG_OUT_OF_SCOPE_HINT or any(s in t for s in _LANG_OUT_OF_SCOPE_HINT) for t in tokens)
    if out_of_scope:
        return ("YES", f"language={rec.language!r} out of scope")

    # Unrecognized but not in-scope: conservative → don't exclude, route to human via D-criteria.
    return ("NO", f"language={rec.language!r} unrecognized; assumed in-scope")


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
