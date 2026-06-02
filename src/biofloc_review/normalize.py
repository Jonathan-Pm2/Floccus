from __future__ import annotations

import re
import unicodedata

_DOI_PREFIXES = ("https://doi.org/", "http://doi.org/", "doi:", "DOI:")
_WS = re.compile(r"\s+")
_PUNCT = re.compile(r"[^\w\s]", flags=re.UNICODE)


def normalize_doi(doi: str | None) -> str | None:
    if not doi:
        return None
    s = str(doi).strip()
    if not s or s.lower() in {"nan", "none", "null"}:
        return None
    for p in _DOI_PREFIXES:
        if s.lower().startswith(p.lower()):
            s = s[len(p):]
            break
    s = s.strip().lower()
    if not s.startswith("10."):
        return None
    return s


def normalize_title(title: str | None) -> str:
    if not title:
        return ""
    s = unicodedata.normalize("NFKD", str(title))
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = s.lower()
    s = _PUNCT.sub(" ", s)
    s = _WS.sub(" ", s).strip()
    return s


def split_authors(raw: str | None, sep: str | None = None) -> list[str]:
    if not raw or str(raw).lower() in {"nan", "none", ""}:
        return []
    s = str(raw)
    if sep:
        parts = s.split(sep)
    else:
        if ";" in s:
            parts = s.split(";")
        elif " and " in s.lower():
            parts = re.split(r"\s+and\s+", s, flags=re.IGNORECASE)
        else:
            parts = [s]
    return [p.strip() for p in parts if p.strip()]


def split_keywords(raw: str | None) -> list[str]:
    if not raw or str(raw).lower() in {"nan", "none", ""}:
        return []
    s = str(raw)
    for sep in (";", "|", "\n"):
        if sep in s:
            return [k.strip() for k in s.split(sep) if k.strip()]
    if "," in s and len(s) > 40:
        return [k.strip() for k in s.split(",") if k.strip()]
    return [s.strip()] if s.strip() else []


def coerce_year(v) -> int | None:
    if v is None:
        return None
    try:
        s = str(v).strip()
        if not s or s.lower() in {"nan", "none"}:
            return None
        m = re.search(r"\b(19|20)\d{2}\b", s)
        if m:
            return int(m.group(0))
        return int(float(s))
    except (ValueError, TypeError):
        return None


def coerce_bool_oa(v) -> bool | None:
    if v is None:
        return None
    s = str(v).strip().lower()
    if s in {"", "nan", "none"}:
        return None
    if s in {"true", "yes", "open access", "gold", "green", "hybrid", "bronze", "diamond"}:
        return True
    if s in {"false", "no", "closed", "none", "not open access"}:
        return False
    if "open" in s:
        return True
    if "closed" in s:
        return False
    return None
