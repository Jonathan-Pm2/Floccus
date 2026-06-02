from __future__ import annotations

import html
import logging
import os
import re
import threading
import time
from typing import Any

import pyalex
from pyalex import Works
from rapidfuzz import fuzz
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)
from urllib3.exceptions import MaxRetryError

from biofloc_review.normalize import (
    coerce_year,
    normalize_doi,
    normalize_title,
    split_keywords,
)
from biofloc_review.schema import Corpus, PaperRecord, Source

log = logging.getLogger(__name__)

_DEFAULT_EMAIL = "REDACTED_EMAIL"
_MATCH_THRESHOLD = 85.0
_YEAR_TOLERANCE = 1


class _RateLimiter:
    def __init__(self, rps: float) -> None:
        self.min_interval = 1.0 / rps if rps > 0 else 0.0
        self._last = 0.0
        self._lock = threading.Lock()

    def wait(self) -> None:
        if self.min_interval <= 0:
            return
        with self._lock:
            now = time.monotonic()
            wait = self._last + self.min_interval - now
            if wait > 0:
                time.sleep(wait)
            self._last = time.monotonic()


def _rate_limit_rps() -> float:
    try:
        return float(os.environ.get("OPENALEX_RPS", "0.5"))
    except ValueError:
        return 0.5


_LIMITER = _RateLimiter(_rate_limit_rps())


def _configure_polite_pool() -> None:
    email = os.environ.get("OPENALEX_EMAIL", _DEFAULT_EMAIL)
    if email:
        pyalex.config.email = email


_HTML_TAG = re.compile(r"<[^>]+>")


def _clean_text(s: str) -> str:
    s = html.unescape(s)
    s = _HTML_TAG.sub("", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _reconstruct_abstract(inv_index: dict[str, list[int]] | None) -> str | None:
    if not inv_index:
        return None
    positions: dict[int, str] = {}
    for word, locs in inv_index.items():
        for loc in locs:
            positions[loc] = word
    if not positions:
        return None
    raw = " ".join(positions[i] for i in sorted(positions))
    return _clean_text(raw)


def _work_to_record(work: dict[str, Any], corpus: Corpus) -> PaperRecord | None:
    title = (work.get("title") or "").strip()
    if not title:
        return None
    doi_raw = work.get("doi")
    doi = normalize_doi(doi_raw) if doi_raw else None
    authors = [
        a.get("author", {}).get("display_name", "")
        for a in (work.get("authorships") or [])
    ]
    authors = [a for a in authors if a]
    abstract = _reconstruct_abstract(work.get("abstract_inverted_index"))
    venue_obj = work.get("primary_location", {}) or {}
    venue_src = (venue_obj.get("source") or {}) if isinstance(venue_obj, dict) else {}
    venue = venue_src.get("display_name") if isinstance(venue_src, dict) else None
    concepts = [c.get("display_name", "") for c in (work.get("concepts") or [])]
    concepts = [c for c in concepts if c]
    oa = (work.get("open_access") or {}).get("is_oa")
    lang = work.get("language")
    return PaperRecord(
        doi=doi,
        title=title,
        title_norm=normalize_title(title),
        authors=authors,
        year=coerce_year(work.get("publication_year")),
        venue=venue,
        document_type=work.get("type"),
        language=lang,
        abstract=abstract,
        keywords=concepts,
        open_access=oa,
        url=work.get("doi") or (work.get("primary_location", {}) or {}).get("landing_page_url"),
        source=Source.OPENALEX,
        source_id=work.get("id"),
        corpus=corpus,
    )


def search_openalex(query: str, corpus: Corpus, per_page: int = 25, max_results: int = 100) -> list[PaperRecord]:
    _configure_polite_pool()
    out: list[PaperRecord] = []
    pager = Works().search(query).paginate(per_page=per_page, n_max=max_results)
    for page in pager:
        for w in page:
            rec = _work_to_record(w, corpus=corpus)
            if rec:
                out.append(rec)
    return out


_RETRY = dict(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=2, min=2, max=60),
    retry=retry_if_exception_type((Exception, MaxRetryError)),
    reraise=True,
)


@retry(**_RETRY)
def _get_by_doi(doi: str) -> dict[str, Any]:
    _LIMITER.wait()
    return Works()[f"https://doi.org/{doi}"]


@retry(**_RETRY)
def _search_title(title: str, per_page: int = 5) -> list[dict[str, Any]]:
    _LIMITER.wait()
    return Works().search(title).get(per_page=per_page)


def enrich_via_openalex(record: PaperRecord) -> PaperRecord:
    _configure_polite_pool()
    if record.doi:
        try:
            work = _get_by_doi(record.doi)
            enriched = _work_to_record(work, corpus=record.corpus)
            if enriched and enriched.abstract:
                return _merge(record, enriched)
        except Exception as e:
            log.warning("OpenAlex DOI lookup failed for %s: %s: %s", record.doi, type(e).__name__, e)

    try:
        candidates = _search_title(record.title, per_page=5)
    except Exception as e:
        log.warning("OpenAlex search failed for %r: %s: %s", record.title[:60], type(e).__name__, e)
        return record

    best: tuple[float, dict[str, Any]] | None = None
    for c in candidates:
        cand_title = c.get("title") or ""
        score = fuzz.token_set_ratio(record.title_norm, normalize_title(cand_title))
        cand_year = coerce_year(c.get("publication_year"))
        if record.year and cand_year and abs(cand_year - record.year) > _YEAR_TOLERANCE:
            continue
        if best is None or score > best[0]:
            best = (score, c)

    if best is None or best[0] < _MATCH_THRESHOLD:
        return record

    enriched = _work_to_record(best[1], corpus=record.corpus)
    if not enriched:
        return record
    return _merge(record, enriched)


def _merge(original: PaperRecord, enriched: PaperRecord) -> PaperRecord:
    data = original.model_dump()
    if not data.get("doi") and enriched.doi:
        data["doi"] = enriched.doi
    if not data.get("abstract") and enriched.abstract:
        data["abstract"] = enriched.abstract
    if not data.get("keywords") and enriched.keywords:
        data["keywords"] = enriched.keywords
    if data.get("open_access") is None and enriched.open_access is not None:
        data["open_access"] = enriched.open_access
    if not data.get("venue") and enriched.venue:
        data["venue"] = enriched.venue
    if not data.get("document_type") and enriched.document_type:
        data["document_type"] = enriched.document_type
    if not data.get("language") and enriched.language:
        data["language"] = enriched.language
    if not data.get("url") and enriched.url:
        data["url"] = enriched.url
    sources = set(data.get("enriched_from") or [])
    sources.add(Source.OPENALEX.value)
    data["enriched_from"] = [Source(s) for s in sources]
    data["needs_enrichment"] = data.get("abstract") is None
    return PaperRecord.model_validate(data)
