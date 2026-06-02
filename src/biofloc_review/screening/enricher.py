"""Layer 2 metadata enrichment via OpenAlex.

For each record that advances past Layer 1 (INCLUDE + HUMAN_REVIEW), fetches
the FULL OpenAlex Work object and extracts metadata not used in Layer 1:

  - concepts: top topical tags (broader categories)
  - topics: newer, more specific topic hierarchy
  - keywords: curated keyword list
  - referenced_works_count: number of papers cited
  - cited_by_count: how many papers cite this one
  - abstract_full: complete abstract (not truncated)
  - is_oa, oa_url, pdf_url: open access status and PDF link if available

This module does NOT download PDFs. PDF download is reserved for Layer 3.

Reuses the rate limiter and polite-pool configuration from
sources/openalex.py to honor the same 0.5 rps default that protects us
from HTTP 429 bans (manuscript L527-545).
"""
from __future__ import annotations

import logging
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field

import requests
from pyalex import Works
from tenacity import retry, stop_after_attempt, wait_exponential

from biofloc_review.schema import PaperRecord
from biofloc_review.sources.openalex import (
    _LIMITER,
    _configure_polite_pool,
    _reconstruct_abstract,
)

log = logging.getLogger(__name__)


@dataclass
class EnrichedRecord:
    """OpenAlex + Europe PMC + arxiv enriched metadata for Layer 2 screening."""
    doi: str | None

    # --- OpenAlex ---
    oa_concepts: list[str] = field(default_factory=list)
    oa_topics: list[str] = field(default_factory=list)
    oa_keywords: list[str] = field(default_factory=list)
    oa_referenced_works_count: int = 0
    oa_cited_by_count: int = 0
    oa_abstract_full: str | None = None
    oa_is_oa: bool | None = None
    oa_url: str | None = None
    oa_pdf_url: str | None = None
    oa_publication_date: str | None = None
    oa_ok: bool = False

    # --- Europe PMC ---
    pmc_id: str | None = None
    pmc_pmid: str | None = None
    pmc_mesh_terms: list[str] = field(default_factory=list)
    pmc_has_fulltext: bool = False
    pmc_fulltext_url: str | None = None
    pmc_is_oa: bool | None = None
    pmc_ok: bool = False

    # --- arxiv ---
    arxiv_id: str | None = None
    arxiv_pdf_url: str | None = None
    arxiv_categories: list[str] = field(default_factory=list)
    arxiv_ok: bool = False

    # --- meta ---
    enriched: bool = False
    error: str | None = None


# --- OpenAlex ----------------------------------------------------------------


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=2, min=2, max=30),
    reraise=True,
)
def _get_work_by_doi(doi: str) -> dict:
    _LIMITER.wait()
    return Works()[f"https://doi.org/{doi}"]


def enrich_via_openalex(doi: str) -> dict:
    """Fetch OpenAlex Work by DOI. Returns dict of oa_* fields."""
    _configure_polite_pool()
    try:
        w = _get_work_by_doi(doi)
    except Exception as e:
        log.warning("OpenAlex lookup failed for %s: %s", doi, e)
        return {"oa_ok": False, "error": f"openalex: {type(e).__name__}: {e}"}

    oa = w.get("open_access") or {}
    primary = w.get("primary_location") or {}
    best = w.get("best_oa_location") or {}
    return {
        "oa_concepts": [c.get("display_name") for c in (w.get("concepts") or []) if c.get("display_name")],
        "oa_topics": [t.get("display_name") for t in (w.get("topics") or []) if t.get("display_name")],
        "oa_keywords": [k.get("display_name") for k in (w.get("keywords") or []) if k.get("display_name")],
        "oa_referenced_works_count": int(w.get("referenced_works_count") or 0),
        "oa_cited_by_count": int(w.get("cited_by_count") or 0),
        "oa_abstract_full": _reconstruct_abstract(w.get("abstract_inverted_index")),
        "oa_is_oa": oa.get("is_oa"),
        "oa_url": oa.get("oa_url"),
        "oa_pdf_url": primary.get("pdf_url") or best.get("pdf_url"),
        "oa_publication_date": w.get("publication_date"),
        "oa_ok": True,
    }


# --- Europe PMC --------------------------------------------------------------


_EPMC_SEARCH = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=2, min=2, max=30),
    reraise=True,
)
def enrich_via_pmc(doi: str) -> dict:
    """Query Europe PMC by DOI. Returns dict of pmc_* fields.

    Europe PMC is free with no auth required and generous rate limits
    (recommended max 2 req/s sustained). Returns MeSH terms and full-text
    availability for papers indexed in PMC or Europe PMC.
    """
    try:
        r = requests.get(
            _EPMC_SEARCH,
            params={
                "query": f'DOI:"{doi}"',
                "format": "json",
                "resultType": "core",
                "pageSize": 1,
            },
            timeout=15,
        )
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        log.warning("Europe PMC lookup failed for %s: %s", doi, e)
        return {"pmc_ok": False, "error": f"pmc: {type(e).__name__}: {e}"}

    results = (data.get("resultList") or {}).get("result") or []
    if not results:
        return {"pmc_ok": True}  # ok=True but no hit
    w = results[0]

    mesh = [
        m.get("descriptorName") for m in
        (w.get("meshHeadingList") or {}).get("meshHeading", [])
        if m.get("descriptorName")
    ]
    pmc_id = w.get("pmcid")
    pmid = w.get("pmid")
    # Fix: hasTextMinedTerms=Y solo significa que el abstract está minado por
    # NLP, NO que hay fulltext. La señal real es inEPMC/inPMC/pmcid.
    has_fulltext = (w.get("inEPMC") == "Y" or
                    w.get("inPMC") == "Y" or
                    bool(pmc_id))
    fulltext_url = _extract_pmc_fulltext_url(w, pmc_id)
    is_oa = w.get("isOpenAccess") == "Y"

    return {
        "pmc_id": pmc_id,
        "pmc_pmid": pmid,
        "pmc_mesh_terms": mesh,
        "pmc_has_fulltext": has_fulltext,
        "pmc_fulltext_url": fulltext_url,
        "pmc_is_oa": is_oa,
        "pmc_ok": True,
    }


# Style preference for fulltext URLs: PDF > HTML (HTML is harder to parse for
# equation extraction in the next pipeline phase).
_STYLE_PREF = ("pdf", "html")
# Availability codes that mean "free/open access without subscription":
#   OA = Open access; F = Free; FREE = Free
# Skip:
#   S = Subscription required
_FREE_AVAILABILITY = {"OA", "F", "FREE"}


def _extract_pmc_fulltext_url(work: dict, pmc_id: str | None) -> str | None:
    """Extract an OA fulltext URL from EPMC `fullTextUrlList`.

    Returns the first URL matching:
      availability_code ∈ {OA, F, FREE}
      AND document_style ∈ {pdf, html}
    Preferring pdf over html, and Europe_PMC/PMC sites over external.
    Falls back to the legacy PMC URL if no OA URL is found but pmc_id exists.
    """
    fl = (work.get("fullTextUrlList") or {}).get("fullTextUrl") or []
    if not fl:
        return f"https://europepmc.org/article/PMC/{pmc_id}" if pmc_id else None

    free_candidates = [
        u for u in fl
        if (u.get("availabilityCode") or "").upper() in _FREE_AVAILABILITY
        and (u.get("documentStyle") or "").lower() in _STYLE_PREF
    ]
    if not free_candidates:
        # Fallback: legacy URL when pmc_id known.
        return f"https://europepmc.org/article/PMC/{pmc_id}" if pmc_id else None

    # Sort by (style preference, site preference)
    def _rank(u: dict) -> tuple:
        style = (u.get("documentStyle") or "").lower()
        site = (u.get("site") or "").lower()
        style_idx = _STYLE_PREF.index(style) if style in _STYLE_PREF else 99
        site_idx = 0 if "europe_pmc" in site or "pmc" in site else 1
        return (style_idx, site_idx)

    free_candidates.sort(key=_rank)
    return free_candidates[0].get("url")


# --- arxiv -------------------------------------------------------------------


_ARXIV_QUERY = "http://export.arxiv.org/api/query"
_ATOM_NS = {"atom": "http://www.w3.org/2005/Atom",
            "arxiv": "http://arxiv.org/schemas/atom"}


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=2, min=2, max=30),
    reraise=True,
)
def enrich_via_arxiv(doi: str) -> dict:
    """Query arxiv API by DOI. Returns dict of arxiv_* fields.

    Most non-CS papers don't have arxiv versions, so misses are common
    and expected. When a match exists, we get the LaTeX-derived PDF
    which is much higher quality than publisher PDFs for equation
    extraction.

    arxiv recommends max 3 req/s with at least 3-second pauses between
    requests for batch use.
    """
    try:
        r = requests.get(
            _ARXIV_QUERY,
            params={"search_query": f"doi:{doi}", "max_results": 1},
            timeout=15,
        )
        r.raise_for_status()
        root = ET.fromstring(r.content)
    except Exception as e:
        log.warning("arxiv lookup failed for %s: %s", doi, e)
        return {"arxiv_ok": False, "error": f"arxiv: {type(e).__name__}: {e}"}

    entries = root.findall("atom:entry", _ATOM_NS)
    if not entries:
        return {"arxiv_ok": True}  # ok but no hit
    entry = entries[0]

    arxiv_id_full = entry.findtext("atom:id", default="", namespaces=_ATOM_NS)
    arxiv_id = arxiv_id_full.rsplit("/", 1)[-1] if arxiv_id_full else None
    pdf_url = None
    for link in entry.findall("atom:link", _ATOM_NS):
        if link.get("title") == "pdf":
            pdf_url = link.get("href")
            break
    categories = [
        c.get("term") for c in entry.findall("atom:category", _ATOM_NS)
        if c.get("term")
    ]

    return {
        "arxiv_id": arxiv_id,
        "arxiv_pdf_url": pdf_url,
        "arxiv_categories": categories,
        "arxiv_ok": True,
    }


# --- Coordinator -------------------------------------------------------------


_SOURCES = {
    "openalex": enrich_via_openalex,
    "pmc": enrich_via_pmc,
    "arxiv": enrich_via_arxiv,
}

DEFAULT_SOURCES: tuple[str, ...] = ("openalex", "pmc")
# arxiv is opt-in: it rate-limits aggressively (3s between calls) and our
# wastewater/biofloc corpus has <5% expected overlap with arxiv preprints,
# making the throughput cost not worth it for default Layer 2. Enable
# explicitly via the CLI flag --with-arxiv when needed (e.g. for the
# Poincaré-biplot paper that mines mathematical modeling literature).


def enrich_record(rec: PaperRecord,
                  sources: tuple[str, ...] = DEFAULT_SOURCES) -> EnrichedRecord:
    """Coordinate enrichment from the requested sources.

    Args:
        rec: PaperRecord with at least DOI populated.
        sources: ordered tuple of source names to query. Default:
            ("openalex", "pmc"). Add "arxiv" explicitly if you want to
            include it (slow, low hit rate for this corpus).
    """
    if not rec.doi:
        return EnrichedRecord(doi=None, error="no DOI on record")

    merged: dict = {"doi": rec.doi}
    errors: list[str] = []

    for name in sources:
        fn = _SOURCES.get(name)
        if fn is None:
            errors.append(f"unknown source: {name}")
            continue
        try:
            result = fn(rec.doi)
        except Exception as e:
            errors.append(f"{name}: {type(e).__name__}: {e}")
            continue
        if "error" in result:
            errors.append(result.pop("error"))
        merged.update(result)

    merged["enriched"] = any(
        merged.get(f"{name}_ok", False) for name in sources
    )
    if errors:
        merged["error"] = " | ".join(errors)

    return EnrichedRecord(**merged)
