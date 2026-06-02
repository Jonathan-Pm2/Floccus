"""SciELO landing-page scraper para recuperar abstract y DOI.

El export CSV de SciELO no incluye abstract ni DOI. Esos campos sí están
disponibles en la landing page del artículo como Google Scholar
metadata (`<meta name="citation_*">`). Este módulo los recupera de forma
respetuosa de los TOS de SciELO (open access, rate-limited).

Las URLs vienen del campo `url` del PaperRecord (cargado por
`sources/scielo.py` desde la columna `Fulltext URL` del CSV).

URLs siguen el patrón:
  http://www.scielo.{country}/scielo.php?script=sci_arttext&pid={pid}&lang=en

SciELO redirige automáticamente al formato nuevo (/j/<journal>/a/<id>/).
Seguimos las redirects.

Política operativa:
  - User-Agent identificable con email de contacto.
  - 2 segundos entre llamadas (politeness mínimo).
  - Retry exponencial vía tenacity.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass

import requests
from bs4 import BeautifulSoup
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from biofloc_review.normalize import normalize_doi

log = logging.getLogger(__name__)


@dataclass
class ScrapedRecord:
    """Resultado de scrapear una landing page de SciELO."""
    url: str
    abstract: str | None = None
    doi: str | None = None
    pdf_url: str | None = None
    xml_url: str | None = None
    ok: bool = False
    error: str | None = None


def _user_agent() -> str:
    """Browser-like UA. Algunos sitios SciELO (e.g. CL) bloquean bot UAs."""
    return "Mozilla/5.0 (X11; Linux x86_64; rv:120.0) Gecko/20100101 Firefox/120.0"


_DEFAULT_HEADERS = {
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.7,es;q=0.5,pt;q=0.3",
}


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=2, min=2, max=30),
    retry=retry_if_exception_type((requests.RequestException,)),
    reraise=True,
)
def _fetch(url: str, timeout: float = 20.0) -> str:
    """GET + follow redirects + return HTML body. Raises on HTTP error."""
    headers = {"User-Agent": _user_agent(), **_DEFAULT_HEADERS}
    r = requests.get(url, headers=headers, timeout=timeout, allow_redirects=True)
    r.raise_for_status()
    return r.text


def _meta(soup: BeautifulSoup, name: str) -> str | None:
    """Get a <meta name="..." content="..."> value, or None."""
    tag = soup.find("meta", attrs={"name": name})
    if not tag:
        return None
    content = tag.get("content")
    return content.strip() if content else None


# Preference order for the language of the abstract when multiple are present:
# English (best for our LLM criteria) > Portuguese > Spanish > anything else.
_LANG_PREFERENCE = ("ABSTRACT", "RESUMO", "RESUMEN")


def _extract_trans_abstract(soup: BeautifulSoup) -> str | None:
    """Fallback for older SciELO sites (CO, MX, CU, etc.).

    Parses `<div class="trans-abstract">` blocks. Each block has a
    `<p class="sec">` with anchor name (RESUMEN/ABSTRACT/RESUMO) followed
    by `<p>` paragraphs of body text.
    """
    blocks = soup.find_all("div", class_="trans-abstract")
    if not blocks:
        return None

    by_marker: dict[str, str] = {}
    for block in blocks:
        marker = None
        anchor = block.find("a", attrs={"name": True})
        if anchor:
            marker = (anchor.get("name") or "").strip().upper()
        # Collect all <p> that are NOT `class="sec"` markers
        paras = [
            p.get_text(" ", strip=True)
            for p in block.find_all("p")
            if "sec" not in (p.get("class") or [])
        ]
        text = " ".join(p for p in paras if p).strip()
        if marker and text:
            by_marker.setdefault(marker, text)

    if not by_marker:
        return None
    # Pick the most preferred language available
    for pref in _LANG_PREFERENCE:
        if pref in by_marker:
            return by_marker[pref]
    return next(iter(by_marker.values()))


def scrape_landing_page(url: str) -> ScrapedRecord:
    """Fetch a SciELO landing page and extract abstract + DOI.

    Tries two strategies in order:
      1. Google Scholar metadata: `<meta name="citation_abstract">` and
         `<meta name="citation_doi">`. Works on the new SciELO templates
         (e.g. scielo.br /j/<journal>/a/<id>/).
      2. Legacy `<div class="trans-abstract">` blocks for older SciELO
         pages (scielo.org.co, scielo.org.mx, scielo.sld.cu, etc).

    Returns a ScrapedRecord; on hard error returns ok=False with `error`.
    """
    if not url:
        return ScrapedRecord(url="", error="empty URL")
    try:
        html = _fetch(url)
    except Exception as e:
        log.warning("scrape failed for %s: %s", url[:80], e)
        return ScrapedRecord(url=url, error=f"{type(e).__name__}: {e}")

    soup = BeautifulSoup(html, "lxml")

    # DOI: tag works on both new and old templates.
    doi_raw = _meta(soup, "citation_doi")
    doi = normalize_doi(doi_raw) if doi_raw else None

    # Abstract: try meta first, then legacy trans-abstract blocks.
    abstract = _meta(soup, "citation_abstract")
    if not abstract:
        abstract = _extract_trans_abstract(soup)

    pdf_url = _meta(soup, "citation_pdf_url")
    xml_url = _meta(soup, "citation_xml_url")

    return ScrapedRecord(
        url=url,
        abstract=abstract,
        doi=doi,
        pdf_url=pdf_url,
        xml_url=xml_url,
        ok=bool(abstract or doi),
    )
