"""Multi-tier fulltext PDF fetcher.

Given a record with enrichment metadata, tries to download the full text
PDF following this preference order:

  1. ``oa_pdf_url`` from OpenAlex (direct Open Access PDF).
  2. ``pmc_fulltext_url`` from Europe PMC (Open Access via EPMC).
  3. DOI resolved through the USAL EZproxy (institutional subscription).

Each tier is attempted in turn. The first tier that delivers a valid PDF
response (200 + ``application/pdf`` content type, non-empty body) wins.
On all tiers exhausted without success, the result is marked ``manual``
and the user must fetch the paper through a browser session.

Output files are written to ``out_dir / {record_id}.pdf``. A small
``FetchResult`` dataclass records the strategy that worked (or failed),
the source URL attempted, the HTTP status, and the local path when
successful.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

import requests

from biofloc_review.sources.ezproxy import rewrite_for_ezproxy

log = logging.getLogger(__name__)


# Minimum size in bytes for a download to be considered a valid PDF body.
# Some publishers return tiny stub pages or HTML error blobs with the
# wrong content-type; this gates against those.
_MIN_PDF_BYTES = 4096


@dataclass
class FetchResult:
    record_id: str
    status: str          # "ok" | "manual" | "error"
    strategy: str        # "oa_openalex" | "epmc" | "doi_ezproxy" | "none"
    url_attempted: str | None = None
    http_status: int | None = None
    pdf_path: Path | None = None
    bytes_downloaded: int = 0
    error: str | None = None
    attempts: list[dict] = field(default_factory=list)


def _is_pdf_response(r: requests.Response) -> bool:
    ct = (r.headers.get("content-type") or "").lower()
    if "application/pdf" in ct:
        return True
    # Some publishers serve PDFs as application/octet-stream with a
    # Content-Disposition filename ending in .pdf.
    cd = (r.headers.get("content-disposition") or "").lower()
    if "application/octet-stream" in ct and ".pdf" in cd:
        return True
    return False


def _save_pdf(r: requests.Response, out_path: Path) -> int:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with out_path.open("wb") as f:
        for chunk in r.iter_content(chunk_size=64 * 1024):
            if chunk:
                f.write(chunk)
                n += len(chunk)
    return n


def _try_download(session: requests.Session, url: str,
                   out_path: Path, timeout: float = 60.0,
                   stream: bool = True) -> tuple[bool, requests.Response, int]:
    """Single download attempt. Returns (ok, response, bytes_written)."""
    r = session.get(url, timeout=timeout, allow_redirects=True, stream=stream)
    if r.status_code != 200:
        return False, r, 0
    if not _is_pdf_response(r):
        return False, r, 0
    n = _save_pdf(r, out_path)
    if n < _MIN_PDF_BYTES:
        # Bogus / truncated body. Remove and treat as failure.
        try:
            out_path.unlink(missing_ok=True)
        except OSError:
            pass
        return False, r, n
    return True, r, n


def fetch_fulltext(
    record_id: str,
    oa_pdf_url: str | None,
    pmc_fulltext_url: str | None,
    doi: str | None,
    session: requests.Session,
    out_dir: Path,
    delay: float = 1.0,
) -> FetchResult:
    """Fetch the fulltext PDF for a single record using the multi-tier strategy."""
    out_path = out_dir / f"{record_id}.pdf"
    result = FetchResult(record_id=record_id, status="manual", strategy="none")

    # Tier 1 — OpenAlex direct OA
    if oa_pdf_url:
        try:
            ok, r, n = _try_download(session, oa_pdf_url, out_path)
            result.attempts.append({
                "strategy": "oa_openalex",
                "url": oa_pdf_url,
                "http_status": r.status_code,
                "content_type": r.headers.get("content-type"),
                "bytes": n,
                "ok": ok,
            })
            if ok:
                result.status = "ok"
                result.strategy = "oa_openalex"
                result.url_attempted = oa_pdf_url
                result.http_status = r.status_code
                result.pdf_path = out_path
                result.bytes_downloaded = n
                return result
        except Exception as e:
            log.debug("oa_openalex failed for %s: %s", record_id, e)
            result.attempts.append({
                "strategy": "oa_openalex", "url": oa_pdf_url,
                "error": f"{type(e).__name__}: {e}",
            })

    if delay:
        time.sleep(delay)

    # Tier 2 — Europe PMC
    if pmc_fulltext_url:
        try:
            ok, r, n = _try_download(session, pmc_fulltext_url, out_path)
            result.attempts.append({
                "strategy": "epmc",
                "url": pmc_fulltext_url,
                "http_status": r.status_code,
                "content_type": r.headers.get("content-type"),
                "bytes": n,
                "ok": ok,
            })
            if ok:
                result.status = "ok"
                result.strategy = "epmc"
                result.url_attempted = pmc_fulltext_url
                result.http_status = r.status_code
                result.pdf_path = out_path
                result.bytes_downloaded = n
                return result
        except Exception as e:
            log.debug("epmc failed for %s: %s", record_id, e)
            result.attempts.append({
                "strategy": "epmc", "url": pmc_fulltext_url,
                "error": f"{type(e).__name__}: {e}",
            })

    if delay:
        time.sleep(delay)

    # Tier 3 — DOI via EZproxy
    if doi:
        doi_url = f"https://doi.org/{doi}"
        proxy_url = rewrite_for_ezproxy(doi_url)
        try:
            ok, r, n = _try_download(session, proxy_url, out_path)
            result.attempts.append({
                "strategy": "doi_ezproxy",
                "url": proxy_url,
                "http_status": r.status_code,
                "content_type": r.headers.get("content-type"),
                "bytes": n,
                "ok": ok,
            })
            if ok:
                result.status = "ok"
                result.strategy = "doi_ezproxy"
                result.url_attempted = proxy_url
                result.http_status = r.status_code
                result.pdf_path = out_path
                result.bytes_downloaded = n
                return result
        except Exception as e:
            log.debug("doi_ezproxy failed for %s: %s", record_id, e)
            result.attempts.append({
                "strategy": "doi_ezproxy", "url": proxy_url,
                "error": f"{type(e).__name__}: {e}",
            })

    # All tiers exhausted. Cleanup any partial file.
    if out_path.exists():
        try:
            out_path.unlink(missing_ok=True)
        except OSError:
            pass
    return result
