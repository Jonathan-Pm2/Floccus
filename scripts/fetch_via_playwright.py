"""Layer 3 fulltext fetcher via Playwright + USAL EZproxy.

Strategy: USAL EZproxy does not persist its session through cookies, so
the previous "log in once, reuse forever" approach fails. Instead, this
script prompts for the user's USAL credentials once at startup (via
``getpass``; credentials live only in process memory) and re-authenticates
through the CAS login form whenever the browser is redirected to it.

For each pending HumanReview record with a DOI:

  1. Navigate to the EZproxy-rewritten DOI URL in a headless Chromium.
  2. If the page lands on the CAS login form (idUSAL), auto-fill the
     credentials and submit. Wait for the redirect back to the
     publisher.
  3. Once on the publisher's landing, intercept any response with
     ``Content-Type: application/pdf`` and save it to disk.

If no PDF response appears after settling, the record is marked
``playwright_miss`` with a reason. Results are written to
``data/processed/fulltext_log.parquet`` in place, with a checkpoint
every 10 records (resumable).

Credentials are NEVER written to disk. Pass them via environment
variables ``USAL_USER`` and ``USAL_PASS`` to skip the prompt
(useful for unattended runs), or let ``getpass`` ask interactively.
"""
from __future__ import annotations

import argparse
import os
import re
import sys
import time
from getpass import getpass
from pathlib import Path

import pandas as pd
from playwright.sync_api import sync_playwright, TimeoutError as PWTimeout, Page

from biofloc_review.sources.ezproxy import rewrite_for_ezproxy

ROOT = Path(__file__).resolve().parent.parent
LOG_PATH = ROOT / "data" / "processed" / "fulltext_log.parquet"
DEC_PATH = ROOT / "data" / "processed" / "biofloc_final_decisions.parquet"
OUT_DIR = ROOT / "data" / "fulltext"

MIN_PDF_BYTES = 4096
NAV_TIMEOUT_MS = 60_000
SETTLE_SECONDS = 5.0
CAS_HOST = "identidad.usal.es"


def get_credentials() -> tuple[str, str]:
    user = os.environ.get("USAL_USER")
    pwd = os.environ.get("USAL_PASS")
    if user and pwd:
        return user, pwd
    print("USAL credentials (kept in memory only, never written to disk):")
    user = input("  username: ").strip()
    pwd = getpass("  password: ")
    return user, pwd


def handle_cas_if_needed(page: Page, user: str, pwd: str) -> bool:
    """If page is on the CAS login form, submit credentials. Returns True if it handled CAS."""
    if CAS_HOST not in page.url:
        return False
    try:
        page.wait_for_selector("input[name='adAS_username']",
                                timeout=15_000, state="visible")
    except PWTimeout:
        return False
    page.fill("input[name='adAS_username']", user)
    page.fill("input[name='adAS_password']", pwd)
    page.click("input[name='adAS_submit']")
    try:
        page.wait_for_url(lambda u: CAS_HOST not in u, timeout=30_000)
    except PWTimeout:
        return False
    return True


def build_publisher_pdf_url(current_url: str, doi: str) -> str | None:
    """Construct the publisher PDF URL deterministically from the landing URL + DOI.

    Most publishers expose the article PDF behind a JS button (not a
    plain ``<a href>``), so HTML scraping is unreliable. But the PDF
    URL itself follows a fixed pattern per publisher domain. As long as
    the user is authenticated through EZproxy, navigating to the
    constructed URL serves the PDF.

    Patterns covered (handles ~80% of the HumanReview corpus):
      - ScienceDirect (Elsevier): /science/article/pii/{PII}/pdfft
      - linkinghub (Elsevier redirect): /retrieve/pii/{PII} -> ScienceDirect pdfft
      - MDPI: append /pdf to the article URL
      - Wiley OnlineLibrary: /doi/{doi} -> /doi/pdfdirect/{doi}
      - Springer Link: /article/{doi} -> /content/pdf/{doi}.pdf
      - Taylor & Francis: /doi/full/{doi} or /doi/abs/{doi} -> /doi/pdf/{doi}
    """
    url = current_url

    # ScienceDirect / Elsevier
    if "sciencedirect-com.usal" in url or "linkinghub-elsevier-com.usal" in url:
        # Find the PII either in the current URL or use the doi suffix
        m = re.search(r"/pii/(S\d[\d\w]+)", url)
        if m:
            pii = m.group(1)
            return (f"https://www-sciencedirect-com.usal.idm.oclc.org"
                    f"/science/article/pii/{pii}/pdfft")

    # MDPI
    if "mdpi-com.usal" in url:
        # Strip query/fragment, append /pdf
        base = url.split("?")[0].split("#")[0].rstrip("/")
        if not base.endswith("/pdf"):
            return base + "/pdf"
        return base

    # Wiley OnlineLibrary
    if "onlinelibrary-wiley-com.usal" in url and doi:
        return (f"https://onlinelibrary-wiley-com.usal.idm.oclc.org"
                f"/doi/pdfdirect/{doi}")

    # Springer Link
    if "link-springer-com.usal" in url and doi:
        return (f"https://link-springer-com.usal.idm.oclc.org"
                f"/content/pdf/{doi}.pdf")

    # Taylor & Francis
    if "tandfonline-com.usal" in url and doi:
        return (f"https://www-tandfonline-com.usal.idm.oclc.org"
                f"/doi/pdf/{doi}")

    return None


def find_self_pdf_link(page: Page, doi: str) -> str | None:
    """HTML-scrape fallback when the publisher pattern is unknown."""
    doi_tail = doi.split("/", 1)[-1].lower() if doi else ""
    pii = ""
    if "/pii/" in page.url:
        pii = page.url.split("/pii/", 1)[1].split("/")[0].split("?")[0]
    links = page.eval_on_selector_all(
        "a[href]",
        """els => els.map(a => a.href).filter(h => /\\/pdf/i.test(h))""",
    )
    if not links:
        return None
    for h in links:
        if doi_tail and doi_tail in h.lower():
            return h
    for h in links:
        if pii and pii in h:
            return h
    return links[0]


def download_via_navigation(page: Page, pdf_url: str,
                             out_path: Path) -> tuple[bool, int, str]:
    """Navigate the *page* itself to the PDF URL.

    This is more robust than ``context.request.get`` because anti-bot
    layers (Cloudflare, Akamai) typically allow same-page navigation
    that originates from the live tab but block ad-hoc request.get
    calls as suspicious. We attach a response listener and let the
    publisher serve the PDF inline.
    """
    captured: dict = {"saved": False, "bytes": 0, "url": "", "http": 0}

    def on_resp(resp):
        if captured["saved"]:
            return
        ct = (resp.headers.get("content-type") or "").lower()
        if "application/pdf" not in ct:
            return
        try:
            body = resp.body()
        except Exception:
            return
        if len(body) < MIN_PDF_BYTES:
            return
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(body)
        captured["saved"] = True
        captured["bytes"] = len(body)
        captured["url"] = resp.url
        captured["http"] = resp.status

    page.on("response", on_resp)
    last_http = 0
    try:
        try:
            r = page.goto(pdf_url, timeout=NAV_TIMEOUT_MS,
                          wait_until="domcontentloaded")
            last_http = r.status if r else 0
        except PWTimeout:
            pass
        except Exception:
            pass
        time.sleep(4.0)  # PDF.js viewer or async stream completion
    finally:
        page.remove_listener("response", on_resp)

    if captured["saved"]:
        return True, captured["bytes"], captured["url"]
    return False, 0, f"nav_no_pdf_http_{last_http}"


def fetch_pdf_for(page: Page, doi: str, out_path: Path,
                   user: str, pwd: str) -> tuple[bool, int, str]:
    """Navigate to the EZproxy-rewritten DOI; auto-CAS if needed; capture PDF.

    Two paths to a PDF body:
      (a) The publisher serves it inline at landing → captured by a
          response listener on the page.
      (b) The publisher serves an HTML landing with a PDF link → we
          locate the self-PDF link and download it via context.request
          (so we reuse the authenticated session cookies).
    """
    proxy_url = rewrite_for_ezproxy(f"https://doi.org/{doi}")
    captured: dict = {"saved": False, "bytes": 0, "url": proxy_url, "http": 0}

    def on_response(resp):
        if captured["saved"]:
            return
        ct = (resp.headers.get("content-type") or "").lower()
        if "application/pdf" not in ct:
            return
        try:
            body = resp.body()
        except Exception:
            return
        if len(body) < MIN_PDF_BYTES:
            return
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(body)
        captured["saved"] = True
        captured["bytes"] = len(body)
        captured["url"] = resp.url
        captured["http"] = resp.status

    page.on("response", on_response)

    nav_reason = ""
    try:
        page.goto(proxy_url, timeout=NAV_TIMEOUT_MS, wait_until="domcontentloaded")
        if handle_cas_if_needed(page, user, pwd):
            time.sleep(2.0)
    except PWTimeout:
        nav_reason = "nav_timeout"
    except Exception as e:
        nav_reason = type(e).__name__

    if not captured["saved"]:
        time.sleep(SETTLE_SECONDS)
    page.remove_listener("response", on_response)

    # Path (a): inline PDF response captured.
    if captured["saved"]:
        return True, captured["bytes"], captured["url"]

    # Path (b1): try the publisher-specific PDF URL pattern first.
    # This is deterministic and bypasses the JS-only "View PDF" buttons
    # that most modern publisher sites use.
    try:
        page.wait_for_timeout(1500)
        pdf_url = build_publisher_pdf_url(page.url, doi)
    except Exception:
        pdf_url = None
    if pdf_url:
        ok, nbytes, info = download_via_navigation(page, pdf_url, out_path)
        if ok:
            return True, nbytes, info
        # Fall through to HTML scraping if the pattern URL failed.

    # Path (b2): fall back to scraping a /pdf link from the HTML.
    try:
        pdf_link = find_self_pdf_link(page, doi)
    except Exception:
        pdf_link = None
    if pdf_link:
        ok, nbytes, info = download_via_navigation(page, pdf_link, out_path)
        if ok:
            return True, nbytes, info
        return False, 0, f"link_dl_fail:{info}"

    return False, 0, nav_reason or "no_pdf_link_found"


def run_fetch(limit: int | None, dry_run: bool = False,
               headless: bool = False) -> None:
    log = pd.read_parquet(LOG_PATH)
    dec = pd.read_parquet(DEC_PATH)
    pool = (dec[dec["final_decision"] == "HUMAN_REVIEW"]
            .merge(log[["record_id", "status"]], on="record_id", how="left"))
    pool = pool[(pool["status"] == "manual") & pool["doi"].notna()]
    if limit:
        pool = pool.head(limit)
    print(f"[fetch] Targets: {len(pool)} records (limit={limit or 'none'}, "
          f"headless={headless})")

    if dry_run:
        for _, row in pool.head(5).iterrows():
            print(f"  would fetch {row['record_id'][:12]} doi={row['doi']}")
        return

    user, pwd = get_credentials()

    log_idx = log.set_index("record_id")
    hits = 0
    with sync_playwright() as p:
        b = p.chromium.launch(headless=headless)
        ctx = b.new_context(accept_downloads=True)
        page = ctx.new_page()

        for i, (_, row) in enumerate(pool.iterrows(), 1):
            rid, doi = row["record_id"], row["doi"]
            out_path = OUT_DIR / f"{rid}.pdf"
            try:
                ok, nbytes, info = fetch_pdf_for(page, doi, out_path, user, pwd)
            except Exception as e:
                ok, nbytes, info = False, 0, f"{type(e).__name__}: {str(e)[:60]}"

            if ok:
                hits += 1
                log_idx.loc[rid, "status"] = "ok"
                log_idx.loc[rid, "strategy"] = "playwright_ezproxy"
                log_idx.loc[rid, "url_attempted"] = info
                log_idx.loc[rid, "http_status"] = 200
                log_idx.loc[rid, "pdf_path"] = str(out_path)
                log_idx.loc[rid, "bytes_downloaded"] = nbytes
                msg = f"OK {nbytes//1024}KB"
            else:
                log_idx.loc[rid, "strategy"] = "playwright_miss"
                log_idx.loc[rid, "error"] = info
                msg = f"MISS {info}"
            print(f"  [{i:>4}/{len(pool)}] {rid[:12]} {msg}")
            if i % 10 == 0:
                log_idx.reset_index().to_parquet(LOG_PATH, index=False)
            time.sleep(1.5)

        ctx.close()
        b.close()

    log_idx.reset_index().to_parquet(LOG_PATH, index=False)
    print(f"\n[fetch] Done. hits={hits}/{len(pool)} ({hits/max(1,len(pool))*100:.1f}%)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None,
                    help="Limit number of records (for smoke tests).")
    ap.add_argument("--dry-run", action="store_true",
                    help="Just list what would be fetched.")
    ap.add_argument("--headless", action="store_true",
                    help="Run Chromium headless (default: UI visible to "
                         "bypass publisher anti-bot challenges).")
    args = ap.parse_args()
    run_fetch(limit=args.limit, dry_run=args.dry_run, headless=args.headless)


if __name__ == "__main__":
    main()
