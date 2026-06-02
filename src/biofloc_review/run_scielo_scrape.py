"""CLI: scrapear landing pages de SciELO para recuperar abstract + DOI.

Reads the canonical parquet, identifies SciELO records that lack abstract
(by source=='scielo' and abstract is None/empty), fetches each landing
page, extracts abstract and DOI via Google Scholar metadata tags, and
writes:

  (1) `biofloc_scielo_scraped.parquet` — scraping results audit trail
      (one row per scraped URL with abstract, doi, pdf_url, xml_url, ok, error).
  (2) `biofloc_final_primary_v2.parquet` — canonical parquet con abstract/doi
      recuperados para SciELO. Listo para re-screening con run_screening.

Política operativa:
  - 2 seg entre llamadas por defecto (`--delay`).
  - User-Agent identificable con email.
  - Resumable: `--resume` salta record_id ya presentes en el parquet (1).

Usage:
  uv run python -m biofloc_review.run_scielo_scrape \\
    data/processed/biofloc_final_primary.parquet --delay 2.0
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import pandas as pd

from biofloc_review.dedup.embeddings import _record_to_point_id
from biofloc_review.schema import PaperRecord
from biofloc_review.sources.scielo_scraper import ScrapedRecord, scrape_landing_page

log = logging.getLogger(__name__)


def _parquet_to_records(path: Path) -> list[PaperRecord]:
    df = pd.read_parquet(path)
    out: list[PaperRecord] = []
    for _, row in df.iterrows():
        d = {
            k: (None if isinstance(v, float) and pd.isna(v) else v)
            for k, v in row.to_dict().items()
        }
        out.append(PaperRecord.model_validate(d))
    return out


def _row_for(rid: str, scraped: ScrapedRecord) -> dict:
    return {
        "record_id": rid,
        "url": scraped.url,
        "abstract_scraped": scraped.abstract,
        "doi_scraped": scraped.doi,
        "pdf_url_scraped": scraped.pdf_url,
        "xml_url_scraped": scraped.xml_url,
        "scrape_ok": scraped.ok,
        "scrape_error": scraped.error,
    }


def _save_partial(rows: list[dict], out_path: Path) -> None:
    if not rows:
        return
    df = pd.DataFrame(rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)


def _merge_into_canonical(
    canonical_records: list[PaperRecord],
    scraped_by_rid: dict[str, ScrapedRecord],
) -> tuple[list[PaperRecord], dict]:
    """Merge scraped abstracts and DOIs into PaperRecord objects.

    Returns (updated_records, stats).
    """
    updated: list[PaperRecord] = []
    n_abstract_added = 0
    n_doi_added = 0
    for rec in canonical_records:
        rid = _record_to_point_id(rec)
        scraped = scraped_by_rid.get(rid)
        if not scraped or not scraped.ok:
            updated.append(rec)
            continue
        data = rec.model_dump()
        changed = False
        if not data.get("abstract") and scraped.abstract:
            data["abstract"] = scraped.abstract
            data["needs_enrichment"] = False
            n_abstract_added += 1
            changed = True
        if not data.get("doi") and scraped.doi:
            data["doi"] = scraped.doi
            n_doi_added += 1
            changed = True
        if changed:
            # Track that this record was enriched from scielo-scrape.
            enriched_from = list(data.get("enriched_from") or [])
            # PaperRecord enriched_from is list[Source]; we use the scielo
            # value because the scrape IS from SciELO. The original record
            # was also scielo, so it's a self-enrichment from the landing.
            updated.append(PaperRecord.model_validate(data))
        else:
            updated.append(rec)
    stats = {
        "n_abstract_added": n_abstract_added,
        "n_doi_added": n_doi_added,
    }
    return updated, stats


def _records_to_parquet(records: list[PaperRecord], path: Path) -> None:
    df = pd.DataFrame([r.model_dump(mode="json") for r in records])
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Scrapear SciELO landing pages para recuperar abstract + DOI"
    )
    parser.add_argument("canonical_parquet", type=Path,
                        help="Parquet canónico (biofloc_final_primary.parquet)")
    parser.add_argument("--scraped-out", type=Path,
                        default=Path("data/processed/biofloc_scielo_scraped.parquet"),
                        help="Parquet con los resultados crudos del scrape (audit trail)")
    parser.add_argument("--canonical-out", type=Path,
                        default=Path("data/processed/biofloc_final_primary_v2.parquet"),
                        help="Parquet canónico actualizado con abstracts/DOIs SciELO")
    parser.add_argument("--delay", type=float, default=2.0,
                        help="Segundos entre calls a SciELO (politeness; default: 2.0)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limitar a N records (testing)")
    parser.add_argument("--resume", action="store_true",
                        help="Saltar record_id ya presentes en --scraped-out")
    parser.add_argument("--checkpoint-every", type=int, default=50)
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    if not args.canonical_parquet.exists():
        raise SystemExit(f"Parquet no encontrado: {args.canonical_parquet}")

    log.info("Leyendo %s", args.canonical_parquet)
    records = _parquet_to_records(args.canonical_parquet)
    log.info("  %d records totales", len(records))

    sci = [r for r in records if r.source.value == "scielo" and not r.abstract]
    log.info("  %d SciELO sin abstract (candidatos a scrape)", len(sci))

    already_done: set[str] = set()
    existing_rows: list[dict] = []
    if args.resume and args.scraped_out.exists():
        ex = pd.read_parquet(args.scraped_out)
        already_done = set(ex.record_id.astype(str).tolist())
        existing_rows = ex.to_dict(orient="records")
        log.info("  --resume: %d ya scrapeados", len(already_done))

    pending = [r for r in sci if _record_to_point_id(r) not in already_done]
    if args.limit:
        pending = pending[: args.limit]
        log.info("  --limit: %d records a scrapear", len(pending))

    log.info("Pendientes: %d (delay=%.1fs)", len(pending), args.delay)

    rows: list[dict] = list(existing_rows)
    completed = 0
    last_checkpoint = 0
    n_ok = 0
    n_err = 0
    t0 = time.time()

    for rec in pending:
        rid = _record_to_point_id(rec)
        scraped = scrape_landing_page(rec.url or "")
        rows.append(_row_for(rid, scraped))
        if scraped.ok:
            n_ok += 1
        else:
            n_err += 1
        completed += 1
        if completed % 25 == 0 or completed == len(pending):
            elapsed = time.time() - t0
            rate = completed / max(elapsed, 1e-6)
            eta = (len(pending) - completed) / max(rate, 1e-6)
            log.info("  scraped %d/%d (ok=%d, err=%d, %.2f rec/s, eta %.0fs)",
                     completed, len(pending), n_ok, n_err, rate, eta)
        if (completed - last_checkpoint) >= args.checkpoint_every:
            _save_partial(rows, args.scraped_out)
            last_checkpoint = completed
        time.sleep(args.delay)

    _save_partial(rows, args.scraped_out)
    log.info("Escritos %d resultados → %s", len(rows), args.scraped_out)

    # Merge into canonical
    scraped_by_rid = {
        row["record_id"]: ScrapedRecord(
            url=row.get("url") or "",
            abstract=row.get("abstract_scraped"),
            doi=row.get("doi_scraped"),
            pdf_url=row.get("pdf_url_scraped"),
            xml_url=row.get("xml_url_scraped"),
            ok=bool(row.get("scrape_ok")),
            error=row.get("scrape_error"),
        )
        for row in rows
    }
    updated, merge_stats = _merge_into_canonical(records, scraped_by_rid)
    _records_to_parquet(updated, args.canonical_out)
    log.info("Parquet canónico v2 → %s (abstracts añadidos: %d, DOIs añadidos: %d)",
             args.canonical_out, merge_stats["n_abstract_added"],
             merge_stats["n_doi_added"])

    summary = {
        "scielo_total_in_canonical": sum(1 for r in records if r.source.value == "scielo"),
        "scielo_without_abstract_before": len(sci),
        "scraped_this_run": completed,
        "scrape_ok": n_ok,
        "scrape_err": n_err,
        "abstracts_added": merge_stats["n_abstract_added"],
        "dois_added": merge_stats["n_doi_added"],
        "elapsed_seconds": round(time.time() - t0, 1),
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
