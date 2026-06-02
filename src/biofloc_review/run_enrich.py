"""CLI for Layer 2 metadata enrichment.

Reads a screened parquet (Layer 1 output), filters to records that advance
(INCLUDE + HUMAN_REVIEW), queries OpenAlex for each DOI, and writes an
enriched parquet that joins by record_id.

Pattern follows run_judge.py / run_screening.py: argparse + ThreadPoolExecutor
+ resumable + periodic checkpoint + summary JSON.

Usage:
  uv run python -m biofloc_review.run_enrich \\
    data/processed/biofloc_screened.parquet \\
    --workers 1 --rps 0.5
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

from biofloc_review.dedup.embeddings import _record_to_point_id
from biofloc_review.schema import PaperRecord
from biofloc_review.screening.enricher import (
    DEFAULT_SOURCES,
    EnrichedRecord,
    enrich_record,
)

log = logging.getLogger(__name__)


def _row_for(rid: str, e: EnrichedRecord) -> dict:
    return {
        "record_id": rid,
        "doi": e.doi,
        # OpenAlex
        "oa_concepts": e.oa_concepts,
        "oa_topics": e.oa_topics,
        "oa_keywords": e.oa_keywords,
        "oa_referenced_works_count": e.oa_referenced_works_count,
        "oa_cited_by_count": e.oa_cited_by_count,
        "oa_abstract_full": e.oa_abstract_full,
        "oa_is_oa": e.oa_is_oa,
        "oa_url": e.oa_url,
        "oa_pdf_url": e.oa_pdf_url,
        "oa_publication_date": e.oa_publication_date,
        "oa_ok": e.oa_ok,
        # Europe PMC
        "pmc_id": e.pmc_id,
        "pmc_pmid": e.pmc_pmid,
        "pmc_mesh_terms": e.pmc_mesh_terms,
        "pmc_has_fulltext": e.pmc_has_fulltext,
        "pmc_fulltext_url": e.pmc_fulltext_url,
        "pmc_is_oa": e.pmc_is_oa,
        "pmc_ok": e.pmc_ok,
        # arxiv
        "arxiv_id": e.arxiv_id,
        "arxiv_pdf_url": e.arxiv_pdf_url,
        "arxiv_categories": e.arxiv_categories,
        "arxiv_ok": e.arxiv_ok,
        # meta
        "enriched": e.enriched,
        "error": e.error,
    }


def _build_rid_index(full_parquet: Path) -> dict[str, PaperRecord]:
    """Read the canonical parquet and index records by their UUID5 point_id.

    record_id in the screened parquet was computed via
    dedup.embeddings._record_to_point_id, so we replay that mapping here.
    """
    df = pd.read_parquet(full_parquet)
    out: dict[str, PaperRecord] = {}
    for _, row in df.iterrows():
        d = {
            k: (None if isinstance(v, float) and pd.isna(v) else v)
            for k, v in row.to_dict().items()
        }
        try:
            rec = PaperRecord.model_validate(d)
        except Exception:
            continue
        out[_record_to_point_id(rec)] = rec
    return out


def _save_partial(rows: list[dict], out_path: Path) -> None:
    if not rows:
        return
    df = pd.DataFrame(rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Layer 2: enriquecer records con metadata completa de OpenAlex (sin descargar PDFs)"
    )
    parser.add_argument(
        "screened_parquet", type=Path,
        help="Parquet de Layer 1 (típicamente biofloc_screened.parquet)",
    )
    parser.add_argument(
        "--full-parquet", type=Path,
        default=Path("data/processed/biofloc_final_primary.parquet"),
        help="Parquet canónico (para tener title/doi/etc. originales)",
    )
    parser.add_argument(
        "--out", type=Path, default=None,
        help="Parquet de salida (default: data/processed/biofloc_enriched.parquet)",
    )
    parser.add_argument(
        "--decisions", nargs="+",
        default=["INCLUDE", "HUMAN_REVIEW"],
        help="Decisiones de Layer 1 que avanzan a Layer 2 (default: INCLUDE HUMAN_REVIEW)",
    )
    parser.add_argument("--workers", type=int, default=1,
                        help="Trabajadores concurrentes (default: 1; OpenAlex throttles fácil)")
    parser.add_argument("--rps", type=float, default=0.5,
                        help="OpenAlex requests/sec (default: 0.5; manuscript-validated)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limitar a N records (testing)")
    parser.add_argument("--resume", action="store_true",
                        help="Saltar record_id ya enriquecidos en el parquet de salida")
    parser.add_argument("--checkpoint-every", type=int, default=50)
    parser.add_argument("--with-arxiv", action="store_true",
                        help="Incluir arxiv en la enrichment (opt-in: arxiv rate-limita "
                             "agresivo y nuestro corpus tiene <5% overlap esperado).")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    if args.rps is not None:
        os.environ["OPENALEX_RPS"] = str(args.rps)

    out_path = args.out or Path("data/processed/biofloc_enriched.parquet")

    if not args.screened_parquet.exists():
        raise SystemExit(f"Screened parquet no encontrado: {args.screened_parquet}")
    if not args.full_parquet.exists():
        raise SystemExit(f"Full parquet no encontrado: {args.full_parquet}")

    log.info("Leyendo screened %s", args.screened_parquet)
    screened = pd.read_parquet(args.screened_parquet)
    advancing = screened[screened.final_decision_primary.isin(args.decisions)]
    log.info("  %d records advancing (%s)", len(advancing), "+".join(args.decisions))

    log.info("Indexando full parquet %s", args.full_parquet)
    by_rid = _build_rid_index(args.full_parquet)
    log.info("  %d records indexados", len(by_rid))

    rids_to_enrich = list(advancing.record_id.astype(str))

    already_done: set[str] = set()
    existing_rows: list[dict] = []
    if args.resume and out_path.exists():
        ex = pd.read_parquet(out_path)
        already_done = set(ex.record_id.astype(str).tolist())
        existing_rows = ex.to_dict(orient="records")
        log.info("  --resume: %d ya enriquecidos", len(already_done))

    pending = [rid for rid in rids_to_enrich if rid not in already_done]
    if args.limit is not None:
        pending = pending[: args.limit]
        log.info("  --limit: procesando solo %d", len(pending))

    log.info("Pendientes: %d (workers=%d, rps=%.1f)", len(pending), args.workers, args.rps)

    sources = tuple(DEFAULT_SOURCES) + (("arxiv",) if args.with_arxiv else ())
    log.info("Sources activas: %s", sources)

    def _enrich_one(rid: str) -> dict:
        rec = by_rid.get(rid)
        if not rec:
            return _row_for(rid, EnrichedRecord(doi=None, error="record_id no encontrado en full parquet"))
        return _row_for(rid, enrich_record(rec, sources=sources))

    rows: list[dict] = list(existing_rows)
    completed = 0
    last_checkpoint = 0
    t0 = time.time()

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(_enrich_one, rid): rid for rid in pending}
        for fut in as_completed(futures):
            rows.append(fut.result())
            completed += 1
            if completed % 25 == 0 or completed == len(pending):
                elapsed = time.time() - t0
                rate = completed / max(elapsed, 1e-6)
                eta = (len(pending) - completed) / max(rate, 1e-6)
                log.info("  %d/%d (%.2f rec/s, eta %.0fs)",
                         completed, len(pending), rate, eta)
            if (completed - last_checkpoint) >= args.checkpoint_every:
                _save_partial(rows, out_path)
                last_checkpoint = completed

    _save_partial(rows, out_path)
    log.info("Escritos %d enriquecidos → %s", len(rows), out_path)

    res = pd.DataFrame(rows)
    summary = {
        "total_attempted": len(res),
        "enriched_ok": int(res.oa_enriched.sum()) if "oa_enriched" in res else 0,
        "with_pdf_url": int(res.oa_pdf_url.notna().sum()) if "oa_pdf_url" in res else 0,
        "is_oa_true": int(res.oa_is_oa.fillna(False).sum()) if "oa_is_oa" in res else 0,
        "errors": int(res.oa_error.notna().sum()) if "oa_error" in res else 0,
        "elapsed_seconds": round(time.time() - t0, 1),
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
