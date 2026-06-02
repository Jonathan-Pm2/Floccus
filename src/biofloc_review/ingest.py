from __future__ import annotations

import argparse
import json
import logging
import os
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

from biofloc_review.schema import Corpus, PaperRecord, Source
from biofloc_review.sources.ieee import load_ieee
from biofloc_review.sources.openalex import enrich_via_openalex
from biofloc_review.sources.scielo import load_scielo
from biofloc_review.sources.scopus import load_scopus
from biofloc_review.sources.wos import load_wos

log = logging.getLogger(__name__)


BIOFLOC_FILES = {
    "scopus": ["scopus_export_*.csv"],
    "wos":    ["WOS.xls"],
    "ieee":   ["IEEE.csv"],
    "scielo": ["Scielo.csv"],
}

BIPLOT_FILES = {
    "scopus": ["biplot/[Ss]copus_biplot.csv"],
    "wos":    ["biplot/[Ww][Oo][Ss]_biplot.xls"],
    "ieee":   ["biplot/[Ii][Ee][Ee][Ee]_biplot.csv"],
    "scielo": ["biplot/[Ss]cielo_biplot.csv"],
}


def _resolve(data_dir: Path, patterns: list[str]) -> list[Path]:
    found: list[Path] = []
    for pat in patterns:
        if any(ch in pat for ch in "*?["):
            found.extend(sorted(data_dir.glob(pat)))
        else:
            p = data_dir / pat
            if p.exists():
                found.append(p)
    return found


def ingest_corpus(data_dir: Path, corpus: Corpus) -> list[PaperRecord]:
    files_map = BIOFLOC_FILES if corpus is Corpus.BIOFLOC else BIPLOT_FILES
    records: list[PaperRecord] = []
    loader_for = {
        "scopus": load_scopus,
        "wos": load_wos,
        "ieee": load_ieee,
        "scielo": load_scielo,
    }
    for source_name, patterns in files_map.items():
        loader = loader_for[source_name]
        paths = _resolve(data_dir, patterns)
        if not paths:
            log.warning("No files for %s/%s under %s", corpus.value, source_name, data_dir)
            continue
        for p in paths:
            log.info("Loading %s from %s", source_name, p.name)
            try:
                batch = loader(p, corpus=corpus)
                records.extend(batch)
                log.info("  -> %d records", len(batch))
            except Exception as e:
                log.exception("Failed to load %s: %s", p, e)
    return records


def _safe_enrich(rec: PaperRecord) -> PaperRecord:
    try:
        return enrich_via_openalex(rec)
    except Exception as e:
        log.debug("enrich failed for %r: %s", rec.title[:50], e)
        return rec


def _already_attempted(rec: PaperRecord) -> bool:
    """Skip records that have abstract OR were already attempted (enriched_from non-empty)."""
    if rec.abstract:
        return True
    if rec.enriched_from:
        return True
    return False


def enrich_records(
    records: list[PaperRecord],
    limit: int | None = None,
    workers: int = 3,
    checkpoint_path: Path | None = None,
    checkpoint_every: int = 50,
) -> list[PaperRecord]:
    out = list(records)
    pending_indexes = [
        i for i, r in enumerate(records)
        if r.needs_enrichment and not _already_attempted(r)
    ]
    if limit is not None:
        pending_indexes = pending_indexes[:limit]
    total = len(pending_indexes)
    skipped = sum(1 for r in records if r.needs_enrichment and _already_attempted(r))
    log.info("Enriching %d records via OpenAlex (workers=%d, skipped already-attempted=%d)",
             total, workers, skipped)
    if total == 0:
        return out

    completed = 0
    hits = 0
    last_checkpoint = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(_safe_enrich, records[i]): i for i in pending_indexes}
        for fut in as_completed(futures):
            i = futures[fut]
            try:
                enriched = fut.result()
                out[i] = enriched
                if enriched.abstract and not records[i].abstract:
                    hits += 1
            except Exception as e:
                log.warning("enrich failed for record %d: %s", i, e)
            completed += 1
            if completed % 50 == 0 or completed == total:
                log.info("  enriched %d/%d (abstract hits: %d, hit-rate %.1f%%)",
                         completed, total, hits, 100.0 * hits / completed)
            if checkpoint_path and (completed - last_checkpoint) >= checkpoint_every:
                save_parquet(out, checkpoint_path)
                last_checkpoint = completed
    if checkpoint_path:
        save_parquet(out, checkpoint_path)
    return out


def records_to_dataframe(records: list[PaperRecord]) -> pd.DataFrame:
    rows = [r.model_dump(mode="json") for r in records]
    df = pd.DataFrame(rows)
    return df


def _parquet_to_records(path: Path) -> list[PaperRecord]:
    df = pd.read_parquet(path)
    out: list[PaperRecord] = []
    for _, row in df.iterrows():
        d = {k: (None if isinstance(v, float) and pd.isna(v) else v) for k, v in row.to_dict().items()}
        out.append(PaperRecord.model_validate(d))
    return out


def save_parquet(records: list[PaperRecord], path: Path) -> None:
    df = records_to_dataframe(records)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    log.info("Wrote %d records → %s", len(df), path)


def summarize(records: list[PaperRecord]) -> dict:
    by_source = Counter(r.source.value for r in records)
    by_corpus = Counter(r.corpus.value for r in records)
    missing_abstract = sum(1 for r in records if r.abstract is None)
    missing_doi = sum(1 for r in records if r.doi is None)
    enriched = sum(1 for r in records if r.enriched_from)
    return {
        "total": len(records),
        "by_source": dict(by_source),
        "by_corpus": dict(by_corpus),
        "missing_abstract": missing_abstract,
        "missing_doi": missing_doi,
        "enriched": enriched,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest sources into unified PaperRecord parquet")
    parser.add_argument("--data-dir", type=Path, default=Path(".."), help="Directory with source CSVs/xls")
    parser.add_argument("--out-dir", type=Path, default=Path("data/processed"), help="Where to write parquet files")
    parser.add_argument("--corpus", choices=["biofloc", "biplot", "both"], default="both")
    parser.add_argument("--enrich", action="store_true", help="Enrich records missing abstract via OpenAlex")
    parser.add_argument("--enrich-limit", type=int, default=None, help="Cap OpenAlex enrichment calls")
    parser.add_argument("--workers", type=int, default=2, help="Thread workers for parallel enrichment")
    parser.add_argument("--rps", type=float, default=0.5, help="Max OpenAlex requests per second (uses OPENALEX_RPS env var)")
    parser.add_argument("--checkpoint-every", type=int, default=50, help="Save parquet every N enrichments")
    parser.add_argument("--resume", action="store_true", help="Resume from existing parquet (skip records with abstract or enriched_from)")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    if args.rps is not None:
        os.environ["OPENALEX_RPS"] = str(args.rps)

    data_dir = args.data_dir.resolve()
    out_dir = args.out_dir.resolve()
    log.info("Data dir: %s", data_dir)
    log.info("Out dir : %s", out_dir)

    targets = [Corpus.BIOFLOC, Corpus.BIPLOT] if args.corpus == "both" else [Corpus(args.corpus)]

    all_records: list[PaperRecord] = []
    for c in targets:
        log.info("=== Corpus: %s ===", c.value)
        corpus_parquet = out_dir / f"{c.value}.parquet"
        if args.resume and corpus_parquet.exists():
            log.info("Resuming from existing parquet: %s", corpus_parquet)
            recs = _parquet_to_records(corpus_parquet)
            log.info("  %d records loaded", len(recs))
        else:
            recs = ingest_corpus(data_dir, c)
        if args.enrich:
            recs = enrich_records(
                recs,
                limit=args.enrich_limit,
                workers=args.workers,
                checkpoint_path=corpus_parquet,
                checkpoint_every=args.checkpoint_every,
            )
        save_parquet(recs, corpus_parquet)
        all_records.extend(recs)

    summary = summarize(all_records)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
