"""Pipeline integrity validator.

Runs sanity checks on the canonical parquet to detect bugs analogous to
the IEEE ``Document Identifier`` collision that we found late in the
screening. The checks are designed to flag, per source:

  - **Source_id uniqueness**: ratio of unique ``source_id`` values to total
    records. A low ratio (<90 %) signals an export field that is not a
    per-paper identifier, which would silently collapse multiple papers
    onto the same ``record_id`` downstream.
  - **DOI population**: fraction of records with a non-null DOI. Some
    sources (e.g. SciELO) legitimately ship without DOIs; this is
    reported but not flagged as an error.
  - **Title uniqueness**: ratio of unique titles. Very low ratios may
    indicate duplicate ingestion of the same export.
  - **Document type distribution**: counts of distinct values. Sources
    where ``document_type`` is null for most records lose the
    deterministic I4 prefilter signal.
  - **Language distribution**: counts of distinct values. Helps detect
    if the export contains records outside the EN/ES/PT scope.

Output is a JSON report. Suspicious findings are printed with a leading
flag so the user can quickly act on them.

Usage:
  uv run python -m biofloc_review.run_validate_pipeline \\
    data/processed/biofloc_final_primary_v3.parquet
"""
from __future__ import annotations

import argparse
import json
import logging
from collections import Counter
from pathlib import Path

import pandas as pd

log = logging.getLogger(__name__)


# Uniqueness ratio below this is flagged as a likely export-key collision
# (à la IEEE Document Identifier).
SOURCE_ID_UNIQUENESS_THRESHOLD = 0.90


def _safe_unique(series: pd.Series) -> int:
    return int(series.dropna().nunique())


def validate_source(df_src: pd.DataFrame, source: str) -> dict:
    """Run checks on the subset of records from one source."""
    n = len(df_src)
    if n == 0:
        return {"source": source, "n": 0, "flags": []}

    flags: list[str] = []

    n_sid = _safe_unique(df_src.source_id)
    n_sid_null = int(df_src.source_id.isna().sum())
    sid_uniq_ratio = n_sid / max(n - n_sid_null, 1)
    if sid_uniq_ratio < SOURCE_ID_UNIQUENESS_THRESHOLD and n_sid > 0:
        flags.append(
            f"source_id uniqueness {sid_uniq_ratio:.0%} (< {SOURCE_ID_UNIQUENESS_THRESHOLD:.0%}) — "
            f"likely export-key collision: {n_sid} distinct values for {n - n_sid_null} non-null records"
        )

    n_doi = _safe_unique(df_src.doi)
    n_doi_null = int(df_src.doi.isna().sum())
    doi_coverage = (n - n_doi_null) / max(n, 1)

    n_title = _safe_unique(df_src.title)
    title_uniq_ratio = n_title / max(n, 1)
    if title_uniq_ratio < 0.95:
        flags.append(
            f"title uniqueness {title_uniq_ratio:.0%} (< 95%) — possible duplicate ingestion"
        )

    doctype_dist = df_src.document_type.value_counts(dropna=False).head(8).to_dict()
    # Normalize NaN → "(null)" for JSON friendliness
    doctype_dist = {("(null)" if pd.isna(k) else str(k)): int(v)
                    for k, v in doctype_dist.items()}
    doctype_null_frac = float(df_src.document_type.isna().sum()) / max(n, 1)
    if doctype_null_frac > 0.5:
        flags.append(
            f"document_type null in {doctype_null_frac:.0%} of records — "
            f"deterministic I4 prefilter unreliable for this source"
        )

    lang_dist = df_src.language.value_counts(dropna=False).head(8).to_dict()
    lang_dist = {("(null)" if pd.isna(k) else str(k)): int(v)
                 for k, v in lang_dist.items()}

    return {
        "source": source,
        "n": int(n),
        "source_id": {
            "unique": n_sid,
            "null_count": n_sid_null,
            "uniqueness_ratio": round(sid_uniq_ratio, 3),
        },
        "doi": {
            "unique": n_doi,
            "null_count": n_doi_null,
            "coverage": round(doi_coverage, 3),
        },
        "title": {
            "unique": n_title,
            "uniqueness_ratio": round(title_uniq_ratio, 3),
        },
        "document_type_top_values": doctype_dist,
        "document_type_null_fraction": round(doctype_null_frac, 3),
        "language_top_values": lang_dist,
        "flags": flags,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pipeline integrity validator — detect IEEE-like bugs per source"
    )
    parser.add_argument(
        "canonical_parquet", type=Path,
        help="Canonical parquet to validate (e.g. biofloc_final_primary_v3.parquet)",
    )
    parser.add_argument(
        "--out-json", type=Path,
        default=Path("data/processed/pipeline_validation.json"),
    )
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    log.info("Leyendo %s", args.canonical_parquet)
    df = pd.read_parquet(args.canonical_parquet)
    log.info("  %d rows totales, %d sources distintos",
             len(df), df.source.nunique())

    report: dict = {
        "total_rows": int(len(df)),
        "sources": {},
        "global_flags": [],
    }

    # Per-source validation
    for source in sorted(df.source.unique()):
        sub = df[df.source == source]
        result = validate_source(sub, source)
        report["sources"][source] = result

    # Global checks
    rid_collisions = df.duplicated(subset=["source", "source_id"]).sum()
    if rid_collisions > 0:
        report["global_flags"].append(
            f"{int(rid_collisions)} rows duplicate (source, source_id) pairs in canonical"
        )

    # Cross-source DOI duplicates (papers visible from multiple bases)
    doi_dup = df[df.doi.notna()].groupby("doi").source.nunique()
    cross_src_dup = int((doi_dup > 1).sum())
    report["global_summary"] = {
        "unique_dois": int(df.doi.dropna().nunique()),
        "papers_with_doi_in_multiple_sources": cross_src_dup,
    }

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    log.info("Reporte guardado → %s", args.out_json)

    # Print quick summary
    print()
    print("=" * 70)
    print(" Pipeline Validation Summary")
    print("=" * 70)
    for src, r in report["sources"].items():
        print(f"\n[{src}]  n={r['n']}")
        sid = r["source_id"]
        print(f"  source_id: {sid['unique']} únicos sobre {r['n'] - sid['null_count']} no-null  "
              f"(uniqueness {sid['uniqueness_ratio']:.0%})")
        doi = r["doi"]
        print(f"  DOI cobertura: {doi['coverage']:.0%}  ({doi['unique']} únicos)")
        ttl = r["title"]
        print(f"  Title uniqueness: {ttl['uniqueness_ratio']:.0%}  ({ttl['unique']} únicos)")
        if r["flags"]:
            for f in r["flags"]:
                print(f"  ⚠️  FLAG: {f}")
        else:
            print(f"  ✅ sin flags")

    if report["global_flags"]:
        print("\n[global flags]")
        for f in report["global_flags"]:
            print(f"  ⚠️  {f}")
    print(f"\nCross-source DOI duplicates (esperable): "
          f"{report['global_summary']['papers_with_doi_in_multiple_sources']} papers en >1 base")


if __name__ == "__main__":
    main()
