from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import pandas as pd

from biofloc_review.dedup.embeddings import (
    collection_name_for,
    ensure_collection,
    find_near_duplicate_pairs,
    index_records,
    qdrant_client,
)
from biofloc_review.schema import PaperRecord

log = logging.getLogger(__name__)


def _parquet_to_records(path: Path) -> list[PaperRecord]:
    df = pd.read_parquet(path)
    out: list[PaperRecord] = []
    for _, row in df.iterrows():
        d = {k: (None if isinstance(v, float) and pd.isna(v) else v) for k, v in row.to_dict().items()}
        out.append(PaperRecord.model_validate(d))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Capa 2: embed primaries into Qdrant and find near-duplicate pairs cross-cluster"
    )
    parser.add_argument("input", type=Path, help="Parquet of records to index (typically *_dedup_primary.parquet)")
    parser.add_argument("--corpus", required=True, choices=["biofloc", "biplot"])
    parser.add_argument("--recreate", action="store_true", help="Drop existing collection before indexing")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--threshold", type=float, default=0.93, help="Cosine sim threshold for near-duplicate")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--out-pairs", type=Path, default=None, help="Where to write candidate pairs JSON")
    parser.add_argument("--skip-search", action="store_true", help="Only index, do not search for pairs")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    if not args.input.exists():
        raise SystemExit(f"Input parquet not found: {args.input}")

    records = _parquet_to_records(args.input)
    log.info("Loaded %d records from %s", len(records), args.input)

    client = qdrant_client()
    coll = collection_name_for(args.corpus)
    ensure_collection(client, coll, recreate=args.recreate)

    n_indexed = index_records(client, coll, records, batch_size=args.batch_size)
    log.info("Indexed %d records", n_indexed)

    if args.skip_search:
        return

    pairs = find_near_duplicate_pairs(
        client, coll, records,
        threshold=args.threshold,
        top_k=args.top_k,
    )

    out_pairs = args.out_pairs or args.input.with_name(f"{args.input.stem}_near_dup_pairs.json")
    out_pairs.parent.mkdir(parents=True, exist_ok=True)
    with out_pairs.open("w", encoding="utf-8") as f:
        json.dump(pairs, f, ensure_ascii=False, indent=2)
    log.info("Wrote %d pairs → %s", len(pairs), out_pairs)

    summary = {
        "indexed": n_indexed,
        "candidate_pairs": len(pairs),
        "threshold": args.threshold,
        "top_score": pairs[0]["score"] if pairs else None,
        "median_score": pairs[len(pairs) // 2]["score"] if pairs else None,
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
