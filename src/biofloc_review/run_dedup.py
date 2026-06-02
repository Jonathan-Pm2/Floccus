from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import pandas as pd

from biofloc_review.dedup import dedup_deterministic
from biofloc_review.schema import PaperRecord

log = logging.getLogger(__name__)


def _parquet_to_records(path: Path) -> list[PaperRecord]:
    df = pd.read_parquet(path)
    out: list[PaperRecord] = []
    for _, row in df.iterrows():
        d = {k: (None if isinstance(v, float) and pd.isna(v) else v) for k, v in row.to_dict().items()}
        out.append(PaperRecord.model_validate(d))
    return out


def _records_to_parquet(records: list[PaperRecord], path: Path) -> None:
    df = pd.DataFrame([r.model_dump(mode="json") for r in records])
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Capa 1 (deterministic) deduplication on a parquet corpus")
    parser.add_argument("input", type=Path, help="Input parquet with PaperRecord rows")
    parser.add_argument("--out-primary", type=Path, default=None, help="Output parquet (only primary per cluster). Default: <input>_dedup_primary.parquet")
    parser.add_argument("--out-full", type=Path, default=None, help="Output parquet (full corpus with cluster_id + is_primary marked). Default: <input>_dedup_full.parquet")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    if not args.input.exists():
        raise SystemExit(f"Input parquet not found: {args.input}")

    out_primary = args.out_primary or args.input.with_name(f"{args.input.stem}_dedup_primary.parquet")
    out_full = args.out_full or args.input.with_name(f"{args.input.stem}_dedup_full.parquet")

    log.info("Reading %s", args.input)
    records = _parquet_to_records(args.input)
    log.info("  %d records loaded", len(records))

    log.info("Running deterministic dedup (Capa 1)...")
    primaries, full_marked, stats = dedup_deterministic(records)

    _records_to_parquet(primaries, out_primary)
    log.info("  primary records → %s (%d rows)", out_primary, len(primaries))

    _records_to_parquet(full_marked, out_full)
    log.info("  full marked      → %s (%d rows)", out_full, len(full_marked))

    print(json.dumps(stats, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
