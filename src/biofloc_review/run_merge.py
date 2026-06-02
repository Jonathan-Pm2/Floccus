from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import pandas as pd

from biofloc_review.dedup.merge_confirmed import merge_confirmed_duplicates
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
    parser = argparse.ArgumentParser(
        description="Apply Capa 3 DUPLICATE verdicts to merge clusters and produce canonical corpus"
    )
    parser.add_argument("full_parquet", type=Path, help="Full parquet from Capa 1 (e.g. <corpus>_dedup_full.parquet)")
    parser.add_argument("judged_pairs", type=Path, help="JSON of judged pairs from Capa 3")
    parser.add_argument("--out-primary", type=Path, default=None,
                        help="Output parquet with canonical primaries only. Default: <input>_final_primary.parquet")
    parser.add_argument("--out-full", type=Path, default=None,
                        help="Output parquet with all records, cluster_id consolidated. Default: <input>_final_full.parquet")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    out_primary = args.out_primary or args.full_parquet.with_name(
        args.full_parquet.stem.replace("_dedup_full", "") + "_final_primary.parquet"
    )
    out_full = args.out_full or args.full_parquet.with_name(
        args.full_parquet.stem.replace("_dedup_full", "") + "_final_full.parquet"
    )

    log.info("Reading parquet %s", args.full_parquet)
    records = _parquet_to_records(args.full_parquet)
    log.info("  %d records loaded", len(records))

    log.info("Reading judged pairs %s", args.judged_pairs)
    pairs = json.loads(args.judged_pairs.read_text())
    log.info("  %d pairs loaded", len(pairs))

    primaries, full_marked, stats = merge_confirmed_duplicates(records, pairs)

    _records_to_parquet(primaries, out_primary)
    log.info("Wrote %d primary records → %s", len(primaries), out_primary)

    _records_to_parquet(full_marked, out_full)
    log.info("Wrote %d full records → %s", len(full_marked), out_full)

    print(json.dumps(stats, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
