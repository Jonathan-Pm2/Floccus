"""Consolida decisiones finales y genera conteos PRISMA.

Combina Layer 1 (biofloc_screened.parquet) + Layer 2 LLM
(biofloc_layer2_screened.parquet) en una decisión consolidada por record:

- Layer 1 INCLUDE → final INCLUDE (no se re-evalúa en L2).
- Layer 1 EXCLUDE → final EXCLUDE.
- Layer 1 HUMAN_REVIEW:
    - Si L2 existe → final = L2 decision.
    - Si L2 no existe (todavía pendiente o falló) → final = HUMAN_REVIEW.

Outputs:
  - biofloc_final_decisions.parquet: un row por record con consolidated decision.
  - prisma_counts.json: conteos por fase PRISMA para el diagrama.

Usage:
  uv run python -m biofloc_review.run_consolidate \\
    data/processed/biofloc_screened.parquet \\
    data/processed/biofloc_layer2_screened.parquet
"""
from __future__ import annotations

import argparse
import json
import logging
from collections import Counter
from pathlib import Path

import pandas as pd

log = logging.getLogger(__name__)


# Constantes de identificación por fuente (de manuscrito L716-740 / Tabla 1).
IDENTIFICATION_BY_SOURCE = {
    "scopus": 6380,
    "wos": 10685,
    "ieee": 545,
    "scielo": 4446,
}
IDENTIFICATION_TOTAL = sum(IDENTIFICATION_BY_SOURCE.values())  # 22.056


def consolidate(layer1: pd.DataFrame, layer2: pd.DataFrame | None) -> pd.DataFrame:
    """Compute final consolidated decision per record."""
    # Dedup Layer 1 by record_id (algunos están duplicados — keep latest by index)
    l1 = layer1.drop_duplicates(subset="record_id", keep="last").copy()
    l1["record_id"] = l1["record_id"].astype(str)

    if layer2 is not None and len(layer2) > 0:
        l2 = layer2.drop_duplicates(subset="record_id", keep="last").copy()
        l2["record_id"] = l2["record_id"].astype(str)
        l2_map = dict(zip(l2["record_id"], l2["l2_final_decision"]))
        l2_reason_map = dict(zip(l2["record_id"], l2["l2_final_reason"]))
    else:
        l2_map = {}
        l2_reason_map = {}

    def _final(row) -> tuple[str, str]:
        rid = str(row["record_id"])
        l1_dec = row["final_decision_primary"]
        if l1_dec == "INCLUDE":
            return ("INCLUDE", "Layer 1 INCLUDE (no re-evaluado en L2)")
        if l1_dec == "EXCLUDE":
            return ("EXCLUDE", row.get("final_reason_primary") or "Layer 1 EXCLUDE")
        # Layer 1 HUMAN_REVIEW
        if rid in l2_map:
            return (l2_map[rid], f"Layer 2: {l2_reason_map.get(rid, '')}")
        return ("HUMAN_REVIEW", "Layer 1 HUMAN_REVIEW; L2 pendiente")

    finals = l1.apply(_final, axis=1, result_type="expand")
    l1["final_decision"] = finals[0]
    l1["final_decision_source"] = finals[1]
    l1["l2_processed"] = l1["record_id"].isin(set(l2_map.keys()))
    return l1


def prisma_counts(consolidated: pd.DataFrame) -> dict:
    """Compute PRISMA flowchart counts from consolidated decisions."""
    total_dedup = len(consolidated)  # 16.544 unique records post-Capa 3

    by_source = {}
    for src, n in IDENTIFICATION_BY_SOURCE.items():
        by_source[src] = n

    # Layer 1 screening (de la columna final_decision_primary)
    l1 = consolidated["final_decision_primary"].value_counts().to_dict()

    # Layer 2 LLM screening (solo aplica a HUMAN_REVIEW de L1)
    l2_processed = consolidated[consolidated.l2_processed]
    l2 = l2_processed["final_decision"].value_counts().to_dict()
    l2_pending = (consolidated.final_decision_primary == "HUMAN_REVIEW") & (~consolidated.l2_processed)

    # Decisión consolidada final
    final = consolidated["final_decision"].value_counts().to_dict()

    return {
        "identification": {
            "by_source": by_source,
            "total": IDENTIFICATION_TOTAL,
        },
        "deduplication": {
            "post_capa1": 17326,  # de manuscrito
            "post_capa3_unique": total_dedup,
            "removed_by_dedup": IDENTIFICATION_TOTAL - total_dedup,
        },
        "screening_layer1": {
            "input": total_dedup,
            "include": int(l1.get("INCLUDE", 0)),
            "exclude": int(l1.get("EXCLUDE", 0)),
            "human_review": int(l1.get("HUMAN_REVIEW", 0)),
            "advanced": int(l1.get("INCLUDE", 0)) + int(l1.get("HUMAN_REVIEW", 0)),
        },
        "screening_layer2": {
            "input": int(l1.get("HUMAN_REVIEW", 0)),
            "l2_processed": int(len(l2_processed)),
            "l2_pending": int(l2_pending.sum()),
            "l2_include": int(l2.get("INCLUDE", 0)),
            "l2_exclude": int(l2.get("EXCLUDE", 0)),
            "l2_human_review": int(l2.get("HUMAN_REVIEW", 0)),
        },
        "final_decisions": {
            "include": int(final.get("INCLUDE", 0)),
            "exclude": int(final.get("EXCLUDE", 0)),
            "human_review": int(final.get("HUMAN_REVIEW", 0)),
            "to_eligibility": int(final.get("INCLUDE", 0)) + int(final.get("HUMAN_REVIEW", 0)),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Consolidar decisiones finales y generar conteos PRISMA"
    )
    parser.add_argument("screened_parquet", type=Path,
                        help="Parquet Layer 1 (biofloc_screened.parquet)")
    parser.add_argument("layer2_parquet", type=Path, nargs="?",
                        default=Path("data/processed/biofloc_layer2_screened.parquet"),
                        help="Parquet Layer 2 LLM (opcional si no existe todavía)")
    parser.add_argument("--out-decisions", type=Path,
                        default=Path("data/processed/biofloc_final_decisions.parquet"))
    parser.add_argument("--out-counts", type=Path,
                        default=Path("data/processed/prisma_counts.json"))
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    log.info("Leyendo Layer 1 %s", args.screened_parquet)
    l1_df = pd.read_parquet(args.screened_parquet)
    log.info("  %d rows, %d record_ids únicos", len(l1_df), l1_df.record_id.nunique())

    if args.layer2_parquet.exists():
        log.info("Leyendo Layer 2 %s", args.layer2_parquet)
        l2_df = pd.read_parquet(args.layer2_parquet)
        log.info("  %d rows L2", len(l2_df))
    else:
        log.warning("Layer 2 parquet no existe: %s — consolidando solo con Layer 1",
                    args.layer2_parquet)
        l2_df = None

    consolidated = consolidate(l1_df, l2_df)
    log.info("Consolidados %d records (deduplicados por record_id)", len(consolidated))

    args.out_decisions.parent.mkdir(parents=True, exist_ok=True)
    consolidated.to_parquet(args.out_decisions, index=False)
    log.info("→ %s", args.out_decisions)

    counts = prisma_counts(consolidated)
    args.out_counts.write_text(json.dumps(counts, indent=2, ensure_ascii=False))
    log.info("→ %s", args.out_counts)

    print()
    print("=" * 60)
    print(" PRISMA COUNTS")
    print("=" * 60)
    print(json.dumps(counts, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
