"""Generate a stratified blind sample for Cohen's kappa validation.

Outputs two paired files:

  1. `validation_sample_blind.csv` — for the human reviewer. Contains the
     minimum input the LLM saw (title, abstract, keywords, venue, year, doi)
     plus EMPTY columns for the human to fill in their verdicts. The LLM's
     decision is NOT included; the human reviews blind.

  2. `validation_sample_truth.parquet` — for the kappa analyzer. Contains
     the LLM's verdicts (final decision + 12 per-criterion answers) keyed
     by the same `review_id`. The human reviewer must NOT see this file.

Stratification: source × final_decision (4 × 3 = 12 cells), with
configurable target per cell (default 42 → ~504 total). When a cell has
fewer records than the target, the whole cell is sampled.

The sample is deterministic via `--seed` (default: 20260603, date-anchored).

Usage:
  uv run python -m biofloc_review.run_validation_sample \\
    data/processed/biofloc_final_decisions.parquet
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd

from biofloc_review.screening.criteria import CRITERIA

log = logging.getLogger(__name__)


# Criteria order for the optional per-criterion review columns. Reads from
# the live criteria module so a removed criterion (e.g. E5) drops out of the
# CSV automatically.
CRITERIA_ORDER = tuple(c.id for c in CRITERIA)


# Columns shown to the human reviewer — the same input the LLM evaluated.
BLIND_INPUT_COLS = ("review_id", "source", "year", "doi", "title",
                    "abstract", "keywords", "venue")


# Columns the human fills in. The decision column is mandatory; the
# per-criterion columns are optional and may be left blank.
HUMAN_OUTPUT_COLS_MANDATORY = ("human_decision",)
HUMAN_OUTPUT_COLS_OPTIONAL = tuple(f"human_{c}" for c in CRITERIA_ORDER) + ("human_notes",)


def stratified_sample(df: pd.DataFrame, by: list[str], target_per_cell: int,
                       seed: int) -> pd.DataFrame:
    """Return a stratified sample with `target_per_cell` records per cell
    of the cross-tabulation defined by `by`.

    Cells smaller than the target are taken in full.
    """
    rng = np.random.default_rng(seed)
    out_chunks = []
    for key, group in df.groupby(by, dropna=False, observed=True):
        n_target = min(target_per_cell, len(group))
        if n_target == 0:
            continue
        sampled = group.sample(n=n_target, random_state=int(rng.integers(0, 2**31)))
        out_chunks.append(sampled)
    return pd.concat(out_chunks, ignore_index=False)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sample stratified records for blind kappa validation"
    )
    parser.add_argument(
        "decisions_parquet", type=Path,
        help="Consolidated decisions parquet (biofloc_final_decisions.parquet)",
    )
    parser.add_argument(
        "--canonical-parquet", type=Path,
        default=Path("data/processed/biofloc_final_primary_v3.parquet"),
        help="Canonical parquet to pull abstract/keywords text from "
             "(consolidated decisions only stores has_abstract bool).",
    )
    parser.add_argument(
        "--per-cell", type=int, default=42,
        help="Records sampled per (source × decision) cell (default: 42 → ~504 total)",
    )
    parser.add_argument(
        "--seed", type=int, default=20260603,
        help="Random seed for reproducibility (default: 20260603)",
    )
    parser.add_argument(
        "--out-blind", type=Path,
        default=Path("data/processed/validation_sample_blind.csv"),
        help="CSV for human reviewer (no LLM verdicts)",
    )
    parser.add_argument(
        "--out-truth", type=Path,
        default=Path("data/processed/validation_sample_truth.parquet"),
        help="Parquet with LLM verdicts for kappa analyzer (DO NOT show to human)",
    )
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    if not args.decisions_parquet.exists():
        raise SystemExit(f"Decisions parquet no encontrado: {args.decisions_parquet}")

    log.info("Leyendo decisiones consolidadas: %s", args.decisions_parquet)
    df = pd.read_parquet(args.decisions_parquet)
    log.info("  %d records, %d únicos por record_id", len(df), df.record_id.nunique())

    # Join with canonical to pull abstract + keywords text (consolidated
    # parquet only carries has_abstract bool).
    log.info("Leyendo canónico para texto: %s", args.canonical_parquet)
    canon = pd.read_parquet(args.canonical_parquet)
    # Compute the same record_id used downstream so we can merge.
    from biofloc_review.dedup.embeddings import _record_to_point_id
    from biofloc_review.schema import PaperRecord
    rid_to_text: dict[str, dict] = {}
    for _, row in canon.iterrows():
        d = {k: (None if isinstance(v, float) and pd.isna(v) else v)
             for k, v in row.to_dict().items()}
        try:
            rec = PaperRecord.model_validate(d)
        except Exception:
            continue
        rid_to_text[_record_to_point_id(rec)] = {
            "abstract": rec.abstract,
            "keywords": ", ".join(rec.keywords) if rec.keywords else "",
        }
    df["abstract"] = df.record_id.astype(str).map(lambda r: rid_to_text.get(r, {}).get("abstract"))
    df["keywords"] = df.record_id.astype(str).map(lambda r: rid_to_text.get(r, {}).get("keywords", ""))
    log.info("  Texto recuperado para %d/%d records",
             df.abstract.notna().sum(), len(df))

    # Drop records without abstract — human can't review blind without one
    pre = len(df)
    df = df[df.abstract.notna() & (df.abstract.astype(str).str.strip() != "")]
    post = len(df)
    if pre != post:
        log.info("  Filtrados %d sin abstract (no revisables blind)", pre - post)

    # Stratified sample by source × final_decision
    log.info("Estratificando por source × final_decision con %d por cell",
             args.per_cell)
    sample = stratified_sample(
        df,
        by=["source", "final_decision"],
        target_per_cell=args.per_cell,
        seed=args.seed,
    )

    # Add a stable blind identifier
    rng = np.random.default_rng(args.seed)
    sample = sample.copy()
    sample["review_id"] = [str(uuid4()) for _ in range(len(sample))]

    log.info("Tamaño de muestra: %d records", len(sample))
    log.info("Distribución por (source, decisión):")
    dist = sample.groupby(["source", "final_decision"], observed=True).size()
    for (src, dec), n in dist.items():
        log.info("  %-10s × %-13s : %d", src, dec, n)

    # ---- Build blind CSV for human reviewer ----
    blind = sample[list(BLIND_INPUT_COLS)].copy()
    # Add empty human output columns
    for col in HUMAN_OUTPUT_COLS_MANDATORY:
        blind[col] = ""
    for col in HUMAN_OUTPUT_COLS_OPTIONAL:
        blind[col] = ""

    args.out_blind.parent.mkdir(parents=True, exist_ok=True)
    blind.to_csv(args.out_blind, index=False)
    log.info("CSV para revisor humano → %s (NO contiene veredictos LLM)",
             args.out_blind)

    # ---- Build truth parquet for kappa analyzer ----
    truth_cols = ["review_id", "record_id", "source", "year", "doi", "title",
                  "final_decision", "final_decision_source"]
    # Add all LLM per-criterion answers
    for c in CRITERIA_ORDER:
        ans_col = f"{c}_answer"
        if ans_col in sample.columns:
            truth_cols.append(ans_col)

    truth = sample[truth_cols].copy()
    args.out_truth.parent.mkdir(parents=True, exist_ok=True)
    truth.to_parquet(args.out_truth, index=False)
    log.info("Parquet de veredictos LLM (NO compartir con revisor) → %s",
             args.out_truth)

    print()
    print("=" * 70)
    print(" Instrucciones para el revisor humano")
    print("=" * 70)
    print(f"""
1. Abrí {args.out_blind} en LibreOffice / Excel / pandas / herramienta preferida.
2. Para cada fila, leé title + abstract + keywords + venue + año.
3. Aplicá los criterios PRISMA (Tabla del manuscrito) y decidí en
   `human_decision`: ESCRIBÍ "cumple" o "no cumple".
     - "cumple"    = el paper cumple los criterios → debería avanzar a la
                     siguiente fase del cribado (incluido o derivado a humano).
     - "no cumple" = el paper NO cumple, debería descartarse.
4. OPCIONAL: en las columnas `human_I1..D3` podés marcar por criterio con
   YES | NO | UNCLEAR. Si no querés, dejá vacías.
5. Cuando termines, guardá el CSV (puede ser el mismo path).
6. Corré `uv run python -m biofloc_review.run_kappa_analysis \\
   {args.out_blind} {args.out_truth}` para calcular κ Cohen contra el LLM.

IMPORTANTE: durante la revisión, NO mires el parquet de truth (revelaría
las decisiones del LLM y sesgaría la validación).
""")


if __name__ == "__main__":
    main()
