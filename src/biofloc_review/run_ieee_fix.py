"""Fix focalizado del bug IEEE record_id collision.

El export CSV de IEEE Xplore usa `Document Identifier` como campo de
identificador único, pero ese valor es un ID de sesión/batch que se
repite entre papers distintos (530 papers → solo 11 valores únicos).
Bajo la regla `_record_to_point_id` actual, 478 papers IEEE colapsan al
mismo UUID y heredan veredictos de cribado.

Este script corrige el problema sin re-cargar IEEE.csv:

  1. Lee el parquet canónico v2 y aísla las filas IEEE.
  2. Recomputa `source_id` por fila usando el orden de preferencia:
     PDF Link (url) > DOI > hash(title_norm). Mismo orden que en la
     versión corregida de `sources/ieee.py:_ieee_source_id`.
  3. Escribe `biofloc_final_primary_v3.parquet` con los IEEE corregidos
     y el resto del corpus sin tocar.
  4. Elimina las filas IEEE de `biofloc_screened.parquet` para que el
     pase L1 las re-cribe sobre los `record_id` corregidos.

Después de este script: correr `run_screening.py` sobre v3 con --resume
(re-cribará solo los IEEE, el resto está ya en el output parquet).
"""
from __future__ import annotations

import argparse
import hashlib
import logging
from pathlib import Path

import pandas as pd

from biofloc_review.normalize import normalize_title

log = logging.getLogger(__name__)


def _new_ieee_source_id(row: pd.Series) -> str | None:
    """Same logic as `sources/ieee.py:_ieee_source_id` but reading from
    a PaperRecord row in a parquet (fields: url, doi, title)."""
    url = row.get("url")
    if url is not None and pd.notna(url):
        s = str(url).strip()
        if s and s.lower() not in {"nan", "none", ""}:
            return f"pdf:{s}"
    doi = row.get("doi")
    if doi is not None and pd.notna(doi):
        s = str(doi).strip()
        if s and s.lower() not in {"nan", "none", ""}:
            return f"doi:{s}"
    title = row.get("title")
    if title is not None and pd.notna(title):
        t = str(title).strip()
        if t:
            h = hashlib.sha1(normalize_title(t).encode("utf-8")).hexdigest()[:16]
            return f"titlehash:{h}"
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Fix IEEE record_id collision (in-place v2→v3)")
    parser.add_argument(
        "--canonical-in", type=Path,
        default=Path("data/processed/biofloc_final_primary_v2.parquet"),
    )
    parser.add_argument(
        "--canonical-out", type=Path,
        default=Path("data/processed/biofloc_final_primary_v3.parquet"),
    )
    parser.add_argument(
        "--screened", type=Path,
        default=Path("data/processed/biofloc_screened.parquet"),
        help="Screened parquet del que se eliminan las filas IEEE (para forzar re-screening).",
    )
    parser.add_argument(
        "--screened-backup", type=Path,
        default=Path("data/processed/biofloc_screened_pre_ieee_fix.parquet"),
    )
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    # 1. Read canonical v2
    log.info("Leyendo canónico v2: %s", args.canonical_in)
    df = pd.read_parquet(args.canonical_in)
    log.info("  %d rows", len(df))

    ieee_mask = df.source == "ieee"
    n_ieee = ieee_mask.sum()
    log.info("  %d rows IEEE detectadas", n_ieee)

    # 2. Recompute source_id for IEEE rows
    n_unique_before = df.loc[ieee_mask, "source_id"].nunique()
    log.info("  source_id IEEE únicos ANTES del fix: %d", n_unique_before)

    new_sids = df.loc[ieee_mask].apply(_new_ieee_source_id, axis=1)
    df.loc[ieee_mask, "source_id"] = new_sids.values

    n_unique_after = df.loc[ieee_mask, "source_id"].nunique()
    n_null_after = df.loc[ieee_mask, "source_id"].isna().sum()
    log.info("  source_id IEEE únicos DESPUÉS del fix: %d (nulls: %d)",
             n_unique_after, n_null_after)

    if n_unique_after < n_ieee * 0.95:
        log.warning("Aún hay colisiones en IEEE source_id (%d únicos de %d)",
                    n_unique_after, n_ieee)

    # 3. Write canonical v3
    args.canonical_out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.canonical_out, index=False)
    log.info("Escrito canónico v3: %s", args.canonical_out)

    # 4. Remove IEEE rows from screened parquet (backup first)
    if args.screened.exists():
        log.info("Backup screened: %s → %s", args.screened, args.screened_backup)
        screened = pd.read_parquet(args.screened)
        screened.to_parquet(args.screened_backup, index=False)
        log.info("  backup: %d rows", len(screened))

        ieee_screened_mask = screened.source == "ieee"
        n_remove = ieee_screened_mask.sum()
        log.info("  Eliminando %d rows IEEE del screened parquet", n_remove)

        screened_filtered = screened[~ieee_screened_mask]
        screened_filtered.to_parquet(args.screened, index=False)
        log.info("  screened post-filter: %d rows (eliminadas %d)",
                 len(screened_filtered), n_remove)
    else:
        log.warning("Screened parquet no encontrado: %s", args.screened)

    log.info("\nFix IEEE listo. Próximos pasos:")
    log.info("  1. run_screening.py %s --resume --workers 2", args.canonical_out)
    log.info("  2. run_enrich.py + run_screening_layer2.py sobre los IEEE re-screened")
    log.info("  3. run_consolidate.py para regenerar prisma_counts.json")


if __name__ == "__main__":
    main()
