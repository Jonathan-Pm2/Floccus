"""Refresh focalizado: re-consulta Europe PMC para todos los records de
`biofloc_enriched.parquet` y actualiza pmc_* en sitio.

Útil cuando la lógica de extracción de URL fulltext mejora (sin tener que
re-correr OpenAlex que es lento por rate-limit). EPMC no tiene rate limit
estricto; aceptamos ~2 req/s.

Usage:
  uv run python -m biofloc_review.run_pmc_refresh \\
    data/processed/biofloc_enriched.parquet
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

from biofloc_review.screening.enricher import enrich_via_pmc

log = logging.getLogger(__name__)


PMC_COLS = [
    "pmc_id", "pmc_pmid", "pmc_mesh_terms",
    "pmc_has_fulltext", "pmc_fulltext_url", "pmc_is_oa", "pmc_ok",
]


def _refresh_one(doi: str) -> dict:
    """Wrapper that handles missing DOI gracefully."""
    if not doi:
        return {"pmc_ok": False}
    return enrich_via_pmc(doi)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Refresh Europe PMC fields en biofloc_enriched.parquet"
    )
    parser.add_argument("enriched_parquet", type=Path,
                        help="Parquet enriched de Layer 2 (in-place update)")
    parser.add_argument("--out", type=Path, default=None,
                        help="Parquet de salida (default: sobreescribe el input)")
    parser.add_argument("--workers", type=int, default=4,
                        help="Trabajadores concurrentes (default: 4; EPMC tolera)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limitar a N records (testing)")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    if not args.enriched_parquet.exists():
        raise SystemExit(f"Parquet no encontrado: {args.enriched_parquet}")

    out_path = args.out or args.enriched_parquet
    log.info("Leyendo %s", args.enriched_parquet)
    df = pd.read_parquet(args.enriched_parquet)
    log.info("  %d records", len(df))

    # Solo refrescamos records con DOI (sin DOI no podemos consultar EPMC)
    candidates_idx = df[df.doi.notna()].index.tolist()
    if args.limit:
        candidates_idx = candidates_idx[: args.limit]
    log.info("Records a refrescar: %d (con DOI)", len(candidates_idx))

    # Lanzar en paralelo
    results: dict[int, dict] = {}
    completed = 0
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {
            ex.submit(_refresh_one, df.loc[i, "doi"]): i
            for i in candidates_idx
        }
        for fut in as_completed(futures):
            i = futures[fut]
            try:
                results[i] = fut.result()
            except Exception as e:
                log.warning("error en idx %d: %s", i, e)
                results[i] = {"pmc_ok": False}
            completed += 1
            if completed % 100 == 0 or completed == len(candidates_idx):
                elapsed = time.time() - t0
                rate = completed / max(elapsed, 1e-6)
                eta = (len(candidates_idx) - completed) / max(rate, 1e-6)
                log.info("  refresh %d/%d (%.1f rec/s, eta %.0fs)",
                         completed, len(candidates_idx), rate, eta)

    # Aplicar los resultados al DataFrame
    log.info("Aplicando resultados al DataFrame...")
    for i, r in results.items():
        for col in PMC_COLS:
            if col in r:
                df.at[i, col] = r[col]

    df.to_parquet(out_path, index=False)
    log.info("Guardado → %s", out_path)

    # Stats finales
    summary = {
        "total": len(df),
        "candidates_refreshed": completed,
        "pmc_ok_after": int(df.pmc_ok.fillna(False).sum()),
        "pmc_has_fulltext_after": int(df.pmc_has_fulltext.fillna(False).sum()),
        "pmc_fulltext_url_after": int(df.pmc_fulltext_url.notna().sum()),
        "elapsed_seconds": round(time.time() - t0, 1),
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
