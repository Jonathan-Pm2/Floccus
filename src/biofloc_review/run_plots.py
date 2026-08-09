from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import pandas as pd

from biofloc_review.plots import apply_paper_style, save_fig
from biofloc_review.plots.figures import (
    fig_capa2_scores_by_verdict,
    fig_capa3_confidence,
    fig_cluster_size_histogram,
    fig_dedup_waterfall,
    fig_source_distribution,
)

log = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate all paper figures from processed parquets")
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--lang", choices=["es", "en"], default="es")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()
    if args.out_dir is None:
        args.out_dir = Path("docs") / ("figures_en" if args.lang == "en" else "figures")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    apply_paper_style()
    log.info("Style applied. Loading data from %s", args.data_dir)

    biofloc = pd.read_parquet(args.data_dir / "biofloc.parquet")
    biplot  = pd.read_parquet(args.data_dir / "biplot.parquet")
    biofloc_dedup_full = pd.read_parquet(args.data_dir / "biofloc_dedup_full.parquet")
    biplot_dedup_full  = pd.read_parquet(args.data_dir / "biplot_dedup_full.parquet")
    biofloc_judged = json.loads((args.data_dir / "biofloc_dedup_primary_near_dup_pairs_judged.json").read_text())
    biplot_judged  = json.loads((args.data_dir / "biplot_dedup_primary_near_dup_pairs_judged.json").read_text())

    # Tamaños finales tras Capa 3 (se podrían leer de los stats; los hardcodeamos por ahora)
    waterfall_stages = {
        "biofloc": {"identif": 22056, "capa1": 17326, "capa3": 17075},
        "biplot":  {"identif": 1401,  "capa1": 1235,  "capa3": 1222},
    }

    log.info("Generating figures into %s", args.out_dir)

    TITLES = {
        "es": dict(
            fig3a="biofloc — Capa 2 scores por veredicto Capa 3",
            fig3b="biplot — Capa 2 scores por veredicto Capa 3",
            fig4a="biofloc — Confianza del juez LLM por veredicto",
            fig4b="biplot — Confianza del juez LLM por veredicto",
            fig5a="biofloc — Tamaños de cluster (Capa 1)",
            fig5b="biplot — Tamaños de cluster (Capa 1)",
        ),
        "en": dict(
            fig3a="biofloc — Layer 2 scores by Layer 3 verdict",
            fig3b="biplot — Layer 2 scores by Layer 3 verdict",
            fig4a="biofloc — LLM judge confidence by verdict",
            fig4b="biplot — LLM judge confidence by verdict",
            fig5a="biofloc — Cluster sizes (Layer 1)",
            fig5b="biplot — Cluster sizes (Layer 1)",
        ),
    }[args.lang]

    fig1 = fig_source_distribution(biofloc, biplot, lang=args.lang)
    save_fig(fig1, args.out_dir, "fig01_source_distribution")

    fig2 = fig_dedup_waterfall(waterfall_stages, lang=args.lang)
    save_fig(fig2, args.out_dir, "fig02_dedup_waterfall")

    fig3a = fig_capa2_scores_by_verdict(biofloc_judged, TITLES["fig3a"], lang=args.lang)
    save_fig(fig3a, args.out_dir, "fig03a_capa2_scores_biofloc")

    fig3b = fig_capa2_scores_by_verdict(biplot_judged, TITLES["fig3b"], lang=args.lang)
    save_fig(fig3b, args.out_dir, "fig03b_capa2_scores_biplot")

    fig4a = fig_capa3_confidence(biofloc_judged, TITLES["fig4a"], lang=args.lang)
    save_fig(fig4a, args.out_dir, "fig04a_capa3_confidence_biofloc")

    fig4b = fig_capa3_confidence(biplot_judged, TITLES["fig4b"], lang=args.lang)
    save_fig(fig4b, args.out_dir, "fig04b_capa3_confidence_biplot")

    fig5a = fig_cluster_size_histogram(biofloc_dedup_full, TITLES["fig5a"], lang=args.lang)
    save_fig(fig5a, args.out_dir, "fig05a_cluster_sizes_biofloc")

    fig5b = fig_cluster_size_histogram(biplot_dedup_full, TITLES["fig5b"], lang=args.lang)
    save_fig(fig5b, args.out_dir, "fig05b_cluster_sizes_biplot")

    log.info("Done.")


if __name__ == "__main__":
    main()
