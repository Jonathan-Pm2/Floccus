"""Search advanced modeling-method keywords in the eligible corpus.

Complements scripts/extract_methods.py (which works on downloaded PDFs)
by searching the same kind of patterns over the full eligible corpus's
title + abstract + keywords. This catches papers that the OA fulltext
fetcher could not download (e.g. paywalled Elsevier journals) and is
the right tool to estimate the prevalence of cutting-edge techniques
(PINN, NeuralODE, hyperbolic geometry, etc.) without depending on the
OA-availability sample.

Patterns covered:
  - PINN (physics-informed neural networks)
  - NeuralODE / NeuralOperator (Fourier Neural Operator, DeepONet, ...)
  - GraphNN
  - Hyperbolic geometry / Poincaré ball / disk
  - Riemannian geometry / manifold optimization
  - Manifold learning (UMAP, t-SNE, isomap)
  - Transformer / self-attention
  - Biplot (Gabriel, HJ, GH)
  - Stochastic differential models (Langevin, Fokker-Planck, MCMC)
  - Digital Twin / cyber-physical systems

Output: data/processed/advanced_methods_abstracts.parquet with one row
per eligible record and binary has_* columns. Useful for cross-tabbing
against the thematic clusters and against the screening decisions.

Usage:
  uv run python scripts/search_advanced_methods.py \\
    data/processed/biofloc_final_decisions.parquet \\
    data/processed/biofloc_final_primary_v3.parquet
"""
from __future__ import annotations

import argparse
import logging
import re
from pathlib import Path

import pandas as pd

from biofloc_review.dedup.embeddings import _record_to_point_id
from biofloc_review.schema import PaperRecord

log = logging.getLogger(__name__)


PATTERNS: dict[str, str] = {
    "PINN": (
        r"\b(physics[-\s]informed\s+neural|"
        r"\bPINN[s]?\b|"
        r"physics[-\s]informed\s+(machine|deep)\s+learning)\b"
    ),
    "NeuralODE": (
        r"\b(neural\s+ODE|"
        r"neural\s+ordinary\s+differential|"
        r"\bNODE\b|"
        r"neural[-\s]network\s+ODE)\b"
    ),
    "NeuralOperator": (
        r"\b(neural\s+operator|"
        r"fourier\s+neural\s+operator|"
        r"\bFNO\b|"
        r"DeepONet|"
        r"graph\s+neural\s+operator)\b"
    ),
    "GraphNN": (
        r"\b(graph\s+neural\s+network|"
        r"\bGNN[s]?\b|"
        r"graph\s+convolutional|"
        r"\bGCN\b)\b"
    ),
    "Hyperbolic": (
        r"\b(hyperbolic\s+(geometry|space|embedding|biplot|neural)|"
        r"disco?\s+de\s+poincar|"
        r"poincar[eé]\s+(disk|ball|biplot|embedding))\b"
    ),
    "Riemannian": (
        r"\b(riemannian\s+(manifold|geometry|optimization|metric)|"
        r"riemann\s+manifold)\b"
    ),
    "ManifoldLearning": (
        r"\b(manifold\s+learning|"
        r"\bUMAP\b|"
        r"t[-\s]SNE\b|"
        r"isomap\b|"
        r"tangent\s+space)\b"
    ),
    "Transformer": (
        r"\b(transformer\s+(model|architecture|network)|"
        r"attention\s+mechanism|"
        r"self[-\s]attention|"
        r"\bBERT\b)\b"
    ),
    "Biplot": (
        r"\b(biplot|"
        r"GH[-\s]biplot|"
        r"HJ[-\s]biplot|"
        r"gabriel\s+biplot)\b"
    ),
    "StochasticBio": (
        r"\b(stochastic\s+differential\s+equation|"
        r"langevin|"
        r"fokker[-\s]planck|"
        r"monte[-\s]carlo\s+simulation|"
        r"markov\s+chain\s+monte)\b"
    ),
    "DigitalTwin": (
        r"\b(digital\s+twin|"
        r"cyber[-\s]physical)\b"
    ),
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Search advanced modeling-method keywords in the eligible corpus"
    )
    parser.add_argument(
        "decisions_parquet", type=Path,
        help="Consolidated decisions parquet (biofloc_final_decisions.parquet)",
    )
    parser.add_argument(
        "canonical_parquet", type=Path,
        help="Canonical parquet with abstracts (biofloc_final_primary_v3.parquet)",
    )
    parser.add_argument(
        "--decisions", nargs="+",
        default=["INCLUDE", "HUMAN_REVIEW"],
        help="Final decisions whose records enter the analysis",
    )
    parser.add_argument(
        "--out", type=Path,
        default=Path("data/processed/advanced_methods_abstracts.parquet"),
    )
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    log.info("Reading consolidated decisions: %s", args.decisions_parquet)
    dec = pd.read_parquet(args.decisions_parquet)
    dec["record_id"] = dec.record_id.astype(str)
    advancing_ids = set(
        dec[dec.final_decision.isin(args.decisions)].record_id.tolist()
    )
    log.info("  %d records advancing", len(advancing_ids))

    log.info("Reading canonical parquet: %s", args.canonical_parquet)
    canon = pd.read_parquet(args.canonical_parquet)

    rows: list[dict] = []
    for _, row in canon.iterrows():
        d = {
            k: (None if isinstance(v, float) and pd.isna(v) else v)
            for k, v in row.to_dict().items()
        }
        try:
            rec = PaperRecord.model_validate(d)
        except Exception:
            continue
        rid = _record_to_point_id(rec)
        if rid not in advancing_ids:
            continue
        keywords_str = ", ".join(rec.keywords) if rec.keywords else ""
        text = " ".join([rec.title or "", rec.abstract or "", keywords_str])
        out_row = {
            "record_id": rid,
            "title": rec.title or "",
            "year": rec.year,
            "source": rec.source,
            "doi": rec.doi or "",
            "venue": rec.venue or "",
            "has_abstract": bool(rec.abstract and len(rec.abstract) > 50),
        }
        for name, pat in PATTERNS.items():
            rx = re.compile(pat, re.IGNORECASE)
            out_row[f"has_{name}"] = bool(rx.search(text))
        rows.append(out_row)

    df = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.out, index=False)

    log.info("Saved → %s", args.out)
    log.info("  total rows: %d", len(df))
    log.info("  with abstract: %d", int(df.has_abstract.sum()))
    log.info("\nMatches per pattern:")
    for name in PATTERNS:
        n = int(df[f"has_{name}"].sum())
        log.info("  %-18s: %4d (%.2f%%)", name, n, n / max(len(df), 1) * 100)


if __name__ == "__main__":
    main()
