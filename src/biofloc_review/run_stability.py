"""Análisis de estabilidad del clasificador a lo largo del corpus.

Genera tres paneles diagnósticos para defender en el manuscrito que la
distribución no-uniforme de decisiones a lo largo del orden de procesamiento
es propiedad del corpus, no deriva del clasificador:

  Panel A — Tasa INCLUDE/EXCLUDE/HUMAN_REVIEW por cuartil del corpus.
  Panel B — Tasa YES por criterio en ventanas deslizantes (CV bajo = estable).
  Panel C — Tasa INCLUDE por source y por year (composición del corpus).
  Panel D — Posición de cada source en el parquet (orden de ingesta).

Usage:
  uv run python -m biofloc_review.run_stability \\
    data/processed/biofloc_screened.parquet --window 1000
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

LLM_CRITERIA = ("I1", "I2", "I3", "E1", "E2", "E3", "E4", "D2", "D3")
DET_CRITERIA = ("I4", "E5", "D1")
ALL_CRITERIA = LLM_CRITERIA + DET_CRITERIA


def panel_a_quartiles(df: pd.DataFrame) -> pd.DataFrame:
    """Decision rate by corpus quartile."""
    q = pd.qcut(np.arange(len(df)), q=4, labels=["Q1", "Q2", "Q3", "Q4"])
    return pd.crosstab(q, df.final_decision_primary, normalize="index") * 100


def panel_b_criterion_stability(df: pd.DataFrame, window: int = 1000) -> dict:
    """Per-criterion YES rate across sliding windows.

    Returns dict mapping criterion -> dict with 'positions', 'rates', 'mean',
    'std', 'cv' (coefficient of variation, percent of mean).
    """
    n = len(df)
    if n < window * 2:
        log.warning("Corpus too small for sliding-window analysis: %d records", n)
    starts = np.arange(0, n - window + 1, window // 2)

    out = {}
    for c in LLM_CRITERIA:
        rates = []
        for s in starts:
            sub = df.iloc[s:s + window]
            if len(sub) == 0:
                continue
            yes = (sub[f"{c}_answer"] == "YES").sum()
            rates.append(100.0 * yes / len(sub))
        arr = np.asarray(rates, dtype=float)
        mean = float(arr.mean()) if len(arr) else 0.0
        std = float(arr.std()) if len(arr) else 0.0
        cv = (100.0 * std / mean) if mean > 0 else 0.0
        out[c] = {
            "positions": (starts + window // 2).tolist(),
            "rates": arr.tolist(),
            "mean": mean,
            "std": std,
            "cv": cv,
        }
    return out


def panel_c_by_source_year(df: pd.DataFrame) -> dict:
    """INCLUDE rate by source and by year."""
    by_src = (df.groupby("source")
                .apply(lambda g: 100.0 * (g.final_decision_primary == "INCLUDE").mean(),
                       include_groups=False)
                .sort_values(ascending=False)
                .to_dict())
    by_year = (df.dropna(subset=["year"]).groupby("year")
                 .apply(lambda g: 100.0 * (g.final_decision_primary == "INCLUDE").mean(),
                        include_groups=False)
                 .to_dict())
    return {"by_source": by_src, "by_year": by_year}


def panel_d_source_positions(df: pd.DataFrame) -> dict:
    """Position of each source in the parquet (proves ingest-order hypothesis)."""
    df_idx = df.reset_index(drop=True)
    out = {}
    for src in df_idx.source.unique():
        idx = df_idx.index[df_idx.source == src].to_numpy()
        out[src] = {
            "min": int(idx.min()),
            "max": int(idx.max()),
            "median": int(np.median(idx)),
            "count": int(len(idx)),
        }
    return out


def render_figures(df: pd.DataFrame,
                   panel_a: pd.DataFrame,
                   panel_b: dict,
                   panel_c: dict,
                   panel_d: dict,
                   out_dir: Path,
                   stem: str = "fig06_stability") -> None:
    """Render the 4-panel diagnostic figure."""
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    # Panel A: decision by quartile (stacked bar)
    ax = axes[0, 0]
    panel_a[["INCLUDE", "HUMAN_REVIEW", "EXCLUDE"]].plot(
        kind="bar", stacked=True, ax=ax,
        color=["#2ca02c", "#ff7f0e", "#7f7f7f"], width=0.8,
    )
    ax.set_xlabel("Cuartil del corpus")
    ax.set_ylabel("% de records")
    ax.set_title("(A) Decisión por cuartil del corpus")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_xticklabels(panel_a.index, rotation=0)

    # Panel B: per-criterion YES rate in sliding windows
    ax = axes[0, 1]
    for c, info in panel_b.items():
        ax.plot(info["positions"], info["rates"], label=c, lw=1.2)
    ax.set_xlabel("Posición en corpus (centro de ventana)")
    ax.set_ylabel("% YES en ventana de 1000")
    ax.set_title("(B) Estabilidad por criterio (CV bajo = clasificador estable)")
    ax.legend(loc="upper right", fontsize=7, ncol=2)
    ax.set_ylim(bottom=0)

    # Panel C: INCLUDE rate by source
    ax = axes[1, 0]
    sources = list(panel_c["by_source"].keys())
    rates = list(panel_c["by_source"].values())
    bars = ax.bar(sources, rates, color="#2ca02c", alpha=0.7)
    for b, r in zip(bars, rates):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.3,
                f"{r:.1f}%", ha="center", fontsize=9)
    ax.set_ylabel("% INCLUDE")
    ax.set_title("(C) Tasa INCLUDE por fuente")
    ax.set_ylim(0, max(rates) * 1.3 if rates else 100)

    # Panel D: source positions in the parquet
    ax = axes[1, 1]
    df_idx = df.reset_index(drop=True)
    for src in panel_d.keys():
        idx = df_idx.index[df_idx.source == src]
        ax.scatter(idx, [src] * len(idx), s=1, alpha=0.3)
    ax.set_xlabel("Posición en parquet")
    ax.set_ylabel("source")
    ax.set_title("(D) Orden de fuentes en el parquet")

    fig.suptitle(
        f"Estabilidad del cribado a lo largo del corpus (n={len(df)})",
        fontsize=11, fontweight="bold",
    )
    fig.tight_layout()

    for ext in ("png", "pdf"):
        path = out_dir / f"{stem}.{ext}"
        fig.savefig(path, dpi=120, bbox_inches="tight")
        log.info("Wrote %s", path)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Análisis de estabilidad del cribado a lo largo del corpus"
    )
    parser.add_argument("parquet", type=Path,
                        help="Parquet de salida del cribado (biofloc_screened.parquet)")
    parser.add_argument("--window", type=int, default=1000,
                        help="Tamaño de ventana para análisis de estabilidad")
    parser.add_argument("--figures-dir", type=Path, default=Path("docs/figures"),
                        help="Carpeta de salida para PNG/PDF")
    parser.add_argument("--stem", default="fig06_stability",
                        help="Stem del nombre de archivo de la figura")
    parser.add_argument("--no-figures", action="store_true",
                        help="Solo imprimir stats, no generar figura")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    if not args.parquet.exists():
        raise SystemExit(f"Parquet no encontrado: {args.parquet}")

    df = pd.read_parquet(args.parquet)
    log.info("Cargados %d records de %s", len(df), args.parquet)

    a = panel_a_quartiles(df)
    b = panel_b_criterion_stability(df, window=args.window)
    c = panel_c_by_source_year(df)
    d = panel_d_source_positions(df)

    print()
    print("=" * 60)
    print(f" ANÁLISIS DE ESTABILIDAD ({len(df)} records)")
    print("=" * 60)

    print("\n(A) Decisión por cuartil del corpus (%):")
    print(a.round(1).to_string())

    print("\n(B) Estabilidad del clasificador por criterio (ventana", args.window, "):")
    print(f"{'crit':<6}{'mean':>8}{'std':>7}{'CV':>7}{'verdict':>14}")
    for crit, info in b.items():
        verdict = "estable" if info["cv"] < 15 else "variable" if info["cv"] < 30 else "INESTABLE"
        print(f"  {crit:<4}{info['mean']:>7.1f}%{info['std']:>6.2f}{info['cv']:>6.1f}%  {verdict:>10}")

    print("\n(C) Tasa INCLUDE por fuente (%):")
    for src, rate in c["by_source"].items():
        n = (df.source == src).sum()
        print(f"  {src:<10} {rate:>5.1f}%   (n={n})")

    print("\n(D) Posición de fuentes en parquet:")
    for src, info in d.items():
        print(f"  {src:<10} pos[{info['min']:>5} - {info['max']:>5}]  "
              f"mediana={info['median']:>5}  n={info['count']}")

    if not args.no_figures:
        render_figures(df, a, b, c, d, args.figures_dir, stem=args.stem)
        print(f"\nFigura guardada en {args.figures_dir}/{args.stem}.png / .pdf")


if __name__ == "__main__":
    main()
