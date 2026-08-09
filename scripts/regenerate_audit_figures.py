"""Regenerate fig11, fig13, fig19 with explicit axis labels.

These three figures of the Floccus manuscript were created ad-hoc in
earlier interactive sessions and shipped without `xlabel`/`ylabel`,
which is grounds for revision in SCI journals. This script rebuilds
them from the parquet datasets in ``data/processed/`` with the
academic palette and explicit axis titles. Versioned here so the
provenance survives future sessions.

Pass ``--lang en`` to generate the English-labelled variants into
``docs/figures_en/``.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

AZUL = "#3b7bb8"
AMBAR = "#c8861d"
GRIS = "#6b7280"
VERDE = "#3b8b6a"


CLUSTER_LABELS_EN = {
    -1: "Unclustered",
    0:  "Microalga-bacteria", 1: "Activated sludge", 2: "Periphyton",
    3:  "Biofloc aquaculture", 4: "AGS", 5: "Energy/LCA",
    6:  "Anaerobic digestion", 7: "EPS/microplastics", 8: "Microbial ecology",
    9:  "MBR", 10: "Biofilm/MBBR", 11: "N2O/GHG",
    12: "Biodegradation", 13: "Pharmaceuticals", 14: "Settlers/CFD",
    15: "Floc structure", 16: "Bulking", 17: "Textiles",
    18: "Anammox", 19: "SBR", 20: "Nitri/denitri",
    21: "Control/fuzzy", 22: "NN/ML", 23: "Plant simulation",
    24: "Phosphorus/EBPR",
}


def _strings(lang: str) -> dict:
    if lang == "en":
        return dict(
            fig11_xlabel="Modelling method",
            fig11_ylabel="Thematic cluster",
            fig11_title="Modelling methods per thematic cluster\n"
                         "(% of downloaded OA PDFs)",
            fig11_cbar="% of PDFs mentioning the method",
            fig13_xlabel="Method (B)",
            fig13_ylabel="Method (A)",
            fig13_title="Co-occurrence of modelling methods\n"
                         "Cell (A, B) = # papers mentioning both methods "
                         "(n=296 PDFs)",
            fig13_cbar="# papers mentioning both methods",
            fig19_xlabel_a="Bibliographic database",
            fig19_ylabel="% of the set",
            fig19_title_a="Distribution by bibliographic database",
            fig19_xlabel_b="Final decision",
            fig19_title_b="Distribution by final decision",
            fig19_legend_noise="Unclustered",
            fig19_legend_grouped="Clustered",
            fig19_suptitle="Comparative profile of the unclustered set (noise) vs clustered",
        )
    return dict(
        fig11_xlabel="Método de modelado",
        fig11_ylabel="Cluster temático",
        fig11_title="Métodos de modelado por cluster temático\n"
                     "(% sobre PDFs OA libres descargados)",
        fig11_cbar="% PDFs con mención del método",
        fig13_xlabel="Método (B)",
        fig13_ylabel="Método (A)",
        fig13_title="Co-ocurrencia de métodos de modelado\n"
                     "Celda (A, B) = # papers que mencionan ambos métodos "
                     "(n=296 PDFs)",
        fig13_cbar="# papers que mencionan ambos métodos",
        fig19_xlabel_a="Base bibliográfica",
        fig19_ylabel="% del conjunto",
        fig19_title_a="Distribución por base bibliográfica",
        fig19_xlabel_b="Decisión final",
        fig19_title_b="Distribución por decisión final",
        fig19_legend_noise="No agrupado",
        fig19_legend_grouped="Agrupado",
        fig19_suptitle="Perfil comparativo del conjunto no agrupado (ruido) vs agrupado",
    )


def regen_fig11_methods_by_cluster(out_dir: Path, lang: str = "es") -> None:
    """Heatmap clusters x methods (% of cluster PDFs mentioning each method)."""
    S = _strings(lang)
    mc = pd.read_parquet(ROOT / "data" / "processed" / "methods_by_cluster.parquet")
    method_cols = ["ODE", "PDE", "ABM", "PBM", "CFD", "ML_ANN", "PINN",
                   "Stochastic", "ASM"]
    mc = mc.sort_values("n_pdfs", ascending=True).reset_index(drop=True)

    def _label_for(r):
        cl = int(r["cluster"])
        if lang == "en":
            lbl = CLUSTER_LABELS_EN.get(cl, str(r["label"]))
        else:
            lbl = r["label"]
        tag = f"C{cl:02d}" if cl >= 0 else "C-1"
        return f"{tag} {lbl} (n={r['n_pdfs']})"

    row_labels = [_label_for(r) for _, r in mc.iterrows()]

    pct = mc[method_cols].div(mc["n_pdfs"].replace(0, np.nan), axis=0) * 100.0
    pct = pct.fillna(0).round(0).astype(int)

    fig, ax = plt.subplots(figsize=(9, 7.5))
    im = ax.imshow(pct.values, cmap="YlOrRd", aspect="auto", vmin=0, vmax=80)

    ax.set_xticks(range(len(method_cols)), method_cols, rotation=35, ha="right")
    ax.set_yticks(range(len(row_labels)), row_labels, fontsize=8.5)
    ax.set_xlabel(S["fig11_xlabel"], fontsize=10.5, labelpad=8)
    ax.set_ylabel(S["fig11_ylabel"], fontsize=10.5, labelpad=8)
    ax.set_title(S["fig11_title"], fontsize=11, pad=10)

    for i in range(pct.shape[0]):
        for j in range(pct.shape[1]):
            v = pct.iat[i, j]
            if v > 0:
                color = "white" if v >= 50 else "black"
                ax.text(j, i, f"{v:.0f}", ha="center", va="center",
                        fontsize=7.5, color=color)

    cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cb.set_label(S["fig11_cbar"], fontsize=9)

    fig.tight_layout()
    fig.savefig(out_dir / "fig11_methods_by_cluster.pdf", bbox_inches="tight")
    fig.savefig(out_dir / "fig11_methods_by_cluster.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig11] regenerated -> {out_dir / 'fig11_methods_by_cluster.pdf'}")


def regen_fig13_cooccurrence(out_dir: Path, lang: str = "es") -> None:
    """Heatmap of method co-occurrence (lower-triangular) with axis labels."""
    S = _strings(lang)
    co = np.load(ROOT / "data" / "processed" / "methods_cooccurrence.npy")
    methods = ["ASM", "ODE", "ML_ANN", "Stochastic", "CFD", "PDE", "ABM",
               "PBM", "PINN"]

    mask = np.triu(np.ones_like(co, dtype=bool), k=1)
    co_masked = np.where(mask, np.nan, co.astype(float))

    fig, ax = plt.subplots(figsize=(7.5, 6.2))
    im = ax.imshow(co_masked, cmap="YlOrRd", aspect="equal", vmin=0, vmax=30)

    ax.set_xticks(range(len(methods)), methods, rotation=35, ha="right")
    ax.set_yticks(range(len(methods)), methods)
    ax.set_xlabel(S["fig13_xlabel"], fontsize=10.5, labelpad=8)
    ax.set_ylabel(S["fig13_ylabel"], fontsize=10.5, labelpad=8)
    ax.set_title(S["fig13_title"], fontsize=10.5, pad=10)

    for i in range(len(methods)):
        for j in range(len(methods)):
            if j > i:
                continue
            v = int(co[i, j])
            if v == 0:
                continue
            on_diag = (i == j)
            color = "white" if (on_diag or v >= 18) else "black"
            weight = "bold" if on_diag else "normal"
            ax.text(j, i, str(v), ha="center", va="center",
                    fontsize=8.5, color=color, fontweight=weight)

    cb = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.02)
    cb.set_label(S["fig13_cbar"], fontsize=9)

    fig.tight_layout()
    fig.savefig(out_dir / "fig13_cooccurrence.pdf", bbox_inches="tight")
    fig.savefig(out_dir / "fig13_cooccurrence.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig13] regenerated -> {out_dir / 'fig13_cooccurrence.pdf'}")


def regen_fig19_noise_profile(out_dir: Path, lang: str = "es") -> None:
    """Two side-by-side bar charts comparing noise vs grouped subsets."""
    S = _strings(lang)
    noise = pd.read_parquet(ROOT / "data" / "processed" / "noise_records.parquet")
    # The grouped counterpart needs full eligible corpus + cluster assignments.
    cluster_assign = pd.read_parquet(
        ROOT / "data" / "processed" / "eligible_clusters_assignment.parquet")
    decisions = pd.read_parquet(
        ROOT / "data" / "processed" / "biofloc_final_decisions.parquet")

    elig = decisions[decisions["final_decision"].isin(["INCLUDE", "HUMAN_REVIEW"])]
    elig = elig.merge(cluster_assign[["record_id", "cluster"]],
                       on="record_id", how="left")
    elig["group"] = np.where(elig["cluster"] == -1, S["fig19_legend_noise"], S["fig19_legend_grouped"])

    sources = ["Scopus", "WoS", "IEEE", "SciELO"]
    src_map = {"scopus": "Scopus", "wos": "WoS", "ieee": "IEEE", "scielo": "SciELO"}
    elig["src_norm"] = elig["source"].str.lower().map(src_map).fillna(elig["source"])

    def pct_by(df, col, categories):
        n = len(df)
        if n == 0:
            return [0.0] * len(categories)
        return [100.0 * (df[col] == c).sum() / n for c in categories]

    noise_set = elig[elig["group"] == S["fig19_legend_noise"]]
    group_set = elig[elig["group"] == S["fig19_legend_grouped"]]

    pct_noise_src = pct_by(noise_set, "src_norm", sources)
    pct_group_src = pct_by(group_set, "src_norm", sources)

    decision_order = ["INCLUDE", "HUMAN_REVIEW"]
    pct_noise_dec = pct_by(noise_set, "final_decision", decision_order)
    pct_group_dec = pct_by(group_set, "final_decision", decision_order)

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(11, 4.2))
    x_a = np.arange(len(sources))
    w = 0.38
    bars_an = ax_a.bar(x_a - w / 2, pct_noise_src, w, color=AMBAR,
                        edgecolor="white",
                        label=f"{S['fig19_legend_noise']} (n={len(noise_set):,})".replace(",", " "))
    bars_ag = ax_a.bar(x_a + w / 2, pct_group_src, w, color=AZUL,
                        edgecolor="white",
                        label=f"{S['fig19_legend_grouped']} (n={len(group_set):,})".replace(",", " "))
    ax_a.set_xticks(x_a, sources)
    ax_a.set_xlabel(S["fig19_xlabel_a"], fontsize=10.5, labelpad=8)
    ax_a.set_ylabel(S["fig19_ylabel"], fontsize=10.5, labelpad=8)
    ax_a.set_title(S["fig19_title_a"], fontsize=11, pad=8)
    ax_a.legend(loc="upper right", fontsize=9, frameon=False)
    ax_a.grid(axis="y", alpha=0.3, linestyle=":")
    ax_a.set_axisbelow(True)
    for bars, vals in [(bars_an, pct_noise_src), (bars_ag, pct_group_src)]:
        for b, v in zip(bars, vals):
            ax_a.text(b.get_x() + b.get_width() / 2, v + 1.5,
                       f"{v:.0f}%", ha="center", fontsize=8.5)

    x_b = np.arange(len(decision_order))
    bars_bn = ax_b.bar(x_b - w / 2, pct_noise_dec, w, color=AMBAR,
                        edgecolor="white", label=S["fig19_legend_noise"])
    bars_bg = ax_b.bar(x_b + w / 2, pct_group_dec, w, color=AZUL,
                        edgecolor="white", label=S["fig19_legend_grouped"])
    ax_b.set_xticks(x_b, decision_order)
    ax_b.set_xlabel(S["fig19_xlabel_b"], fontsize=10.5, labelpad=8)
    ax_b.set_ylabel(S["fig19_ylabel"], fontsize=10.5, labelpad=8)
    ax_b.set_title(S["fig19_title_b"], fontsize=11, pad=8)
    ax_b.legend(loc="upper left", fontsize=9, frameon=False)
    ax_b.grid(axis="y", alpha=0.3, linestyle=":")
    ax_b.set_axisbelow(True)
    for bars, vals in [(bars_bn, pct_noise_dec), (bars_bg, pct_group_dec)]:
        for b, v in zip(bars, vals):
            ax_b.text(b.get_x() + b.get_width() / 2, v + 1.5,
                       f"{v:.0f}%", ha="center", fontsize=8.5)

    fig.suptitle(S["fig19_suptitle"], y=1.02, fontsize=11.5)
    fig.tight_layout()
    fig.savefig(out_dir / "fig19_noise_profile.pdf", bbox_inches="tight")
    fig.savefig(out_dir / "fig19_noise_profile.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig19] regenerated -> {out_dir / 'fig19_noise_profile.pdf'}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lang", choices=["es", "en"], default="es")
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args()

    out_dir = args.out_dir or (
        ROOT / "docs" / ("figures_en" if args.lang == "en" else "figures"))
    out_dir.mkdir(parents=True, exist_ok=True)

    regen_fig11_methods_by_cluster(out_dir, lang=args.lang)
    regen_fig13_cooccurrence(out_dir, lang=args.lang)
    regen_fig19_noise_profile(out_dir, lang=args.lang)


if __name__ == "__main__":
    main()
