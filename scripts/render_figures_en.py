"""Render English variants of ad-hoc figures used in the Floccus manuscript.

These figures were originally produced in interactive sessions with Spanish
labels embedded. This script regenerates them in English from the persisted
parquet datasets, writing PDFs/PNGs to ``docs/figures_en/``.

Covered figures:
  fig06_stability_layer1_full     fig06_stability_preliminary
  fig08_stability_consolidated    fig08_temporal_eligible
  fig09_publishers_eligible       fig12_methods_temporal
  fig14_c03_profile               fig15_c03_subclusters
  fig16_citations_by_cluster      fig17_top_venues
  fig18_decision_by_cluster       fig20_criteria_audit
  fig21_c15_subclusters
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "processed"
OUT = ROOT / "docs" / "figures_en"
OUT.mkdir(parents=True, exist_ok=True)

AZUL = "#3b7bb8"
AMBAR = "#c8861d"
GRIS = "#6b7280"
VERDE = "#3b8b6a"
ROJO = "#c14a3a"

PALETTE_DEC = {"INCLUDE": "#3b8b6a", "HUMAN_REVIEW": "#c8861d", "EXCLUDE": "#6b7280"}

CLUSTER_LABELS_EN = {
    -1: "Unclustered",
    0: "Microalga-bacteria",
    1: "Activated sludge",
    2: "Periphyton",
    3: "Biofloc aquaculture",
    4: "AGS",
    5: "Energy/LCA",
    6: "Anaerobic digestion",
    7: "EPS/microplastics",
    8: "Microbial ecology",
    9: "MBR",
    10: "Biofilm/MBBR",
    11: "N2O/GHG",
    12: "Biodegradation",
    13: "Pharmaceuticals",
    14: "Settlers/CFD",
    15: "Floc structure",
    16: "Bulking",
    17: "Textiles",
    18: "Anammox",
    19: "SBR",
    20: "Nitri/denitri",
    21: "Control/fuzzy",
    22: "NN/ML",
    23: "Plant simulation",
    24: "Phosphorus/EBPR",
}

CLUSTER_LABELS_LONG_EN = {
    1: "Activated sludge reactor (general)",
    9: "Membrane bioreactors (MBR)",
    24: "Phosphorus removal / EBPR",
    6: "Anaerobic digestion / biogas",
    21: "Process control / fuzzy logic",
    18: "Anammox / partial nitritation",
    13: "Pharmaceuticals and antibiotic resistance",
    12: "Biodegradation kinetics",
    3: "Biofloc in aquaculture",
    2: "Periphyton and river ecology",
    23: "Plant simulation (GPS-X)",
    10: "Biofilm and moving/fixed bed",
    4: "Aerobic granular sludge (AGS)",
    8: "Microbial ecology in WWTPs",
    5: "Energy and LCA of WWTPs",
    11: "N2O emissions / greenhouse gases",
    0: "Microalga-bacteria consortia",
    22: "Neural networks / ML",
    14: "Secondary settlers / CFD",
    17: "Textile wastewater",
    15: "Floc structure / rheology",
    19: "Sequencing batch reactor (SBR)",
    7: "EPS, microplastics, nanoparticles",
    16: "Bulking and filamentous bacteria",
    20: "Nitrification / denitrification",
}


def _save(fig: plt.Figure, name: str) -> None:
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(OUT / f"{name}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[{name}] -> {OUT / (name + '.pdf')}")


# ---------------------------------------------------------------------------
# fig06 / fig08_stability_consolidated — 4-panel screening stability
# ---------------------------------------------------------------------------
def _render_stability(df: pd.DataFrame, name: str) -> None:
    n = len(df)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"Screening stability across the corpus (n={n:,})",
                 fontsize=13, fontweight="bold")

    # Panel A: decision by corpus quartile
    ax = axes[0, 0]
    df = df.reset_index(drop=True).copy()
    df["q"] = pd.qcut(df.index, 4, labels=["Q1", "Q2", "Q3", "Q4"])
    pivot = (df.groupby(["q", "final_decision"]).size()
                .unstack(fill_value=0))
    for d in ["INCLUDE", "HUMAN_REVIEW", "EXCLUDE"]:
        if d not in pivot.columns:
            pivot[d] = 0
    pivot = pivot[["INCLUDE", "HUMAN_REVIEW", "EXCLUDE"]]
    pct = pivot.div(pivot.sum(axis=1), axis=0) * 100
    bottom = np.zeros(4)
    for d in ["INCLUDE", "HUMAN_REVIEW", "EXCLUDE"]:
        vals = pct[d].values
        ax.bar(pct.index.astype(str), vals, bottom=bottom,
               label=d, color=PALETTE_DEC[d], edgecolor="white")
        bottom += vals
    ax.set_title("(A) Decision by corpus quartile")
    ax.set_xlabel("Corpus quartile")
    ax.set_ylabel("% of records")
    ax.legend(loc="upper right", fontsize=9, frameon=False)
    ax.set_ylim(0, 100)

    # Panel B: stability by criterion (rolling window of 1000)
    ax = axes[0, 1]
    crits = ["I1", "I2", "I3", "E1", "E2", "E3", "E4", "D2", "D3"]
    window = 1000
    centers = []
    series = {c: [] for c in crits}
    for c in crits:
        col = f"{c}_answer"
        if col not in df.columns:
            continue
        yes = (df[col] == "YES").astype(float).values
        for i in range(0, len(yes) - window, window // 2):
            v = 100.0 * yes[i:i + window].mean()
            series[c].append(v)
            if c == crits[0]:
                centers.append(i + window // 2)
    for c in crits:
        if series[c]:
            ax.plot(centers, series[c], label=c, linewidth=1.4)
    ax.set_title("(B) Stability by criterion (low CV = stable classifier)")
    ax.set_xlabel("Corpus position (window centre)")
    ax.set_ylabel("% YES within 1000-record window")
    ax.legend(loc="upper right", ncol=2, fontsize=8, frameon=False)
    ax.set_ylim(0, 100)

    # Panel C: INCLUDE rate by source
    ax = axes[1, 0]
    src_counts = df.groupby("source").apply(
        lambda g: 100.0 * (g["final_decision"] == "INCLUDE").mean())
    src_counts = src_counts.sort_values(ascending=False)
    bars = ax.bar(src_counts.index, src_counts.values, color=VERDE,
                  edgecolor="white")
    for b, v in zip(bars, src_counts.values):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.3,
                f"{v:.1f}%", ha="center", fontsize=10)
    ax.set_title("(C) INCLUDE rate by source")
    ax.set_xlabel("Source")
    ax.set_ylabel("% INCLUDE")
    ax.set_ylim(0, max(src_counts.values) * 1.25)

    # Panel D: source ordering in parquet
    ax = axes[1, 1]
    colors = {"scopus": AZUL, "wos": AMBAR, "ieee": VERDE, "scielo": ROJO}
    df["pos"] = np.arange(len(df))
    for src in ["scopus", "wos", "ieee", "scielo"]:
        sub = df[df["source"] == src]
        ax.scatter(sub["pos"], [src] * len(sub),
                   c=colors.get(src, GRIS), s=4, alpha=0.4)
    ax.set_title("(D) Source ordering in the parquet")
    ax.set_xlabel("Position in parquet")
    ax.set_ylabel("source")

    fig.tight_layout()
    _save(fig, name)


def fig06_stability_layer1_full() -> None:
    df = pd.read_parquet(DATA / "biofloc_final_decisions.parquet")
    _render_stability(df, "fig06_stability_layer1_full")


def fig06_stability_preliminary() -> None:
    df = pd.read_parquet(DATA / "biofloc_final_decisions.parquet")
    _render_stability(df, "fig06_stability_preliminary")


def fig08_stability_consolidated() -> None:
    df = pd.read_parquet(DATA / "biofloc_final_decisions.parquet")
    df = df[df["l2_processed"].fillna(False)] if "l2_processed" in df.columns else df
    _render_stability(df, "fig08_stability_consolidated")


# ---------------------------------------------------------------------------
# fig08_temporal_eligible — temporal distribution of eligible corpus
# ---------------------------------------------------------------------------
def fig08_temporal_eligible() -> None:
    df = pd.read_parquet(DATA / "biofloc_final_decisions.parquet")
    df = df[df["final_decision"].isin(["INCLUDE", "HUMAN_REVIEW"])].copy()
    df["year_int"] = pd.to_numeric(df["year"], errors="coerce")
    df = df.dropna(subset=["year_int"])
    df["year_int"] = df["year_int"].astype(int)
    yrs = range(int(df["year_int"].min()), int(df["year_int"].max()) + 1)

    hr_counts = df[df["final_decision"] == "HUMAN_REVIEW"]["year_int"].value_counts()
    inc_counts = df[df["final_decision"] == "INCLUDE"]["year_int"].value_counts()
    hr_vals = [hr_counts.get(y, 0) for y in yrs]
    inc_vals = [inc_counts.get(y, 0) for y in yrs]

    fig, ax = plt.subplots(figsize=(10, 4.2))
    ax.bar(list(yrs), hr_vals, color=GRIS, label="HUMAN_REVIEW", edgecolor="white", linewidth=0.3)
    ax.bar(list(yrs), inc_vals, bottom=hr_vals, color=AZUL, label="INCLUDE", edgecolor="white", linewidth=0.3)
    ax.set_title(f"Temporal distribution of the eligible corpus (n={len(df):,})")
    ax.set_xlabel("Publication year")
    ax.set_ylabel("Number of records")
    ax.legend(loc="upper left", fontsize=10, frameon=False)
    fig.tight_layout()
    _save(fig, "fig08_temporal_eligible")


# ---------------------------------------------------------------------------
# fig09_publishers_eligible — top 10 publishers by DOI prefix
# ---------------------------------------------------------------------------
def fig09_publishers_eligible() -> None:
    df = pd.read_parquet(DATA / "biofloc_final_decisions.parquet")
    df = df[df["final_decision"].isin(["INCLUDE", "HUMAN_REVIEW"])].copy()

    PUB_MAP = {
        "10.1016": "Elsevier",
        "10.2166": "IWA Publishing",
        "10.1007": "Springer",
        "10.1002": "Wiley", "10.1111": "Wiley",
        "10.1109": "IEEE",
        "10.1080": "Taylor & Francis",
        "10.3390": "MDPI",
        "10.1021": "ACS",
    }

    def classify(doi):
        if not isinstance(doi, str) or "/" not in doi:
            return "no DOI"
        prefix = doi.split("/")[0]
        return PUB_MAP.get(prefix, "others")

    df["publisher"] = df["doi"].apply(classify)
    counts = df.groupby(["publisher", "final_decision"]).size().unstack(fill_value=0)
    if "INCLUDE" not in counts.columns:
        counts["INCLUDE"] = 0
    if "HUMAN_REVIEW" not in counts.columns:
        counts["HUMAN_REVIEW"] = 0
    counts["total"] = counts.sum(axis=1)
    counts = counts.sort_values("total", ascending=False).head(10)
    counts = counts.iloc[::-1]

    fig, ax = plt.subplots(figsize=(10, 5))
    y = np.arange(len(counts))
    ax.barh(y, counts["HUMAN_REVIEW"], color=GRIS, label="HUMAN_REVIEW", edgecolor="white")
    ax.barh(y, counts["INCLUDE"], left=counts["HUMAN_REVIEW"], color=AZUL,
            label="INCLUDE", edgecolor="white")
    ax.set_yticks(y, counts.index)
    ax.set_title("Top 10 publishers in the eligible corpus (by DOI prefix)")
    ax.set_xlabel("Number of records")
    ax.set_ylabel("Publisher / repository")
    ax.legend(loc="lower right", title="Consolidated decision", fontsize=9)
    fig.tight_layout()
    _save(fig, "fig09_publishers_eligible")


# ---------------------------------------------------------------------------
# fig10_umap_clusters — UMAP 2D + HDBSCAN
# ---------------------------------------------------------------------------
def fig10_umap_clusters() -> None:
    umap = np.load(DATA / "eligible_umap2d.npy")
    clusters = np.load(DATA / "eligible_clusters.npy")
    n = len(clusters)

    fig, ax = plt.subplots(figsize=(12, 8))
    cmap = plt.cm.tab20
    unique = sorted(set(int(c) for c in clusters))

    for c in unique:
        mask = clusters == c
        if c == -1:
            ax.scatter(umap[mask, 0], umap[mask, 1], c="#cccccc", s=8,
                       alpha=0.5, label=f"C-1: Unclustered (n={mask.sum()})")
        else:
            color = cmap((c % 20) / 20)
            ax.scatter(umap[mask, 0], umap[mask, 1], c=[color], s=10,
                       alpha=0.8, edgecolor="white", linewidth=0.2)
    handles = []
    from matplotlib.lines import Line2D
    for c in unique:
        if c == -1:
            color = "#cccccc"
            label = f"C-1: Unclustered"
        else:
            color = cmap((c % 20) / 20)
            label = f"C{c:02d}: {CLUSTER_LABELS_LONG_EN.get(c, CLUSTER_LABELS_EN.get(c, str(c)))}"
        handles.append(Line2D([0], [0], marker="o", color="w",
                              markerfacecolor=color, markersize=8, label=label))
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.02, 1),
              fontsize=8.5, frameon=False)
    ax.set_title(f"Thematic map of the eligible corpus (n={n:,})   UMAP 2D + HDBSCAN")
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    _save(fig, "fig10_umap_clusters")


# ---------------------------------------------------------------------------
# fig12_methods_temporal — temporal adoption of modelling methods
# ---------------------------------------------------------------------------
def fig12_methods_temporal() -> None:
    """Share of downloaded full texts mentioning each method, per period,
    with exact Clopper-Pearson 95% intervals.

    Computed from papers_with_methods.parquet (previously the values were
    typed in by hand). The two earliest periods are pooled (n = 6 and 15)
    because n = 6 alone does not support an estimate; every interval is
    shown so that overlapping periods are not read as trends (reviewer 2).
    """
    from scipy.stats import beta

    p = pd.read_parquet(DATA / "papers_with_methods.parquet").dropna(subset=["year"])
    bins, periods = [0, 2009, 2014, 2019, 2100], ["<2010", "2010–2014", "2015–2019", "2020–2026"]
    p["period"] = pd.cut(p.year, bins, labels=periods)
    n = p.groupby("period", observed=False).size().reindex(periods).to_numpy()
    shown = {"ASM": ("has_ASM", "o"), "ODE": ("has_ODE", "s"),
             "ML / ANN": ("has_ML_ANN", "^"), "PINN": ("has_PINN", "*")}

    def cp(k, m):
        lo = beta.ppf(.025, k, m - k + 1) if k else 0.0
        hi = beta.ppf(.975, k + 1, m - k) if k < m else 1.0
        return lo, hi

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(13, 4.4), gridspec_kw={"width_ratios": [2.1, 1]})
    x = np.arange(len(periods))
    for j, (lab, (col, mk)) in enumerate(shown.items()):
        k = p.groupby("period", observed=False)[col].sum().reindex(periods).to_numpy().astype(int)
        pct = 100 * k / n
        lo, hi = np.array([cp(kk, nn) for kk, nn in zip(k, n)]).T * 100
        off = (j - 1.5) * 0.09
        ax_a.errorbar(x + off, pct, yerr=[pct - lo, hi - pct], marker=mk, markersize=7,
                      linewidth=1.5, capsize=3, label=lab)
    ax_a.set_xticks(x, periods)
    ax_a.set_title("Share of full texts mentioning each method (95% CI)")
    ax_a.set_xlabel("Period"); ax_a.set_ylabel("% of full texts")
    ax_a.set_ylim(0, 75)
    ax_a.legend(ncol=2, fontsize=9, frameon=False, loc="upper right")
    ax_a.grid(True, alpha=0.3, linestyle=":")

    bars = ax_b.bar(periods, n, color=AZUL, edgecolor="white")
    for bb, v in zip(bars, n):
        ax_b.text(bb.get_x() + bb.get_width() / 2, v + 2, f"{v}", ha="center", fontsize=10)
    ax_b.set_title("Full texts per period (n)")
    ax_b.set_xlabel("Period"); ax_b.set_ylabel("Number of full texts")
    ax_b.set_ylim(0, max(n) * 1.15)
    fig.tight_layout()
    _save(fig, "fig12_methods_temporal")


# ---------------------------------------------------------------------------
# fig14_c03_profile — temporal + top venues for C03
# ---------------------------------------------------------------------------
def fig14_c03_profile() -> None:
    df = pd.read_parquet(DATA / "c03_biofloc_papers.parquet")
    df["year_int"] = pd.to_numeric(df["year"], errors="coerce").dropna().astype(int)

    yrs = sorted(df["year_int"].dropna().astype(int).unique())
    counts = df["year_int"].value_counts().sort_index()
    venues = df["venue"].fillna("").str.upper()
    venues = venues[venues.str.len() > 0].value_counts().head(12)
    venues = venues.iloc[::-1]

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(13, 5))
    ax_a.bar(counts.index, counts.values, color=VERDE, edgecolor="white")
    ax_a.set_title(f"C03 Biofloc aquaculture — temporal distribution (n={len(df)})")
    ax_a.set_xlabel("Year")
    ax_a.set_ylabel("Number of papers")

    y = np.arange(len(venues))
    bars = ax_b.barh(y, venues.values, color=AZUL, edgecolor="white")
    ax_b.set_yticks(y, [v[:50] for v in venues.index], fontsize=8.5)
    for b, v in zip(bars, venues.values):
        ax_b.text(v + 0.5, b.get_y() + b.get_height() / 2,
                  str(v), va="center", fontsize=9)
    ax_b.set_title("Top 12 venues in C03")
    ax_b.set_xlabel("Number of papers")
    fig.tight_layout()
    _save(fig, "fig14_c03_profile")


# ---------------------------------------------------------------------------
# fig15_c03_subclusters — UMAP scatter of C03 sub-clusters
# ---------------------------------------------------------------------------
def fig15_c03_subclusters() -> None:
    df = pd.read_parquet(DATA / "c03_subclusters.parquet")
    sc_labels = {
        -1: "Unclustered",
        0:  "Recirculating aquaculture / tilapia (Iberoam.)",
        1:  "Biofloc + Litopenaeus vannamei (Iberoam.)",
        2:  "Stocking density + bioeconomics",
        3:  "Modern biofloc + AI/IoT + multitrophic",
    }
    sc_colors = {-1: "#cccccc", 0: "#3b7bb8", 1: "#c8861d", 2: "#3b8b6a", 3: "#c14a3a"}

    fig, ax = plt.subplots(figsize=(10, 6.5))
    for sc, lbl in sc_labels.items():
        sub = df[df["sub_cluster"] == sc]
        ax.scatter(sub["umap_x"], sub["umap_y"], c=sc_colors[sc],
                   s=45, edgecolor="white", linewidth=0.6,
                   label=f"SC{sc}: {lbl} (n={len(sub)})")
    ax.set_title(f"Sub-structure of core cluster C03 Biofloc aquaculture (n={len(df)})")
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), fontsize=9, frameon=False)
    fig.tight_layout()
    _save(fig, "fig15_c03_subclusters")


# ---------------------------------------------------------------------------
# fig16_citations_by_cluster — bibliometric impact scatter
# ---------------------------------------------------------------------------
def fig16_citations_by_cluster() -> None:
    df = pd.read_parquet(DATA / "citations_by_cluster.parquet")
    df = df[df["n"] >= 5].copy()
    core = df["cluster"].isin([3, 15])
    fig, ax = plt.subplots(figsize=(11, 6.5))
    ax.scatter(df[~core]["n"], df[~core]["mean_cit"],
               s=90, c=AZUL, edgecolor="black", linewidth=0.5,
               label="Other clusters")
    ax.scatter(df[core]["n"], df[core]["mean_cit"],
               s=140, c=AMBAR, edgecolor="black", linewidth=0.6,
               label="Core clusters (C03 + C15)")
    for _, r in df.iterrows():
        cl = int(r["cluster"])
        lbl = "C-1" if cl == -1 else f"C{cl:02d}"
        ax.annotate(lbl, (r["n"], r["mean_cit"]),
                    xytext=(5, 4), textcoords="offset points", fontsize=8.5)
    ax.set_title("Bibliometric impact by cluster in the eligible corpus (clusters n≥5)")
    ax.set_xlabel("Cluster size (n papers)")
    ax.set_ylabel("Average citations per paper (OpenAlex)")
    ax.legend(loc="upper right", fontsize=10, frameon=True)
    ax.grid(True, alpha=0.3, linestyle=":")
    fig.tight_layout()
    _save(fig, "fig16_citations_by_cluster")


# ---------------------------------------------------------------------------
# fig17_top_venues — top 20 venues
# ---------------------------------------------------------------------------
def fig17_top_venues() -> None:
    df = pd.read_parquet(DATA / "eligible_with_venues.parquet")
    df = df[df["venue_norm"].notna() & (df["venue_norm"] != "")].copy()
    n_papers = len(df)
    n_venues = df["venue_norm"].nunique()
    top = df["venue_norm"].value_counts().head(20).iloc[::-1]

    def title_case(s):
        return " ".join(w.capitalize() for w in str(s).split())

    labels = [title_case(v) for v in top.index]

    fig, ax = plt.subplots(figsize=(11, 7))
    y = np.arange(len(top))
    colors = []
    for lbl in labels:
        if lbl == "Water Research":
            colors.append(AZUL)
        elif "Latin American" in lbl:
            colors.append(AMBAR)
        else:
            colors.append(GRIS)
    bars = ax.barh(y, top.values, color=colors, edgecolor="white")
    ax.set_yticks(y, labels, fontsize=9)
    for b, v in zip(bars, top.values):
        ax.text(v + 2, b.get_y() + b.get_height() / 2,
                str(v), va="center", fontsize=8.5)
    ax.set_title(f"Top 20 venues in the eligible corpus (n={n_papers:,} papers; {n_venues} distinct venues)")
    ax.set_xlabel("Number of papers in the eligible corpus")
    from matplotlib.patches import Patch
    handles = [
        Patch(facecolor=AZUL, label="Water Research (top)"),
        Patch(facecolor=AMBAR, label="LatAm J Aquatic Res (only Iberoamerican in top 20)"),
        Patch(facecolor="#3b8b6a", label="Ecological Modelling (target journal)"),
        Patch(facecolor=GRIS, label="Other journals"),
    ]
    ax.legend(handles=handles, loc="lower right", fontsize=9)
    fig.tight_layout()
    _save(fig, "fig17_top_venues")


# ---------------------------------------------------------------------------
# fig18_decision_by_cluster — INCLUDE/HUMAN_REVIEW composition by cluster
# ---------------------------------------------------------------------------
def fig18_decision_by_cluster() -> None:
    df = pd.read_parquet(DATA / "cluster_decision_breakdown.parquet")
    df = df[df["cluster"] != -1].copy()
    df = df.sort_values("pct_include", ascending=True).reset_index(drop=True)

    labels = []
    for _, r in df.iterrows():
        cl = int(r["cluster"])
        en = CLUSTER_LABELS_EN.get(cl, str(r["label"]))
        labels.append(f"C{cl:02d} {en}")

    fig, ax = plt.subplots(figsize=(11, 8.5))
    y = np.arange(len(df))
    ax.barh(y, df["n_include"], color=VERDE, label="INCLUDE", edgecolor="white")
    ax.barh(y, df["n_hr"], left=df["n_include"], color=GRIS, label="HUMAN_REVIEW",
            edgecolor="white")
    for i, r in df.iterrows():
        ax.text(r["n_total"] + 4, i, f"{r['pct_include']:.0f}%",
                va="center", fontsize=9)
    ax.set_yticks(y, labels, fontsize=9)
    ax.set_title("INCLUDE/HUMAN_REVIEW composition by cluster (sorted by INCLUDE rate)")
    ax.set_xlabel("Number of papers")
    ax.legend(loc="lower right", fontsize=10)
    fig.tight_layout()
    _save(fig, "fig18_decision_by_cluster")


# ---------------------------------------------------------------------------
# fig20_criteria_audit — firing rates per criterion
# ---------------------------------------------------------------------------
def fig20_criteria_audit() -> None:
    df = pd.read_parquet(DATA / "biofloc_final_decisions.parquet")
    n_total = len(df)
    n_excl = (df["final_decision"] == "EXCLUDE").sum()

    crits = [("I1", "Biol. flocs"), ("I2", "Quant. modelling"),
             ("I3", "Aqueous system"), ("I4", "Article/review"),
             ("E1", "Model=organism"), ("E2", "Exp. without model"),
             ("E3", "Out of water"), ("E4", "Chem. flocculation"),
             ("D1", "Short abstract"), ("D2", "Ambiguous context"),
             ("D3", "Method unspecified")]
    pct = []
    for c, _ in crits:
        col = f"{c}_answer"
        if col not in df.columns:
            pct.append(0)
            continue
        pct.append(100.0 * (df[col] == "YES").mean())

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(14, 5.2))
    colors_a = []
    for c, _ in crits:
        if c.startswith("I"):
            colors_a.append(AZUL)
        elif c.startswith("E"):
            colors_a.append(AMBAR)
        else:
            colors_a.append(GRIS)
    bars = ax_a.bar([c for c, _ in crits], pct, color=colors_a, edgecolor="white")
    for b, v in zip(bars, pct):
        ax_a.text(b.get_x() + b.get_width() / 2, v + 1.5,
                  f"{v:.0f}", ha="center", fontsize=10)
    ax_a.set_title(f"Firing rate per criterion (over n={n_total:,})")
    ax_a.set_ylabel("% of corpus with YES")
    ax_a.set_xticks(range(len(crits)),
                    [f"{c}\n{lbl}" for c, lbl in crits], rotation=45,
                    ha="right", fontsize=8.5)
    ax_a.set_ylim(0, 105)
    ax_a.grid(axis="y", alpha=0.3, linestyle=":")

    excl_df = df[df["final_decision"] == "EXCLUDE"]
    e_share = []
    for e in ["E1", "E2", "E3", "E4"]:
        col = f"{e}_answer"
        if col not in excl_df.columns:
            e_share.append(0)
        else:
            e_share.append(100.0 * (excl_df[col] == "YES").mean())
    bars = ax_b.bar(["E1", "E2", "E3", "E4"], e_share, color=AMBAR,
                    edgecolor="white")
    for b, v in zip(bars, e_share):
        ax_b.text(b.get_x() + b.get_width() / 2, v + 1.5,
                  f"{v:.0f}%", ha="center", fontsize=10)
    ax_b.set_title(f"Which E criterion drives exclusions (over n={n_excl:,})")
    ax_b.set_ylabel("% of EXCLUDE rows where it fires YES")
    ax_b.grid(axis="y", alpha=0.3, linestyle=":")
    ax_b.set_ylim(0, max(e_share) * 1.2 if e_share else 100)
    fig.tight_layout()
    _save(fig, "fig20_criteria_audit")


# ---------------------------------------------------------------------------
# fig21_c15_subclusters — UMAP scatter of C15 sub-clusters
# ---------------------------------------------------------------------------
def fig21_c15_subclusters() -> None:
    df = pd.read_parquet(DATA / "c15_subclusters.parquet")
    sc_labels = {
        -1: "Unclustered",
        0:  "Cohesive sediments / fractal estuarine",
        1:  "Rheology / yield stress",
        2:  "Population Balance + bioflocculation",
        3:  "Microbial structure / EPS / dynamics",
    }
    sc_colors = {-1: "#cccccc", 0: "#3b7bb8", 1: "#c8861d", 2: "#3b8b6a", 3: "#c14a3a"}

    fig, ax = plt.subplots(figsize=(9, 7))
    for sc, lbl in sc_labels.items():
        sub = df[df["sub_cluster"] == sc]
        ax.scatter(sub["umap_x"], sub["umap_y"], c=sc_colors[sc],
                   s=45, edgecolor="white", linewidth=0.6,
                   label=f"SC{sc}: {lbl} (n={len(sub)})")
    ax.set_title(f"Sub-structure of cluster C15 Floc structure (n={len(df)})")
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), fontsize=9, frameon=False)
    fig.tight_layout()
    _save(fig, "fig21_c15_subclusters")


def main() -> None:
    fig06_stability_layer1_full()
    fig06_stability_preliminary()
    fig08_stability_consolidated()
    fig08_temporal_eligible()
    fig09_publishers_eligible()
    fig10_umap_clusters()
    fig12_methods_temporal()
    fig14_c03_profile()
    fig15_c03_subclusters()
    fig16_citations_by_cluster()
    fig17_top_venues()
    fig18_decision_by_cluster()
    fig20_criteria_audit()
    fig21_c15_subclusters()


if __name__ == "__main__":
    main()
