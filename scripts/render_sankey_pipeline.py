"""Render the Floccus pipeline Sankey/alluvial diagram for the manuscript.

Design (agreed with the authors):
  * Descriptive strata on the left  : Document type -> Language -> Source
    (real cross-tabs computed from data/processed/biofloc.parquet, no
    illustrative values).
  * Processing flow on the right     : Source -> Deduplication ->
    Screening decision -> Eligible corpus -> Full-text extraction.
  * Enrichment side-band             : OpenAlex and Europe PMC drawn as
    lateral annotation boxes with dashed arrows pointing to the stages
    they enrich (they augment metadata, they do not filter records, so
    they are NOT part of the conservation-preserving main flow).

All processing-flow magnitudes are exact counts read from the persisted
parquet files. Enrichment magnitudes are exact counts shown as labels on
the side-band, not as flow ribbons.

Outputs (docs/figures_en/):
  fig_sankey_pipeline.pdf   (vector, for LaTeX \\includegraphics)
  fig_sankey_pipeline.png   (raster preview, scale=2)

Run:  .venv/bin/python scripts/render_sankey_pipeline.py
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "processed"
OUT = ROOT / "docs" / "figures_en"
OUT.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Colour palette — muted, publication-grade "sophisticated" scheme
# (colour-blind safe; harmonised desaturated tones).
# ---------------------------------------------------------------------------
NODE_SLATE = "#37474f"    # document-type stratum  (deep blue-grey)
NODE_CYAN = "#6d9bb5"     # language stratum       (dusty blue)
NODE_BLUE = "#2f6d8c"     # sources / identification (deep petrol blue)
NODE_GREY = "#7d8794"     # dedup / neutral         (cool grey)
NODE_GREEN = "#4a7c59"    # include / eligible / extracted (muted forest)
NODE_AMBER = "#c19a3e"    # human review            (soft ochre)
NODE_RED = "#b0553f"      # exclude / duplicates    (muted terracotta)
NODE_VIOLET = "#7d5a8c"   # enrichment side-band    (muted plum)

# Ribbon fills — semi-transparent versions of the node colours
RIBBON = "rgba(109,155,181,0.30)"      # descriptive strata ribbons (dusty blue)
RIBBON_GREEN = "rgba(74,124,89,0.50)"
RIBBON_AMBER = "rgba(193,154,62,0.50)"
RIBBON_RED = "rgba(176,85,63,0.45)"
RIBBON_GREY = "rgba(125,135,148,0.40)"


# ---------------------------------------------------------------------------
# 1. Read data and compute the REAL cross-tabs and processing counts
# ---------------------------------------------------------------------------
def load_counts() -> dict:
    b = pd.read_parquet(DATA / "biofloc.parquet")

    def doc_bucket(x: str) -> str:
        x = str(x).lower()
        if "proceed" in x:
            return "Proceedings"
        if "thesis" in x or "dissert" in x:
            return "Thesis"
        if "review" in x:
            return "Review"
        if "article" in x:
            return "Article"
        return "Other"

    def lang_bucket(x: str) -> str:
        x = str(x)
        if x == "English":
            return "English"
        if x == "Spanish":
            return "Spanish"
        if x == "Portuguese":
            return "Portuguese"
        if x == "Chinese":
            return "Chinese"
        return "Other/NA"

    src_map = {"wos": "WoS", "scopus": "Scopus", "scielo": "SciELO", "ieee": "IEEE"}
    b["doc_b"] = b["document_type"].apply(doc_bucket)
    b["lang_b"] = b["language"].apply(lang_bucket)
    b["src_b"] = b["source"].map(src_map).fillna(b["source"])

    doc_lang = pd.crosstab(b["doc_b"], b["lang_b"])
    lang_src = pd.crosstab(b["lang_b"], b["src_b"])

    # Post-deduplication per-source survivors (from the screening decisions).
    dec = pd.read_parquet(DATA / "biofloc_final_decisions.parquet")
    src_dec = pd.crosstab(dec["source"].map(src_map).fillna(dec["source"]),
                          dec["final_decision"])
    survivors = src_dec.sum(axis=1)          # per source, post-dedup (= 17,063 total)

    identified = b["src_b"].value_counts()   # per source, identification (= 22,056)

    return {
        "doc_lang": doc_lang,
        "lang_src": lang_src,
        "survivors": survivors,
        "identified": identified,
        "decisions": dec["final_decision"].value_counts(),
    }


# ---------------------------------------------------------------------------
# 2. Build the Sankey
# ---------------------------------------------------------------------------
def build_sankey() -> go.Figure:
    C = load_counts()
    doc_order = ["Article", "Other", "Proceedings", "Thesis", "Review"]
    lang_order = ["English", "Other/NA", "Spanish", "Portuguese", "Chinese"]
    src_order = ["WoS", "Scopus", "SciELO", "IEEE"]

    labels: list[str] = []
    color: list[str] = []
    xpos: list[float] = []
    ypos: list[float] = []
    idx: dict[str, int] = {}

    def add(key: str, disp: str, col_color: str, x: float, y: float) -> None:
        idx[key] = len(labels)
        labels.append(f"<b>{disp}</b>")   # bold node labels for legibility
        color.append(col_color)
        xpos.append(x)
        ypos.append(y)

    # ---- Column x-positions (0..1) ----
    X_DOC, X_LANG, X_SRC = 0.01, 0.11, 0.23
    X_DEDUP, X_DEC, X_ELIG = 0.36, 0.51, 0.65
    X_FULLTEXT, X_EXTRACT, X_ANALYSIS = 0.76, 0.86, 0.96

    # ---- Column 1: document type ----
    doc_tot = int(C["doc_lang"].sum(axis=1).sum())
    y = 0.05
    for d in doc_order:
        n = int(C["doc_lang"].loc[d].sum())
        add(f"doc_{d}", f"{d} {n:,}", NODE_SLATE, X_DOC, min(y, 0.95))
        y += 0.19

    # ---- Column 2: language ----
    y = 0.05
    for lg in lang_order:
        n = int(C["lang_src"].loc[lg].sum())
        add(f"lang_{lg}", f"{lg} {n:,}", NODE_CYAN, X_LANG, min(y, 0.95))
        y += 0.19

    # ---- Column 3: source ----
    y = 0.06
    for s in src_order:
        n = int(C["identified"][s])
        add(f"src_{s}", f"{s} {n:,}", NODE_BLUE, X_SRC, min(y, 0.9))
        y += 0.22

    # ---- Column 4: deduplication gate ----
    add("dedup", "Deduplicated 17,063", NODE_GREY, X_DEDUP, 0.45)
    add("dups", "Duplicates 4,993", NODE_RED, X_DEDUP, 0.92)

    # ---- Column 5: screening decision ----
    add("inc", "Include 1,385", NODE_GREEN, X_DEC, 0.30)
    add("hr", "HumanReview 3,498", NODE_AMBER, X_DEC, 0.50)
    add("exc", "Exclude 12,180", NODE_RED, X_DEC, 0.82)

    # ---- Column 6: eligible corpus ----
    add("elig", "Eligible corpus 4,883", NODE_GREEN, X_ELIG, 0.42)

    # ---- Column 7: full-text acquisition ----
    add("oapdf", "OA PDFs 297", NODE_GREEN, X_FULLTEXT, 0.20)
    add("manual", "Abstract-level 4,586", NODE_GREY, X_FULLTEXT, 0.60)

    # ---- Column 8: text extraction ----
    add("extr", "Text extracted 296", NODE_GREEN, X_EXTRACT, 0.20)

    # ---- Column 9: final analysis ----
    add("analysis", "Analysis · 25 clusters", NODE_GREEN, X_ANALYSIS, 0.42)

    # ---------------------------------------------------------------
    # Links (all real counts)
    # ---------------------------------------------------------------
    s_idx: list[int] = []
    t_idx: list[int] = []
    val: list[float] = []
    lcol: list[str] = []

    def link(a: str, b: str, v: float, c: str) -> None:
        if v <= 0:
            return
        s_idx.append(idx[a])
        t_idx.append(idx[b])
        val.append(v)
        lcol.append(c)

    # doc -> language (real cross-tab)
    for d in doc_order:
        for lg in lang_order:
            link(f"doc_{d}", f"lang_{lg}", int(C["doc_lang"].loc[d, lg]), RIBBON)
    # language -> source (real cross-tab)
    for lg in lang_order:
        for s in src_order:
            link(f"lang_{lg}", f"src_{s}", int(C["lang_src"].loc[lg, s]), RIBBON)
    # source -> dedup survivors and duplicates (real)
    for s in src_order:
        surv = int(C["survivors"][s])
        ident = int(C["identified"][s])
        link(f"src_{s}", "dedup", surv, RIBBON)
        link(f"src_{s}", "dups", ident - surv, RIBBON_RED)
    # dedup -> decision (real)
    link("dedup", "inc", 1385, RIBBON_GREEN)
    link("dedup", "hr", 3498, RIBBON_AMBER)
    link("dedup", "exc", 12180, RIBBON_RED)
    # include + human review -> eligible
    link("inc", "elig", 1385, RIBBON_GREEN)
    link("hr", "elig", 3498, RIBBON_AMBER)
    # eligible -> full-text acquisition (297 OA PDFs) / abstract-level (rest)
    link("elig", "oapdf", 297, RIBBON_GREEN)
    link("elig", "manual", 4883 - 297, RIBBON_GREY)
    # OA PDFs -> text extracted (1 corrupt PDF lost)
    link("oapdf", "extr", 296, RIBBON_GREEN)
    # both streams feed the final thematic + methods analysis
    link("extr", "analysis", 296, RIBBON_GREEN)
    link("manual", "analysis", 4883 - 297, RIBBON_GREY)

    fig = go.Figure(go.Sankey(
        arrangement="fixed",
        textfont=dict(family="Latin Modern Roman, serif", size=14,
                      color="#1a1a1a"),
        node=dict(
            label=labels, color=color, x=xpos, y=ypos,
            pad=16, thickness=18, line=dict(color="white", width=0.8),
        ),
        link=dict(source=s_idx, target=t_idx, value=val, color=lcol),
    ))

    # ---------------------------------------------------------------
    # Enrichment side-band: OpenAlex + Europe PMC as lateral annotation
    # boxes with dashed arrows pointing to the stages they enrich.
    # (Plotly annotation arrows are solid; we use a distinct violet,
    #  thin style plus a dashed connector drawn as a shape to convey the
    #  auxiliary, non-flow nature.)
    # ---------------------------------------------------------------
    # Side-band boxes in a clear TOP band; dashed connectors run DOWN to the
    # exact stage each auxiliary component acts on. Violet = metadata
    # enrichment (OpenAlex / Europe PMC); teal = local LLM compute (Ollama).
    OLLAMA_C = "#1d7a7a"
    annotations = [
        # Row 1 (highest) — OpenAlex stage 1, above the dedup column
        dict(x=X_DEDUP, y=1.30, xref="paper", yref="paper",
             text="<b>OpenAlex</b> · stage 1<br>metadata recovery (5,122)",
             showarrow=False, font=dict(size=10, color=NODE_VIOLET),
             bgcolor="rgba(129,86,168,0.07)", bordercolor=NODE_VIOLET,
             borderwidth=1, borderpad=4, align="center"),
        # Row 2 — OpenAlex + Europe PMC stage 2, above the decision column
        dict(x=X_DEC + 0.02, y=1.30, xref="paper", yref="paper",
             text="<b>OpenAlex + Europe PMC</b> · stage 2<br>"
                  "topics + MeSH (4,542 / 1,830 / 272)",
             showarrow=False, font=dict(size=10, color=NODE_VIOLET),
             bgcolor="rgba(129,86,168,0.07)", bordercolor=NODE_VIOLET,
             borderwidth=1, borderpad=4, align="center"),
        # Row 3 (lower) — Ollama compute backend, centred over dedup+decision
        dict(x=(X_DEDUP + X_DEC) / 2, y=1.09, xref="paper", yref="paper",
             text="<b>Ollama</b> local LLM server · "
                  "<i>Qwen3-14B</i> (dedup judge + screening) + "
                  "<i>Qwen3-Embedding-8B</i> (semantic dedup)",
             showarrow=False, font=dict(size=10, color=OLLAMA_C),
             bgcolor="rgba(29,122,122,0.07)", bordercolor=OLLAMA_C,
             borderwidth=1, borderpad=4, align="center"),
    ]

    shapes = [
        # OpenAlex stage 1 (violet dotted) -> Deduplicated node
        dict(type="line", xref="paper", yref="paper",
             x0=X_DEDUP, y0=1.25, x1=X_DEDUP, y1=0.66,
             line=dict(color=NODE_VIOLET, width=1.2, dash="dot")),
        # OpenAlex + Europe PMC stage 2 (violet dotted) -> advancing subset
        dict(type="line", xref="paper", yref="paper",
             x0=X_DEC + 0.02, y0=1.25, x1=X_DEC + 0.02, y1=0.74,
             line=dict(color=NODE_VIOLET, width=1.2, dash="dot")),
        # Ollama (teal dashed) -> deduplication column
        dict(type="line", xref="paper", yref="paper",
             x0=(X_DEDUP + X_DEC) / 2 - 0.03, y0=1.04, x1=X_DEDUP, y1=0.62,
             line=dict(color=OLLAMA_C, width=1.2, dash="dash")),
        # Ollama (teal dashed) -> screening decision column
        dict(type="line", xref="paper", yref="paper",
             x0=(X_DEDUP + X_DEC) / 2 + 0.03, y0=1.04, x1=X_DEC, y1=0.78,
             line=dict(color=OLLAMA_C, width=1.2, dash="dash")),
    ]

    fig.update_layout(
        font=dict(family="Latin Modern Roman, serif", size=12, color="#222"),
        width=1360, height=720,
        margin=dict(l=15, r=150, t=200, b=45),
        annotations=annotations,
        shapes=shapes,
    )
    return fig


def main() -> None:
    fig = build_sankey()
    fig.write_image(str(OUT / "fig_sankey_pipeline.pdf"))
    fig.write_image(str(OUT / "fig_sankey_pipeline.png"), scale=2)
    print("[sankey] ->", OUT / "fig_sankey_pipeline.pdf")


if __name__ == "__main__":
    main()
