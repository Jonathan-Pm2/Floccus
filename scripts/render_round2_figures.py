"""Regenerate the manuscript's data-driven figure and table sources for round 2.

Only the sources whose numbers change with the round-2 corpus
(scripts/build_round2_corpus.py) are written, EN and ES variants alike:

  fig07_prisma.tex / fig07_prisma_es.tex     PRISMA 2020 flow (TikZ)
  tab_corpus_clusters.tex / _es.tex          thematic clusters of the fitted
                                             partition (4,682 embedded records
                                             minus those merged away)
  fig_sankey_pipeline.pdf / .png             scripts/render_sankey_pipeline.py
  fig10_umap_clusters.pdf / .png             scripts/render_figures_en.py, with
                                             the merged-away records masked

tab_methods_distribution and fig12_methods_temporal depend on the 297 full
texts, none of which is merged away (round2_counts.json: fulltext), so they
are not regenerated. The TikZ and table templates reproduce the round-1
sources byte for byte when fed the round-1 counts (tests/test_render_round2_figures.py).

Run:  .venv/bin/python scripts/render_round2_figures.py \\
          [--out-dir ../algorithms/revision2/generated] [--check-round1 ../algorithms/docs/figures_en]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "processed"
sys.path.insert(0, str(ROOT / "scripts"))

DEFAULT_OUT = ROOT.parent / "algorithms" / "revision2" / "generated"
FOCUS = (3, 15)

LABELS_EN = {
    1: "Activated-sludge reactor (general)", 9: "Membrane bioreactors (MBR)", 24: "Phosphorus removal / EBPR",
    6: "Anaerobic digestion / biogas", 21: "Process control / fuzzy logic", 18: "Anammox / partial nitritation",
    13: "Pharmaceuticals and antibiotic resistance", 12: "Biodegradation kinetics", 3: "Biofloc in aquaculture",
    2: "Periphyton and freshwater ecology", 23: "Plant simulation (GPS-X)", 10: "Biofilm and moving/fixed bed",
    4: "Aerobic granular sludge (AGS)", 8: "Microbial ecology in WWTPs", 5: "Energy and life-cycle of WWTPs",
    11: "$N_2O$ / GHG emissions", 0: "Microalgae-bacteria consortia", 22: "Neural networks / ML",
    14: "Secondary settlers / CFD", 17: "Textile wastewater", 15: "Floc structure / rheology",
    19: "Sequencing batch reactor (SBR)", 7: "EPS, microplastics, nanoparticles",
    16: "Bulking and filamentous bacteria", 20: "Nitrification / denitrification",
}
LABELS_ES = {
    1: "Reactor de lodos activados (general)", 9: "Biorreactores de membrana (MBR)", 24: "Remoción de fósforo / EBPR",
    6: "Digestión anaerobia / biogás", 21: "Control de procesos / lógica difusa", 18: "Anammox / nitritación parcial",
    13: "Fármacos y resistencia a antibióticos", 12: "Cinética de biodegradación", 3: "Biofloc en acuicultura",
    2: "Perifiton y ecología de agua dulce", 23: "Simulación de planta (GPS-X)", 10: "Biopelícula y lecho móvil/fijo",
    4: "Lodos granulares aerobios (AGS)", 8: "Ecología microbiana en EDAR", 5: "Energía y ciclo de vida de EDAR",
    11: "Emisiones de $N_2O$ / GEI", 0: "Consorcios microalgas-bacterias", 22: "Redes neuronales / ML",
    14: "Decantadores secundarios / CFD", 17: "Aguas residuales textiles", 15: "Estructura / reología del flóculo",
    19: "Reactor discontinuo secuencial (SBR)", 7: "EPS, microplásticos, nanopartículas",
    16: "Bulking y bacterias filamentosas", 20: "Nitrificación / desnitrificación",
}

PRISMA_TEXT = {
    "en": dict(
        ident="Records identified from databases", phase_ident="Identif.",
        dedup="Unique records after multi-layer\n      deduplication", dups="Duplicates removed",
        dups_note="Layers 1+2+3 + SciELO recovery",
        l1="Layer 1 screening --- 11 binary PRISMA criteria", l1excl="Excluded at Layer 1", phase_screen="Screening",
        l1out="Advanced to consolidation / Layer 2", l2="Layer 2 LLM screening --- enriched input",
        l2excl_note="L2 \\textsc{Include} promoted", final="Consolidated to full-text eligibility", phase_elig="Eligib.",
        caption=("PRISMA 2020 flow diagram of the identification, deduplication and\n"
                 "    screening pipeline. The Layer 2 LLM screening ran in full on the ${l2}$\n"
                 "    records derived from Layer 1: ${l2excl}$ were $\\textsc{{Excluded}}$, ${l2inc}$\n"
                 "    promoted to $\\textsc{{Include}}$, and ${hr}$ (${stay}\\%$) preserved as\n"
                 "    $\\textsc{{HumanReview}}$. The total movement rate (${move}\\%$, mostly toward\n"
                 "    $\\textsc{{Exclude}}$) quantifies the contribution of the second pass with\n"
                 "    enriched metadata on the borderline set."),
    ),
    "es": dict(
        ident="Registros identificados en bases de datos", phase_ident="Identif.",
        dedup="Registros únicos tras la deduplicación\n      multicapa", dups="Duplicados eliminados",
        dups_note="Capas 1+2+3 + recuperación SciELO",
        l1="Cribado de Capa 1 --- 11 criterios PRISMA binarios", l1excl="Excluidos en la Capa 1", phase_screen="Cribado",
        l1out="Avanzan a consolidación / Capa 2", l2="Cribado LLM de Capa 2 --- entrada enriquecida",
        l2excl_note="L2 promovidos a \\textsc{Include}", final="Consolidados a elegibilidad a texto completo",
        phase_elig="Elegib.",
        caption=("Diagrama de flujo PRISMA 2020 del pipeline de identificación,\n"
                 "    deduplicación y cribado. El cribado LLM de Capa 2 se ejecutó en su\n"
                 "    totalidad sobre los ${l2}$ registros derivados de la Capa 1: ${l2excl}$\n"
                 "    fueron \\textsc{{Excluded}}, ${l2inc}$ promovidos a \\textsc{{Include}} y\n"
                 "    ${hr}$ (${stay}\\%$) conservados como \\textsc{{HumanReview}}. La tasa de\n"
                 "    movimiento total (${move}\\%$, mayoritariamente hacia \\textsc{{Exclude}})\n"
                 "    cuantifica la contribución de la segunda pasada con metadatos\n"
                 "    enriquecidos sobre el conjunto limítrofe."),
    ),
}
ROUND1_HEADER = ("%% PRISMA 2020 flow diagram — Biofloc systematic review.\n"
                 "%% Generated from data/processed/prisma_counts.json.\n"
                 "%% State: post-Layer-2 LLM complete + IEEE re-screening (530/530 unique record_id).\n")
ROUND2_HEADER = ("%% PRISMA 2020 flow diagram — Biofloc systematic review.\n"
                 "%% Generated from data/processed/round2_counts.json (scripts/render_round2_figures.py).\n"
                 "%% State: round 2 — 98 verified IEEE Xplore duplicates merged (retention rule), "
                 "E5 language criterion retired and its 31 records re-screened.\n")

PRISMA_TEMPLATE = r"""{header}%% Input: \input{{figures_en/fig07_prisma.tex}}
\begin{{figure}}[H]
  \centering
  \resizebox{{\textwidth}}{{!}}{{% fit the text width exactly
  \begin{{tikzpicture}}[
    node distance=0.5cm and 0.7cm,
    every node/.style={{font=\scriptsize}},
    main/.style   ={{rectangle, draw, rounded corners=2pt, align=center,
                    fill=teal!18, minimum width=5.5cm, minimum height=0.9cm,
                    inner sep=3pt, line width=0.4pt}},
    outbox/.style ={{rectangle, draw, rounded corners=2pt, align=center,
                    fill=black!85, text=white, minimum width=5.5cm,
                    minimum height=0.9cm, inner sep=3pt, line width=0.4pt,
                    font=\scriptsize\bfseries}},
    excl/.style   ={{rectangle, draw, rounded corners=2pt, align=left,
                    fill=red!8, minimum width=4.0cm, minimum height=0.9cm,
                    inner sep=3pt, line width=0.4pt}},
    pending/.style={{rectangle, draw, dashed, rounded corners=2pt, align=center,
                    fill=gray!12, minimum width=5.5cm, minimum height=0.9cm,
                    inner sep=3pt, font=\scriptsize\itshape}},
    phase/.style  ={{font=\footnotesize\bfseries, anchor=east, text=teal!50!black}},
    arr/.style    ={{-{{Stealth[length=2mm]}}, thick, gray!60!black}},
    arrd/.style   ={{-{{Stealth[length=2mm]}}, thick, dashed, gray!60!black}},
  ]
    %% ----- Identification -----
    \node[main] (ident) {{{t[ident]} ($n={ident}$) \\
      \footnotesize Scopus ${scopus}$ \;\textbar\; WoS ${wos}$ \;\textbar\;
      IEEE Xplore ${ieee}$ \;\textbar\; SciELO ${scielo}$}};
    \node[phase] at ([xshift=-0.5cm]ident.west) {{{t[phase_ident]}}};

    %% ----- Deduplication -----
    \node[main, below=of ident] (dedup) {{{t[dedup]} ($n={unique}$)}};
    \node[excl, right=of dedup] (deduplicates) {{{t[dups]} ($n={dups}$) \\
      \footnotesize {dups_note}}};
    \draw[arr] (ident) -- (dedup);
    \draw[arr] (dedup.east) -- (deduplicates.west);

    %% ----- Layer 1 Screening -----
    \node[main, below=of dedup] (l1) {{{t[l1]} \\
      $(n={unique})$}};
    \node[excl, right=of l1] (l1excl) {{{t[l1excl]} ($n={l1excl}$) \\
      \footnotesize $\exists e\in\mathcal{{E}}: e=\textsc{{Yes}}$}};
    \draw[arr] (dedup) -- (l1);
    \draw[arr] (l1.east) -- (l1excl.west);
    \node[phase] at ([xshift=-0.5cm]l1.west) {{{t[phase_screen]}}};

    %% ----- Layer 1 output -----
    \node[main, below=of l1] (l1out)
      {{{t[l1out]} ($n={l1adv}$) \\
       \footnotesize \textsc{{Include}} ${l1inc}$ + \textsc{{HumanReview}} ${l1hr}$}};
    \draw[arr] (l1) -- (l1out);

    %% ----- Layer 2 Screening -----
    \node[main, below=of l1out] (l2)
      {{{t[l2]} \\
       OpenAlex + Europe PMC \;($n={l1hr}$)}};
    \node[excl, right=of l2] (l2excl)
      {{L2 \textsc{{Excluded}} ($n={l2excl}$) \\
       \footnotesize {t[l2excl_note]} ($n={l2inc}$)}};
    \draw[arr] (l1out) -- (l2);
    \draw[arr] (l2.east) -- (l2excl.west);

    %% ----- Consolidated final -----
    \node[outbox, below=of l2] (final)
      {{{t[final]} ($n={elig}$) \\
       \footnotesize \textsc{{Include}} ${inc}$ + \textsc{{HumanReview}} ${hr}$}};
    \draw[arr] (l2) -- (final);
    \node[phase] at ([xshift=-0.5cm]final.west) {{{t[phase_elig]}}};

  \end{{tikzpicture}}}}
  \caption{{{caption}}}
  \label{{fig:prisma}}
\end{{figure}}
"""

CLUSTER_TEXT = {
    "en": dict(
        caption=("Thematic structure of the eligible corpus obtained by\n"
                 "    \\textsc{{Umap}}+\\textsc{{Hdbscan}} on the ${n}$ dense embeddings\n"
                 "    of the records that advanced to eligibility. Labels were assigned\n"
                 "    manually by inspecting the top tokens and representative titles\n"
                 "    of each \\emph{{cluster}}; the ordering reflects descending size.\n"
                 "    The ``Median year'' column characterises the temporal maturity\n"
                 "    of each sub-topic."),
        head="ID & Thematic label & $n$ & \\% & Median year \\\\",
        noise="--  & \\emph{{Non-grouped (noise)}} & ${noise}$ & ${noise_pct}$ & --- \\\\",
        total="\\multicolumn{{2}}{{l}}{{\\textbf{{Total eligible corpus}}}} & $\\mathbf{{{n}}}$ & $\\mathbf{{100.0}}$ & --- \\\\",
        foot="\\multicolumn{5}{l}{\\scriptsize $^{\\bigstar}$ Clusters structurally aligned with the research question.} \\\\",
    ),
    "es": dict(
        caption=("Estructura temática del corpus elegible obtenida mediante\n"
                 "    \\textsc{{Umap}}+\\textsc{{Hdbscan}} sobre los ${n}$ embeddings densos\n"
                 "    de los registros que avanzaron a elegibilidad. Las etiquetas se\n"
                 "    asignaron manualmente inspeccionando los tokens principales y los\n"
                 "    títulos representativos de cada \\emph{{clúster}}; el orden refleja el\n"
                 "    tamaño descendente. La columna ``Año mediano'' caracteriza la madurez\n"
                 "    temporal de cada subtema."),
        head="ID & Etiqueta temática & $n$ & \\% & Año mediano \\\\",
        noise="--  & \\emph{{No agrupados (ruido)}} & ${noise}$ & ${noise_pct}$ & --- \\\\",
        total="\\multicolumn{{2}}{{l}}{{\\textbf{{Total corpus elegible}}}} & $\\mathbf{{{n}}}$ & $\\mathbf{{100.0}}$ & --- \\\\",
        foot="\\multicolumn{5}{l}{\\scriptsize $^{\\bigstar}$ Clústeres estructuralmente alineados con la pregunta de investigación.} \\\\",
    ),
}

CLUSTER_TEMPLATE = """\\begin{{table}}[H]
  \\centering
  \\caption{{{caption}}}
  \\label{{tab:corpus-clusters}}
  \\scriptsize
  \\begin{{tabular}}{{rlrrr}}
    \\toprule
    {head}
    \\midrule
{rows}
    \\midrule
    {noise}
    \\midrule
    {total}
    \\bottomrule
    {foot}
  \\end{{tabular}}
\\end{{table}}
"""


# ---------------------------------------------------------------- pure helpers

def tex_int(n: int) -> str:
    """Thousands separator as the manuscript writes it: 17{,}063."""
    return f"{int(n):,}".replace(",", "{,}")


def prisma_tex(c: dict, lang: str, header: str = ROUND2_HEADER, dups_note: str | None = None) -> str:
    """c: identified, by_source{scopus,wos,ieee,scielo}, unique, l1_include, l1_exclude,
    l1_human_review, l2_include, l2_exclude, include, human_review."""
    t = PRISMA_TEXT[lang]
    l1adv = c["l1_include"] + c["l1_human_review"]
    elig = c["include"] + c["human_review"]
    stay = round(100 * c["human_review"] / c["l1_human_review"], 1)
    nums = dict(ident=tex_int(c["identified"]), scopus=tex_int(c["by_source"]["scopus"]),
                wos=tex_int(c["by_source"]["wos"]), ieee=tex_int(c["by_source"]["ieee"]),
                scielo=tex_int(c["by_source"]["scielo"]), unique=tex_int(c["unique"]),
                dups=tex_int(c["identified"] - c["unique"]), l1excl=tex_int(c["l1_exclude"]),
                l1adv=tex_int(l1adv), l1inc=tex_int(c["l1_include"]), l1hr=tex_int(c["l1_human_review"]),
                l2excl=tex_int(c["l2_exclude"]), l2inc=tex_int(c["l2_include"]), elig=tex_int(elig),
                inc=tex_int(c["include"]), hr=tex_int(c["human_review"]))
    caption = t["caption"].format(l2=nums["l1hr"], l2excl=nums["l2excl"], l2inc=nums["l2inc"], hr=nums["hr"],
                                  stay=f"{stay:.1f}", move=f"{100 - stay:.1f}")
    return PRISMA_TEMPLATE.format(header=header, t=t, caption=caption,
                                  dups_note=dups_note or t["dups_note"], **nums)


def cluster_rows(sizes: dict[int, int], medians: dict[int, int], labels: dict[int, str]) -> str:
    """Rows ordered by descending size, then cluster id; C03 and C15 in bold with a star."""
    n = sum(sizes.values())
    lines = []
    for k in sorted(sizes, key=lambda c: (-sizes[c], c)):
        if k < 0:
            continue
        pct = f"{100 * sizes[k] / n:.1f}"
        if k in FOCUS:
            lines.append(f"    C{k:02d}$^{{\\bigstar}}$ & \\textbf{{{labels[k]}}} & $\\mathbf{{{sizes[k]}}}$ "
                         f"& $\\mathbf{{{pct}}}$ & ${medians[k]}$ \\\\")
        else:
            lines.append(f"    C{k:02d} & {labels[k]} & ${sizes[k]}$ & ${pct}$ & ${medians[k]}$ \\\\")
    return "\n".join(lines)


def cluster_tex(sizes: dict[int, int], medians: dict[int, int], lang: str) -> str:
    t = CLUSTER_TEXT[lang]
    n = sum(sizes.values())
    noise = sizes.get(-1, 0)
    labels = LABELS_EN if lang == "en" else LABELS_ES
    return CLUSTER_TEMPLATE.format(
        caption=t["caption"].format(n=tex_int(n)), head=t["head"],
        rows=cluster_rows(sizes, medians, labels),
        noise=t["noise"].format(noise=tex_int(noise), noise_pct=f"{100 * noise / n:.1f}"),
        total=t["total"].format(n=tex_int(n)), foot=t["foot"])


# ---------------------------------------------------------------- data

def round1_prisma_counts() -> dict:
    p = json.loads((DATA / "prisma_counts.json").read_text())
    return {"identified": p["identification"]["total"], "by_source": p["identification"]["by_source"],
            "unique": p["deduplication"]["post_capa3_unique"],
            "l1_include": p["screening_layer1"]["include"], "l1_exclude": p["screening_layer1"]["exclude"],
            "l1_human_review": p["screening_layer1"]["human_review"],
            "l2_include": p["screening_layer2"]["l2_include"], "l2_exclude": p["screening_layer2"]["l2_exclude"],
            "include": p["final_decisions"]["include"], "human_review": p["final_decisions"]["human_review"]}


def round2_prisma_counts(counts: dict) -> dict:
    p = counts["prisma"]
    r1 = round1_prisma_counts()
    return {"identified": p["identified"], "by_source": r1["by_source"], "unique": p["unique"]["new"],
            "l1_include": p["screening_layer1"]["include"]["new"], "l1_exclude": p["screening_layer1"]["exclude"]["new"],
            "l1_human_review": p["screening_layer1"]["human_review"]["new"],
            "l2_include": p["screening_layer2"]["l2_include"]["new"], "l2_exclude": p["screening_layer2"]["l2_exclude"]["new"],
            "include": p["final_specificity_mode"]["include"]["new"],
            "human_review": p["final_specificity_mode"]["human_review"]["new"]}


def fitted_partition(dropped: set[str]) -> tuple[pd.DataFrame, np.ndarray]:
    """Archived fitted partition (4,682) minus the merged-away records, and the keep mask
    in the order of eligible_clusters.npy / eligible_umap2d.npy."""
    a = pd.read_parquet(DATA / "eligible_clusters_assignment.parquet", columns=["record_id", "cluster", "year"])
    a["record_id"] = a.record_id.astype(str)
    ids = [r for r in (DATA / "eligible_record_ids.txt").read_text().split("\n") if r]
    keep = np.array([r not in dropped for r in ids])
    return a[~a.record_id.isin(dropped)].reset_index(drop=True), keep


def sizes_and_medians(a: pd.DataFrame) -> tuple[dict[int, int], dict[int, int]]:
    sizes = {int(k): int(v) for k, v in a.cluster.value_counts().items()}
    medians = {int(k): int(g.year.median()) for k, g in a[a.cluster >= 0].groupby("cluster")}
    return sizes, medians


def check_round1(figdir: Path) -> None:
    """The templates must reproduce the round-1 sources exactly."""
    c = round1_prisma_counts()
    a, _ = fitted_partition(set())
    sizes, med = sizes_and_medians(a)
    for name, text in (("fig07_prisma.tex", prisma_tex(c, "en", ROUND1_HEADER)),
                       ("fig07_prisma_es.tex", prisma_tex(c, "es", ROUND1_HEADER)),
                       ("tab_corpus_clusters.tex", cluster_tex(sizes, med, "en")),
                       ("tab_corpus_clusters_es.tex", cluster_tex(sizes, med, "es"))):
        ref = (figdir / name).read_text()
        if ref != text:
            sys.exit(f"{name}: template does not reproduce the round-1 source; stopping")
        print(f"  round-1 reproduction ok: {name}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--check-round1", type=Path, default=ROOT.parent / "algorithms" / "docs" / "figures_en",
                    help="round-1 figures_en directory used to verify the templates (skip with '')")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if str(args.check_round1):
        check_round1(args.check_round1)

    counts = json.loads((DATA / "round2_counts.json").read_text())
    dec = pd.read_parquet(DATA / "round2_final_decisions.parquet",
                          columns=["record_id", "round2_change", "in_round2_corpus"])
    dropped = set(dec.loc[~dec.in_round2_corpus, "record_id"].astype(str))
    n_merged = counts["corrections"]["ieee_verified_merges"]["records_leaving"]
    notes = {"en": f"Layers 1+2+3 + SciELO recovery + {n_merged} IEEE merges",
             "es": f"Capas 1+2+3 + recuperación SciELO + {n_merged} fusiones IEEE"}

    c2 = round2_prisma_counts(counts)
    for lang, name in (("en", "fig07_prisma.tex"), ("es", "fig07_prisma_es.tex")):
        (args.out_dir / name).write_text(prisma_tex(c2, lang, ROUND2_HEADER, notes[lang]))
        print(f"  wrote {name}")

    a, keep = fitted_partition(dropped)
    sizes, med = sizes_and_medians(a)
    for lang, name in (("en", "tab_corpus_clusters.tex"), ("es", "tab_corpus_clusters_es.tex")):
        (args.out_dir / name).write_text(cluster_tex(sizes, med, lang))
        print(f"  wrote {name} (n = {len(a)})")

    ft = counts["fulltext"]
    if ft["methods_tables_and_fig12_affected"]:
        print("  WARNING: full-text set changed; tab_methods_distribution / fig12 must be regenerated by hand")
    import render_sankey_pipeline as sankey
    fig = sankey.build_sankey(DATA / "round2_final_decisions.parquet", ft["oa_pdfs"]["new"], ft["text_extracted"]["new"])
    fig.write_image(str(args.out_dir / "fig_sankey_pipeline.pdf"))
    fig.write_image(str(args.out_dir / "fig_sankey_pipeline.png"), scale=2)
    print("  wrote fig_sankey_pipeline.pdf/.png")

    import render_figures_en as rfe
    rfe.OUT = args.out_dir
    rfe.fig10_umap_clusters(keep=keep)
    print(f"-> {args.out_dir}")


if __name__ == "__main__":
    main()
