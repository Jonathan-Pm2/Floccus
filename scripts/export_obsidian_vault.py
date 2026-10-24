"""Export the Floccus eligible corpus as a two-layer Obsidian knowledge graph.

Layer 1 -- corpus: one note per eligible record, per thematic cluster, per
sub-cluster and per canonical reference, linked by cluster membership,
C03--C15 bridge score and the modelling methods detected in full text.

Layer 2 -- equations: one note per equation family, per labelled equation of
the manuscript and per coupling operator, linked as in the multi-scale
hierarchy of the paper.

The two layers meet through the Methods notes: a paper that mentions a
population balance in its full text links to the PBM method note, which links
to the PBM family note, which links to every PBM equation. Nothing here is
inferred; every edge is read from a persisted pipeline output or from the
manuscript's own equation catalogue.

Obsidian renders [[wikilinks]] as graph edges. Open the output folder with
"Open folder as vault". Use the graph filters (path:Clusters, path:Equations,
tag:#proposed) to switch between layers.

Run:
    .venv/bin/python scripts/export_obsidian_vault.py --out ../floccus-knowledge-graph
    .venv/bin/python scripts/export_obsidian_vault.py --figure ../algorithms/docs/figures_en/fig_knowledge_graph.pdf
"""
from __future__ import annotations

import argparse
import re
import shutil
from collections import defaultdict
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "processed"

# --------------------------------------------------------------------------
# Static knowledge: cluster labels, equation catalogue, hierarchy edges.
# All taken from the manuscript (tab:corpus-clusters, tab:c15-subclusters,
# the equation labels and fig:framework-hierarchy); nothing invented here.
# --------------------------------------------------------------------------
CLUSTER_LABEL = {
    -1: "Transversal (non-grouped)",
    0: "Microalgae-bacteria consortia", 1: "Activated-sludge reactor (general)",
    2: "Periphyton and freshwater ecology", 3: "Biofloc in aquaculture",
    4: "Aerobic granular sludge (AGS)", 5: "Energy and life-cycle of WWTPs",
    6: "Anaerobic digestion / biogas", 7: "EPS, microplastics, nanoparticles",
    8: "Microbial ecology in WWTPs", 9: "Membrane bioreactors (MBR)",
    10: "Biofilm and moving/fixed bed", 11: "N2O / GHG emissions",
    12: "Biodegradation kinetics", 13: "Pharmaceuticals and antibiotic resistance",
    14: "Secondary settlers / CFD", 15: "Floc structure / rheology",
    16: "Bulking and filamentous bacteria", 17: "Textile wastewater",
    18: "Anammox / partial nitritation", 19: "Sequencing batch reactor (SBR)",
    20: "Nitrification / denitrification", 21: "Process control / fuzzy logic",
    22: "Neural networks / ML", 23: "Plant simulation (GPS-X)",
    24: "Phosphorus removal / EBPR",
}
CORE_CLUSTERS = {3, 15}

SUBCLUSTER_LABEL = {
    (3, 0): "Ibero-American production culture (block A)",
    (3, 1): "Ibero-American production culture (block A)",
    (3, 2): "Modern quantitative block (B)",
    (3, 3): "Modern quantitative block (B)",
    (15, 0): "Cohesive sediments / estuarine fractal",
    (15, 1): "Rheology / yield stress",
    (15, 2): "Population balance + bioflocculation",
    (15, 3): "Microbial structure / EPS / dynamics",
}

FAMILY = {
    "monod": "Monod kinetics (ODE)",
    "asm": "Activated-sludge models (ASM)",
    "pbm": "Population balance of flocs (PBM)",
    "langevin": "Langevin / Fokker-Planck",
    "cfd": "Navier-Stokes / CFD",
    "pinn": "Data-driven and physics-informed (PINN)",
    "floccus": "Floccus screening algorithm",
}

# equation label -> (family key, short title, proposed?)
EQUATIONS = {
    "growth": ("monod", "Biomass balance (chemostat)", False),
    "substrate": ("monod", "Substrate balance", False),
    "monod": ("monod", "Monod saturation law", False),
    "monod-nondim": ("monod", "Non-dimensional Monod system", False),
    "asm": ("asm", "Petersen-matrix ASM system", False),
    "nh4": ("asm", "Biofloc ammonium balance", False),
    "pbm": ("pbm", "Smoluchowski population balance", False),
    "agg-kernel": ("pbm", "Saffman-Turner aggregation kernel", False),
    "langevin": ("langevin", "Langevin SDE for a passive floc", False),
    "fokker": ("langevin", "Fokker-Planck density equation", False),
    "continuity": ("cfd", "Continuity equation", False),
    "ns": ("cfd", "Incompressible Navier-Stokes", False),
    "coupling-meanfield": ("pbm", "Mean-field closure Langevin to PBM", False),
    "coupling-rans": ("cfd", "RANS k-epsilon closure and eddy diffusivity", False),
    "coupled-pbm-cfd": ("pbm", "Coupled population-balance-hydrodynamic equation", False),
    "biomass-moment": ("pbm", "Biomass as third moment of n(L)", True),
    "thiele": ("pbm", "Thiele modulus of a floc", True),
    "effectiveness": ("pbm", "Effectiveness factor of a spherical floc", True),
    "coupling-pbm-asm": ("asm", "PBM to ASM coupling via active biomass", True),
    "dissipation-power": ("cfd", "Mean dissipation from aeration power", True),
    "kla": ("cfd", "Oxygen transfer k_La (van't Riet)", True),
    "oxygen": ("asm", "Dissolved-oxygen balance with active sink", True),
    "closed-system": ("pinn", "Closed aeration-floc-nitrogen system", True),
    "pinn": ("pinn", "Generic PINN composite loss", False),
    "pinn-pbm-residual": ("pinn", "PINN residual on the biofloc PBM", False),
    "pinn-pbm-loss": ("pinn", "PINN loss with conservation penalties", False),
    "pinn-coupled-loss": ("pinn", "Multi-residual PINN loss on the closed system", True),
    "wls": ("pinn", "Weighted least-squares calibration", False),
    "sobol": ("pinn", "Sobol sensitivity indices", False),
    "cosine": ("floccus", "Cosine similarity of record embeddings", False),
    "screening-rule": ("floccus", "Asymmetric AND/OR screening rule", False),
    "remap": ("floccus", "Unclear remapping operators", False),
    "kappa": ("floccus", "Cohen's kappa agreement", False),
    "clustering": ("floccus", "UMAP+HDBSCAN cluster assignment", False),
    "bridge": ("floccus", "C03-C15 bridge score", False),
}

# coupling operators: (note title, from family, to family, equations, proposed?)
COUPLINGS = [
    ("Mean field (Langevin to Monod)", "langevin", "monod", [], False),
    ("Multi-component (Monod to ASM)", "monod", "asm", ["asm"], False),
    ("Mean ensemble (Langevin to Fokker-Planck)", "langevin", "langevin", ["fokker"], False),
    ("Emergent kernels (ASM to PBM)", "asm", "pbm", ["coupling-meanfield"], False),
    ("Statistical closure (Fokker-Planck to PBM)", "langevin", "pbm", ["coupling-meanfield"], False),
    ("Advection-diffusion (PBM to CFD)", "pbm", "cfd", ["coupling-rans", "coupled-pbm-cfd"], False),
    ("Operator N (PINN to any level)", "pinn", "langevin", ["pinn", "pinn-pbm-residual"], False),
    ("Thiele effectiveness (PBM to ASM)", "pbm", "asm",
     ["biomass-moment", "thiele", "effectiveness", "coupling-pbm-asm"], True),
    ("Aeration power (P to epsilon and kLa)", "cfd", "asm",
     ["dissipation-power", "kla", "oxygen"], True),
    ("Closed hierarchy (single input P)", "pinn", "pinn",
     ["closed-system", "pinn-coupled-loss"], True),
]

# full-text method flag -> families it evidences
METHOD_FAMILIES = {
    "ASM": ["asm"], "ODE": ["monod"], "PBM": ["pbm"], "CFD": ["cfd"],
    "ML_ANN": ["pinn"], "PINN": ["pinn"], "Stochastic": ["langevin"],
    "PDE": ["langevin", "cfd"], "ABM": ["langevin"],
}
ABSTRACT_METHOD_FAMILIES = {
    "has_PINN": "pinn", "has_NeuralODE": "pinn", "has_NeuralOperator": "pinn",
    "has_GraphNN": "pinn", "has_StochasticBio": "langevin", "has_DigitalTwin": "pinn",
    "has_Biplot": "floccus",
}

BRIDGE_THRESHOLD = 0.7


def _safe(label: str) -> str:
    """Note names double as file names: Obsidian forbids '/' in both."""
    return label.replace(" / ", " – ").replace("/", "-")


CLUSTER_LABEL = {k: _safe(v) for k, v in CLUSTER_LABEL.items()}
SUBCLUSTER_LABEL = {k: _safe(v) for k, v in SUBCLUSTER_LABEL.items()}
FAMILY = {k: _safe(v) for k, v in FAMILY.items()}
COUPLINGS = [(_safe(t), a, b, e, pr) for t, a, b, e, pr in COUPLINGS]


# --------------------------------------------------------------------------
def slug(text: str, limit: int = 80) -> str:
    """Obsidian-safe file stem: drop forbidden characters, collapse spaces."""
    text = re.sub(r'[\\/:*?"<>|#^\[\]]', " ", str(text))
    text = re.sub(r"\s+", " ", text).strip().rstrip(".")
    return text[:limit].rstrip() or "untitled"


def cluster_note_name(c: int) -> str:
    return f"C{c:02d} {CLUSTER_LABEL[c]}" if c >= 0 else "Transversal (non-grouped)"


def family_note_name(key: str) -> str:
    return FAMILY[key]


def eq_note_name(label: str) -> str:
    return f"eq {label}"


def write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


# --------------------------------------------------------------------------
def load() -> dict:
    assign = pd.read_parquet(DATA / "eligible_clusters_assignment.parquet")
    dec = pd.read_parquet(DATA / "biofloc_final_decisions.parquet")[
        ["record_id", "doi", "year", "venue", "source", "language", "final_decision"]]
    eligible = dec[dec["final_decision"].isin(["INCLUDE", "HUMAN_REVIEW"])]
    papers = eligible.merge(assign[["record_id", "cluster", "title"]], on="record_id", how="left")
    papers["cluster"] = papers["cluster"].fillna(-2).astype(int)   # -2 = eligible, no embedding
    # titles for records without an embedding come from the decisions table
    missing_title = papers["title"].isna()
    if missing_title.any():
        t = pd.read_parquet(DATA / "biofloc_final_decisions.parquet")[["record_id", "title"]]
        papers = papers.drop(columns="title").merge(t, on="record_id", how="left")

    sub = pd.concat([
        pd.read_parquet(DATA / "c03_subclusters.parquet")[["record_id", "sub_cluster"]].assign(core=3),
        pd.read_parquet(DATA / "c15_subclusters.parquet")[["record_id", "sub_cluster", "cit"]].assign(core=15),
    ])
    bridges = pd.read_parquet(DATA / "c03_c15_bridges.parquet")[["record_id", "sim_c03", "sim_c15", "bridge_score"]]
    methods = pd.read_parquet(DATA / "papers_with_methods.parquet")
    adv = pd.read_parquet(DATA / "advanced_methods_abstracts.parquet")
    summary = pd.read_parquet(DATA / "eligible_clusters_summary.parquet")
    cites = pd.read_parquet(DATA / "citations_by_cluster.parquet")
    topics = pd.read_parquet(DATA / "clusters_vs_oa_topics.parquet")
    decisions = pd.read_parquet(DATA / "cluster_decision_breakdown.parquet")
    mbc = pd.read_parquet(DATA / "methods_by_cluster.parquet")
    canon = pd.read_csv(DATA / "refs_keys_by_cluster.csv")
    return dict(papers=papers, sub=sub, bridges=bridges, methods=methods, adv=adv,
                summary=summary, cites=cites, topics=topics, decisions=decisions,
                mbc=mbc, canon=canon)


# --------------------------------------------------------------------------
def build_vault(out: Path, d: dict) -> dict:
    if out.exists():
        shutil.rmtree(out)
    papers, sub, bridges = d["papers"], d["sub"], d["bridges"]
    sub_by_id = {r.record_id: r for r in sub.itertuples()}
    bridge_by_id = {r.record_id: r for r in bridges.itertuples()}
    methods_by_id = {r.record_id: r for r in d["methods"].itertuples()}
    adv_by_id = {r.record_id: r for r in d["adv"].itertuples()}

    stats = defaultdict(int)
    paper_note: dict[str, str] = {}          # record_id -> note name
    used_names: set[str] = set()
    cluster_members: dict[int, list[str]] = defaultdict(list)
    sub_members: dict[tuple, list[str]] = defaultdict(list)
    bridge_members: list[tuple[float, str]] = []
    method_members: dict[str, list[str]] = defaultdict(list)
    family_paper_count: dict[str, int] = defaultdict(int)

    # ---------------- papers ----------------
    for r in papers.itertuples():
        year = int(r.year) if pd.notna(r.year) else 0
        base = f"{year} {slug(r.title, 70)}" if year else slug(r.title, 75)
        name = base
        if name in used_names:
            name = f"{base} ({r.record_id[:8]})"
        used_names.add(name)
        paper_note[r.record_id] = name
        c = int(r.cluster)
        folder = "Papers/no-embedding" if c == -2 else ("Papers/transversal" if c == -1 else f"Papers/C{c:02d}")

        links, tags = [], ["paper", f"decision/{r.final_decision.lower()}"]
        fm = {"record_id": r.record_id, "doi": r.doi if pd.notna(r.doi) else "", "year": year,
              "source": r.source, "venue": r.venue if pd.notna(r.venue) else "",
              "language": r.language if pd.notna(r.language) else "", "decision": r.final_decision,
              "cluster": c}
        if c >= -1:
            links.append(f"[[{cluster_note_name(c)}]]")
            cluster_members[c].append(name)
        if r.record_id in sub_by_id:
            s = sub_by_id[r.record_id]
            sc = int(s.sub_cluster)
            if sc >= 0:
                sname = f"C{int(s.core):02d}-SC{sc} {SUBCLUSTER_LABEL[(int(s.core), sc)]}"
                links.append(f"[[{sname}]]"); sub_members[(int(s.core), sc)].append(name)
                fm["subcluster"] = f"SC{sc}"
            if int(s.core) == 15 and pd.notna(getattr(s, "cit", None)):
                fm["citations_openalex"] = int(s.cit)
        if r.record_id in bridge_by_id:
            b = bridge_by_id[r.record_id]
            fm["bridge_score"] = round(float(b.bridge_score), 3)
            fm["sim_c03"] = round(float(b.sim_c03), 3); fm["sim_c15"] = round(float(b.sim_c15), 3)
            if b.bridge_score >= BRIDGE_THRESHOLD and c not in CORE_CLUSTERS:
                links.append("[[C03-C15 Bridge]]"); tags.append("bridge")
                bridge_members.append((float(b.bridge_score), name))
        meth = []
        if r.record_id in methods_by_id:
            m = methods_by_id[r.record_id]
            for key in METHOD_FAMILIES:
                if getattr(m, f"has_{key}", False):
                    meth.append(key); method_members[key].append(name)
                    for fam in METHOD_FAMILIES[key]:
                        family_paper_count[fam] += 1
        if r.record_id in adv_by_id:
            a = adv_by_id[r.record_id]
            for col, fam in ABSTRACT_METHOD_FAMILIES.items():
                if getattr(a, col, False):
                    key = col[4:]
                    if key not in meth:
                        meth.append(key); method_members[key].append(name)
                        family_paper_count[fam] += 1
        if meth:
            fm["methods"] = meth
            links += [f"[[Method {k}]]" for k in meth]
            tags.append("has-method")
        fm_txt = "\n".join(f"{k}: {v!r}" if isinstance(v, str) else f"{k}: {v}" for k, v in fm.items())
        body = (f"---\n{fm_txt}\ntags: [{', '.join(tags)}]\n---\n"
                f"# {r.title}\n\n"
                + (f"DOI: https://doi.org/{r.doi}\n\n" if pd.notna(r.doi) and r.doi else "")
                + "**Links:** " + " · ".join(links) + "\n")
        write(out / folder / f"{name}.md", body)
        stats["papers"] += 1

    # ---------------- clusters ----------------
    summ = d["summary"].set_index("cluster"); cites = d["cites"].set_index("cluster")
    topics = d["topics"].set_index("cluster"); decs = d["decisions"].set_index("cluster")
    mbc = d["mbc"].set_index("cluster")
    for c in sorted(set(papers["cluster"]) - {-2}):
        name = cluster_note_name(c)
        members = cluster_members[c]
        rows = []
        if c in cites.index:
            ci = cites.loc[c]
            rows.append(f"| citations (OpenAlex) | total {int(ci.total_cit)} · mean {ci.mean_cit} · median {int(ci.median_cit)} · max {int(ci.max_cit)} |")
        if c in decs.index:
            de = decs.loc[c]
            rows.append(f"| screening decisions | Include {int(de.n_include)} ({de.pct_include}%) · HumanReview {int(de.n_hr)} ({de.pct_hr}%) |")
        if c in summ.index:
            rows.append(f"| median year | {int(summ.loc[c].year_median)} |")
        if c in topics.index:
            t = topics.loc[c]
            rows.append(f"| OpenAlex topics | {t.top1_oa_topic} ({int(t.top1_count)}) · {t.top2_oa_topic} ({int(t.top2_count)}) · {t.top3_oa_topic} ({int(t.top3_count)}) |")
        fam_links = []
        if c in mbc.index:
            m = mbc.loc[c]
            for key in METHOD_FAMILIES:
                n = int(m.get(key, 0))
                if n:
                    fam_links.append(f"[[Method {key}]] ({n}/{int(m.n_pdfs)} full texts)")
        tags = ["cluster"] + (["core"] if c in CORE_CLUSTERS else [])
        body = (f"---\ncluster: {c}\nn: {len(members)}\ntags: [{', '.join(tags)}]\n---\n"
                f"# {name}\n\n| | |\n|---|---|\n" + "\n".join(rows) + "\n\n"
                + ("**Modelling methods detected in full text:** " + " · ".join(fam_links) + "\n\n" if fam_links else "")
                + ("**Sub-clusters:** " + " · ".join(f"[[C{c:02d}-SC{sc} {lbl}]]" for (cc, sc), lbl in SUBCLUSTER_LABEL.items() if cc == c) + "\n\n" if c in CORE_CLUSTERS else "")
                + (f"**Bridge hub:** [[C03-C15 Bridge]]\n\n" if c in CORE_CLUSTERS else "")
                + f"## Members ({len(members)})\n\n" + "\n".join(f"- [[{n}]]" for n in sorted(members)) + "\n")
        write(out / "Clusters" / f"{name}.md", body); stats["clusters"] += 1

    # ---------------- sub-clusters ----------------
    for (c, sc), lbl in SUBCLUSTER_LABEL.items():
        name = f"C{c:02d}-SC{sc} {lbl}"
        members = sub_members[(c, sc)]
        body = (f"---\ncluster: {c}\nsubcluster: {sc}\nn: {len(members)}\ntags: [subcluster]\n---\n"
                f"# {name}\n\nParent: [[{cluster_note_name(c)}]]\n\n## Members ({len(members)})\n\n"
                + "\n".join(f"- [[{n}]]" for n in sorted(members)) + "\n")
        write(out / "Clusters" / "Subclusters" / f"{name}.md", body); stats["subclusters"] += 1

    # ---------------- bridge hub ----------------
    bridge_members.sort(reverse=True)
    body = (f"---\ntags: [hub, bridge]\nthreshold: {BRIDGE_THRESHOLD}\nn: {len(bridge_members)}\n---\n"
            f"# C03-C15 Bridge\n\nPapers outside the two core clusters with bridge score "
            f"b(d) = min(sim_C03, sim_C15) >= {BRIDGE_THRESHOLD} (Eq. eq:bridge). "
            f"Cores: [[{cluster_note_name(3)}]] · [[{cluster_note_name(15)}]] · equation [[eq bridge]]\n\n"
            + "\n".join(f"- {b:.3f} · [[{n}]]" for b, n in bridge_members) + "\n")
    write(out / "Hubs" / "C03-C15 Bridge.md", body)

    # ---------------- canon (references cited by the manuscript) ----------------
    for r in d["canon"].itertuples():
        c = int(r.cluster); year = int(r.year) if pd.notna(r.year) else 0
        name = f"{slug(r.lead_author, 30)} {year} ({r.key})"
        body = (f"---\nbibkey: {r.key}\ncluster: {c}\nyear: {year}\ncitations: {int(r.orig_cit) if pd.notna(r.orig_cit) else 0}\ntags: [canon]\n---\n"
                f"# {r.orig_title}\n\nCited in the manuscript as `\\cite{{{r.key}}}`.\n\n"
                f"**Cluster:** [[{cluster_note_name(c) if c in CLUSTER_LABEL else 'Transversal (non-grouped)'}]]\n")
        write(out / "Canon" / f"{name}.md", body); stats["canon"] += 1

    # ---------------- methods (the hinge between layers) ----------------
    for key, fams in {**METHOD_FAMILIES, **{k[4:]: [v] for k, v in ABSTRACT_METHOD_FAMILIES.items()}}.items():
        members = method_members.get(key, [])
        body = (f"---\nmethod: {key}\nn_papers: {len(members)}\ntags: [method]\n---\n"
                f"# Method {key}\n\n**Equation family:** " + " · ".join(f"[[{family_note_name(f)}]]" for f in fams)
                + f"\n\n## Papers mentioning it ({len(members)})\n\n" + "\n".join(f"- [[{n}]]" for n in sorted(members)) + "\n")
        write(out / "Methods" / f"Method {key}.md", body); stats["methods"] += 1

    # ---------------- equation families ----------------
    for fkey, fname in FAMILY.items():
        eqs = [l for l, (f, _, _) in EQUATIONS.items() if f == fkey]
        coup = [t for t, a, b, _, _ in COUPLINGS if fkey in (a, b)]
        body = (f"---\nfamily: {fkey}\nn_equations: {len(eqs)}\npapers_evidencing: {family_paper_count.get(fkey, 0)}\ntags: [family]\n---\n"
                f"# {fname}\n\n## Equations\n\n" + "\n".join(f"- [[{eq_note_name(l)}]] — {EQUATIONS[l][1]}" + (" *(proposed)*" if EQUATIONS[l][2] else "") for l in eqs)
                + "\n\n## Coupling operators\n\n" + "\n".join(f"- [[{t}]]" for t in coup)
                + "\n\n## Methods evidencing this family in the corpus\n\n"
                + "\n".join(f"- [[Method {k}]]" for k, fs in METHOD_FAMILIES.items() if fkey in fs) + "\n")
        write(out / "Equations" / "Families" / f"{fname}.md", body); stats["families"] += 1

    # ---------------- equations ----------------
    for label, (fkey, title, proposed) in EQUATIONS.items():
        tags = ["equation"] + (["proposed"] if proposed else [])
        coup = [t for t, _, _, eqs, _ in COUPLINGS if label in eqs]
        body = (f"---\nlabel: eq:{label}\nfamily: {fkey}\nproposed: {str(proposed).lower()}\ntags: [{', '.join(tags)}]\n---\n"
                f"# {title}\n\n`\\ref{{eq:{label}}}` — family [[{family_note_name(fkey)}]]"
                + (" — **proposed in this paper**" if proposed else "")
                + ("\n\n**Coupling operators:** " + " · ".join(f"[[{t}]]" for t in coup) if coup else "") + "\n")
        write(out / "Equations" / f"{eq_note_name(label)}.md", body); stats["equations"] += 1

    # ---------------- coupling operators ----------------
    for title, a, b, eqs, proposed in COUPLINGS:
        tags = ["coupling"] + (["proposed"] if proposed else [])
        body = (f"---\nfrom: {a}\nto: {b}\nproposed: {str(proposed).lower()}\ntags: [{', '.join(tags)}]\n---\n"
                f"# {title}\n\n[[{family_note_name(a)}]] → [[{family_note_name(b)}]]"
                + (" — **proposed in this paper**" if proposed else "")
                + ("\n\n**Equations:** " + " · ".join(f"[[{eq_note_name(l)}]]" for l in eqs) if eqs else "") + "\n")
        write(out / "Equations" / "Couplings" / f"{title}.md", body); stats["couplings"] += 1

    # ---------------- README ----------------
    write(out / "README.md", f"""# Floccus knowledge graph

Generated by `scripts/export_obsidian_vault.py` from the persisted outputs of
the Floccus pipeline (Zenodo 10.5281/zenodo.21926426). Open this folder in
Obsidian with *Open folder as vault* and switch to Graph view.

| layer | folder | notes |
|---|---|---|
| corpus | `Papers/` | {stats['papers']} eligible records, one note each, grouped by cluster |
| corpus | `Clusters/` | {stats['clusters']} thematic clusters + {stats['subclusters']} sub-clusters |
| corpus | `Canon/` | {stats['canon']} references cited by the manuscript, mapped to clusters |
| hinge | `Methods/` | {stats['methods']} modelling methods detected in full text / abstracts |
| equations | `Equations/Families/` | {stats['families']} equation families |
| equations | `Equations/` | {stats['equations']} labelled equations of the manuscript |
| equations | `Equations/Couplings/` | {stats['couplings']} coupling operators (edges made explicit) |
| hub | `Hubs/C03-C15 Bridge` | {len(bridge_members)} papers with bridge score >= {BRIDGE_THRESHOLD} |

## Useful graph filters

- `path:Equations` — the multi-scale hierarchy alone
- `path:Clusters OR path:Equations/Families OR path:Methods` — cluster-to-family view
- `tag:#proposed` — the equations and operators proposed in this paper
- `tag:#bridge` — the C03-C15 bridge papers
- `path:Papers/C03 OR path:Papers/C15` — the two core clusters

Every edge is read from a pipeline output (cluster assignment, sub-clusters,
bridge scores, full-text method census) or from the manuscript's equation
catalogue. No edge is inferred.
""")
    stats["bridge_papers"] = len(bridge_members)
    return stats


# --------------------------------------------------------------------------
def check_links(out: Path) -> tuple[int, list[str]]:
    """Every [[wikilink]] must resolve to an existing note."""
    names = {p.stem for p in out.rglob("*.md")}
    broken, total = [], 0
    for p in out.rglob("*.md"):
        for link in re.findall(r"\[\[([^\]|#]+)", p.read_text(encoding="utf-8")):
            total += 1
            if link not in names:
                broken.append(f"{p.relative_to(out)} -> {link}")
    return total, broken


# --------------------------------------------------------------------------
def build_figure(path: Path, d: dict, lang: str = "en") -> None:
    """Static cluster-to-family graph for the manuscript (networkx + matplotlib).

    The six equation families are pinned on a hexagon so the spring layout
    only has to place the clusters around them; otherwise the mutually
    coupled families collapse into an unreadable centre.
    """
    import math
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import networkx as nx

    SHORT = {"monod": "Monod\n(ODE)", "asm": "ASM", "pbm": "PBM", "langevin": "Langevin /\nFokker–Planck",
             "cfd": "Navier–Stokes\n/ CFD", "pinn": "PINN /\ndata-driven"}
    if lang == "es":
        SHORT = {**SHORT, "monod": "Monod\n(EDO)", "pinn": "PINN /\nbasados en datos"}
    mbc = d["mbc"].set_index("cluster")
    summ = d["summary"].set_index("cluster")
    G = nx.Graph()
    for c in summ.index:
        G.add_node(("c", int(c)), kind="cluster", n=int(summ.loc[c].n), core=int(c) in CORE_CLUSTERS)
    fams = [k for k in FAMILY if k != "floccus"]
    for fkey in fams:
        G.add_node(("f", fkey), kind="family")
    for c in mbc.index:
        if int(c) == -1 or ("c", int(c)) not in G:
            continue
        m = mbc.loc[c]
        fam_w = defaultdict(int)
        for key, fs in METHOD_FAMILIES.items():
            n = int(m.get(key, 0))
            for f in fs:
                fam_w[f] += n
        for f, w in fam_w.items():
            if w:
                G.add_edge(("c", int(c)), ("f", f), weight=w, kind="evidence")
    for title, a, b, _, proposed in COUPLINGS:
        if a != b:
            G.add_edge(("f", a), ("f", b), weight=3, kind="proposed" if proposed else "coupling")
    nb = int((d["bridges"]["bridge_score"] >= BRIDGE_THRESHOLD).sum())
    G.add_edge(("c", 3), ("c", 15), weight=nb, kind="bridge")

    # families pinned on a hexagon (order follows the hierarchy: elementary -> mean-field -> macroscopic -> hybrid)
    order = ["monod", "langevin", "asm", "pbm", "cfd", "pinn"]
    fixed = {("f", k): (0.42 * math.cos(math.pi / 2 + i * 2 * math.pi / 6),
                        0.42 * math.sin(math.pi / 2 + i * 2 * math.pi / 6)) for i, k in enumerate(order)}
    pos = nx.spring_layout(G, pos=fixed, fixed=list(fixed), seed=20260603, k=0.9, iterations=300, weight="weight")
    # clusters on an evenly spaced outer ring: keep the angular order the
    # spring layout found (each cluster near the families it evidences) but
    # equalise the spacing so every node and label is visible.
    cl_nodes = [n for n in G.nodes if n[0] == "c"]
    cl_nodes.sort(key=lambda n: math.atan2(pos[n][1], pos[n][0]))
    start = math.atan2(pos[cl_nodes[0]][1], pos[cl_nodes[0]][0])
    R = 0.88
    for i, n in enumerate(cl_nodes):
        ang = start + i * 2 * math.pi / len(cl_nodes)
        pos[n] = (R * math.cos(ang), R * math.sin(ang))

    fig, ax = plt.subplots(figsize=(10.5, 10.5))
    for kind, style in (("evidence", dict(color="#9aa5b1", alpha=.5)),
                        ("coupling", dict(color="#2f6d8c", alpha=.95)),
                        ("proposed", dict(color="#b0553f", alpha=.95, style="dashed")),
                        ("bridge", dict(color="#c19a3e", alpha=.95))):
        es = [(u, v) for u, v, k in G.edges(data="kind") if k == kind]
        ws = [0.5 + 0.3 * G[u][v]["weight"] ** 0.5 for u, v in es]
        nx.draw_networkx_edges(G, pos, edgelist=es, width=ws, edge_color=style["color"],
                               alpha=style["alpha"], style=style.get("style", "solid"), ax=ax)
    cl = [n for n, k in G.nodes(data="kind") if k == "cluster"]
    fa = [n for n, k in G.nodes(data="kind") if k == "family"]
    nx.draw_networkx_nodes(G, pos, nodelist=cl, node_size=[80 + 2.4 * G.nodes[n]["n"] for n in cl],
                           node_color=["#b0553f" if G.nodes[n]["core"] else "#6d9bb5" for n in cl],
                           edgecolors="white", linewidths=.8, ax=ax)
    nx.draw_networkx_nodes(G, pos, nodelist=fa, node_size=1500, node_shape="s", node_color="#4a7c59",
                           edgecolors="white", linewidths=1.2, ax=ax)
    nx.draw_networkx_labels(G, pos, {n: f"C{n[1]:02d}" for n in cl}, font_size=7.5, ax=ax)
    for n in fa:                      # family labels sit just below their square, in dark text
        x, y = pos[n]
        ax.text(x, y - 0.075, SHORT[n[1]], ha="center", va="top", fontsize=8.5, fontweight="bold", color="#1f3b2a")
    ax.set_xlim(-1.0, 1.0); ax.set_ylim(-1.02, 1.0); ax.set_aspect("equal"); ax.set_axis_off()
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    fig.savefig(path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    print(f"[figure] {path}  nodes={G.number_of_nodes()} edges={G.number_of_edges()} bridge_papers={nb}")


# --------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=DATA / "obsidian_vault", help="vault folder (recreated)")
    ap.add_argument("--figure", type=Path, default=None, help="also write a static cluster-family graph (PDF+PNG)")
    ap.add_argument("--no-vault", action="store_true", help="skip the vault, only draw the figure")
    ap.add_argument("--lang", choices=["en", "es"], default="en", help="language of the figure labels")
    args = ap.parse_args()
    d = load()
    if not args.no_vault:
        stats = build_vault(args.out, d)
        total, broken = check_links(args.out)
        print("[vault]", args.out)
        for k, v in stats.items():
            print(f"  {k:<14} {v}")
        print(f"  wikilinks     {total}  broken {len(broken)}")
        for b in broken[:10]:
            print("   !!", b)
    if args.figure:
        build_figure(args.figure, d, args.lang)


if __name__ == "__main__":
    main()
