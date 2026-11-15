"""Results of a finished run: counts, decisions, figures (plotly) and validation.

The biofloc example project ships the published results from the tracked
``release_data/`` folder (identifiers and decisions only, no abstracts).
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go

from floccus_app.i18n import t

EXAMPLE_FILES = (
    "prisma_counts.json",
    "screening_decisions.csv",
    "eligible_clusters.csv",
    "validation_sample.csv",
)
# How the published biofloc run was configured (see src/biofloc_review/dedup).
EXAMPLE_RUN_INFO = {
    "provider": "ollama",
    "generation_model": "qwen3:14b",
    "embedding_model": "qwen3-embedding:8b",
    "temperature": 0,
    "max_output_tokens": 400,
    "operating_mode": "specificity",
}
SOURCES = ("scopus", "wos", "ieee", "scielo")
SOURCE_LABELS = {"scopus": "Scopus", "wos": "Web of Science", "ieee": "IEEE Xplore", "scielo": "SciELO"}
CRITERIA_ORDER = ("I1", "I2", "I3", "I4", "E1", "E2", "E3", "E4", "D1", "D2", "D3")


def release_data_dir() -> Path:
    import biofloc_review

    return Path(biofloc_review.__file__).resolve().parents[2] / "release_data"


def published_dir(project: Path) -> Path:
    return project / "results" / "published"


def install_example_results(project: Path) -> None:
    """Copy the published biofloc results into the project."""
    src = release_data_dir()
    out = published_dir(project)
    out.mkdir(parents=True, exist_ok=True)
    for name in EXAMPLE_FILES:
        shutil.copyfile(src / name, out / name)
    (out / "run_info.json").write_text(json.dumps(EXAMPLE_RUN_INFO, indent=2), encoding="utf-8")


@dataclass
class Results:
    counts: dict
    decisions: pd.DataFrame
    eligible: pd.DataFrame | None
    validation: pd.DataFrame | None
    info: dict
    kind: str = "published"  # "published" (shipped example) or "latest" (own run)


def load_results(project: Path) -> Results | None:
    """Shipped example results first, else the project's latest finished run."""
    d = published_dir(project)
    kind = "published"
    if not (d / "prisma_counts.json").is_file():
        d, kind = project / "results" / "latest", "latest"
        if not (d / "screening_decisions.csv").is_file():
            return None
        if "final_specificity_mode" not in json.loads((d / "prisma_counts.json").read_text(encoding="utf-8")):
            return None

    def opt_csv(name: str) -> pd.DataFrame | None:
        return pd.read_csv(d / name, low_memory=False) if (d / name).is_file() else None

    info_path = d / "run_info.json"
    return Results(
        counts=json.loads((d / "prisma_counts.json").read_text(encoding="utf-8")),
        decisions=pd.read_csv(d / "screening_decisions.csv", low_memory=False),
        eligible=opt_csv("eligible_clusters.csv"),
        validation=opt_csv("validation_sample.csv"),
        info=json.loads(info_path.read_text(encoding="utf-8")) if info_path.is_file() else {},
        kind=kind,
    )


def _unique(c: dict) -> int:
    dd = c["deduplication"]
    return dd.get("after_round2_ieee_merges") or dd["after_layers2_3_and_merge"]


def pipeline_steps(c: dict, lang: str) -> list[dict]:
    l1, final = c["screening_layer1"], c["final_specificity_mode"]
    return [
        {"step": t("step_import", lang), "what": t("step_import_what", lang),
         "in": len(c["identification"]["by_source"]), "out": c["identification"]["total"]},
        {"step": t("step_dedup", lang), "what": t("step_dedup_what", lang),
         "in": c["identification"]["total"], "out": _unique(c)},
        {"step": t("step_l1", lang), "what": t("step_l1_what", lang),
         "in": l1["input"], "out": l1["include"] + l1["human_review"]},
        {"step": t("step_l2", lang), "what": t("step_l2_what", lang),
         "in": (c.get("screening_layer2") or {}).get("input", 0), "out": final["to_eligibility"]},
    ]


def _sources(c: dict) -> list[str]:
    by_src = c["identification"]["by_source"]
    ordered = [k for k in SOURCES if k in by_src]
    return ordered + [k for k in by_src if k not in ordered]


def fig_flow(c: dict, lang: str) -> go.Figure:
    by_src, dedup = c["identification"]["by_source"], c.get("deduplicated_by_source", {})
    l1, l2 = c["screening_layer1"], c.get("screening_layer2") or {}
    sources = _sources(c)
    n = len(sources)
    unique, dup, inc1, rev1, exc1, inc, rev, exc = range(n, n + 8)
    labels = [SOURCE_LABELS.get(s, s) for s in sources] + [
        t("n_unique", lang), t("n_duplicates", lang),
        t("n_l1_include", lang), t("n_l1_review", lang), t("n_l1_exclude", lang),
        t("n_include", lang), t("n_review", lang), t("n_exclude", lang),
    ]
    links = []
    for i, s in enumerate(sources):
        kept = dedup.get(s, 0)
        links += [(i, unique, kept), (i, dup, by_src[s] - kept)]
    l2_review = l2.get("human_review", l1["human_review"]) if l2.get("input") else l1["human_review"]
    links += [
        (unique, inc1, l1["include"]), (unique, rev1, l1["human_review"]), (unique, exc1, l1["exclude"]),
        (inc1, inc, l1["include"]), (rev1, inc, l2.get("include", 0)), (rev1, rev, l2_review),
        (rev1, exc, l2.get("exclude", 0)), (exc1, exc, l1["exclude"]),
    ]
    links = [x for x in links if x[2] > 0]
    fig = go.Figure(go.Sankey(
        node={"label": labels, "pad": 18},
        link={"source": [a for a, _, _ in links], "target": [b for _, b, _ in links],
              "value": [v for _, _, v in links]},
    ))
    fig.update_layout(title=t("fig_flow", lang), height=480, margin={"l": 10, "r": 10, "t": 50, "b": 10})
    return fig


def _bar(x, y, title: str, color: str | None = None) -> go.Figure:
    fig = go.Figure(go.Bar(x=list(x), y=list(y), text=list(y), textposition="auto",
                           marker_color=color))
    fig.update_layout(title=title, height=380, margin={"l": 10, "r": 10, "t": 50, "b": 10})
    return fig


def fig_sources(c: dict, lang: str) -> go.Figure:
    by_src = c["identification"]["by_source"]
    sources = _sources(c)
    return _bar([SOURCE_LABELS.get(s, s) for s in sources], [by_src[s] for s in sources], t("fig_sources", lang))


def fig_final_decisions(c: dict, lang: str) -> go.Figure:
    f = c["final_specificity_mode"]
    fig = _bar([t("n_include", lang), t("n_review", lang), t("n_exclude", lang)],
               [f["include"], f["human_review"], f["exclude"]], t("fig_final", lang))
    fig.data[0].marker.color = ["#4c9a6a", "#d9a441", "#b5523b"]
    return fig


def active_decisions(decisions: pd.DataFrame) -> pd.DataFrame:
    if "round2_change" in decisions:
        return decisions[decisions["round2_change"] != "ieee_merged_away"]
    return decisions


def fig_criteria(decisions: pd.DataFrame, lang: str, criteria_ids=None) -> go.Figure:
    d = active_decisions(decisions)
    ids = [cid for cid in (criteria_ids or CRITERIA_ORDER) if f"l1_{cid}" in d]
    yes = [int((d[f"l1_{cid}"] == "YES").sum()) for cid in ids]
    return _bar(ids, yes, t("fig_criteria", lang), "#5b7fa6")


def fig_clusters(eligible: pd.DataFrame, lang: str) -> go.Figure:
    counts = eligible["cluster_label"].fillna("—").value_counts().sort_index()
    return _bar(counts.index, counts.values, t("fig_clusters", lang), "#7a6aa6")


def validation_crosstab(validation: pd.DataFrame, reviewer: str) -> pd.DataFrame:
    return pd.crosstab(validation["pipeline_decision"], validation[reviewer].fillna("—"))
