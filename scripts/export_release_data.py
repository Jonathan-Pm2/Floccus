"""Export the derived decision tables published in release_data/.

The raw Scopus, Web of Science, IEEE Xplore and SciELO exports cannot be
redistributed, and neither can the titles, abstracts, keywords or author
lists they contain (see docs/DATA.md). This script therefore writes, from the
pipeline artefacts in data/processed/ and the three returned human-labelling
workbooks, compact UTF-8 CSV tables that carry IDENTIFIERS AND DECISIONS ONLY:

  screening_decisions.csv   all 17,063 deduplicated records: Layer-1 and
                            Layer-2 criterion verdicts, final decision
                            (specificity mode) and sensitivity-mode decision
  eligible_clusters.csv     the 4,883 eligible records and their thematic
                            cluster (201 IEEE Xplore records assigned later
                            are flagged)
  validation_sample.csv     the 465-record blind validation sample: stratum,
                            sampling weight, pipeline decisions, reviewer_1,
                            reviewer_2 and the two baselines
  dedup_audit_pairs.csv     the 110 audited deduplication pairs
  c03_manual_screening.csv  the 141 records of cluster C03 screened by hand
  prisma_counts.json        PRISMA flow counts (both operating modes)

Identifier columns: record_id (stable UUID5 used throughout the pipeline),
source, source_id (identifier in the source database), doi, year. Free text
(titles, abstracts, keywords, LLM evidence/reasoning, reviewer comments) and
personal names are never written. The expected manuscript counts are checked
at the end; a mismatch is reported and the exit status is 1.

Run:  .venv/bin/python scripts/export_release_data.py \\
          --reviewer2-workbook <second reviewer's returned workbook.xlsx> \\
          --dedup-workbook <returned deduplication audit workbook.xlsx> \\
          --c03-workbook <returned C03 screening workbook.xlsx> \\
          [--out-dir release_data]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import uuid
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from compute_recall_wss import DATA, load  # noqa: E402
from make_dedup_audit_workbook import DECISIONS as DEDUP_DECISIONS  # noqa: E402
from operating_modes_recall import sensitivity_decision  # noqa: E402
from second_reviewer_agreement import has_problems, validate_decisions  # noqa: E402

CRITERIA = ("I1", "I2", "I3", "I4", "E1", "E2", "E3", "E4", "E5", "D1", "D2", "D3")
ELIGIBLE = ("INCLUDE", "HUMAN_REVIEW")
ID_COLS = ["record_id", "source", "source_id", "doi", "year"]

EXPECTED = {"deduplicated_records": 17063, "eligible_records": 4883, "include": 1385,
            "validation_sample": 465, "dedup_audit_pairs": 110, "c03_records": 141}

LEVEL_EN = {"Incluir": "include", "Dudoso": "uncertain", "Excluir": "exclude"}
BINARY_EN = {"cumple": "advance", "no cumple": "exclude"}
DEDUP_EN = dict(zip(DEDUP_DECISIONS, ("same_work", "different_works", "undetermined")))
REASON_CODE = {
    "E1 Organismo modelo": "E1", "E2 Sin modelación formal": "E2",
    "E3 Sistema no acuoso": "E3", "E4 Coagulación química pura": "E4",
    "No cumple I1 (objeto)": "not_I1", "No cumple I2 (modelo)": "not_I2",
    "No cumple I3 (contexto)": "not_I3", "No cumple I4 (tipo de documento)": "not_I4",
    "Otro (ver comentario)": "other",
}


# --------------------------------------------------------------- helpers
def reason_code(v) -> str | None:
    if v is None or (isinstance(v, float) and pd.isna(v)) or str(v).strip() == "":
        return None
    v = str(v).strip()
    if v not in REASON_CODE:
        raise SystemExit(f"unknown exclusion reason in workbook: {v!r}")
    return REASON_CODE[v]


def decision_reason(text: str) -> tuple[str, str]:
    """Split the pipeline's (Spanish) decision trace into (layer, English reason code)."""
    layer = "L2" if text.startswith("Layer 2:") else "L1"
    t = text.removeprefix("Layer 2:").strip()
    ids = lambda s: ",".join(re.findall(r"[IED]\d", s))  # noqa: E731
    if t.startswith("Layer 1 INCLUDE") or t.startswith("todos I cumplidos"):
        return layer, "all_inclusion_met"
    if t.startswith("exclusión con incertidumbre"):
        return layer, f"exclusion_unclear:{ids(t)}"
    if t.startswith("exclusión por"):
        return layer, f"excluded_by:{ids(t)}"
    if t.startswith("inclusión incompleta"):
        return layer, f"inclusion_not_met:{ids(t)}"
    if t.startswith("derivación por"):
        return layer, f"routed_by:{ids(t)}"
    raise SystemExit(f"unrecognised decision trace: {text!r}")


def source_ids() -> pd.DataFrame:
    """record_id -> source_id, rebuilt exactly as the pipeline builds record_id
    (uuid5 of source + source_id, else the normalised title)."""
    src = pd.read_parquet(DATA / "biofloc_final_primary_v3.parquet",
                          columns=["source", "source_id", "title_norm"])
    src["record_id"] = [str(uuid.uuid5(uuid.NAMESPACE_URL, f"{s}:{sid or (tn or '')[:80]}"))
                        for s, sid, tn in zip(src.source, src.source_id, src.title_norm)]
    src = src[["record_id", "source_id"]].drop_duplicates("record_id")
    # A malformed IEEE Xplore export row carries its abstract in the PDF-link
    # field, so its source_id is free text: never publish such a value.
    bad = src.source_id.str.contains(r"\s", regex=True, na=False) | (src.source_id.str.len() > 200)
    src.loc[bad, "source_id"] = pd.NA
    return src


def read_workbook(path: Path, sheet: str, levels, expected_n) -> pd.DataFrame:
    reg = pd.read_excel(path, sheet_name=sheet)
    dec, problems = validate_decisions(reg, expected_n=expected_n, levels=levels)
    if has_problems(problems):
        raise SystemExit(f"{path.name}: incomplete or invalid workbook {problems}")
    return dec[["n", "r2_level", "r2_reason"]]


# ---------------------------------------------------------------- tables
def screening_decisions() -> pd.DataFrame:
    dec = pd.read_parquet(DATA / "biofloc_final_decisions.parquet")
    dec = dec.merge(source_ids(), on="record_id", how="left", validate="one_to_one")
    l2 = pd.read_parquet(DATA / "biofloc_layer2_screened.parquet",
                         columns=["record_id", "l2_final_decision"]
                         + [f"l2_{c}_answer" for c in CRITERIA])
    l2 = l2.drop_duplicates("record_id", keep="last")
    e2 = pd.read_parquet(DATA / "e2_v2_answers.parquet", columns=["record_id", "E2_v2"])
    dec = (dec.merge(l2, on="record_id", how="left", validate="one_to_one")
              .merge(e2, on="record_id", how="left", validate="one_to_one"))
    layer_reason = dec.final_decision_source.map(decision_reason)
    out = pd.DataFrame({
        "record_id": dec.record_id, "source": dec.source, "source_id": dec.source_id,
        "doi": dec.doi, "year": dec.year.astype("Int64"),
        "dedup_cluster_id": dec.cluster_id,
        "document_type": dec.document_type,
        "language": dec.language.replace({"nan": pd.NA}),
        **{f"l1_{c}": dec[f"{c}_answer"] for c in CRITERIA},
        "l1_decision": dec.final_decision_primary,
        **{f"l2_{c}": dec[f"l2_{c}_answer"] for c in CRITERIA},
        "l2_decision": dec.l2_final_decision,
        "final_decision": dec.final_decision,
        "decision_layer": layer_reason.str[0],
        "decision_reason": layer_reason.str[1],
        "e2_sensitivity": dec.E2_v2,
        "sensitivity_decision": sensitivity_decision(dec).to_numpy(),
    })
    out["eligible"] = out.final_decision.isin(ELIGIBLE)
    out["eligible_sensitivity"] = out.sensitivity_decision.isin(ELIGIBLE)
    return out


def eligible_clusters(screen: pd.DataFrame) -> pd.DataFrame:
    base = pd.read_parquet(DATA / "eligible_clusters_assignment.parquet", columns=["record_id", "cluster"])
    base["assigned_later"] = False
    base["assignment_method"] = "umap_hdbscan_fit"
    base["membership_strength"] = pd.NA
    ieee = pd.read_parquet(DATA / "ieee_cluster_assignment.parquet",
                           columns=["record_id", "cluster", "strength"])
    ieee = ieee.rename(columns={"strength": "membership_strength"})
    ieee["assigned_later"] = True
    ieee["assignment_method"] = "umap_transform_hdbscan_approximate_predict"
    cl = pd.concat([base, ieee], ignore_index=True)
    if cl.record_id.duplicated().any():
        raise SystemExit("a record has two cluster assignments")
    out = screen.loc[screen.eligible, ID_COLS + ["final_decision"]].merge(
        cl, on="record_id", how="left", validate="one_to_one")
    out["cluster"] = out.cluster.astype("Int64")
    out["cluster_label"] = out.cluster.map(lambda c: pd.NA if pd.isna(c) else
                                           ("noise" if c < 0 else f"C{int(c):02d}"))
    return out[ID_COLS + ["final_decision", "cluster", "cluster_label", "assigned_later",
                          "assignment_method", "membership_strength"]]


def validation_sample(screen: pd.DataFrame, r2_path: Path) -> pd.DataFrame:
    df, _ = load()                      # review_id, human_decision, stratum, w, ...
    truth = pd.read_parquet(DATA / "validation_sample_truth.parquet", columns=["review_id", "record_id"])
    df = df.merge(truth, on="review_id", how="left", validate="one_to_one")
    df["N_c"] = df.groupby("stratum")["w"].transform("first") * df.groupby("stratum")["w"].transform("size")

    key = pd.read_csv(DATA / "validation_reviewer2_key.csv")
    r2 = read_workbook(r2_path, "Registros", list(LEVEL_EN), key["n"].tolist()).merge(key, on="n")
    kw = pd.read_parquet(DATA / "baseline_keyword_decisions.parquet",
                         columns=["review_id", "advance_keyword", "advance_keyword_strict"])
    sp = pd.read_parquet(DATA / "baseline_single_prompt_answers.parquet", columns=["record_id", "answer"])

    df = (df.drop(columns=["source"])
            .merge(r2, on="review_id", how="left", validate="one_to_one")
            .merge(kw, on="review_id", how="left", validate="one_to_one")
            .merge(sp, on="record_id", how="left", validate="one_to_one")
            .merge(screen[ID_COLS + ["final_decision", "sensitivity_decision"]]
                   .rename(columns={"final_decision": "pipeline_check"}),
                   on="record_id", how="left", validate="one_to_one"))
    if (df.final_decision != df.pipeline_check).any():
        raise SystemExit("validation sample decisions differ from the decision table")
    return pd.DataFrame({
        "review_id": df.review_id, **{c: df[c] for c in ID_COLS},
        "stratum": df.stratum, "stratum_population": df.N_c.round().astype(int),
        "sampling_weight": df.w,
        "pipeline_decision": df.final_decision,
        "pipeline_decision_sensitivity": df.sensitivity_decision,
        "reviewer_1": df.human_decision.str.strip().str.lower().map(BINARY_EN),
        "reviewer_2": df.r2_level.map(LEVEL_EN),
        "reviewer_2_reason_code": df.r2_reason.map(reason_code),
        "baseline_keyword": df.advance_keyword.map({True: "advance", False: "exclude"}),
        "baseline_keyword_strict": df.advance_keyword_strict.map({True: "advance", False: "exclude"}),
        "baseline_single_prompt": df.answer,
    }).sort_values("review_id").reset_index(drop=True)


def dedup_audit(path: Path) -> pd.DataFrame:
    key = pd.read_csv(DATA / "dedup_audit_key.csv")
    lab = read_workbook(path, "Pares", DEDUP_DECISIONS, key["n"].tolist())
    df = key.merge(lab, on="n", how="left", validate="one_to_one")
    return pd.DataFrame({
        "pair_id": df.n, "stratum": df.stratum, "layer": df.layer,
        "match_basis": df.match_basis, "cosine": df.cosine,
        "judge_verdict": df.system_verdict, "judge_confidence": df.judge_confidence.astype("Int64"),
        "a_ref": df.a_ref, "b_ref": df.b_ref, "a_dedup_cluster_id": df.a_cluster,
        "b_dedup_cluster_id": df.b_cluster,
        "audit_label": df.r2_level.map(DEDUP_EN),
    })


def c03_screening(screen: pd.DataFrame, path: Path) -> pd.DataFrame:
    key = pd.read_csv(DATA / "c03_screening_key.csv", usecols=["n", "record_id"])
    lab = read_workbook(path, "Registros", list(LEVEL_EN), key["n"].tolist())
    df = (key.merge(lab, on="n", how="left", validate="one_to_one")
             .merge(screen[ID_COLS + ["final_decision", "sensitivity_decision"]],
                    on="record_id", how="left", validate="one_to_one"))
    return pd.DataFrame({
        "item": df.n, **{c: df[c] for c in ID_COLS},
        "pipeline_decision": df.final_decision,
        "pipeline_decision_sensitivity": df.sensitivity_decision,
        "label": df.r2_level.map(LEVEL_EN),
        "reason_code": df.r2_reason.map(reason_code),
    })


def prisma(screen: pd.DataFrame) -> dict:
    p = json.loads((DATA / "prisma_counts.json").read_text())
    vc = lambda s: {k.lower(): int(v) for k, v in s.value_counts().items()}  # noqa: E731
    return {
        "identification": {"by_source": p["identification"]["by_source"],
                           "total": p["identification"]["total"]},
        "deduplication": {"after_layer1_deterministic": p["deduplication"]["post_capa1"],
                          "after_layers2_3_and_merge": p["deduplication"]["post_capa3_unique"],
                          "removed": p["deduplication"]["removed_by_dedup"]},
        "screening_layer1": p["screening_layer1"],
        "screening_layer2": {k.removeprefix("l2_"): v for k, v in p["screening_layer2"].items()},
        "final_specificity_mode": {**vc(screen.final_decision),
                                   "to_eligibility": int(screen.eligible.sum())},
        "final_sensitivity_mode": {**vc(screen.sensitivity_decision),
                                   "to_eligibility": int(screen.eligible_sensitivity.sum())},
        "deduplicated_by_source": {k: int(v) for k, v in screen.source.value_counts().items()},
        "note": "Counts recomputed from screening_decisions.csv except identification and "
                "deduplication, copied from the pipeline's PRISMA counts. "
                "screening_layer2.input counts the Layer-1 HUMAN_REVIEW records re-screened.",
    }


# ------------------------------------------------------------------ main
def write_csv(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, index=False, encoding="utf-8", lineterminator="\n")
    print(f"  {path.name:<26} {len(df):>6} rows  {len(df.columns):>3} cols")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reviewer2-workbook", type=Path, required=True)
    ap.add_argument("--dedup-workbook", type=Path, required=True)
    ap.add_argument("--c03-workbook", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, default=ROOT / "release_data")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    screen = screening_decisions()
    clusters = eligible_clusters(screen)
    sample = validation_sample(screen, args.reviewer2_workbook)
    audit = dedup_audit(args.dedup_workbook)
    c03 = c03_screening(screen, args.c03_workbook)
    counts = prisma(screen)

    print(f"-> {args.out_dir}")
    write_csv(screen, args.out_dir / "screening_decisions.csv")
    write_csv(clusters, args.out_dir / "eligible_clusters.csv")
    write_csv(sample, args.out_dir / "validation_sample.csv")
    write_csv(audit, args.out_dir / "dedup_audit_pairs.csv")
    write_csv(c03, args.out_dir / "c03_manual_screening.csv")
    (args.out_dir / "prisma_counts.json").write_text(json.dumps(counts, indent=2) + "\n", encoding="utf-8")
    print("  prisma_counts.json")

    observed = {"deduplicated_records": len(screen), "eligible_records": len(clusters),
                "include": int((screen.final_decision == "INCLUDE").sum()),
                "validation_sample": len(sample), "dedup_audit_pairs": len(audit),
                "c03_records": len(c03)}
    checks = {
        "source_id withheld (free text) for at most 1 record": screen.source_id.isna().sum() <= 1,
        "every eligible record has a cluster": clusters.cluster.notna().all(),
        "IEEE records assigned later": int(clusters.assigned_later.sum()) == 201,
        "C03 table matches cluster 3": set(c03.record_id) == set(clusters.record_id[clusters.cluster == 3]),
        "every sample record labelled by both reviewers":
            sample.reviewer_1.notna().all() and sample.reviewer_2.notna().all(),
        "every audit pair labelled": audit.audit_label.notna().all(),
        "every C03 record labelled": c03.label.notna().all(),
    }
    ok = True
    print("\ncount checks (expected = manuscript):")
    for k, exp in EXPECTED.items():
        good = observed[k] == exp
        ok &= good
        print(f"  {k:<22} {observed[k]:>6}  expected {exp:>6}  {'ok' if good else 'MISMATCH'}")
    for k, good in checks.items():
        ok &= bool(good)
        print(f"  {k:<48} {'ok' if good else 'FAILED'}")
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
