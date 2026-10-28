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

With --round2 the tables reflect the round-2 corpus
(scripts/build_round2_corpus.py): screening_decisions.csv keeps all 17,063
round-1 records and adds round2_change (unchanged / ieee_merged_away /
e5_rescreened) and merged_into; the 98 merged-away records are flagged and
excluded from every count; eligible_clusters.csv and prisma_counts.json are
computed on the 16,965 retained records.

Run:  .venv/bin/python scripts/export_release_data.py \\
          --reviewer2-workbook <second reviewer's returned workbook.xlsx> \\
          --dedup-workbook <returned deduplication audit workbook.xlsx> \\
          --c03-workbook <returned C03 screening workbook.xlsx> \\
          [--out-dir release_data] [--round2]
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
EXPECTED_ROUND2 = {**EXPECTED, "deduplicated_records": 16965, "eligible_records": 4883, "include": 1395}
ASSIGNED_LATER = {False: 201, True: 225}       # 201 IEEE; round 2: 199 IEEE + 26 E5
ROUND2_DECISIONS = DATA / "round2_final_decisions.parquet"
ROUND2_COUNTS = DATA / "round2_counts.json"

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
def screening_decisions(round2: bool = False) -> pd.DataFrame:
    dec = pd.read_parquet(ROUND2_DECISIONS if round2 else DATA / "biofloc_final_decisions.parquet")
    dec = dec.merge(source_ids(), on="record_id", how="left", validate="one_to_one")
    l2cols = ["record_id", "l2_final_decision"] + [f"l2_{c}_answer" for c in CRITERIA]
    l2 = pd.read_parquet(DATA / "biofloc_layer2_screened.parquet", columns=l2cols)
    l2 = l2.drop_duplicates("record_id", keep="last")
    if round2:
        # Layer-2 answers of the E5 records re-screened without the language criterion
        e5 = pd.read_parquet(DATA / "e5_rescreen_decisions.parquet")
        e5 = e5[e5.l2_processed_new.astype(bool)].assign(record_id=e5.record_id.astype(str))
        e5 = e5.reindex(columns=l2cols)          # E5 has no Layer-2 answer: its column stays empty
        l2 = pd.concat([l2[~l2.record_id.isin(e5.record_id)], e5], ignore_index=True)
        dec = dec.merge(l2, on="record_id", how="left", validate="one_to_one")
        dec["sensitivity_decision_r2"] = dec.sensitivity_decision
    else:
        e2 = pd.read_parquet(DATA / "e2_v2_answers.parquet", columns=["record_id", "E2_v2"])
        dec = (dec.merge(l2, on="record_id", how="left", validate="one_to_one")
                  .merge(e2, on="record_id", how="left", validate="one_to_one"))
        dec["sensitivity_decision_r2"] = sensitivity_decision(dec).to_numpy()
        dec["round2_change"], dec["merged_into"], dec["in_round2_corpus"] = "unchanged", None, True
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
        "sensitivity_decision": dec.sensitivity_decision_r2.to_numpy(),
        "round2_change": dec.round2_change,
        "merged_into": dec.merged_into,
    })
    retained = dec.in_round2_corpus.to_numpy().astype(bool)
    out["eligible"] = out.final_decision.isin(ELIGIBLE) & retained
    out["eligible_sensitivity"] = out.sensitivity_decision.isin(ELIGIBLE) & retained
    return out


def retained(screen: pd.DataFrame) -> pd.DataFrame:
    """Records of the (round-2) corpus: the merged-away IEEE duplicates are flagged, not counted."""
    return screen[screen.round2_change != "ieee_merged_away"]


def eligible_clusters(screen: pd.DataFrame, round2: bool = False) -> pd.DataFrame:
    base = pd.read_parquet(DATA / "eligible_clusters_assignment.parquet", columns=["record_id", "cluster"])
    base["assigned_later"] = False
    base["assignment_method"] = "umap_hdbscan_fit"
    base["membership_strength"] = pd.NA
    ieee = pd.read_parquet(DATA / "ieee_cluster_assignment.parquet",
                           columns=["record_id", "cluster", "strength"])
    ieee = ieee.rename(columns={"strength": "membership_strength"})
    ieee["assigned_later"] = True
    ieee["assignment_method"] = "umap_transform_hdbscan_approximate_predict"
    frames = [base, ieee]
    if round2:   # the 26 E5 records newly eligible, mapped with the same procedure (rescreen_e5.py)
        e5 = pd.read_parquet(DATA / "e5_cluster_assignment.parquet",
                             columns=["record_id", "cluster", "cluster_strength"])
        e5 = e5.rename(columns={"cluster_strength": "membership_strength"})
        e5["assigned_later"] = True
        e5["assignment_method"] = "umap_transform_hdbscan_approximate_predict"
        frames.append(e5)
    cl = pd.concat(frames, ignore_index=True)
    cl["record_id"] = cl.record_id.astype(str)
    if cl.record_id.duplicated().any():
        raise SystemExit("a record has two cluster assignments")
    out = screen.loc[screen.eligible, ID_COLS + ["final_decision"]].merge(
        cl, on="record_id", how="left", validate="one_to_one")
    out["cluster"] = out.cluster.astype("Int64")
    out["cluster_label"] = out.cluster.map(lambda c: pd.NA if pd.isna(c) else
                                           ("noise" if c < 0 else f"C{int(c):02d}"))
    return out[ID_COLS + ["final_decision", "cluster", "cluster_label", "assigned_later",
                          "assignment_method", "membership_strength"]]


def validation_sample(screen: pd.DataFrame, r2_path: Path, round2: bool = False) -> pd.DataFrame:
    # review_id, human_decision, stratum, w, ...; round 2 re-weights to the retained corpus
    df, _ = load(population=retained(screen) if round2 else None)
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


def prisma(screen: pd.DataFrame, round2: bool = False) -> dict:
    p = json.loads((DATA / "prisma_counts.json").read_text())
    vc = lambda s: {k.lower(): int(v) for k, v in s.value_counts().items()}  # noqa: E731
    kept = retained(screen)
    out = {
        "identification": {"by_source": p["identification"]["by_source"],
                           "total": p["identification"]["total"]},
        "deduplication": {"after_layer1_deterministic": p["deduplication"]["post_capa1"],
                          "after_layers2_3_and_merge": p["deduplication"]["post_capa3_unique"],
                          "removed": p["deduplication"]["removed_by_dedup"]},
        "screening_layer1": p["screening_layer1"],
        "screening_layer2": {k.removeprefix("l2_"): v for k, v in p["screening_layer2"].items()},
        "final_specificity_mode": {**vc(kept.final_decision),
                                   "to_eligibility": int(kept.eligible.sum())},
        "final_sensitivity_mode": {**vc(kept.sensitivity_decision),
                                   "to_eligibility": int(kept.eligible_sensitivity.sum())},
        "deduplicated_by_source": {k: int(v) for k, v in kept.source.value_counts().items()},
        "note": "Counts recomputed from screening_decisions.csv except identification and "
                "deduplication, copied from the pipeline's PRISMA counts. "
                "screening_layer2.input counts the Layer-1 HUMAN_REVIEW records re-screened.",
    }
    if round2:
        r2 = json.loads(ROUND2_COUNTS.read_text())["prisma"]
        new = lambda block: {k: v["new"] for k, v in block.items()}  # noqa: E731
        out["deduplication"]["ieee_verified_merges_round2"] = r2["duplicates_removed"]["ieee_verified_merges"]["new"]
        out["deduplication"]["after_round2_ieee_merges"] = r2["unique"]["new"]
        out["deduplication"]["removed"] = r2["duplicates_removed"]["total"]["new"]
        out["screening_layer1"] = new(r2["screening_layer1"])
        out["screening_layer2"] = {k.removeprefix("l2_"): v for k, v in new(r2["screening_layer2"]).items()}
        out["round2"] = {"records_flagged_ieee_merged_away": int((screen.round2_change == "ieee_merged_away").sum()),
                         "records_e5_rescreened": int((screen.round2_change == "e5_rescreened").sum()),
                         "source": "scripts/build_round2_corpus.py (data/processed/round2_counts.json)"}
        out["note"] += (" Round 2: the 98 merged-away IEEE duplicates are flagged in "
                        "screening_decisions.csv (round2_change) and excluded from every count.")
    return out


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
    ap.add_argument("--round2", action="store_true",
                    help="export the round-2 corpus (build_round2_corpus.py) instead of the round-1 one")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    expected = EXPECTED_ROUND2 if args.round2 else EXPECTED

    screen = screening_decisions(args.round2)
    clusters = eligible_clusters(screen, args.round2)
    sample = validation_sample(screen, args.reviewer2_workbook, args.round2)
    audit = dedup_audit(args.dedup_workbook)
    c03 = c03_screening(screen, args.c03_workbook)
    counts = prisma(screen, args.round2)

    print(f"-> {args.out_dir}")
    write_csv(screen, args.out_dir / "screening_decisions.csv")
    write_csv(clusters, args.out_dir / "eligible_clusters.csv")
    write_csv(sample, args.out_dir / "validation_sample.csv")
    write_csv(audit, args.out_dir / "dedup_audit_pairs.csv")
    write_csv(c03, args.out_dir / "c03_manual_screening.csv")
    (args.out_dir / "prisma_counts.json").write_text(json.dumps(counts, indent=2) + "\n", encoding="utf-8")
    print("  prisma_counts.json")

    kept = retained(screen)
    observed = {"deduplicated_records": len(kept), "eligible_records": len(clusters),
                "include": int((kept.final_decision == "INCLUDE").sum()),
                "validation_sample": len(sample), "dedup_audit_pairs": len(audit),
                "c03_records": len(c03)}
    checks = {
        "source_id withheld (free text) for at most 1 record": screen.source_id.isna().sum() <= 1,
        "every eligible record has a cluster": clusters.cluster.notna().all(),
        "records assigned to clusters later": int(clusters.assigned_later.sum()) == ASSIGNED_LATER[args.round2],
        "merged-away records carry merged_into": bool(
            (screen.loc[screen.round2_change == "ieee_merged_away", "merged_into"].notna()).all()),
        "C03 table matches cluster 3": set(c03.record_id) == set(clusters.record_id[clusters.cluster == 3]),
        "every sample record labelled by both reviewers":
            sample.reviewer_1.notna().all() and sample.reviewer_2.notna().all(),
        "every audit pair labelled": audit.audit_label.notna().all(),
        "every C03 record labelled": c03.label.notna().all(),
    }
    ok = True
    print("\ncount checks (expected = manuscript):")
    for k, exp in expected.items():
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
