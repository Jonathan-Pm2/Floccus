"""Round-2 canonical corpus and every corpus-dependent count of the manuscript.

Two corrections decided by the authors after round 1 are applied to the
published decision table (biofloc_final_decisions.parquet, 17,063 records):

1. IEEE duplicates. The IEEE Xplore records never reached the embedding layer
   of the deduplication (identifier collision, scripts/ieee_integrity_check.py).
   Of the 145 candidate pairs found when they are embedded with their own
   vectors, ONLY the 98 judged DUPLICATE whose two records share a DOI or a
   normalised title are merged here (the 26 LLM-only verdicts are not).
   Each merged group keeps ONE record, chosen by the retention rule below;
   the retained record keeps its own screening decision, the other record
   leaves the corpus (round2_change = ieee_merged_away, merged_into = the
   retained record_id).

   Retention rule (in this order of priority, first difference decides):
     1. member of the 465-record blind validation sample;
     2. member of the clustered set (4,682 embedded + 201 IEEE assigned);
     3. record with a DOI;
     4. the IEEE Xplore record;
     5. tie-break: lexicographically smallest record_id.
   The rule preserves the archived analyses: no validation-sample record and
   as few clustered records as possible leave the corpus. The outcome of the
   Layer-1 primary rule (completeness, then source priority), which the
   integrity check used, is reported for comparison.

2. E5 re-screen. The 31 Scopus records excluded solely by the retired
   language criterion E5 were re-screened without it (scripts/rescreen_e5.py):
   11 INCLUDE, 15 HUMAN_REVIEW, 5 EXCLUDE (through Layer 2). The 23 records
   excluded by E5 + E2 only were re-asked the corrected E2 for the
   sensitivity mode (15 rescued to HUMAN_REVIEW). All 54 carry
   round2_change = e5_rescreened.

Every reported quantity is recomputed OLD -> NEW with the existing scripts
(imported): PRISMA counts (run_consolidate.prisma_counts), the stratum
re-weighted validation metrics (compute_recall_wss, second_reviewer_agreement,
baselines_compare, wss_reconciliation), the thematic clusters
(eligible_clusters_assignment + assign_ieee_clusters / rescreen_e5 mappings),
the bridge score and its citation check (bridge_score,
bridge_citations_crossref, cached Crossref references), the language table and
the supplementary-search overlap. Qdrant (papers_biofloc) must be running for
the bridge block.

Run:  .venv/bin/python scripts/build_round2_corpus.py \\
          --reviewer2-workbook <second reviewer's returned workbook.xlsx> [--quick]
Out:  data/processed/round2_final_decisions.parquet
      data/processed/round2_counts.json
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import date
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "processed"
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import compute_recall_wss as crw  # noqa: E402
import second_reviewer_agreement as sra  # noqa: E402
from baselines_compare import ROWS as BASELINE_ROWS  # noqa: E402
from baselines_compare import advance_vectors, score  # noqa: E402
from bridge_citations_crossref import clean_doi, link_matrix, pair_density  # noqa: E402
from bridge_score import bridge, fetch_vectors, unit  # noqa: E402
from ieee_integrity_check import DSU, archived_dsu, point_id, same_work  # noqa: E402
from operating_modes_recall import sensitivity_decision  # noqa: E402
from wss_reconciliation import reconcile  # noqa: E402

OUT_DECISIONS = DATA / "round2_final_decisions.parquet"
OUT_COUNTS = DATA / "round2_counts.json"
ELIGIBLE = ("INCLUDE", "HUMAN_REVIEW")
UNCHANGED, MERGED_AWAY, E5_RESCREENED = "unchanged", "ieee_merged_away", "e5_rescreened"
RETENTION_RULE = ("validation-sample member > clustered record (4,682 embedded + 201 IEEE assigned) "
                  "> record with DOI > IEEE Xplore record > smallest record_id")
DECLARED = ("English", "Spanish", "Portuguese", "Chinese")
NOT_DECLARED = {"", "nan", "none", "undefined"}
FOCUS = (3, 15)
BRIDGE_THRESH = (0.5, 0.7)
IDENTIFIED = 22056
POST_LAYER1 = 17326
REFERENCES = {"reviewer1": "First reviewer", "reviewer2_binary": "Second reviewer",
              "consensus_both": "Consensus", "union_either": "Union"}


# ---------------------------------------------------------------- pure helpers

def retention_key(rec: dict) -> tuple:
    """Priority tuple of the retention rule (higher wins)."""
    return (bool(rec.get("in_sample")), bool(rec.get("in_clustered")),
            bool(rec.get("has_doi")), rec.get("source") == "ieee")


def retain(group: list[dict]) -> tuple[str, list[str]]:
    """Apply the retention rule to one merged group of record dicts.

    Each dict carries record_id, source, in_sample, in_clustered, has_doi.
    Returns (retained record_id, dropped record_ids in input order)."""
    if len(group) < 2:
        raise ValueError("a merged group has at least two records")
    ranked = sorted(group, key=lambda r: (tuple(not k for k in retention_key(r)), r["record_id"]))
    keep = ranked[0]["record_id"]
    return keep, [r["record_id"] for r in group if r["record_id"] != keep]


def merged_groups(cluster_of: dict[str, str], dsu: DSU, pairs: list[tuple[str, str]]) -> list[list[str]]:
    """Groups of record_ids (size > 1) after unioning the cluster pairs onto a copy of dsu.

    cluster_of maps record_id -> Layer-1 cluster id; dsu already holds the
    archived Layer-3 merges. Groups come out sorted for determinism."""
    new = DSU()
    new.parent = dict(dsu.parent)
    for a, b in pairs:
        new.union(a, b)
    by_root: dict[str, list[str]] = {}
    for rid, c in cluster_of.items():
        by_root.setdefault(new.find(c), []).append(rid)
    return sorted((sorted(g) for g in by_root.values() if len(g) > 1), key=lambda g: g[0])


def verified_pairs(pairs: list[dict], verdicts: dict[str, dict]) -> list[dict]:
    """Candidate pairs judged DUPLICATE (judge cache) whose two records share a DOI or a
    normalised title (same_doi_or_title, computed by ieee_integrity_check.same_work on the
    actual candidate records, which may be Layer-1 primaries absorbed by Layer 3)."""
    out = []
    for p in pairs:
        v = verdicts.get(f"{p['a_cluster']}|{p['b_cluster']}", {}).get("verdict")
        if v == "DUPLICATE" and p.get("same_doi_or_title", "no") != "no":
            out.append({**p, "verification": p["same_doi_or_title"]})
    return out


def language_tokens(value) -> list[str]:
    """Declared languages of a record ('Spanish, English' counts in each); [] when not declared."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    s = str(value).strip()
    if s.lower() in NOT_DECLARED:
        return []
    return [t.strip() for t in s.split(",") if t.strip()]


def language_table(languages: pd.Series) -> dict[str, int]:
    """Rows of the manuscript's language table: the four main languages, Other, Not declared.

    Records declaring two languages are counted in each; any declared value
    outside the four main languages (including ISO codes such as 'en') is Other."""
    rows = Counter({k: 0 for k in (*DECLARED, "Other", "Not declared")})
    for v in languages:
        toks = language_tokens(v)
        if not toks:
            rows["Not declared"] += 1
        for t in toks:
            rows[t if t in DECLARED else "Other"] += 1
    return dict(rows)


def decision_counts(s: pd.Series) -> dict[str, int]:
    c = s.value_counts()
    out = {k: int(c.get(k, 0)) for k in ("INCLUDE", "HUMAN_REVIEW", "EXCLUDE")}
    out["eligible"] = out["INCLUDE"] + out["HUMAN_REVIEW"]
    return out


def cluster_label(k) -> str:
    return "noise" if int(k) < 0 else f"C{int(k):02d}"


def old_new(old, new) -> dict:
    return {"old": old, "new": new}


# ---------------------------------------------------------------- corpus

def load_published() -> tuple[pd.DataFrame, pd.DataFrame]:
    dec = pd.read_parquet(DATA / "biofloc_final_decisions.parquet")
    dec["record_id"] = dec.record_id.astype(str)
    v3 = pd.read_parquet(DATA / "biofloc_final_primary_v3.parquet")
    v3["record_id"] = [point_id(s, i, t) for s, i, t in zip(v3.source, v3.source_id, v3.title_norm)]
    v3 = v3.drop_duplicates("record_id").set_index("record_id")
    missing = set(dec.record_id) - set(v3.index)
    if missing:
        sys.exit(f"{len(missing)} decision records not found in the canonical corpus; stopping")
    return dec, v3.loc[dec.record_id]


def apply_e5(dec: pd.DataFrame, e5: pd.DataFrame) -> pd.DataFrame:
    """Overwrite the decisions of the 54 E5-affected records with the re-screen outcome."""
    out = dec.copy()
    e5 = e5.set_index(e5.record_id.astype(str))
    rows = out.record_id.isin(e5.index)
    idx = out.index[rows]
    rid = out.loc[idx, "record_id"]
    out.loc[idx, "final_decision_primary"] = e5.loc[rid, "l1_new_decision"].to_numpy()
    out.loc[idx, "final_reason_primary"] = e5.loc[rid, "l1_new_reason"].to_numpy()
    out.loc[idx, "final_triggers_primary"] = e5.loc[rid, "l1_new_triggers"].to_numpy()
    out.loc[idx, "final_decision"] = e5.loc[rid, "final_decision_new"].to_numpy()
    out.loc[idx, "final_decision_source"] = e5.loc[rid, "final_decision_source_new"].to_numpy()
    out.loc[idx, "l2_processed"] = e5.loc[rid, "l2_processed_new"].to_numpy().astype(bool)
    out.loc[idx, "round2_change"] = E5_RESCREENED
    out.loc[idx, "round2_note"] = e5.loc[rid, "affected_set"].to_numpy()
    return out


def sensitivity_column(dec: pd.DataFrame, e5: pd.DataFrame | None = None) -> tuple[pd.Series, pd.Series]:
    """Sensitivity-mode decision and corrected-E2 answer for every record.

    Published rule (operating_modes_recall.sensitivity_decision) everywhere; the
    54 E5-affected records take the decision of the E5 re-screen when e5 is given."""
    e2 = pd.read_parquet(DATA / "e2_v2_answers.parquet", columns=["record_id", "E2_v2"])
    e2["record_id"] = e2.record_id.astype(str)
    if e5 is not None:
        e5v2 = e5[["record_id", "E2_v2"]].assign(record_id=e5.record_id.astype(str))
        e2 = pd.concat([e5v2, e2[~e2.record_id.isin(e5v2.record_id)]], ignore_index=True)
    m = dec[["record_id", "final_decision"] + [f"E{i}_answer" for i in range(1, 6)]].merge(
        e2.drop_duplicates("record_id"), on="record_id", how="left")
    m.index = dec.index
    sens = sensitivity_decision(m)
    if e5 is not None:
        e5s = e5.set_index(e5.record_id.astype(str))["sensitivity_decision_new"]
        rows = dec.record_id.isin(e5s.index)
        sens[rows] = e5s.loc[dec.loc[rows, "record_id"]].to_numpy()
    return sens, m["E2_v2"]


def as_record(row: pd.Series):
    from biofloc_review.schema import PaperRecord
    d = {k: (None if isinstance(v, float) and pd.isna(v) else v) for k, v in row.items()}
    for k in ("authors", "keywords", "enriched_from", "merged_from"):
        if isinstance(d.get(k), np.ndarray):
            d[k] = d[k].tolist()
    return PaperRecord.model_validate(d)


def ieee_merges(dec: pd.DataFrame, v3: pd.DataFrame, sample: set[str], clustered: set[str]) -> tuple[pd.DataFrame, dict]:
    """Flag the records that leave the corpus through the 98 verified IEEE merges."""
    from biofloc_review.dedup.deterministic import _select_primary

    l1 = pd.read_parquet(DATA / "biofloc_dedup_primary.parquet", columns=["cluster_id"])
    l1_clusters = set(l1.cluster_id)
    judged = json.loads((DATA / "biofloc_dedup_primary_near_dup_pairs_judged.json").read_text())
    dsu = archived_dsu(judged, l1_clusters)
    cluster_of = dict(zip(dec.record_id, v3.cluster_id))
    roots = {dsu.find(c) for c in cluster_of.values()}
    if len(roots) != len(dec):
        sys.exit("archived Layer-3 merge not reproduced; stopping")

    integrity = json.loads((DATA / "ieee_integrity.json").read_text())
    verdicts = json.loads((DATA / "ieee_counterfactual_judged.json").read_text())
    candidates = integrity["counterfactual_layer2"]["pairs"]
    pairs = verified_pairs(candidates, verdicts)
    groups = merged_groups(cluster_of, dsu, [(p["a_cluster"], p["b_cluster"]) for p in pairs])

    out = dec.copy()
    info = dec.set_index("record_id")
    rows_of = {rid: {"record_id": rid, "source": info.at[rid, "source"], "in_sample": rid in sample,
                     "in_clustered": rid in clustered,
                     "has_doi": isinstance(info.at[rid, "doi"], str) and info.at[rid, "doi"].strip() != ""}
               for g in groups for rid in g}
    dropped_rule, dropped_l1, detail = [], [], []
    for g in groups:
        keep, drop = retain([rows_of[r] for r in g])
        prim = g[_select_primary([as_record(v3.loc[r]) for r in g])]
        dropped_rule += drop
        dropped_l1 += [r for r in g if r != prim]
        for r in drop:
            out.loc[out.record_id == r, ["round2_change", "merged_into"]] = [MERGED_AWAY, keep]
        detail.append({"records": [{**rows_of[r], "decision": info.at[r, "final_decision"]} for r in g],
                       "retained": keep, "layer1_rule_would_retain": prim})
    dec_of = lambda ids: dict(Counter(info.loc[ids, "final_decision"]))  # noqa: E731
    summary = {
        "rule": RETENTION_RULE,
        "candidate_pairs": len(candidates),
        "verdicts": dict(Counter(verdicts[f"{p['a_cluster']}|{p['b_cluster']}"]["verdict"] for p in candidates)),
        "verified_duplicate_pairs_applied": len(pairs),
        "verification": dict(Counter(p["verification"] for p in pairs)),
        "groups_merged": len(groups), "group_sizes": dict(Counter(len(g) for g in groups)),
        "records_leaving": len(dropped_rule),
        "records_leaving_by_decision": dec_of(dropped_rule),
        "records_leaving_by_source": dict(Counter(info.loc[dropped_rule, "source"])),
        "retained_by_source": dict(Counter(info.loc[[d["retained"] for d in detail], "source"])),
        "groups_touching_validation_sample": int(sum(any(rows_of[r]["in_sample"] for r in g) for g in groups)),
        "groups_touching_clustered_set": int(sum(any(rows_of[r]["in_clustered"] for r in g) for g in groups)),
        "groups_with_two_clustered_records": int(sum(sum(rows_of[r]["in_clustered"] for r in g) >= 2 for g in groups)),
        "validation_sample_records_dropped": {"without_rule_layer1_primary": int(sum(r in sample for r in dropped_l1)),
                                              "with_rule": int(sum(r in sample for r in dropped_rule))},
        "clustered_records_dropped": {"without_rule_layer1_primary": int(sum(r in clustered for r in dropped_l1)),
                                      "with_rule": int(sum(r in clustered for r in dropped_rule))},
        "records_leaving_without_rule_by_decision": dec_of(dropped_l1),
        "groups_where_rule_and_layer1_differ": int(sum(d["retained"] != d["layer1_rule_would_retain"] for d in detail)),
        "detail": detail,
    }
    return out, summary


def build_round2(workbook_unused=None) -> tuple[pd.DataFrame, dict, dict]:
    dec, v3 = load_published()
    dec["round2_change"] = UNCHANGED
    dec["round2_note"] = None
    dec["merged_into"] = None
    e5 = pd.read_parquet(DATA / "e5_rescreen_decisions.parquet")
    e5["record_id"] = e5.record_id.astype(str)
    dec = apply_e5(dec, e5)
    dec["sensitivity_decision"], dec["E2_v2"] = sensitivity_column(dec, e5)

    sample = set(pd.read_parquet(DATA / "validation_sample_truth.parquet").record_id.astype(str))
    clustered = set(r for r in (DATA / "eligible_record_ids.txt").read_text().split("\n") if r)
    clustered |= set(pd.read_parquet(DATA / "ieee_cluster_assignment.parquet").record_id.astype(str))
    dec, merges = ieee_merges(dec, v3, sample, clustered)
    dec["in_round2_corpus"] = dec.round2_change != MERGED_AWAY
    e5_summary = {
        "e5_only_rescreened": int((dec.round2_note == "e5_only").sum()),
        "e5_plus_e2_only_reasked_e2": int((dec.round2_note == "e5_plus_e2_only").sum()),
        "e5_only_new_decisions": dict(Counter(e5[e5.affected_set == "e5_only"].final_decision_new)),
        "sensitivity_rescued_e5_plus_e2_only": int(((e5.affected_set == "e5_plus_e2_only")
                                                    & (e5.sensitivity_decision_new == "HUMAN_REVIEW")).sum()),
    }
    return dec, merges, e5_summary


# ---------------------------------------------------------------- blocks

def prisma_block(old: pd.DataFrame, new: pd.DataFrame, new_all: pd.DataFrame, merges: dict) -> dict:
    from biofloc_review.run_consolidate import prisma_counts
    p_old, p_new = prisma_counts(old), prisma_counts(new)
    scielo_collapse = POST_LAYER1 - 251 - len(old)
    old_sens = decision_counts(old.sensitivity_decision)
    new_sens = decision_counts(new.sensitivity_decision)
    before_merge = decision_counts(new_all.final_decision)
    before_merge_sens = decision_counts(new_all.sensitivity_decision)
    exclusions = ("E1", "E3", "E4")

    def scope(d: pd.DataFrame, with_e5: bool) -> dict:
        other = d[[f"{c}_answer" for c in exclusions + (("E5",) if with_e5 else ())]].eq("YES").any(axis=1)
        only_e2 = (d.E2_answer == "YES") & ~other
        adv = d.final_decision.isin(ELIGIBLE)
        return {"scope": int((only_e2 | adv).sum()), "rescuable_e2_only": int(only_e2.sum()),
                "advancing": int(adv.sum()), "answered": int(d.E2_v2[only_e2 | adv].notna().sum())}

    return {
        "identified": IDENTIFIED,
        "duplicates_removed": {
            "layer1_deterministic": IDENTIFIED - POST_LAYER1,
            "layers2_3_net": 251, "scielo_identical_record_id_collapse": scielo_collapse,
            "ieee_verified_merges": old_new(0, merges["records_leaving"]),
            "total": old_new(IDENTIFIED - len(old), IDENTIFIED - len(new)),
        },
        "unique": old_new(len(old), len(new)),
        "screened": old_new(len(old), len(new)),
        "screening_layer1": {k: old_new(p_old["screening_layer1"][k], p_new["screening_layer1"][k])
                             for k in p_old["screening_layer1"]},
        "screening_layer2": {k: old_new(p_old["screening_layer2"][k], p_new["screening_layer2"][k])
                             for k in p_old["screening_layer2"]},
        "final_specificity_mode": {k: old_new(p_old["final_decisions"][k], p_new["final_decisions"][k])
                                   for k in p_old["final_decisions"]},
        "final_specificity_mode_after_e5_before_ieee_merges": before_merge,
        "sensitivity_mode": {k: old_new(old_sens[k], new_sens[k]) for k in old_sens},
        "sensitivity_mode_after_e5_before_ieee_merges": before_merge_sens,
        "sensitivity_mode_workload": old_new(old_sens["eligible"], new_sens["eligible"]),
        "sensitivity_mode_extra_records_vs_specificity": old_new(
            old_sens["eligible"] - p_old["final_decisions"]["to_eligibility"],
            new_sens["eligible"] - p_new["final_decisions"]["to_eligibility"]),
        "decision_dependent_rescreened_e2": old_new(scope(old, True), scope(new, False)),
        "shares_of_unique": {k: old_new(round(100 * p_old["final_decisions"][k] / len(old), 1),
                                        round(100 * p_new["final_decisions"][k] / len(new), 1))
                             for k in ("include", "human_review", "exclude", "to_eligibility")},
        "layer2_movement": {
            "stayed_human_review_pct": old_new(
                round(100 * p_old["screening_layer2"]["l2_human_review"] / p_old["screening_layer2"]["input"], 1),
                round(100 * p_new["screening_layer2"]["l2_human_review"] / p_new["screening_layer2"]["input"], 1))},
    }


def sources_block(old: pd.DataFrame, new: pd.DataFrame, e5: pd.DataFrame, dropped: set[str]) -> dict:
    cit = pd.read_parquet(DATA / "eligible_with_citations.parquet", columns=["record_id", "oa_cited_by_count"])
    cit["record_id"] = cit.record_id.astype(str)
    e5c = pd.read_parquet(DATA / "e5_enriched.parquet", columns=["record_id", "oa_cited_by_count"])
    e5c["record_id"] = e5c.record_id.astype(str)
    cit_new = pd.concat([cit[~cit.record_id.isin(dropped)], e5c[~e5c.record_id.isin(cit.record_id)]])
    out = {}
    for src in ("scopus", "wos", "ieee", "scielo"):
        o, n = old[old.source == src], new[new.source == src]
        co, cn = decision_counts(o.final_decision), decision_counts(n.final_decision)
        row = {"unique": old_new(len(o), len(n))}
        row.update({k: old_new(co[k], cn[k]) for k in co})
        row["eligible_share_of_eligible_set_pct"] = old_new(
            round(100 * co["eligible"] / (old.final_decision.isin(ELIGIBLE)).sum(), 1),
            round(100 * cn["eligible"] / (new.final_decision.isin(ELIGIBLE)).sum(), 1))
        row["decision_shares_within_source_pct"] = {
            k: old_new(round(100 * co[k] / len(o), 2), round(100 * cn[k] / len(n), 2))
            for k in ("INCLUDE", "HUMAN_REVIEW", "EXCLUDE")}
        if src == "scielo":
            def scielo_cit(table: pd.DataFrame, c: pd.DataFrame) -> dict:
                el = table[table.final_decision.isin(ELIGIBLE)][["record_id", "source"]].merge(c, on="record_id", how="left")
                s = el[el.source == "scielo"]
                tot = el.oa_cited_by_count.fillna(0).sum()
                return {"eligible": int(len(s)), "without_citation": int((s.oa_cited_by_count.fillna(0) == 0).sum()),
                        "share_of_eligible_citations_pct": round(100 * s.oa_cited_by_count.fillna(0).sum() / tot, 2),
                        "eligible_records_with_citation_data": int(el.oa_cited_by_count.notna().sum())}
            row["citations"] = old_new(scielo_cit(old, cit), scielo_cit(new, cit_new))
        out[src] = row
    out["note"] = ("E5 records' OpenAlex citation counts come from e5_enriched.parquet; records "
                   "without an enrichment row count as 0 citations, as in eligible_with_citations.parquet")
    return out


def language_block(old: pd.DataFrame, new: pd.DataFrame) -> dict:
    def table(d: pd.DataFrame) -> dict:
        el = d[d.final_decision.isin(ELIGIBLE)]
        inc = d[d.final_decision == "INCLUDE"]
        return {"deduplicated": language_table(d.language), "eligible": language_table(el.language),
                "include": language_table(inc.language)}

    def not_declared_by_source(d: pd.DataFrame) -> dict:
        nd = d[d.language.map(lambda v: not language_tokens(v))]
        return {"total": int(len(nd)), **{k: int(v) for k, v in nd.source.value_counts().items()}}

    def bilingual(d: pd.DataFrame) -> int:
        return int(d.language.map(lambda v: len(language_tokens(v)) > 1).sum())

    t_old, t_new = table(old), table(new)
    return {
        "rows": {stage: {lang: old_new(t_old[stage][lang], t_new[stage][lang]) for lang in t_old[stage]}
                 for stage in t_old},
        "records_declaring_two_languages": old_new(bilingual(old), bilingual(new)),
        "not_declared_by_source": old_new(not_declared_by_source(old), not_declared_by_source(new)),
        "ieee_without_language_reconciliation": {
            "manuscript_528": "IEEE records whose language field is null (480) or the literal string 'nan' (48)",
            "ieee_integrity_480": "counts only null fields (language.isna()) on the canonical corpus",
            "verdict": "528 is the right count of IEEE records without a declared language; "
                       "480 undercounts because 48 IEEE rows store the string 'nan'",
            "old": {"null": int((old.source.eq("ieee") & old.language.isna()).sum()),
                    "string_nan": int((old.source.eq("ieee") & old.language.astype(str).str.lower().eq("nan")
                                       & old.language.notna()).sum())},
        },
        "rule": "declared language only; two-language records counted in each language; values outside "
                "English/Spanish/Portuguese/Chinese (including the ISO code 'en') are Other",
    }


def validation_block(new: pd.DataFrame, old_full: pd.DataFrame, workbook: Path) -> dict:
    sample_old, n_old = crw.load()
    sample_new, n_new = crw.load(population=new)
    truth = pd.read_parquet(DATA / "validation_sample_truth.parquet", columns=["record_id", "final_decision"])
    truth["record_id"] = truth.record_id.astype(str)
    chk = truth.rename(columns={"final_decision": "design_decision"}).merge(
        new[["record_id", "final_decision", "sensitivity_decision", "round2_change"]], on="record_id", how="left")
    old_sens = old_full.set_index("record_id").loc[chk.record_id, "sensitivity_decision"].to_numpy()
    same_dec = bool((chk.final_decision.to_numpy() == chk.design_decision.to_numpy()).all())
    sample_check = {"n": int(len(chk)), "records_in_round2_corpus": int(chk.final_decision.notna().sum()),
                    "records_merged_away": int((chk.round2_change == MERGED_AWAY).sum()),
                    "records_e5_rescreened": int((chk.round2_change == E5_RESCREENED).sum()),
                    "specificity_decisions_unchanged": bool(same_dec),
                    "sensitivity_decisions_unchanged": bool((chk.sensitivity_decision.to_numpy() == old_sens).all())}
    if sample_check["records_merged_away"] or not same_dec:
        sys.exit("validation sample touched by round 2; the raw-sample statistics would change")

    strata = pd.DataFrame({"n_c": sample_old.stratum.value_counts()})
    strata["N_c_old"] = (sample_old.groupby("stratum").w.first() * strata.n_c).round().astype(int)
    strata["N_c_new"] = (sample_new.groupby("stratum").w.first() * strata.n_c).round().astype(int)
    strata_out = {s: {"n_c": int(r.n_c), "N_c": old_new(int(r.N_c_old), int(r.N_c_new))} for s, r in strata.iterrows()}

    # reviewer 2 join, as second_reviewer_agreement.main
    key = pd.read_csv(DATA / "validation_reviewer2_key.csv")
    reg = pd.read_excel(workbook, sheet_name="Registros")
    r2, problems = sra.validate_decisions(reg, expected_n=key["n"].tolist())
    if sra.has_problems(problems):
        sys.exit(f"reviewer-2 workbook incomplete or invalid: {problems}")
    r2 = r2.merge(key, on="n", how="left")
    df = sample_new.copy()
    df["sens_decision"] = sra.sensitivity_decisions(df)
    sub = df.reset_index().merge(r2, on="review_id", how="inner").set_index("index")
    if len(sub) != len(df):
        sys.exit("reviewer-2 join lost rows")
    agree_new = sra.analyse(sub)
    agree_old = json.loads((DATA / "second_reviewer_agreement.json").read_text())

    no_ci = lambda d: {k: v for k, v in d.items() if not k.endswith("_ci95")}  # noqa: E731
    raw_same = {
        "inter_human_binary_point_estimates": no_ci(agree_old["inter_human_binary"]) == no_ci(agree_new["inter_human_binary"]),
        "inter_human_binary_with_bootstrap_ci": agree_old["inter_human_binary"] == agree_new["inter_human_binary"],
        "r2_levels_vs_r1": agree_old["r2_levels_vs_r1"] == agree_new["r2_levels_vs_r1"],
        "r2_vs_pipeline_point_estimates": all(no_ci(agree_old["r2_vs_pipeline"][m]) == no_ci(agree_new["r2_vs_pipeline"][m])
                                              for m in agree_old["r2_vs_pipeline"]),
        "r2_vs_pipeline_with_bootstrap_ci": agree_old["r2_vs_pipeline"] == agree_new["r2_vs_pipeline"],
        "pipeline_vs_reference_raw": all(
            agree_old["pipeline_vs_reference"][r][m]["raw"] == agree_new["pipeline_vs_reference"][r][m]["raw"]
            for r in REFERENCES for m in ("specificity", "sensitivity")),
    }

    n_adv = {"specificity": {"old": int(old_full.final_decision.isin(ELIGIBLE).sum()),
                             "new": int(new.final_decision.isin(ELIGIBLE).sum())},
             "sensitivity": {"old": int(old_full.sensitivity_decision.isin(ELIGIBLE).sum()),
                             "new": int(new.sensitivity_decision.isin(ELIGIBLE).sum())}}
    wss_old = json.loads((DATA / "wss_reconciliation.json").read_text())
    table7, wss = {}, {}
    for ref, label in REFERENCES.items():
        table7[label] = {"relevant_in_sample": agree_new["pipeline_vs_reference"][ref]["n_relevant_sample"]}
        wss[label] = {}
        for mode in ("specificity", "sensitivity"):
            bo, bn = agree_old["pipeline_vs_reference"][ref][mode], agree_new["pipeline_vs_reference"][ref][mode]
            rec_new = reconcile(bn["reweighted"], bn["reweighted_ci95"], n_adv[mode]["new"], n_new)
            rec_old = wss_old["references"][label][mode]
            wss[label][mode] = {k: old_new(rec_old[k], rec_new[k]) for k in rec_new}
            table7[label][mode] = {
                "recall": old_new(bo["reweighted"]["recall"], bn["reweighted"]["recall"]),
                "recall_ci95": old_new(bo["reweighted_ci95"]["recall"], bn["reweighted_ci95"]["recall"]),
                "specificity": old_new(bo["reweighted"]["specificity"], bn["reweighted"]["specificity"]),
                "specificity_ci95": old_new(bo["reweighted_ci95"]["specificity"], bn["reweighted_ci95"]["specificity"]),
                "wss_exact_workload": old_new(rec_old["wss_exact_workload"], rec_new["wss_exact_workload"]),
                "wss_exact_workload_ci95": old_new(rec_old["wss_exact_workload_ci95"], rec_new["wss_exact_workload_ci95"]),
                "wss_reweighted_as_printed_round1": old_new(bo["reweighted"]["wss"], bn["reweighted"]["wss"]),
            }
    r1 = table7["First reviewer"]
    table5 = {mode: {"recall": r1[mode]["recall"], "recall_ci95": r1[mode]["recall_ci95"],
                     "specificity": r1[mode]["specificity"], "specificity_ci95": r1[mode]["specificity_ci95"],
                     "records_to_human_review": n_adv[mode]}
              for mode in ("specificity", "sensitivity")}
    table5["raw_sample_sensitivity_mapping_A"] = {
        mode: agree_new["pipeline_vs_reference"]["reviewer1"][mode]["raw"]["recall"] for mode in table5}

    # baselines (Table of the manuscript): reweighted on the new weights
    base_old = json.loads((DATA / "baselines_compare.json").read_text())
    vecs, _ = advance_vectors(sample_new)
    table_baselines = {}
    for name in BASELINE_ROWS:
        s_new, s_old = score(sample_new, vecs[name], n_new), base_old["rows"][name]
        exact = name.startswith("pipeline_")
        mode = name.removeprefix("pipeline_")
        row = {
            "sens_raw": old_new(s_old["raw"]["recall"], s_new["raw"]["recall"]),
            "spec_raw": old_new(s_old["raw"]["specificity"], s_new["raw"]["specificity"]),
            "recall_reweighted": old_new(s_old["reweighted"]["recall"], s_new["reweighted"]["recall"]),
            "recall_ci95": old_new(s_old["reweighted_ci95"]["recall"], s_new["reweighted_ci95"]["recall"]),
            "specificity_reweighted": old_new(s_old["reweighted"]["specificity"], s_new["reweighted"]["specificity"]),
            "specificity_ci95": old_new(s_old["reweighted_ci95"]["specificity"], s_new["reweighted_ci95"]["specificity"]),
            "kappa_vs_human": old_new(s_old["kappa_vs_human"], s_new["kappa_vs_human"]),
        }
        if exact:
            row["share_sent_to_review_exact"] = old_new(n_adv[mode]["old"] / n_old, n_adv[mode]["new"] / n_new)
            row["wss_exact_workload"] = old_new(
                wss["First reviewer"][mode]["wss_exact_workload"]["old"],
                wss["First reviewer"][mode]["wss_exact_workload"]["new"])
            row["wss_reweighted_as_printed_round1"] = old_new(s_old["reweighted"]["wss"], s_new["reweighted"]["wss"])
        else:
            row["share_sent_to_review_reweighted"] = old_new(s_old["share_advanced_corpus"], s_new["share_advanced_corpus"])
            row["wss_reweighted"] = old_new(s_old["reweighted"]["wss"], s_new["reweighted"]["wss"])
        table_baselines[name] = row

    return {
        "n_population": old_new(n_old, n_new),
        "sample_check": sample_check,
        "strata": strata_out,
        "raw_sample_statistics_unchanged": raw_same,
        "inter_human_binary": agree_new["inter_human_binary"],
        "r2_vs_pipeline_kappas": {m: {k: v for k, v in agree_new["r2_vs_pipeline"][m].items() if k.startswith("kappa")}
                                  for m in agree_new["r2_vs_pipeline"]},
        "records_advanced_exact": n_adv,
        "wss_formula": "WSS = (N - n_adv)/N - (1 - R_hat); N and n_adv exact for the two pipeline modes, "
                       "R_hat stratum re-weighted; baselines keep the re-weighted share (sample estimate)",
        "table_operating_modes": table5,
        "table_human_references": table7,
        "wss_reconciliation": wss,
        "table_baselines": table_baselines,
        "n_bootstrap": crw.N_BOOT,
    }


def cluster_frames(new: pd.DataFrame, dropped: set[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(old assignment: 4,682 fitted + 201 IEEE mapped; new: + 26 E5 mapped - merged away)."""
    base = pd.read_parquet(DATA / "eligible_clusters_assignment.parquet", columns=["record_id", "cluster", "year"])
    base["record_id"] = base.record_id.astype(str)
    base["method"] = "fit"
    ieee = pd.read_parquet(DATA / "ieee_cluster_assignment.parquet", columns=["record_id", "cluster"])
    ieee["record_id"] = ieee.record_id.astype(str)
    ieee["method"] = "ieee_mapped"
    e5 = pd.read_parquet(DATA / "e5_cluster_assignment.parquet", columns=["record_id", "cluster"])
    e5["record_id"] = e5.record_id.astype(str)
    e5["method"] = "e5_mapped"
    years = new.set_index("record_id").year
    for f in (ieee, e5):
        f["year"] = years.reindex(f.record_id).to_numpy()
    old = pd.concat([base, ieee], ignore_index=True)
    allnew = pd.concat([base, ieee, e5], ignore_index=True)
    allnew = allnew[~allnew.record_id.isin(dropped)].reset_index(drop=True)
    return old, allnew


def clusters_block(old_assign: pd.DataFrame, new_assign: pd.DataFrame, old: pd.DataFrame, new: pd.DataFrame) -> dict:
    info = pd.read_parquet(DATA / "eligible_clusters_info.parquet")
    ids = sorted(int(c) for c in info.cluster) + [-1]

    def sizes(a: pd.DataFrame) -> dict:
        c = a.cluster.value_counts()
        return {cluster_label(k): int(c.get(k, 0)) for k in ids}

    def table6(a: pd.DataFrame) -> dict:
        n = len(a)
        rows = {}
        for k in ids:
            m = a[a.cluster == k]
            rows[cluster_label(k)] = {"n": int(len(m)), "pct": round(100 * len(m) / n, 1),
                                      "year_median": (int(m.year.median()) if k >= 0 and m.year.notna().any() else None)}
        rows["total"] = n
        return rows

    emb_old, emb_new = old_assign[old_assign.method == "fit"], new_assign[new_assign.method == "fit"]
    ieee_old, ieee_new = old_assign[old_assign.method == "ieee_mapped"], new_assign[new_assign.method == "ieee_mapped"]
    e5_new = new_assign[new_assign.method == "e5_mapped"]
    elig_new = new[new.final_decision.isin(ELIGIBLE)].record_id
    elig_old = old[old.final_decision.isin(ELIGIBLE)].record_id
    unassigned = set(elig_new) - set(new_assign.record_id)
    if unassigned:
        sys.exit(f"{len(unassigned)} eligible round-2 records have no cluster assignment; stopping")

    def ieee_paragraph(a: pd.DataFrame) -> dict:
        c = a.cluster.value_counts()
        others = {cluster_label(k): int(v) for k, v in c.items() if k not in (21, 22, 23, -1)}
        return {"n": int(len(a)), "C21": int(c.get(21, 0)), "C22": int(c.get(22, 0)), "C23": int(c.get(23, 0)),
                "unclustered": int(c.get(-1, 0)), "others_total": int(sum(others.values())),
                "others_n_clusters": len(others), "others": others,
                "C03": int(c.get(3, 0)), "C15": int(c.get(15, 0))}

    t_old, t_new = table6(emb_old), table6(emb_new)
    return {
        "embedded_partition_fit": {"n": old_new(len(emb_old), len(emb_new)),
                                   "sizes": {k: old_new(sizes(emb_old)[k], sizes(emb_new)[k]) for k in sizes(emb_old)},
                                   "table6_rows": {k: old_new(t_old[k], t_new[k]) for k in t_old},
                                   "clustered_share_pct": old_new(round(100 * (emb_old.cluster >= 0).mean(), 1),
                                                                  round(100 * (emb_new.cluster >= 0).mean(), 1))},
        "all_assigned_fit_plus_mapped": {"n": old_new(len(old_assign), len(new_assign)),
                                         "sizes": {k: old_new(sizes(old_assign)[k], sizes(new_assign)[k])
                                                   for k in sizes(old_assign)}},
        "focus_unchanged": {"C03": old_new(sizes(emb_old)["C03"], sizes(emb_new)["C03"]),
                            "C15": old_new(sizes(emb_old)["C15"], sizes(emb_new)["C15"]),
                            "C03_members_identical": set(emb_old[emb_old.cluster == 3].record_id) == set(emb_new[emb_new.cluster == 3].record_id),
                            "C15_members_identical": set(emb_old[emb_old.cluster == 15].record_id) == set(emb_new[emb_new.cluster == 15].record_id),
                            "mapped_into_C03_or_C15": int(new_assign[(new_assign.method != "fit") & new_assign.cluster.isin(FOCUS)].shape[0])},
        "eligible": old_new(int(len(elig_old)), int(len(elig_new))),
        "eligible_without_archived_embedding": old_new(int(len(ieee_old)), int(len(ieee_new) + len(e5_new))),
        "eligible_without_archived_embedding_by_origin": {"ieee": old_new(len(ieee_old), len(ieee_new)), "e5": old_new(0, len(e5_new))},
        "ieee_201_paragraph": old_new(ieee_paragraph(ieee_old), ieee_paragraph(ieee_new)),
        "e5_mapped": {"n": int(len(e5_new)), "sizes": {k: int(v) for k, v in Counter(e5_new.cluster.map(cluster_label)).items()}},
        "embedded_records_merged_away": int(len(emb_old) - len(emb_new)),
        "ieee_mapped_records_merged_away": int(len(ieee_old) - len(ieee_new)),
        "merged_away_by_cluster": {k: int(v) for k, v in Counter(
            old_assign[~old_assign.record_id.isin(new_assign.record_id)].cluster.map(cluster_label)).items()},
    }


def supplementary_block(new: pd.DataFrame) -> dict:
    d = pd.read_parquet(DATA / "supp_eaasd_dedup.parquet", columns=["record_id", "status", "match_record_id"])
    summary = json.loads((DATA / "supp_eaasd_summary.json").read_text())
    status = new.set_index("record_id")
    matched = d[d.match_record_id.notna()].copy()
    matched["match_record_id"] = matched.match_record_id.astype(str)
    in_table = matched.match_record_id.isin(status.index)
    merged_away = matched.match_record_id.map(lambda r: r in status.index and status.at[r, "round2_change"] == MERGED_AWAY)
    e5_touched = matched.match_record_id.map(lambda r: r in status.index and status.at[r, "round2_change"] == E5_RESCREENED)
    return {
        "retrieved": summary["n_retrieved"],
        "already_in_corpus": old_new(summary["dedup"]["already_in_corpus_total"], summary["dedup"]["already_in_corpus_total"]),
        "already_in_corpus_by_layer": {"layer1": summary["dedup"]["layer1_deterministic"]["already_in_corpus"],
                                       "layer3": summary["dedup"]["layer3_llm_judge"]["already_in_corpus"]},
        "new_records": summary["n_new"], "new_eligible": summary["n_new_eligible"],
        "new_eligible_by_decision": summary["n_new_eligible_by_decision"],
        "new_eligible_clusters": summary["clusters"]["counts"],
        "matches_resolved_against_decision_table": int(in_table.sum()),
        "matches_to_records_merged_away_in_round2": int(merged_away.sum()),
        "matches_to_e5_rescreened_records": int(e5_touched.sum()),
        "eligible_share_added_pct": old_new(round(100 * summary["n_new_eligible"] / 4883, 2),
                                            round(100 * summary["n_new_eligible"] / int(new.final_decision.isin(ELIGIBLE).sum()), 2)),
        "verdict": "unchanged: the overlap was measured against Layer-1 clusters of the full corpus "
                   "(22,056 records), which the round-2 merges only regroup and the E5 re-screen does not touch; "
                   "a ProQuest record matched to a merged-away record is matched to its retained twin "
                   "(merged_into). The 33 new eligible records stay outside the analysed corpus.",
    }


def bridge_block(old_assign: pd.DataFrame, new_assign: pd.DataFrame, new: pd.DataFrame, old: pd.DataFrame) -> dict:
    emb_old = old_assign[old_assign.method == "fit"][["record_id", "cluster"]].reset_index(drop=True)
    emb_new = new_assign[new_assign.method == "fit"][["record_id", "cluster"]].reset_index(drop=True)
    # archived vectors of the fitted partition (the Qdrant export the clustering used);
    # the Qdrant index is the fallback when the local cache is absent
    ids = [r for r in (DATA / "eligible_record_ids.txt").read_text().split("\n") if r]
    if (DATA / "eligible_vectors.npy").exists():
        vec = dict(zip(ids, np.load(DATA / "eligible_vectors.npy")))
    else:
        vec = fetch_vectors(emb_old.record_id.tolist())
    if set(emb_old.record_id) - set(vec):
        sys.exit(f"{len(set(emb_old.record_id) - set(vec))} archived vectors missing; stopping")
    old_pairs = pd.read_parquet(DATA / "bridge_real_pairs.parquet")
    cr_old = json.loads((DATA / "bridge_citations_crossref.json").read_text())
    refs = pd.read_parquet(DATA / "crossref_refs.parquet")

    def geometry(a: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
        E = unit(np.stack([vec[r] for r in a.record_id]))
        c = a.cluster.to_numpy()
        a_idx, b_idx = np.where(c == 3)[0], np.where(c == 15)[0]
        pool = np.where((c != 3) & (c != 15))[0]
        b = bridge(E, a_idx, b_idx, pool)
        ks = sorted(k for k in set(c) if k >= 0)
        cent = {k: unit(E[c == k].mean(0)) for k in ks}
        rows = []
        for i, x in enumerate(ks):
            for y in ks[i + 1:]:
                p = (c != x) & (c != y)
                s = np.minimum(E[p] @ cent[x], E[p] @ cent[y])
                rows.append({"a": x, "b": y, "n_pool": int(p.sum()), "frac_07": float((s >= .7).mean()),
                             "n_07": int((s >= .7).sum()), "cos_centroids": float(cent[x] @ cent[y])})
        df = pd.DataFrame(rows)
        me = df[(df.a == 3) & (df.b == 15)].iloc[0]
        near = df[(df.cos_centroids - me.cos_centroids).abs() <= 0.03]
        res = {"n_pool": int(len(pool)), "b_ge_0.5": int((b >= .5).sum()), "b_ge_0.7": int((b >= .7).sum()),
               "b_ge_0.7_pct_of_pool": round(100 * float((b >= .7).mean()), 2),
               "b_ge_0.5_pct_of_pool": round(100 * float((b >= .5).mean()), 1),
               "n_pairs": int(len(df)), "rank_thinnest": int((df.frac_07 <= me.frac_07).sum()),
               "median_frac_pct": round(100 * float(df.frac_07.median()), 1),
               "c03_c15_centroid_cos": round(float(me.cos_centroids), 3),
               "similar_distance_pairs": int(len(near)), "rank_among_similar": int((near.frac_07 <= me.frac_07).sum()),
               "similar_pairs_set": sorted(f"C{int(x):02d}-C{int(y):02d}" for x, y in zip(near.a, near.b))}
        return res, df

    def citations(a: pd.DataFrame, table: pd.DataFrame, pairs: pd.DataFrame) -> dict:
        recs = a.merge(table[["record_id", "doi"]], on="record_id", how="left")
        recs["doi"] = recs.doi.map(clean_doi)
        recs = recs[recs.cluster >= 0]
        d = recs.merge(refs, on="doi", how="left")
        d["deposited"] = d.deposited.fillna(False).astype(bool)
        d["refs"] = d.refs.apply(lambda x: list(x) if isinstance(x, (list, np.ndarray)) else [])
        clusters = sorted(d.cluster.unique())
        idx = {c: i for i, c in enumerate(clusters)}
        with_doi = d[d.doi.notna()]
        cluster_of_doi = dict(zip(with_doi.doi, with_doi.cluster))
        citers = d[d.deposited]
        L = link_matrix(citers.cluster.tolist(), citers.refs.tolist(), cluster_of_doi, clusters)
        n_doi = with_doi.cluster.value_counts().reindex(clusters, fill_value=0).to_numpy()
        n_cite = citers.cluster.value_counts().reindex(clusters, fill_value=0).to_numpy()
        df = pd.DataFrame([{"a": x, "b": y, "links": int(L[idx[x], idx[y]] + L[idx[y], idx[x]]),
                            "density": pair_density(L, idx[x], idx[y], n_cite, n_doi)}
                           for x, y in combinations(clusters, 2)])
        df = df.merge(pairs[["a", "b", "cos_centroids"]], on=["a", "b"], how="left")
        me = df[(df.a == 3) & (df.b == 15)].iloc[0]
        near = df[(df.cos_centroids - me.cos_centroids).abs() <= 0.03]

        def cov(c):
            g = d[d.cluster == c]
            return {"records": int(len(g)), "with_doi": int(g.doi.notna().sum()), "references_deposited": int(g.deposited.sum())}
        return {"coverage": {"all_clustered": cov_all(d), "C03": cov(3), "C15": cov(15)},
                "C03_to_C15": int(L[idx[3], idx[15]]), "C15_to_C03": int(L[idx[15], idx[3]]),
                "within_C03": int(L[idx[3], idx[3]]), "within_C15": int(L[idx[15], idx[15]]),
                "rank_from_sparsest": f"{int((df.density <= me.density).sum())}/{len(df)}",
                "distance_matched": {"n_pairs": int(len(near)), "share_with_zero_links": round(float((near.links == 0).mean()), 4),
                                     "pairs_with_zero_links": int((near.links == 0).sum()),
                                     "rank_from_sparsest": int((near.density <= me.density).sum())},
                "dois_not_in_crossref_cache": int((~d.doi.dropna().isin(set(refs.doi))).sum())}

    def cov_all(d):
        return {"records": int(len(d)), "with_doi": int(d.doi.notna().sum()), "references_deposited": int(d.deposited.sum())}

    g_old, pairs_old = geometry(emb_old)
    g_new, pairs_new = geometry(emb_new)
    c_old, c_new = citations(emb_old, old, pairs_old), citations(emb_new, new, pairs_new)
    reproduction = {"b_ge_0.7": g_old["b_ge_0.7"] == 44, "n_pool": g_old["n_pool"] == 4473,
                    "rank_thinnest": g_old["rank_thinnest"] == 26, "similar_pairs": g_old["similar_distance_pairs"] == 31,
                    "crossref": {k: c_old[k] == cr_old[k] for k in ("within_C03", "within_C15", "C03_to_C15", "C15_to_C03")}}
    same_pool = len(emb_old) == len(emb_new)
    return {
        "pool_definition": "fitted partition records (archived vectors) outside C03 and C15",
        "pool_unchanged": same_pool,
        "reproduction_of_published_values": reproduction,
        "geometry": {k: old_new(g_old[k], g_new[k]) for k in g_old if k != "similar_pairs_set"},
        "similar_pairs_set_unchanged": g_old["similar_pairs_set"] == g_new["similar_pairs_set"],
        "crossref": {k: old_new(c_old[k], c_new[k]) for k in c_old},
    }


def fulltext_block(old: pd.DataFrame, new: pd.DataFrame, dropped: set[str]) -> dict:
    fm = pd.read_parquet(DATA / "fulltext_methods.parquet", columns=["record_id", "extract_ok"])
    fm["record_id"] = fm.record_id.astype(str)
    pm = pd.read_parquet(DATA / "papers_with_methods.parquet", columns=["record_id", "cluster", "year"])
    pm["record_id"] = pm.record_id.astype(str)
    c03 = set(pd.read_parquet(DATA / "c03_biofloc_papers.parquet", columns=["record_id"]).record_id.astype(str))
    pdfs_dropped = set(fm.record_id) & dropped
    n_elig_old, n_elig_new = int(old.final_decision.isin(ELIGIBLE).sum()), int(new.final_decision.isin(ELIGIBLE).sum())
    return {
        "oa_pdfs": old_new(int(len(fm)), int(len(fm) - len(pdfs_dropped))),
        "text_extracted": old_new(int(fm.extract_ok.sum()), int(fm[~fm.record_id.isin(dropped)].extract_ok.sum())),
        "abstract_level": old_new(n_elig_old - int(len(fm)), n_elig_new - int(len(fm) - len(pdfs_dropped))),
        "papers_with_methods": old_new(int(len(pm)), int((~pm.record_id.isin(dropped)).sum())),
        "full_texts_merged_away": sorted(pdfs_dropped),
        "c03_records_merged_away": int(len(c03 & dropped)),
        "methods_tables_and_fig12_affected": bool(set(pm.record_id) & dropped),
        "c03_full_texts_accessible": int(len(set(pm.record_id) & c03)),
        "note": "the 26 newly eligible E5 records had no full-text fetch attempt and count as abstract-level",
    }


# ---------------------------------------------------------------- main

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--reviewer2-workbook", type=Path, required=True, help="returned second-reviewer workbook")
    ap.add_argument("--quick", action="store_true", help="200 bootstrap draws instead of 10,000 (development)")
    ap.add_argument("--skip-bridge", action="store_true", help="skip the Qdrant-dependent bridge block")
    args = ap.parse_args()
    if args.quick:
        crw.N_BOOT = sra.N_BOOT = 200

    dec, merges, e5_summary = build_round2()
    old, _ = load_published()
    e5 = pd.read_parquet(DATA / "e5_rescreen_decisions.parquet")
    old["sensitivity_decision"], old["E2_v2"] = sensitivity_column(old)
    new = dec[dec.in_round2_corpus].reset_index(drop=True)
    dropped = set(dec.loc[~dec.in_round2_corpus, "record_id"])
    dec.to_parquet(OUT_DECISIONS, index=False)
    print(f"wrote {OUT_DECISIONS.name}: {len(dec)} rows, {len(new)} in the round-2 corpus, "
          f"{dict(Counter(dec.round2_change))}", flush=True)

    old_assign, new_assign = cluster_frames(new, dropped)
    counts = {
        "date": date.today().isoformat(),
        "corrections": {"ieee_verified_merges": {k: v for k, v in merges.items() if k != "detail"},
                        "e5_rescreen": e5_summary},
        "prisma": prisma_block(old, new, dec, merges),
        "sources": sources_block(old, new, e5, dropped),
        "language": language_block(old, new),
        "clusters": clusters_block(old_assign, new_assign, old, new),
        "supplementary_search": supplementary_block(dec),
        "fulltext": fulltext_block(old, new, dropped),
    }
    print("prisma:", json.dumps(counts["prisma"]["final_specificity_mode"]), flush=True)
    counts["validation"] = validation_block(new, old, args.reviewer2_workbook)
    print("validation done", flush=True)
    if not args.skip_bridge:
        counts["bridge"] = bridge_block(old_assign, new_assign, new, old)
        print("bridge done", flush=True)
    counts["merged_groups_detail"] = merges["detail"]
    OUT_COUNTS.write_text(json.dumps(counts, indent=2, ensure_ascii=False, default=str))
    print(f"wrote {OUT_COUNTS.name}")


if __name__ == "__main__":
    main()
