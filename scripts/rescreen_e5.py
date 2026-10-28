"""Bring the records that criterion E5 (language) excluded on its own through
the published pipeline as if E5 had never existed, and quantify the effect.

Background. Layer 1 (3 Jun 2026), Layer 2 and the consolidation were run while
a deterministic language criterion E5 was still active; E5 was removed from
the protocol later (commit 54995c4) without recomputing decisions. E5 fired on
161 records; for 31 of them (all Scopus) it was the only exclusion criterion.
The manuscript describes eleven criteria (I1-I4, E1-E4, D1-D3).

Stages (each resumable from its own ``e5_*`` output under data/processed/):
  1. select the affected set from biofloc_final_decisions.parquet (E5 = YES and
     no other E = YES) and recompute the Layer-1 decision with E5 dropped using
     the pipeline aggregation rule (screening.decision.aggregate, imported);
  2. enrich the records that now route to HUMAN_REVIEW (OpenAlex + Europe PMC,
     screening.enricher.enrich_record, 0.5 rps, 1 worker);
  3. run the published Layer 2 on them (screening.layer2.screen_record_layer2:
     same model, prompts, temperature = 0, think = False, num_predict = 400;
     the pipeline passes no seed);
  4. ask the corrected E2 prompt (scripts/pilot_prompt_v2.ask, the function that
     produced e2_v2_answers.parquet) on the records whose sensitivity-mode
     decision depends on it: the newly eligible ones (they leave if E2_v2 = YES)
     and the records excluded by E5 + E2 only (rescued if E2_v2 != YES);
  5. embed the newly eligible records (title + abstract, qwen3-embedding:8b) and
     map them into the archived 25-cluster partition with the functions of
     scripts/assign_ieee_clusters.py (UMAP-10 + HDBSCAN refit on the archived
     vectors, ARI = 1.0 check, umap.transform + approximate_predict);
  6. consolidate and write e5_rescreen_decisions.parquet (per-criterion answers)
     and e5_rescreen_results.json (before/after counts, record list, languages).

Nothing in the published tables is modified. LLM/embedding stages stop with a
message when the Ollama server is unreachable; incomplete answers are never
persisted, so a rerun retries them.

Run:  set -a; . ../.env; set +a
      .venv/bin/python scripts/rescreen_e5.py [--workers 2] [--skip-clusters]
Out:  data/processed/e5_rescreen_results.json
      data/processed/e5_rescreen_decisions.parquet
      intermediates: e5_affected_layer1.parquet, e5_enriched.parquet,
      e5_layer2_screened.parquet, e5_e2_v2_answers.parquet,
      e5_eligible_vectors.npy, e5_eligible_record_ids.txt,
      e5_cluster_assignment.parquet
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from biofloc_review.screening.classifier import CriterionAnswer
from biofloc_review.screening.criteria import CRITERIA, EXCLUSION_IDS, LLM_CRITERIA_IDS
from biofloc_review.screening.decision import FinalDecision, aggregate

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "processed"
sys.path.insert(0, str(Path(__file__).resolve().parent))

DECISIONS = DATA / "biofloc_final_decisions.parquet"
CORPUS = DATA / "biofloc_final_primary_v3.parquet"
PUBLISHED_E2V2 = DATA / "e2_v2_answers.parquet"
OPERATING_MODES = DATA / "operating_modes.json"

OUT_L1 = DATA / "e5_affected_layer1.parquet"
OUT_ENRICHED = DATA / "e5_enriched.parquet"
OUT_L2 = DATA / "e5_layer2_screened.parquet"
OUT_E2V2 = DATA / "e5_e2_v2_answers.parquet"
CACHE_VEC = DATA / "e5_eligible_vectors.npy"
CACHE_IDS = DATA / "e5_eligible_record_ids.txt"
OUT_CLUSTERS = DATA / "e5_cluster_assignment.parquet"
OUT_DECISIONS = DATA / "e5_rescreen_decisions.parquet"
OUT_RESULTS = DATA / "e5_rescreen_results.json"

CRITERION_IDS: tuple[str, ...] = tuple(c.id for c in CRITERIA)   # the eleven published criteria
OTHER_EXCLUSION = ("E1", "E2", "E3", "E4")
ELIGIBLE = ("INCLUDE", "HUMAN_REVIEW")
VALID_ANSWERS = {"YES", "NO", "UNCLEAR"}
FOCUS = (3, 15)
EMBED_BATCH = 32

assert "E5" not in CRITERION_IDS and "E5" not in EXCLUSION_IDS, "E5 is still in the criteria set"
warnings.filterwarnings("ignore", message="n_jobs value", category=UserWarning)


# ---------------------------------------------------------------- pure helpers

def _answer_cols(ids) -> list[str]:
    return [f"{c}_answer" for c in ids]


def select_affected(dec: pd.DataFrame) -> pd.Series:
    """E5 = YES and no other exclusion criterion = YES (E5 was the only exclusion reason)."""
    return (dec["E5_answer"] == "YES") & ~dec[_answer_cols(OTHER_EXCLUSION)].eq("YES").any(axis=1)


def select_e2_rescuable(dec: pd.DataFrame) -> pd.Series:
    """E5 = YES and E2 = YES with no other exclusion criterion: in sensitivity mode
    these records depend on the corrected E2 exactly like the published
    'excluded by E2 alone' set (scripts/rescreen_e2_sensitivity_mode.py)."""
    others = [c for c in OTHER_EXCLUSION if c != "E2"]
    return ((dec["E5_answer"] == "YES") & (dec["E2_answer"] == "YES")
            & ~dec[_answer_cols(others)].eq("YES").any(axis=1))


def _cell(row, key, default=None):
    v = row.get(key, default) if hasattr(row, "get") else getattr(row, key, default)
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return default
    return v


def answers_from_row(row) -> dict[str, CriterionAnswer]:
    """Rebuild the Layer-1 CriterionAnswer objects for the eleven criteria (E5 dropped)."""
    out: dict[str, CriterionAnswer] = {}
    for cid in CRITERION_IDS:
        out[cid] = CriterionAnswer(
            criterion_id=cid,
            answer=str(_cell(row, f"{cid}_answer", "UNCLEAR")),
            confidence=int(_cell(row, f"{cid}_confidence", 0)),
            evidence=str(_cell(row, f"{cid}_evidence", "")),
            reasoning=str(_cell(row, f"{cid}_reasoning", "")),
            via=str(_cell(row, f"{cid}_via", "")),
        )
    return out


def recompute_layer1(row) -> FinalDecision:
    """Layer-1 decision with E5 dropped, through the pipeline aggregation rule."""
    return aggregate(answers_from_row(row))


def consolidate_affected(l1: pd.DataFrame, l2: dict[str, tuple[str, str]]) -> pd.DataFrame:
    """Same rule as run_consolidate.consolidate: Layer 1 INCLUDE/EXCLUDE stand,
    Layer 1 HUMAN_REVIEW takes the Layer 2 decision when it exists."""
    out = l1.copy()

    def _final(r) -> tuple[str, str]:
        rid = str(r["record_id"])
        d = r["l1_new_decision"]
        if d == "INCLUDE":
            return ("INCLUDE", "Layer 1 INCLUDE (no re-evaluado en L2)")
        if d == "EXCLUDE":
            return ("EXCLUDE", r.get("l1_new_reason") or "Layer 1 EXCLUDE")
        if rid in l2:
            return (l2[rid][0], f"Layer 2: {l2[rid][1]}")
        return ("HUMAN_REVIEW", "Layer 1 HUMAN_REVIEW; L2 pendiente")

    finals = out.apply(_final, axis=1, result_type="expand")
    out["final_decision_new"] = finals[0]
    out["final_decision_source_new"] = finals[1]
    out["l2_processed_new"] = out["record_id"].astype(str).isin(set(l2))
    return out


def sensitivity_decisions(df: pd.DataFrame) -> pd.Series:
    """Sensitivity-mode decision, mirroring rescreen_e2_sensitivity_mode.py:
    eligible records leave if E2_v2 = YES; E5+E2-only records are rescued to
    HUMAN_REVIEW if E2_v2 is answered and not YES."""
    sens = df["final_decision_new"].copy()
    v2 = df["E2_v2"]
    rescued = df["e2_rescuable"].astype(bool) & v2.notna() & (v2 != "YES")
    newly_excluded = df["final_decision_new"].isin(ELIGIBLE) & (v2 == "YES")
    sens[rescued] = "HUMAN_REVIEW"
    sens[newly_excluded] = "EXCLUDE"
    return sens


# ---------------------------------------------------------------- infrastructure

def check_ollama() -> None:
    """Stop early when the LLM server is unreachable; never print its URL or token."""
    import requests
    from biofloc_review.dedup.ollama_embed import ollama_headers, ollama_url
    try:
        r = requests.get(f"{ollama_url()}/api/tags", headers=ollama_headers(), timeout=20)
        r.raise_for_status()
    except Exception as e:
        sys.exit(f"Ollama unreachable ({type(e).__name__}); stopping without fake results")


def corpus_text() -> pd.DataFrame:
    """record_id (rebuilt as the pipeline does) + the text fields of the corpus."""
    from assign_ieee_clusters import record_id
    src = pd.read_parquet(CORPUS, columns=["source", "source_id", "title", "title_norm",
                                           "abstract", "keywords", "venue"])
    src["record_id"] = [record_id(s, sid, tn) for s, sid, tn in
                        zip(src.source, src.source_id, src.title_norm)]
    src["keywords"] = src.keywords.apply(lambda k: "; ".join(k) if k is not None and len(k) else "")
    return src.drop_duplicates("record_id")


def _load_done(path: Path) -> pd.DataFrame | None:
    if path.exists():
        df = pd.read_parquet(path)
        df["record_id"] = df.record_id.astype(str)
        return df
    return None


# ---------------------------------------------------------------- stages

def stage_layer1(dec: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    affected = select_affected(dec)
    rescuable = select_e2_rescuable(dec)
    flagged = dec["E5_answer"] == "YES"
    others = dec[flagged & ~affected]
    others_changed = [str(r["record_id"]) for _, r in others.iterrows()
                      if recompute_layer1(r).decision != "EXCLUDE"]

    aff = dec[affected].copy()
    aff["record_id"] = aff.record_id.astype(str)
    new = [recompute_layer1(r) for _, r in aff.iterrows()]
    aff["l1_new_decision"] = [d.decision for d in new]
    aff["l1_new_reason"] = [d.reason for d in new]
    aff["l1_new_triggers"] = [",".join(d.triggered_criteria) for d in new]
    aff["e2_rescuable"] = False
    aff.to_parquet(OUT_L1, index=False)

    info = {
        "e5_flagged": int(flagged.sum()),
        "e5_only": int(affected.sum()),
        "e5_only_sources": aff.source.value_counts().to_dict(),
        "e5_with_other_exclusion": int(len(others)),
        "e5_with_other_exclusion_changed_without_e5": len(others_changed),
        "e5_with_other_exclusion_changed_ids": others_changed,
        "e5_plus_e2_only": int(rescuable.sum()),
        "old_decisions": aff.final_decision.value_counts().to_dict(),
        "layer1_new_decisions": aff.l1_new_decision.value_counts().to_dict(),
    }
    print("layer 1:", json.dumps(info, indent=2), flush=True)
    return aff, info


def stage_enrich(rids: list[str], papers: dict, rps: float) -> pd.DataFrame:
    from biofloc_review.run_enrich import _row_for as enrich_row
    from biofloc_review.screening.enricher import DEFAULT_SOURCES, enrich_record

    os.environ["OPENALEX_RPS"] = str(rps)
    done = _load_done(OUT_ENRICHED)
    have = set(done.record_id) if done is not None else set()
    pending = [r for r in rids if r not in have]
    print(f"enrich: {len(rids)} to enrich, {len(have)} done, {len(pending)} pending", flush=True)
    rows = done.to_dict(orient="records") if done is not None else []
    for i, rid in enumerate(pending, 1):
        rows.append(enrich_row(rid, enrich_record(papers[rid], sources=DEFAULT_SOURCES)))
        pd.DataFrame(rows).to_parquet(OUT_ENRICHED, index=False)
        print(f"  enriched {i}/{len(pending)}", flush=True)
    out = pd.DataFrame(rows) if rows else pd.DataFrame(columns=["record_id"])
    out["record_id"] = out.record_id.astype(str)
    return out


def _l2_complete(row: dict) -> bool:
    return all(row.get(f"l2_{cid}_answer") in VALID_ANSWERS for cid in LLM_CRITERIA_IDS)


def stage_layer2(aff: pd.DataFrame, papers: dict, workers: int, model: str | None) -> pd.DataFrame:
    from biofloc_review.run_screening_layer2 import (
        _build_enriched_index, _row_for, _safe_screen,
    )
    hr = aff[aff.l1_new_decision == "HUMAN_REVIEW"]
    rids = hr.record_id.tolist()
    done = _load_done(OUT_L2)
    have = set(done.record_id) if done is not None else set()
    pending = [r for r in rids if r not in have]
    print(f"layer 2: {len(rids)} HUMAN_REVIEW, {len(have)} done, {len(pending)} pending "
          f"(workers={workers})", flush=True)
    rows = done.to_dict(orient="records") if done is not None else []
    if not pending:
        return pd.DataFrame(rows)

    enriched = _build_enriched_index(OUT_ENRICHED) if OUT_ENRICHED.exists() else {}
    l1_rows = {r.record_id: {"final_decision_primary": r.l1_new_decision,
                             "final_reason_primary": r.l1_new_reason} for r in hr.itertuples()}
    incomplete = 0
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_safe_screen, papers[rid], enriched.get(rid), model): rid for rid in pending}
        for i, f in enumerate(as_completed(futs), 1):
            rid = futs[f]
            paper, state = f.result()
            row = _row_for(rid, paper, l1_rows[rid], state)
            if not _l2_complete(row):
                incomplete += 1          # not persisted: a rerun retries it
                print(f"  L2 {i}/{len(pending)} incomplete answers for {rid} (not saved)", flush=True)
                continue
            rows.append(row)
            pd.DataFrame(rows).to_parquet(OUT_L2, index=False)
            print(f"  L2 {i}/{len(pending)} {row['l2_final_decision']} ({time.time() - t0:.0f}s)",
                  flush=True)
    if incomplete:
        sys.exit(f"layer 2: {incomplete} record(s) had LLM errors; rerun to retry. Stopping.")
    return pd.DataFrame(rows)


def stage_e2_v2(scope: pd.DataFrame, text: pd.DataFrame, workers: int) -> pd.DataFrame:
    from pilot_prompt_v2 import ask   # same prompt/parsing that produced e2_v2_answers.parquet

    done = _load_done(OUT_E2V2)
    if done is None:
        done = pd.DataFrame(columns=["record_id", "E2_v2", "raw", "reused_published"])
    published = _load_done(PUBLISHED_E2V2)
    if published is not None:
        reuse = published[published.record_id.isin(scope.record_id) & ~published.record_id.isin(done.record_id)]
        if len(reuse):
            reuse = reuse.assign(reused_published=True)[["record_id", "E2_v2", "raw", "reused_published"]]
            done = pd.concat([done, reuse], ignore_index=True)
    pending = scope[~scope.record_id.isin(done.record_id)]
    pending = pending[["record_id"]].merge(text[["record_id", "title", "abstract", "keywords", "venue"]],
                                           on="record_id", how="left")
    print(f"E2_v2: scope {len(scope)}, done {len(done)}, pending {len(pending)}", flush=True)
    rows, failed = [], 0
    with ThreadPoolExecutor(workers) as ex:
        futs = {ex.submit(ask, r, "E2"): r.record_id for r in pending.itertuples()}
        for f in as_completed(futs):
            try:
                ans, raw = f.result()
            except Exception as e:
                failed += 1
                print(f"  E2_v2 fail {futs[f]}: {type(e).__name__}", flush=True)
                continue
            rows.append({"record_id": futs[f], "E2_v2": ans, "raw": raw, "reused_published": False})
    if rows:
        done = pd.concat([done, pd.DataFrame(rows)], ignore_index=True)
    done.to_parquet(OUT_E2V2, index=False)
    if failed:
        sys.exit(f"E2_v2: {failed} call(s) failed; rerun to retry. Stopping.")
    return done


def stage_clusters(elig: pd.DataFrame, text: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    import numpy as np
    from sklearn.metrics import adjusted_rand_score

    from assign_ieee_clusters import align_labels, fit, predict
    from biofloc_review.dedup.embeddings import _text_for
    from biofloc_review.dedup.ollama_embed import DEFAULT_MODEL, chunked, embed_batch

    recs = elig[["record_id"]].merge(text[["record_id", "title", "abstract"]], on="record_id",
                                     how="left").sort_values("record_id").reset_index(drop=True)
    if recs.title.isna().any():
        sys.exit("cluster stage: eligible records missing from the corpus; stopping")

    cached = [r for r in CACHE_IDS.read_text().split("\n") if r] if CACHE_IDS.exists() else []
    if CACHE_VEC.exists() and cached == recs.record_id.tolist():
        raw = np.load(CACHE_VEC)
        print(f"clusters: using cached vectors ({CACHE_VEC.name})", flush=True)
    else:
        texts = [_text_for(SimpleNamespace(title=t, abstract=a)) for t, a in zip(recs.title, recs.abstract)]
        vecs: list[list[float]] = []
        try:
            for batch in chunked(texts, EMBED_BATCH):
                vecs.extend(embed_batch(batch, model=DEFAULT_MODEL))
        except Exception as e:
            status = getattr(getattr(e, "response", None), "status_code", None)
            sys.exit(f"embedding failed: {type(e).__name__} (HTTP status {status}); stopping")
        raw = np.asarray(vecs, dtype=np.float32)
        np.save(CACHE_VEC, raw)
        CACHE_IDS.write_text("\n".join(recs.record_id))

    V = np.load(DATA / "eligible_vectors.npy")
    ids = [r for r in (DATA / "eligible_record_ids.txt").read_text().split("\n") if r]
    stored = np.load(DATA / "eligible_clusters.npy")
    if raw.shape[1] != V.shape[1]:
        sys.exit(f"dimension mismatch: new {raw.shape[1]} vs stored {V.shape[1]}; stopping")
    already = set(ids) & set(recs.record_id)
    if already:
        sys.exit(f"{len(already)} affected records already in the archived eligible set; stopping")
    Q = (raw / np.linalg.norm(raw, axis=1)[:, None]).astype(np.float32)

    t0 = time.time()
    um, hd = fit(V)
    ari = float(adjusted_rand_score(stored, hd.labels_))
    print(f"clusters: full refit ARI vs stored = {ari:.4f} ({time.time() - t0:.0f}s)", flush=True)
    if ari != 1.0:
        sys.exit("refit changed the archived partition (ARI != 1.0); stopping")
    mapping = align_labels(hd.labels_, stored)
    labels, strength = predict(um, hd, Q)
    labels = np.array([mapping.get(int(p), -1) for p in labels])
    out = pd.DataFrame({"record_id": recs.record_id, "cluster": labels.astype(int),
                        "cluster_strength": strength.astype(float)})
    out.to_parquet(OUT_CLUSTERS, index=False)
    info = {
        "model": DEFAULT_MODEL, "n_stored": int(len(ids)), "n_new": int(len(recs)),
        "full_refit_ari_vs_stored": ari,
        "assignment": "umap.transform + hdbscan.approximate_predict (scripts/assign_ieee_clusters.py)",
        "counts": {f"C{k:02d}" if k >= 0 else "noise": int(v) for k, v in sorted(Counter(labels.tolist()).items())},
        "focus": {f"C{c:02d}": int((labels == c).sum()) for c in FOCUS},
    }
    return out, info


# ---------------------------------------------------------------- reporting

def _label(k: int) -> str:
    return f"C{k:02d}" if k >= 0 else "noise"


def write_outputs(dec: pd.DataFrame, aff: pd.DataFrame, resc: pd.DataFrame, l2: pd.DataFrame,
                  e2v2: pd.DataFrame, clusters: pd.DataFrame | None, info: dict) -> dict:
    l2_map = {}
    if len(l2):
        l2_map = {str(r.record_id): (r.l2_final_decision, r.l2_final_reason) for r in l2.itertuples()}
    cons = consolidate_affected(aff, l2_map)

    # E5+E2-only records: decision unchanged in specificity mode (E2 still fires).
    resc = resc.copy()
    resc["l1_new_decision"] = "EXCLUDE"
    resc["l1_new_reason"] = [recompute_layer1(r).reason for _, r in resc.iterrows()]
    resc["l1_new_triggers"] = [",".join(recompute_layer1(r).triggered_criteria) for _, r in resc.iterrows()]
    resc["e2_rescuable"] = True
    resc = consolidate_affected(resc, {})

    allrows = pd.concat([cons, resc], ignore_index=True)
    allrows["affected_set"] = ["e5_only"] * len(cons) + ["e5_plus_e2_only"] * len(resc)
    v2 = e2v2.drop_duplicates("record_id")[["record_id", "E2_v2"]] if len(e2v2) else \
        pd.DataFrame(columns=["record_id", "E2_v2"])
    allrows = allrows.merge(v2, on="record_id", how="left")
    allrows["E2_v2"] = allrows["E2_v2"].astype(object).where(allrows["E2_v2"].notna(), None)
    allrows["sensitivity_decision_old"] = "EXCLUDE"
    allrows["sensitivity_decision_new"] = sensitivity_decisions(allrows)
    if clusters is not None and len(clusters):
        allrows = allrows.merge(clusters, on="record_id", how="left")
    else:
        allrows["cluster"] = None
        allrows["cluster_strength"] = None
    if len(l2):
        l2cols = [c for c in l2.columns if c.startswith("l2_")]
        allrows = allrows.merge(l2[["record_id"] + l2cols], on="record_id", how="left")
    allrows.to_parquet(OUT_DECISIONS, index=False)

    # ---- counts ----
    before = dec.final_decision.value_counts().to_dict()
    new = cons.final_decision_new.value_counts().to_dict()
    n_inc, n_hr = int(new.get("INCLUDE", 0)), int(new.get("HUMAN_REVIEW", 0))
    after = {"INCLUDE": before.get("INCLUDE", 0) + n_inc,
             "HUMAN_REVIEW": before.get("HUMAN_REVIEW", 0) + n_hr,
             "EXCLUDE": before.get("EXCLUDE", 0) - n_inc - n_hr}
    modes = json.loads(OPERATING_MODES.read_text()) if OPERATING_MODES.exists() else {}
    sens_before = modes.get("sensitivity_mode", {})
    sens_new = allrows.sensitivity_decision_new.value_counts().to_dict()
    s_inc, s_hr = int(sens_new.get("INCLUDE", 0)), int(sens_new.get("HUMAN_REVIEW", 0))
    sens_after = {"INCLUDE": sens_before.get("INCLUDE", 0) + s_inc,
                  "HUMAN_REVIEW": sens_before.get("HUMAN_REVIEW", 0) + s_hr,
                  "EXCLUDE": sens_before.get("EXCLUDE", 0) - s_inc - s_hr}

    def _wl(d: dict) -> int:
        return int(d.get("INCLUDE", 0) + d.get("HUMAN_REVIEW", 0))

    def _rec(r) -> dict:
        return {
            "record_id": r.record_id, "affected_set": r.affected_set, "source": r.source,
            "doi": r.doi if isinstance(r.doi, str) else None,
            "year": int(r.year) if pd.notna(r.year) else None,
            "language": r.language, "e5_evidence": r.E5_evidence,
            "title": r.title,
            "old_decision": r.final_decision, "old_reason": r.final_reason_primary,
            "layer1_new_decision": r.l1_new_decision, "layer1_new_reason": r.l1_new_reason,
            "layer2_decision": (r.l2_final_decision if "l2_final_decision" in allrows.columns
                                and isinstance(r.l2_final_decision, str) else None),
            "new_decision": r.final_decision_new,
            "E2_v2": r.E2_v2 if isinstance(r.E2_v2, str) else None,
            "sensitivity_decision_new": r.sensitivity_decision_new,
            "cluster": _label(int(r.cluster)) if pd.notna(r.cluster) else None,
            "cluster_strength": float(r.cluster_strength) if pd.notna(r.cluster_strength) else None,
        }

    records = [_rec(r) for r in allrows.itertuples()]
    by_lang = {}
    for lang, g in cons.groupby(cons.language.fillna("(none)")):
        by_lang[lang] = {"n": int(len(g)), **{k: int(v) for k, v in g.final_decision_new.value_counts().items()}}
    by_lang_resc = {}
    sub = allrows[allrows.affected_set == "e5_plus_e2_only"]
    for lang, g in sub.groupby(sub.language.fillna("(none)")):
        by_lang_resc[lang] = {"n": int(len(g)),
                              **{k: int(v) for k, v in g.sensitivity_decision_new.value_counts().items()}}

    elig_new = allrows[(allrows.affected_set == "e5_only") & allrows.final_decision_new.isin(ELIGIBLE)]
    results = {
        "date": date.today().isoformat(),
        "rule": "E5 dropped; decision = screening.decision.aggregate over I1-I4, E1-E4, D1-D3",
        "selection": info["layer1"],
        "counts_specificity_mode": {
            "deduplicated": {"before": int(len(dec)), "after": int(len(dec))},
            "eligible": {"before": _wl(before), "after": _wl(after)},
            "INCLUDE": {"before": int(before.get("INCLUDE", 0)), "after": after["INCLUDE"]},
            "HUMAN_REVIEW": {"before": int(before.get("HUMAN_REVIEW", 0)), "after": after["HUMAN_REVIEW"]},
            "EXCLUDE": {"before": int(before.get("EXCLUDE", 0)), "after": after["EXCLUDE"]},
        },
        "counts_sensitivity_mode": {
            "workload": {"before": _wl(sens_before), "after": _wl(sens_after)},
            "before": sens_before, "after": sens_after,
            "newly_eligible_excluded_by_E2_v2": int(((allrows.affected_set == "e5_only")
                                                     & allrows.final_decision_new.isin(ELIGIBLE)
                                                     & (allrows.sensitivity_decision_new == "EXCLUDE")).sum()),
            "e5_plus_e2_only_rescued": int(((allrows.affected_set == "e5_plus_e2_only")
                                            & (allrows.sensitivity_decision_new == "HUMAN_REVIEW")).sum()),
            "e2_v2_answered": int(allrows.E2_v2.notna().sum()),
        },
        "affected_new_decisions": {
            "layer1": cons.l1_new_decision.value_counts().to_dict(),
            "layer2": l2.l2_final_decision.value_counts().to_dict() if len(l2) else {},
            "final": cons.final_decision_new.value_counts().to_dict(),
        },
        "by_language_e5_only": by_lang,
        "by_language_e5_plus_e2_only_sensitivity": by_lang_resc,
        "clusters": info.get("clusters"),
        "clusters_of_new_eligible": {r.record_id: _label(int(r.cluster)) for r in elig_new.itertuples()
                                     if pd.notna(r.cluster)},
        "runtime_seconds": info.get("runtime", {}),
        "records": records,
    }
    OUT_RESULTS.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    return results


# ---------------------------------------------------------------- main

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--workers", type=int, default=2, help="Layer 2 workers (published run: 2)")
    ap.add_argument("--e2-workers", type=int, default=4, help="E2_v2 workers (published run: 4)")
    ap.add_argument("--rps", type=float, default=0.5, help="OpenAlex requests/s (published run: 0.5)")
    ap.add_argument("--model", default=None, help="override the LLM model (default: env, qwen3:14b)")
    ap.add_argument("--skip-llm", action="store_true",
                    help="stop after the Layer-1 recomputation (no enrichment, LLM or embeddings)")
    ap.add_argument("--skip-clusters", action="store_true", help="skip the cluster assignment")
    args = ap.parse_args()
    runtime: dict[str, float] = {}
    t_all = time.time()

    if os.environ.get("FLOCCUS_SCREENING_MODE", "specificity") != "specificity":
        sys.exit("FLOCCUS_SCREENING_MODE must be unset/specificity to reproduce the published Layer 2")

    dec = pd.read_parquet(DECISIONS)
    dec["record_id"] = dec.record_id.astype(str)
    t0 = time.time()
    aff, l1_info = stage_layer1(dec)
    resc = dec[select_e2_rescuable(dec)].copy()
    runtime["layer1"] = round(time.time() - t0, 1)
    info = {"layer1": l1_info}
    if args.skip_llm:
        write_outputs(dec, aff, resc, pd.DataFrame(), pd.DataFrame(), None, info | {"runtime": runtime})
        print(f"wrote {OUT_RESULTS.name} (layer 1 only)")
        return

    check_ollama()
    from biofloc_review.run_screening_layer2 import _build_paper_index
    papers = _build_paper_index(CORPUS)
    missing = [r for r in aff.record_id if r not in papers]
    if missing:
        sys.exit(f"{len(missing)} affected records not found in the corpus parquet; stopping")
    text = corpus_text()

    t0 = time.time()
    hr_rids = aff[aff.l1_new_decision == "HUMAN_REVIEW"].record_id.tolist()
    stage_enrich(hr_rids, papers, args.rps)
    runtime["enrich"] = round(time.time() - t0, 1)

    t0 = time.time()
    l2 = stage_layer2(aff, papers, args.workers, args.model)
    runtime["layer2"] = round(time.time() - t0, 1)

    l2_map = {str(r.record_id): (r.l2_final_decision, r.l2_final_reason) for r in l2.itertuples()} if len(l2) else {}
    cons = consolidate_affected(aff, l2_map)
    elig_new = cons[cons.final_decision_new.isin(ELIGIBLE)]

    t0 = time.time()
    scope = pd.concat([elig_new[["record_id"]], resc[["record_id"]]], ignore_index=True).drop_duplicates()
    e2v2 = stage_e2_v2(scope, text, args.e2_workers)
    runtime["e2_v2"] = round(time.time() - t0, 1)

    clusters = None
    if not args.skip_clusters and len(elig_new):
        t0 = time.time()
        clusters, info["clusters"] = stage_clusters(elig_new, text)
        runtime["clusters"] = round(time.time() - t0, 1)
    runtime["total"] = round(time.time() - t_all, 1)

    res = write_outputs(dec, aff, resc, l2, e2v2, clusters, info | {"runtime": runtime})
    print(json.dumps({k: res[k] for k in ("counts_specificity_mode", "counts_sensitivity_mode",
                                           "affected_new_decisions", "by_language_e5_only",
                                           "clusters_of_new_eligible", "runtime_seconds")},
                     indent=2, ensure_ascii=False))
    print(f"wrote {OUT_RESULTS.name}, {OUT_DECISIONS.name}")


if __name__ == "__main__":
    main()
