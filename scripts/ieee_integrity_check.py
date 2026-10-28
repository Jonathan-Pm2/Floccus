"""Integrity check of the IEEE Xplore records after the identifier collision.

The IEEE export's "Document Identifier" is a content-type label ("IEEE
Conferences", "IEEE Journals", ...), not a per-paper identifier. Point ids in
Qdrant were uuid5("ieee:<source_id>"), so IEEE primaries sharing a label
overwrote each other in the vector index used by Layer 2. run_ieee_fix.py later
re-keyed IEEE records (PDF link > DOI > title hash) and re-screened them.

This script answers, from the archived data, whether the collision changed any
published count:

1. counts: IEEE records identified, Layer-1 primaries, distinct old/new ids;
2. Layer 1: re-runs the deterministic layer on all 22,056 records with the
   corrected IEEE ids and compares partitions and primary choice (Layer 1 keys on
   DOI/title, so ids can only affect the tie-break among equally complete records);
3. Qdrant: colliding point ids, overwritten vectors, which record each surviving
   IEEE point actually holds;
4. Layers 2-3: candidate pairs and DUPLICATE merges that touch an IEEE point id
   or a Layer-1 cluster containing an IEEE record, with DOI/title verification;
5. counterfactual Layer 2: embeds every IEEE primary under its corrected id
   (same model and text as the dedup step) and repeats the near-duplicate search
   (cosine >= 0.93, k = 5, cross-cluster) against the archived index (stale IEEE
   points excluded) and among the IEEE records themselves;
6. screening: one decision per corrected IEEE record_id, titles match, and the
   screening stage never reads vectors (record_id is only an identifier).

Vectors are cached in data/processed/ieee_all_vectors.npy + ieee_all_record_ids.txt;
a rerun does not call the embedding server. The Qdrant container must be running
(scripts/setup.sh). Needs the env (Ollama URL/token) only on the first run.

Run:  .venv/bin/python scripts/ieee_integrity_check.py
Out:  data/processed/ieee_integrity.json
"""
from __future__ import annotations

import json
import sys
import uuid
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "processed"
THRESHOLD, TOP_K = 0.93, 5
COLLECTION = "papers_biofloc"
CACHE_VEC = DATA / "ieee_all_vectors.npy"
CACHE_IDS = DATA / "ieee_all_record_ids.txt"
JUDGED = DATA / "ieee_counterfactual_judged.json"
JUDGE_MODEL = "qwen3:14b"  # Layer-3 judge of the manuscript
ELIGIBLE = ("INCLUDE", "HUMAN_REVIEW")


# ---------------------------------------------------------------- pure helpers

def point_id(source: str, source_id: str | None, title_norm: str | None) -> str:
    """Same rule as dedup.embeddings._record_to_point_id."""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"{source}:{source_id or (title_norm or '')[:80]}"))


def collision_groups(ids: list[str]) -> dict[str, int]:
    """Point id -> number of records sharing it, for ids used more than once."""
    return {k: v for k, v in Counter(ids).items() if v > 1}


def norm_doi(doi) -> str | None:
    if doi is None or (isinstance(doi, float) and np.isnan(doi)):
        return None
    s = str(doi).strip().lower()
    return s or None


def same_work(a: dict, b: dict) -> str:
    """'doi' when both DOIs agree, 'title' when normalised titles agree, else 'no'."""
    da, db = norm_doi(a.get("doi")), norm_doi(b.get("doi"))
    if da and db and da == db:
        return "doi"
    if (a.get("title_norm") or "") and a.get("title_norm") == b.get("title_norm"):
        return "title"
    return "no"


# ---------------------------------------------------------------- steps

def new_source_ids(df: pd.DataFrame) -> pd.Series:
    sys.path.insert(0, str(ROOT / "src"))
    from biofloc_review.run_ieee_fix import _new_ieee_source_id
    return df.apply(_new_ieee_source_id, axis=1)


def layer1_rerun(raw: pd.DataFrame) -> dict:
    from biofloc_review.dedup.deterministic import dedup_deterministic
    from biofloc_review.schema import PaperRecord

    fixed = raw.copy()
    m = fixed.source == "ieee"
    fixed.loc[m, "source_id"] = new_source_ids(fixed.loc[m]).values

    def recs(df):
        out = []
        for row in df.to_dict("records"):
            d = {k: (None if isinstance(v, float) and pd.isna(v) else v) for k, v in row.items()}
            d["cluster_id"], d["is_primary"], d["merged_from"] = None, False, None
            for k in ("authors", "keywords"):
                if d.get(k) is not None:
                    d[k] = list(d[k])
            out.append(PaperRecord.model_validate(d))
        return out

    res = {}
    parts = {}
    for name, df in (("original_ids", raw), ("corrected_ids", fixed)):
        prim, full, _stats = dedup_deterministic(recs(df))
        parts[name] = full
        res[name] = {"primaries": len(prim), "ieee_primaries": sum(p.source.value == "ieee" for p in prim)}
    # partition equality on (source, title_norm, doi, year) record signatures by row order
    def membership(full):
        return Counter((r.cluster_id, r.source.value, r.title_norm, r.year, r.doi) for r in full)
    res["same_partition"] = membership(parts["original_ids"]) == membership(parts["corrected_ids"])
    prim_o = Counter((r.cluster_id, r.source.value, r.title_norm) for r in parts["original_ids"] if r.is_primary)
    prim_c = Counter((r.cluster_id, r.source.value, r.title_norm) for r in parts["corrected_ids"] if r.is_primary)
    res["primary_choice_differs_in_clusters"] = int(sum((prim_o - prim_c).values()))
    return res


def layer1_ieee_merges(full: pd.DataFrame) -> dict:
    ie_clusters = set(full.loc[full.source == "ieee", "cluster_id"])
    g = full[full.cluster_id.isin(ie_clusters)].groupby("cluster_id")
    multi = [c for c, d in g if len(d) > 1]
    rows = []
    for c in multi:
        d = full[full.cluster_id == c].to_dict("records")
        ie = [r for r in d if r["source"] == "ieee"]
        others = [r for r in d if r is not ie[0]]
        rows.append({"cluster_id": c, "size": len(d), "sources": sorted(r["source"] for r in d),
                     "ieee_is_primary": bool(any(r["is_primary"] for r in ie)),
                     "verification": sorted({same_work(ie[0], o) for o in others})})
    return {"ieee_in_multi_record_clusters": int(sum(sum(r == "ieee" for r in x["sources"]) for x in rows)),
            "clusters": len(rows),
            "verified_same_work_all": all("no" not in x["verification"] for x in rows),
            "detail": rows}


def qdrant_state(prim: pd.DataFrame) -> dict:
    from biofloc_review.dedup.embeddings import qdrant_client
    client = qdrant_client()
    n_points = client.count(COLLECTION, exact=True).count
    ie = prim[prim.source == "ieee"]
    groups = collision_groups(ie.old_pid.tolist())
    all_groups = collision_groups(prim.old_pid.tolist())
    pts = client.retrieve(COLLECTION, ids=sorted(set(ie.old_pid)), with_payload=True, with_vectors=True)
    holders = []
    for p in pts:
        members = ie[ie.old_pid == str(p.id)]
        hit = members[members.title == p.payload.get("title")]
        holders.append({"point_id": str(p.id), "records_sharing": len(members),
                        "payload_source_id": p.payload.get("source_id"),
                        "payload_matches_a_member_title": bool(len(hit)),
                        "holder_new_record_id": hit.new_rid.iloc[0] if len(hit) else None})
    return {
        "points_in_collection": int(n_points),
        "layer1_primaries": len(prim),
        "distinct_point_ids_all_sources": int(prim.old_pid.nunique()),
        "ieee_primaries": len(ie),
        "ieee_distinct_point_ids": int(ie.old_pid.nunique()),
        "ieee_collision_group_sizes": sorted(groups.values()),
        "ieee_records_in_collisions": int(sum(groups.values())),
        "ieee_vectors_overwritten": int(len(ie) - ie.old_pid.nunique()),
        "non_ieee_collisions": {"groups": len(all_groups) - len(groups),
                                "records": int(sum(all_groups.values()) - sum(groups.values())),
                                "sources": sorted(set(prim[prim.old_pid.isin(set(all_groups) - set(groups))].source))},
        "surviving_ieee_points": holders,
    }, {str(p.id): np.asarray(p.vector, dtype=np.float32) for p in pts}




class DSU:
    def __init__(self) -> None:
        self.parent: dict[str, str] = {}

    def find(self, x: str) -> str:
        self.parent.setdefault(x, x)
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[ra] = rb

    def n_components(self, items) -> int:
        return len({self.find(x) for x in items})


def archived_dsu(pairs: list[dict], clusters) -> DSU:
    """Replica of merge_confirmed: union Layer-1 clusters of every DUPLICATE pair."""
    d = DSU()
    for c in clusters:
        d.find(c)
    for p in pairs:
        if p.get("judge_verdict") == "DUPLICATE" and p.get("a_cluster") and p.get("b_cluster"):
            d.union(p["a_cluster"], p["b_cluster"])
    return d


def record_dict(row: dict) -> dict:
    out = {}
    for k in ("title", "title_norm", "authors", "year", "venue", "doi", "source", "abstract", "cluster_id"):
        v = row.get(k)
        if isinstance(v, np.ndarray):
            v = v.tolist()
        if isinstance(v, float) and np.isnan(v):
            v = None
        out[k] = v
    return out


def archived_pairs(prim: pd.DataFrame, full_l1: pd.DataFrame, pairs: list[dict], dsu: DSU) -> dict:
    ie_pids = set(prim.loc[prim.source == "ieee", "old_pid"])
    ie_clusters = set(full_l1.loc[full_l1.source == "ieee", "cluster_id"])
    by_pid = prim.drop_duplicates("old_pid", keep="last").set_index("old_pid")
    touching_pid = [p for p in pairs if p["a_id"] in ie_pids or p["b_id"] in ie_pids]
    touching_cl = [p for p in pairs if p["a_cluster"] in ie_clusters or p["b_cluster"] in ie_clusters]
    detail = []
    for p in touching_cl:
        a, b = by_pid.loc[p["a_id"]].to_dict(), by_pid.loc[p["b_id"]].to_dict()
        cl_ieee = p["a_cluster"] if p["a_cluster"] in ie_clusters else p["b_cluster"]
        ieee_rows = full_l1[(full_l1.cluster_id == cl_ieee) & (full_l1.source == "ieee")]
        detail.append({"a_source": a["source"], "b_source": b["source"], "score": p["score"],
                       "verdict": p["judge_verdict"], "same_doi_or_title": same_work(a, b),
                       "a_title": a["title"], "b_title": b["title"], "a_year": a["year"], "b_year": b["year"],
                       "a_doi": a["doi"], "b_doi": b["doi"],
                       "ieee_sibling_title": ieee_rows.title.iloc[0] if len(ieee_rows) else None})
    comp_size = Counter(dsu.find(c) for c in set(full_l1.cluster_id))
    ie_l1 = full_l1.loc[full_l1.source == "ieee", "cluster_id"]
    moved = int(sum(comp_size[dsu.find(c)] > 1 for c in ie_l1))
    return {"candidate_pairs_total": len(pairs),
            "pairs_with_ieee_point_id": len(touching_pid),
            "pairs_touching_cluster_with_ieee_record": len(touching_cl),
            "pairs_touching_detail": detail,
            "ieee_records_in_clusters_merged_by_layer3": moved}


def embed_ieee(ie: pd.DataFrame) -> np.ndarray:
    ids = ie.new_rid.tolist()
    if CACHE_VEC.exists() and CACHE_IDS.exists():
        cached = [r for r in CACHE_IDS.read_text().split("\n") if r]
        if cached == ids:
            print(f"using cached IEEE vectors ({CACHE_VEC.name})")
            return np.load(CACHE_VEC)
    from biofloc_review.dedup.embeddings import _text_for
    from biofloc_review.dedup.ollama_embed import DEFAULT_MODEL, chunked, embed_batch
    texts = [_text_for(SimpleNamespace(title=t, abstract=a if isinstance(a, str) else None))
             for t, a in zip(ie.title, ie.abstract)]
    vecs: list[list[float]] = []
    try:
        for batch in chunked(texts, 32):
            vecs.extend(embed_batch(batch, model=DEFAULT_MODEL))
            print(f"  embedded {len(vecs)}/{len(texts)}", flush=True)
    except Exception as e:  # never print the URL or token  # noqa: BLE001
        status = getattr(getattr(e, "response", None), "status_code", None)
        sys.exit(f"embedding failed: {type(e).__name__} (HTTP status {status}); stopping")
    V = np.asarray(vecs, dtype=np.float32)
    V /= np.linalg.norm(V, axis=1, keepdims=True)
    np.save(CACHE_VEC, V)
    CACHE_IDS.write_text("\n".join(ids))
    return V


def counterfactual(prim: pd.DataFrame, V: np.ndarray, stale: dict[str, np.ndarray]) -> tuple[dict, list[dict]]:
    """Layer-2 search for every IEEE primary with its own vector (k = 5, cosine >= 0.93, cross-cluster)."""
    from qdrant_client.http import models as qm

    from biofloc_review.dedup.embeddings import qdrant_client
    client = qdrant_client()
    ie = prim[prim.source == "ieee"].reset_index(drop=True)
    not_ieee = qm.Filter(must_not=[qm.FieldCondition(key="source", match=qm.MatchValue(value="ieee"))])
    non_ieee = prim[prim.source != "ieee"].drop_duplicates("old_pid", keep="last").set_index("old_pid")
    stale_holder = prim[prim.source == "ieee"].drop_duplicates("old_pid", keep="last").set_index("old_pid")

    repro = []
    for pid, vec in stale.items():
        hit = ie.index[ie.title == stale_holder.loc[pid, "title"]]
        if len(hit):
            repro.append(float(vec @ V[hit[0]] / np.linalg.norm(vec)))

    sims = V @ V.T
    np.fill_diagonal(sims, -np.inf)
    pairs: dict[tuple[str, str], dict] = {}
    best_external, n_above = [], []
    for i, r in ie.iterrows():
        resp = client.query_points(COLLECTION, query=V[i].tolist(), limit=2 * TOP_K, query_filter=not_ieee,
                                   with_payload=True)
        cands = [(float(h.score), "index", non_ieee.loc[str(h.id)].to_dict()) for h in resp.points]
        cands += [(float(sims[i, j]), "ieee", ie.loc[j].to_dict()) for j in np.argsort(-sims[i])[:2 * TOP_K]]
        cands.sort(key=lambda c: -c[0])
        best_external.append(resp.points[0].score if resp.points else float("nan"))
        n_above.append(sum(c[0] >= THRESHOLD for c in cands))
        for score, origin, o in cands[:TOP_K]:
            if score < THRESHOLD:
                break
            if o["cluster_id"] == r.cluster_id:
                continue
            key = tuple(sorted((r.cluster_id, o["cluster_id"])))
            if key in pairs and pairs[key]["score"] >= score:
                continue
            pairs[key] = {"a_cluster": r.cluster_id, "b_cluster": o["cluster_id"], "score": score,
                          "origin": origin, "a": record_dict(r.to_dict()), "b": record_dict(o),
                          "same_doi_or_title": same_work(r.to_dict(), o)}
    out = sorted(pairs.values(), key=lambda p: -p["score"])
    be = np.asarray(best_external)
    summary = {
        "ieee_records_embedded": len(ie),
        "embedding_reproducibility_cosine_vs_archived_point": {"n": len(repro), "min": min(repro), "max": max(repro)},
        "best_external_cosine": {"median": float(np.nanmedian(be)), "max": float(np.nanmax(be)),
                                 "n_at_or_above_threshold": int((be >= THRESHOLD).sum())},
        "ieee_records_with_any_neighbour_above_threshold": int(sum(n > 0 for n in n_above)),
        "ieee_records_with_more_than_k_neighbours_above_threshold": int(sum(n > TOP_K for n in n_above)),
        "new_candidate_pairs": len(out),
        "by_origin": dict(Counter(p["origin"] for p in out)),
        "by_other_source": dict(Counter(p["b"]["source"] for p in out)),
        "same_doi_or_title": dict(Counter(p["same_doi_or_title"] for p in out)),
    }
    return summary, out


def judge(pairs: list[dict]) -> list[dict]:
    """Layer 3 on the new candidate pairs, same judge and prompt; cached by cluster pair."""
    cache = json.loads(JUDGED.read_text()) if JUDGED.exists() else {}
    from biofloc_review.dedup.llm_judge import judge_pair
    for k, p in enumerate(pairs):
        key = f"{p['a_cluster']}|{p['b_cluster']}"
        if key not in cache:
            try:
                v = judge_pair(p["a"], p["b"], p["score"], model=JUDGE_MODEL)
                cache[key] = {"verdict": v.verdict, "confidence": v.confidence, "reasoning": v.reasoning}
            except Exception as e:  # never print the URL or token  # noqa: BLE001
                cache[key] = {"verdict": "ERROR", "confidence": 0, "reasoning": type(e).__name__}
            if k % 10 == 0:
                JUDGED.write_text(json.dumps(cache, indent=1, ensure_ascii=False))
                print(f"  judged {k + 1}/{len(pairs)}", flush=True)
        p.update({f"judge_{x}": cache[key][x] for x in ("verdict", "confidence", "reasoning")})
    JUDGED.write_text(json.dumps(cache, indent=1, ensure_ascii=False))
    return pairs


RANK = {"EXCLUDE": 0, "HUMAN_REVIEW": 1, "INCLUDE": 2}


def impact(pairs: list[dict], dsu: DSU, l1_clusters: set[str], prim_v3: pd.DataFrame,
           verified_only: bool = False) -> dict:
    """Effect on the canonical corpus and on the screening counts of merging the new DUPLICATE pairs.

    Rule "retained": the retained record of each merged group is chosen with the
    Layer-1 rule (_select_primary: completeness, then source priority) and keeps
    its archived screening decision; the other records leave the corpus.
    Rule "most_permissive": the merged group keeps the most permissive decision of
    its members (INCLUDE > HUMAN_REVIEW > EXCLUDE), the PRISMA-conservative choice.
    verified_only restricts the merges to DUPLICATE pairs with the same DOI or title.
    """
    from biofloc_review.dedup.deterministic import _select_primary
    from biofloc_review.schema import PaperRecord

    base_n = dsu.n_components(l1_clusters)
    dec = pd.read_parquet(DATA / "biofloc_final_decisions.parquet", columns=["record_id", "final_decision"])
    dec_map = dict(zip(dec.record_id.astype(str), dec.final_decision))
    v3 = prim_v3.copy()
    v3["record_id"] = [point_id(s, i, t) for s, i, t in zip(v3.source, v3.source_id, v3.title_norm)]
    v3 = v3.drop_duplicates("record_id")
    root_rows: dict[str, list[dict]] = {}
    for row in v3.to_dict("records"):
        root_rows.setdefault(dsu.find(row["cluster_id"]), []).append(row)

    new = DSU()
    new.parent = dict(dsu.parent)
    dups = [p for p in pairs if p["judge_verdict"] == "DUPLICATE"
            and (not verified_only or p["same_doi_or_title"] != "no")]
    for p in dups:
        new.union(p["a_cluster"], p["b_cluster"])
    groups: dict[str, set[str]] = {}
    for root in {dsu.find(c) for c in l1_clusters}:
        groups.setdefault(new.find(root), set()).add(root)
    merged = [g for g in groups.values() if len(g) > 1]

    def as_record(row: dict) -> PaperRecord:
        d = {k: (None if isinstance(v, float) and pd.isna(v) else v) for k, v in row.items() if k != "record_id"}
        for k in ("authors", "keywords", "merged_from"):
            if isinstance(d.get(k), np.ndarray):
                d[k] = d[k].tolist()
        return PaperRecord.model_validate(d)

    clustered = set(np.loadtxt(DATA / "eligible_record_ids.txt", dtype=str))
    sample = set(pd.read_parquet(DATA / "validation_sample_truth.parquet").record_id.astype(str))
    dropped, before, after, detail = Counter(), Counter(), Counter(), []
    dropped_ids: list[str] = []
    for g in merged:
        rows = [r for root in g for r in root_rows.get(root, [])]
        win = _select_primary([as_record(r) for r in rows])
        decisions = [dec_map.get(r["record_id"]) for r in rows]
        before.update(decisions)
        after[max(decisions, key=lambda x: RANK.get(x, -1))] += 1
        for k, r in enumerate(rows):
            if k != win:
                dropped[dec_map.get(r["record_id"])] += 1
                dropped_ids.append(r["record_id"])
        detail.append({"sources": [r["source"] for r in rows], "decisions": decisions,
                       "retained": rows[win]["source"], "titles": [r["title"][:100] for r in rows]})
    agree = sum(len(set(d["decisions"])) == 1 for d in detail)
    eligible_drop = dropped.get("INCLUDE", 0) + dropped.get("HUMAN_REVIEW", 0)
    perm = {k: before.get(k, 0) - after.get(k, 0) for k in RANK}
    return {
        "verified_only": verified_only,
        "duplicate_verdicts": len(dups),
        "verdicts": dict(Counter(p["judge_verdict"] for p in pairs)),
        "duplicate_verdicts_same_doi_or_title": int(sum(p["same_doi_or_title"] != "no" for p in dups)),
        "components_before": base_n, "components_after": new.n_components(l1_clusters),
        "net_additional_merges": base_n - new.n_components(l1_clusters),
        "groups_merged": len(merged),
        "groups_with_identical_decisions": agree,
        "records_leaving_corpus_by_decision": dict(dropped),
        "eligible_records_leaving": int(eligible_drop),
        "counterfactual_counts": {
            "post_capa3_unique": 17063 - (base_n - new.n_components(l1_clusters)),
            "include": 1385 - dropped.get("INCLUDE", 0),
            "human_review": 3498 - dropped.get("HUMAN_REVIEW", 0),
            "exclude": 12180 - dropped.get("EXCLUDE", 0),
            "to_eligibility": 4883 - eligible_drop,
        },
        "most_permissive_rule": {
            "records_leaving_by_decision": perm,
            "counterfactual_counts": {
                "include": 1385 - perm["INCLUDE"], "human_review": 3498 - perm["HUMAN_REVIEW"],
                "exclude": 12180 - perm["EXCLUDE"],
                "to_eligibility": 4883 - perm["INCLUDE"] - perm["HUMAN_REVIEW"]}},
        "leaving_records_among_clustered_4682": len(set(dropped_ids) & clustered),
        "leaving_records_in_validation_sample": len(set(dropped_ids) & sample),
        "groups_touching_validation_sample": int(sum(
            any(r["record_id"] in sample for root in g for r in root_rows.get(root, [])) for g in merged)),
        "merged_group_detail": detail,
    }


def screening_check(prim_v3: pd.DataFrame) -> dict:
    dec = pd.read_parquet(DATA / "biofloc_final_decisions.parquet",
                          columns=["record_id", "source", "title", "final_decision"])
    pre = pd.read_parquet(DATA / "biofloc_screened_pre_ieee_fix.parquet", columns=["record_id", "source"])
    ie = prim_v3[prim_v3.source == "ieee"]
    rid = {point_id("ieee", s, t): title for s, t, title in zip(ie.source_id, ie.title_norm, ie.title)}
    d = dec[dec.source == "ieee"]
    elig = set(np.loadtxt(DATA / "eligible_record_ids.txt", dtype=str))
    src = ROOT / "src" / "biofloc_review"
    reads_vectors = any(k in (src / f).read_text()
                        for f in ("run_screening.py", "run_screening_layer2.py", "run_consolidate.py")
                        for k in ("query_points", "embed_batch", "QdrantClient", "qdrant_client("))
    return {
        "ieee_decisions": len(d), "ieee_distinct_record_ids": int(d.record_id.nunique()),
        "pre_fix_distinct_record_ids": int(pre[pre.source == "ieee"].record_id.nunique()),
        "record_ids_match_corrected_ids": bool(set(d.record_id.astype(str)) == set(rid)),
        "titles_match": bool(all(rid.get(r) == t for r, t in zip(d.record_id.astype(str), d.title))),
        "decisions": d.final_decision.value_counts().to_dict(),
        "eligible": int(d.final_decision.isin(ELIGIBLE).sum()),
        "eligible_with_archived_vector": int(d[d.final_decision.isin(ELIGIBLE)].record_id.astype(str).isin(elig).sum()),
        "screening_code_reads_vectors": reads_vectors,
    }


def main() -> None:
    raw = pd.read_parquet(DATA / "biofloc.parquet")
    full_l1 = pd.read_parquet(DATA / "biofloc_dedup_full.parquet")
    prim = pd.read_parquet(DATA / "biofloc_dedup_primary.parquet")
    prim_v3 = pd.read_parquet(DATA / "biofloc_final_primary_v3.parquet")
    pairs = json.loads((DATA / "biofloc_dedup_primary_near_dup_pairs_judged.json").read_text())

    prim["old_pid"] = [point_id(s, i, t) for s, i, t in zip(prim.source, prim.source_id, prim.title_norm)]
    m = prim.source == "ieee"
    prim["new_rid"] = prim.old_pid
    prim.loc[m, "new_rid"] = [point_id("ieee", s, t) for s, t in
                              zip(new_source_ids(prim.loc[m]), prim.loc[m, "title_norm"])]
    raw_ie = raw[raw.source == "ieee"]
    counts = {"identified_ieee": len(raw_ie),
              "distinct_document_identifier_values": int(raw_ie.source_id.nunique(dropna=False)),
              "distinct_corrected_ids_identified": int(new_source_ids(raw_ie).nunique()),
              "ieee_layer1_primaries": int(m.sum()),
              "ieee_removed_by_layer1": int(len(raw_ie) - m.sum()),
              "ieee_in_canonical_corpus": int((prim_v3.source == "ieee").sum()),
              "ieee_without_language_field": int(prim_v3.loc[prim_v3.source == "ieee", "language"].isna().sum())}
    print("counts:", counts, flush=True)

    l1 = layer1_rerun(raw)
    print("layer 1 rerun:", l1, flush=True)
    l1_merges = layer1_ieee_merges(full_l1)
    q, stale = qdrant_state(prim)
    print("qdrant:", {k: v for k, v in q.items() if k != "surviving_ieee_points"}, flush=True)
    l1_clusters = set(prim.cluster_id)
    dsu = archived_dsu(pairs, l1_clusters)
    assert dsu.n_components(l1_clusters) == len(prim_v3), "DSU replica does not reproduce the archived merge"
    l23 = archived_pairs(prim, full_l1, pairs, dsu)
    print("archived layers 2-3:", {k: v for k, v in l23.items() if k != "pairs_touching_detail"}, flush=True)

    V = embed_ieee(prim[m].reset_index(drop=True))
    cf, new_pairs = counterfactual(prim, V, stale)
    print("counterfactual:", cf, flush=True)
    new_pairs = judge(new_pairs)
    imp = impact(new_pairs, dsu, l1_clusters, prim_v3)
    imp_verified = impact(new_pairs, dsu, l1_clusters, prim_v3, verified_only=True)
    print("impact:", {k: v for k, v in imp.items() if k != "merged_group_detail"}, flush=True)
    scr = screening_check(prim_v3)
    print("screening:", scr, flush=True)

    res = {"counts": counts, "layer1_rerun_with_corrected_ids": l1, "layer1_merges_with_ieee": l1_merges,
           "qdrant": q, "archived_layers_2_3": l23,
           "counterfactual_layer2": {**cf, "judge_model": JUDGE_MODEL,
                                     "pairs": [{k: v for k, v in p.items() if k not in ("a", "b")}
                                               | {"a_title": p["a"]["title"], "b_title": p["b"]["title"],
                                                  "a_source": p["a"]["source"], "b_source": p["b"]["source"],
                                                  "a_year": p["a"]["year"], "b_year": p["b"]["year"],
                                                  "a_doi": p["a"]["doi"], "b_doi": p["b"]["doi"]}
                                               for p in new_pairs]},
           "counterfactual_layer3_impact": imp,
           "counterfactual_layer3_impact_verified_only": imp_verified, "screening": scr}
    (DATA / "ieee_integrity.json").write_text(json.dumps(res, indent=2, ensure_ascii=False, default=str))
    for block in (imp, imp_verified):
        print(json.dumps({k: v for k, v in block.items() if k != "merged_group_detail"}, indent=2))


if __name__ == "__main__":
    main()
