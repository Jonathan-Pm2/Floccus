"""Supplementary search in ProQuest EAASD (Reviewer 2, comment 8), run through the published pipeline.

The Scopus three-axis query (extended with BFT-specific terms) was run on
2026-10-25 in ProQuest's "Earth, Atmospheric & Aquatic Science Database"
(peer reviewed) and returned 950 records. This script measures what that
search adds to the published corpus, reusing the pipeline's own code:

  1. ingest     sources/proquest.py -> PaperRecord (source="proquest", source_id=StoreId)
  2. dedup      Layer 1: dedup.deterministic.cluster_deterministic over the
                corpus (biofloc_final_full, 22,056 records with their merged
                siblings) + the 950 records (DOI, else normalised title + year).
                Layer 2: qwen3-embedding:8b vectors of the Layer-1 survivors
                (embeddings._text_for), cosine >= 0.93, top-5, against the
                Qdrant collection papers_biofloc (READ-ONLY query, no upsert)
                and among the survivors themselves.
                Layer 3: dedup.llm_judge.judge_pair (qwen3:14b) on every
                Layer-2 pair; only DUPLICATE merges (as merge_confirmed does).
  3. screening  new records only. Layer 1 = graph.node_prefilter +
                classifier.ask_criterion for the 9 LLM criteria + decision.aggregate
                (the nodes of graph.screen_record, run through one pool capped
                at 4 concurrent LLM requests), mode "specificity" (published).
                Layer 2 = enricher.enrich_record (OpenAlex + Europe PMC) +
                layer2.ask_criterion_layer2 on the Layer-1 HUMAN_REVIEW records,
                consolidated with run_consolidate.consolidate.
                Optional --with-sensitivity: Layer 1 again in "sensitivity" mode.
  4. clusters   new eligible records (INCLUDE or HUMAN_REVIEW) mapped onto the
                archived 25-cluster partition with assign_ieee_clusters.fit /
                predict / align_labels (UMAP-10 + HDBSCAN refit on the archived
                4,682 vectors, ARI = 1.0 checked; umap.transform +
                hdbscan.approximate_predict).

Resumable: vectors, judge verdicts, screening answers and enrichment are
cached in data/processed/supp_eaasd_* and reused on rerun. Nothing under
data/processed is modified except supp_eaasd_* files. LLM errors are not
cached (retried on rerun) and their messages are redacted (no server URL).

Run:  set -a && . ../.env && set +a && .venv/bin/python scripts/supplementary_search_eaasd.py
      [--xls ../ProQuest/ProQuestDocuments-2026-10-25.xls] [--with-sensitivity]
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "processed"
sys.path.insert(0, str(ROOT / "scripts"))

from biofloc_review.dedup.deterministic import _select_primary, cluster_deterministic  # noqa: E402
from biofloc_review.dedup.embeddings import _record_to_point_id, _text_for  # noqa: E402
from biofloc_review.schema import PaperRecord, Source  # noqa: E402
from biofloc_review.sources.proquest import load_proquest  # noqa: E402

DEFAULT_XLS = ROOT.parent / "ProQuest" / "ProQuestDocuments-2026-10-25.xls"
SEARCH_DATE = "2026-10-25"
DATABASE = "ProQuest Earth, Atmospheric & Aquatic Science Database (EAASD)"
EMBED_MODEL = "qwen3-embedding:8b"
COLLECTION = "papers_biofloc"
THRESHOLD, TOP_K = 0.93, 5
MAX_CONCURRENCY = 4
EMBED_BATCH = 16
ELIGIBLE = ("INCLUDE", "HUMAN_REVIEW")
FOCUS = (3, 15)

P = {
    "records": DATA / "supp_eaasd_records.parquet",
    "vectors": DATA / "supp_eaasd_vectors.npy",
    "vector_ids": DATA / "supp_eaasd_vector_ids.txt",
    "pairs": DATA / "supp_eaasd_near_dup_pairs_judged.json",
    "dedup": DATA / "supp_eaasd_dedup.parquet",
    "answers": DATA / "supp_eaasd_screening_answers.parquet",
    "enriched": DATA / "supp_eaasd_enriched.parquet",
    "decisions": DATA / "supp_eaasd_decisions.parquet",
    "clusters": DATA / "supp_eaasd_cluster_assignment.parquet",
    "summary": DATA / "supp_eaasd_summary.json",
}

log = logging.getLogger("supp_eaasd")


# ---------------------------------------------------------------- pure helpers

def corpus_key(rid: str) -> str:
    return f"corpus:{rid}"


def layer1_overlap(clusters: dict[str, list[PaperRecord]]) -> dict[str, dict]:
    """Classify each new-source record from deterministic clusters (corpus + new).

    A new record in a cluster with any corpus member is already in the corpus
    (L1_corpus). Among new-only clusters, one representative (pipeline's
    _select_primary) survives and the rest are internal duplicates (L1_internal).
    """
    out: dict[str, dict] = {}
    for cluster in clusters.values():
        new = [r for r in cluster if r.source is Source.PROQUEST]
        if not new:
            continue
        old = [r for r in cluster if r.source is not Source.PROQUEST]
        if old:
            match = next((r for r in old if r.is_primary), old[0])
            for r in new:
                out[_record_to_point_id(r)] = {"status": "L1_corpus", "match": _record_to_point_id(match),
                                               "match_title": match.title, "via": "doi" if r.doi else "title+year"}
            continue
        rep = new[_select_primary(new)]
        rep_id = _record_to_point_id(rep)
        for r in new:
            rid = _record_to_point_id(r)
            out[rid] = ({"status": "survivor", "match": None, "match_title": None, "via": None} if rid == rep_id else
                        {"status": "L1_internal", "match": rep_id, "match_title": rep.title,
                         "via": "doi" if r.doi else "title+year"})
    return out


def layer3_resolve(survivors: list[str], dup_pairs: list[tuple[str, str]],
                   rank: dict[str, tuple] | None = None) -> dict[str, dict]:
    """Union-find over DUPLICATE pairs (corpus nodes prefixed "corpus:").

    A survivor whose component holds a corpus node is L3_corpus; otherwise one
    survivor per component is kept (highest rank, then smallest id) and the
    others are L3_internal. Survivors untouched by any pair are not returned.
    """
    parent: dict[str, str] = {}

    def find(x: str) -> str:
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in dup_pairs:
        parent[find(a)] = find(b)
    direct: dict[str, str] = {}
    for a, b in dup_pairs:
        for x, y in ((a, b), (b, a)):
            if not x.startswith("corpus:") and y.startswith("corpus:"):
                direct.setdefault(x, y)
    comps: dict[str, list[str]] = {}
    for x in list(parent):
        comps.setdefault(find(x), []).append(x)
    surv = set(survivors)
    rank = rank or {}
    out: dict[str, dict] = {}
    for members in comps.values():
        corpus_nodes = sorted(m for m in members if m.startswith("corpus:"))
        new = sorted(m for m in members if m in surv)
        if corpus_nodes:
            for m in new:
                out[m] = {"status": "L3_corpus", "match": direct.get(m, corpus_nodes[0]).removeprefix("corpus:")}
        elif len(new) > 1:
            keep = max(new, key=lambda m: (rank.get(m, ()), [-ord(c) for c in m]))
            for m in new:
                if m != keep:
                    out[m] = {"status": "L3_internal", "match": keep}
    return out


def cluster_label(k: int) -> str:
    return f"C{k:02d}" if k >= 0 else "noise"


def redact_error(text: str | None) -> str | None:
    """Keep the exception class of an LLM error, drop its message (may carry the server URL)."""
    if text and text.startswith("LLM error:"):
        return text.split(":")[0] + ": " + text.split(":")[1].strip() + " (message redacted)"
    return text


# ---------------------------------------------------------------- io helpers

def _rows_to_records(df: pd.DataFrame) -> list[PaperRecord]:
    out = []
    for row in df.to_dict(orient="records"):
        d = {k: (None if isinstance(v, float) and pd.isna(v) else v) for k, v in row.items()}
        out.append(PaperRecord.model_validate(d))
    return out


def _write_json(path: Path, obj) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False))
    tmp.replace(path)


def _write_parquet(path: Path, df: pd.DataFrame) -> None:
    tmp = path.with_suffix(".tmp.parquet")
    df.to_parquet(tmp, index=False)
    tmp.replace(path)


def _err_class(e: Exception) -> str:
    status = getattr(getattr(e, "response", None), "status_code", None)
    return f"{type(e).__name__}" + (f" (HTTP {status})" if status else "")


# ---------------------------------------------------------------- step 1-2

def ingest(xls: Path) -> list[PaperRecord]:
    recs = load_proquest(xls)
    if len({r.source_id for r in recs}) != len(recs):
        sys.exit("StoreId is not unique; stopping")
    df = pd.DataFrame([r.model_dump(mode="json") for r in recs])
    df.insert(0, "record_id", [_record_to_point_id(r) for r in recs])
    _write_parquet(P["records"], df)
    log.info("ingested %d records -> %s", len(recs), P["records"].name)
    return recs


def load_corpus() -> tuple[list[PaperRecord], dict[str, dict]]:
    full = _rows_to_records(pd.read_parquet(DATA / "biofloc_final_full.parquet"))
    v3 = _rows_to_records(pd.read_parquet(DATA / "biofloc_final_primary_v3.parquet"))
    by_id: dict[str, dict] = {}
    for r in full + v3:  # v3 last: SciELO-scraped abstracts win
        by_id[_record_to_point_id(r)] = r.model_dump(mode="json")
    return full, by_id


def embed_records(recs: list[PaperRecord]) -> dict[str, np.ndarray]:
    from biofloc_review.dedup.ollama_embed import embed_batch

    ids = [_record_to_point_id(r) for r in recs]
    cache: dict[str, np.ndarray] = {}
    if P["vectors"].exists() and P["vector_ids"].exists():
        cids = [x for x in P["vector_ids"].read_text().split("\n") if x]
        V = np.load(P["vectors"])
        cache = dict(zip(cids, V))
    todo = [(i, r) for i, r in zip(ids, recs) if i not in cache]
    log.info("embeddings: %d cached, %d to embed", len(ids) - len(todo), len(todo))

    def _one(batch):
        return batch, embed_batch([_text_for(r) for _, r in batch], model=EMBED_MODEL, timeout=120)

    batches = [todo[i:i + EMBED_BATCH] for i in range(0, len(todo), EMBED_BATCH)]
    with ThreadPoolExecutor(max_workers=2) as ex:
        futs = [ex.submit(_one, b) for b in batches]
        for n, fut in enumerate(as_completed(futs), 1):
            try:
                batch, vecs = fut.result()
            except Exception as e:
                sys.exit(f"embedding failed: {_err_class(e)}; stopping (cached vectors kept)")
            for (i, _), v in zip(batch, vecs):
                cache[i] = np.asarray(v, dtype=np.float32)
            if n % 5 == 0 or n == len(batches):
                order = list(cache)
                np.save(P["vectors"], np.stack([cache[k] for k in order]))
                P["vector_ids"].write_text("\n".join(order))
                log.info("  embedded batch %d/%d", n, len(batches))
    return {i: cache[i] for i in ids}


def layer2_pairs(recs: list[PaperRecord], vecs: dict[str, np.ndarray]) -> list[dict]:
    from qdrant_client import QdrantClient

    client = QdrantClient(url=os.environ.get("QDRANT_URL", "http://localhost:6333"))
    if not client.collection_exists(COLLECTION):
        sys.exit(f"Qdrant collection {COLLECTION} not found; run `docker start qdrant-biofloc`")
    pairs: list[dict] = []
    ids = [_record_to_point_id(r) for r in recs]
    by_id = dict(zip(ids, recs))
    for rid in ids:
        v = vecs[rid] / np.linalg.norm(vecs[rid])
        resp = client.query_points(COLLECTION, query=v.tolist(), limit=TOP_K,
                                   score_threshold=THRESHOLD, with_payload=True)
        for h in resp.points:
            pl = h.payload or {}
            pairs.append({"a_id": rid, "b_id": corpus_key(str(h.id)), "kind": "corpus",
                          "a_title": by_id[rid].title, "b_title": pl.get("title"),
                          "score": float(h.score), "b_payload": pl})
    # among the new survivors (cosine, same threshold and top-k)
    M = np.stack([vecs[i] / np.linalg.norm(vecs[i]) for i in ids])
    S = M @ M.T
    np.fill_diagonal(S, -1)
    seen = set()
    for a in range(len(ids)):
        for b in np.argsort(-S[a])[:TOP_K]:
            if S[a, b] < THRESHOLD:
                break
            key = tuple(sorted((ids[a], ids[b])))
            if key in seen:
                continue
            seen.add(key)
            pairs.append({"a_id": key[0], "b_id": key[1], "kind": "internal",
                          "a_title": by_id[key[0]].title, "b_title": by_id[key[1]].title,
                          "score": float(S[a, b])})
    log.info("layer 2: %d candidate pairs (%d vs corpus, %d internal)", len(pairs),
             sum(p["kind"] == "corpus" for p in pairs), sum(p["kind"] == "internal" for p in pairs))
    return pairs


def layer3_judge(pairs: list[dict], new_by_id: dict[str, dict], corpus_by_id: dict[str, dict]) -> list[dict]:
    from biofloc_review.dedup.llm_judge import default_llm_model, judge_pair

    cache: dict[str, dict] = {}
    if P["pairs"].exists():
        cache = {f'{p["a_id"]}|{p["b_id"]}': p for p in json.loads(P["pairs"].read_text())
                 if p.get("judge_verdict") in {"DUPLICATE", "NOT_DUPLICATE", "UNCLEAR"} and not p.get("judge_error")}
    model = default_llm_model()

    def _rec(pid: str, payload: dict | None) -> dict | None:
        if pid.startswith("corpus:"):
            return corpus_by_id.get(pid.removeprefix("corpus:")) or payload  # IEEE pre-fix ids: payload only
        return new_by_id.get(pid)

    def _one(p: dict) -> dict:
        q = {k: v for k, v in p.items() if k != "b_payload"}
        try:
            v = judge_pair(_rec(p["a_id"], None), _rec(p["b_id"], p.get("b_payload")), p["score"], model=model)
            return {**q, "judge_verdict": v.verdict, "judge_confidence": v.confidence,
                    "judge_reasoning": v.reasoning, "judge_model": model}
        except Exception as e:
            return {**q, "judge_verdict": "UNCLEAR", "judge_confidence": 0,
                    "judge_reasoning": f"LLM error: {_err_class(e)}", "judge_error": True, "judge_model": model}

    todo = [p for p in pairs if f'{p["a_id"]}|{p["b_id"]}' not in cache]
    log.info("layer 3: %d cached verdicts, %d to judge", len(pairs) - len(todo), len(todo))
    with ThreadPoolExecutor(max_workers=MAX_CONCURRENCY) as ex:
        for n, fut in enumerate(as_completed([ex.submit(_one, p) for p in todo]), 1):
            j = fut.result()
            cache[f'{j["a_id"]}|{j["b_id"]}'] = j
            if n % 10 == 0 or n == len(todo):
                _write_json(P["pairs"], list(cache.values()))
                log.info("  judged %d/%d", n, len(todo))
    if not todo:
        _write_json(P["pairs"], list(cache.values()))
    return [cache[f'{p["a_id"]}|{p["b_id"]}'] for p in pairs]


# ---------------------------------------------------------------- step 3

ANS_COLS = ["record_id", "layer", "mode", "criterion_id", "answer", "confidence", "evidence", "reasoning", "via"]


def _load_answers() -> pd.DataFrame:
    return pd.read_parquet(P["answers"]) if P["answers"].exists() else pd.DataFrame(columns=ANS_COLS)


def run_criteria(recs: list[PaperRecord], layer: str, mode: str, enriched: dict | None = None) -> dict:
    """Return {record_id: {criterion_id: CriterionAnswer}} for one layer/mode, using the cache."""
    from biofloc_review.screening.classifier import CriterionAnswer, ask_criterion
    from biofloc_review.screening.criteria import CRITERIA_BY_ID, LLM_CRITERIA_IDS
    from biofloc_review.screening.graph import node_prefilter
    from biofloc_review.screening.layer2 import ask_criterion_layer2

    os.environ["FLOCCUS_SCREENING_MODE"] = mode
    cache = _load_answers()
    cache = cache[(cache.layer == layer) & (cache["mode"] == mode)]
    have = {(r, c) for r, c in zip(cache.record_id, cache.criterion_id)}
    out: dict[str, dict] = {}
    for row in cache.to_dict(orient="records"):
        out.setdefault(row["record_id"], {})[row["criterion_id"]] = CriterionAnswer(
            criterion_id=row["criterion_id"], answer=row["answer"], confidence=int(row["confidence"]),
            evidence=row["evidence"], reasoning=row["reasoning"], via=row["via"])
    tasks = []
    for r in recs:
        rid = _record_to_point_id(r)
        for cid, ans in node_prefilter({"record": r})["answers"].items():
            out.setdefault(rid, {})[cid] = ans
        tasks += [(rid, r, cid) for cid in LLM_CRITERIA_IDS if (rid, cid) not in have]
    log.info("screening %s/%s: %d records, %d LLM calls to run (%d cached)",
             layer, mode, len(recs), len(tasks), len(have))

    def _one(t):
        rid, r, cid = t
        if layer == "L1":
            return rid, ask_criterion(r, CRITERIA_BY_ID[cid])
        return rid, ask_criterion_layer2(r, enriched.get(rid), CRITERIA_BY_ID[cid])

    new_rows: list[dict] = []
    n_err = 0
    t0 = time.time()

    def _flush():
        if new_rows:
            _write_parquet(P["answers"], pd.concat([_load_answers(), pd.DataFrame(new_rows)], ignore_index=True))
            new_rows.clear()

    with ThreadPoolExecutor(max_workers=MAX_CONCURRENCY) as ex:
        for n, fut in enumerate(as_completed([ex.submit(_one, t) for t in tasks]), 1):
            rid, ans = fut.result()
            if ans.evidence == "(error LLM)":
                n_err += 1
                ans.reasoning = redact_error(ans.reasoning)
            else:
                new_rows.append({"record_id": rid, "layer": layer, "mode": mode, "criterion_id": ans.criterion_id,
                                 "answer": ans.answer, "confidence": int(ans.confidence), "evidence": ans.evidence,
                                 "reasoning": ans.reasoning, "via": ans.via})
            out.setdefault(rid, {})[ans.criterion_id] = ans
            if n % 36 == 0 or n == len(tasks):
                _flush()
                el = time.time() - t0
                log.info("  %d/%d calls (%.1f s/call wall, %d errors)", n, len(tasks), el / n, n_err)
    _flush()
    if n_err:
        log.warning("%d LLM errors (not cached; rerun to retry)", n_err)
    return out


def enrich(recs: list[PaperRecord]) -> dict:
    from biofloc_review.run_enrich import _row_for
    from biofloc_review.run_screening_layer2 import _build_enriched_index
    from biofloc_review.screening.enricher import enrich_record

    have = set(pd.read_parquet(P["enriched"]).record_id) if P["enriched"].exists() else set()
    todo = [r for r in recs if _record_to_point_id(r) not in have]
    log.info("enrichment: %d cached, %d to fetch", len(recs) - len(todo), len(todo))
    rows = []
    for r in todo:
        e = enrich_record(r)
        rows.append(_row_for(_record_to_point_id(r), e))
    if rows:
        old = pd.read_parquet(P["enriched"]) if P["enriched"].exists() else None
        _write_parquet(P["enriched"], pd.concat([old, pd.DataFrame(rows)], ignore_index=True)
                       if old is not None else pd.DataFrame(rows))
    return _build_enriched_index(P["enriched"]) if P["enriched"].exists() else {}


def screen(new: list[PaperRecord], with_sensitivity: bool) -> tuple[pd.DataFrame, dict]:
    from biofloc_review.run_consolidate import consolidate
    from biofloc_review.run_screening import _row_for as l1_row
    from biofloc_review.run_screening_layer2 import _row_for as l2_row
    from biofloc_review.screening.decision import aggregate

    l1 = run_criteria(new, "L1", "specificity")
    l1_rows = []
    for r in new:
        ans = l1[_record_to_point_id(r)]
        l1_rows.append(l1_row(r, {"answers": ans, "final": aggregate(ans)}))
    l1_df = pd.DataFrame(l1_rows)

    hr = [r for r, row in zip(new, l1_rows) if row["final_decision_primary"] == "HUMAN_REVIEW"]
    l2_df = None
    if hr:
        enriched = enrich(hr)
        l2 = run_criteria(hr, "L2", "specificity", enriched)
        l1_by = {row["record_id"]: row for row in l1_rows}
        l2_df = pd.DataFrame([l2_row(_record_to_point_id(r), r, l1_by[_record_to_point_id(r)],
                                     {"answers": l2[_record_to_point_id(r)],
                                      "final": aggregate(l2[_record_to_point_id(r)])}) for r in hr])
    dec = consolidate(l1_df, l2_df)
    for c in [c for c in dec.columns if c.endswith("_reasoning")]:
        dec[c] = dec[c].map(redact_error)
    ref_cols = list(pd.read_parquet(DATA / "biofloc_final_decisions.parquet").columns)
    dec = dec[[c for c in ref_cols if c in dec.columns] + [c for c in dec.columns if c not in ref_cols]]
    _write_parquet(P["decisions"], dec)

    sens: dict = {}
    if with_sensitivity:
        s = run_criteria(new, "L1", "sensitivity")
        sdec = {_record_to_point_id(r): aggregate(s[_record_to_point_id(r)]).decision for r in new}
        os.environ["FLOCCUS_SCREENING_MODE"] = "specificity"
        spec = dict(zip(dec.record_id, dec.final_decision_primary))
        trans = Counter(f"{spec[k]}->{v}" for k, v in sdec.items())
        sens = {"layer1_decisions_sensitivity": dict(Counter(sdec.values())),
                "layer1_decisions_specificity": dict(Counter(spec.values())),
                "transitions_specificity_to_sensitivity": dict(trans),
                "n_eligible_layer1_sensitivity": sum(v in ELIGIBLE for v in sdec.values()),
                "note": "Layer 1 only; Layer 2 was not re-run in sensitivity mode"}
    return dec, sens


# ---------------------------------------------------------------- step 4

def assign_clusters(elig: pd.DataFrame, vecs: dict[str, np.ndarray]) -> pd.DataFrame:
    from assign_ieee_clusters import align_labels, fit, predict
    from sklearn.metrics import adjusted_rand_score

    V = np.load(DATA / "eligible_vectors.npy")
    stored = np.load(DATA / "eligible_clusters.npy")
    if elig.empty:
        return pd.DataFrame(columns=["record_id", "title", "cluster", "cluster_label", "strength"])
    raw = np.stack([vecs[r] for r in elig.record_id])
    Q = (raw / np.linalg.norm(raw, axis=1)[:, None]).astype(np.float32)
    um, hd = fit(V)
    ari = float(adjusted_rand_score(stored, hd.labels_))
    log.info("refit on archived vectors: ARI vs stored = %.4f", ari)
    if ari != 1.0:
        sys.exit("refit does not reproduce the archived partition (ARI != 1.0); stopping")
    mapping = align_labels(hd.labels_, stored)
    labels, strength = predict(um, hd, Q)
    labels = np.array([mapping.get(int(k), -1) for k in labels])
    out = pd.DataFrame({"record_id": elig.record_id.values, "title": elig.title.values,
                        "final_decision": elig.final_decision.values,
                        "cluster": labels.astype(int), "cluster_label": [cluster_label(int(k)) for k in labels],
                        "strength": strength.astype(float)})
    _write_parquet(P["clusters"], out)
    return out


# ---------------------------------------------------------------- main

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--xls", type=Path, default=DEFAULT_XLS)
    ap.add_argument("--with-sensitivity", action="store_true",
                    help="also screen the new records (Layer 1) in sensitivity mode")
    ap.add_argument("--dedup-only", action="store_true", help="stop after deduplication")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    # library warnings may carry the server URL in exception messages
    for name in ("biofloc_review", "httpx", "urllib3"):
        logging.getLogger(name).setLevel(logging.ERROR)
    t0 = time.time()

    recs = ingest(args.xls)
    new_by_id = {_record_to_point_id(r): r for r in recs}
    corpus_full, corpus_by_id = load_corpus()
    log.info("corpus: %d records (final_full), %d indexed by point id", len(corpus_full), len(corpus_by_id))

    # Layer 1
    l1 = layer1_overlap(cluster_deterministic(corpus_full + recs))
    survivors = [rid for rid, v in l1.items() if v["status"] == "survivor"]
    log.info("layer 1: %s", dict(Counter(v["status"] for v in l1.values())))

    # Layer 2 + 3
    vecs = embed_records([new_by_id[s] for s in survivors])
    pairs = layer2_pairs([new_by_id[s] for s in survivors], vecs)
    judged = layer3_judge(pairs, {k: r.model_dump(mode="json") for k, r in new_by_id.items()}, corpus_by_id)
    dup = [(p["a_id"], p["b_id"]) for p in judged if p["judge_verdict"] == "DUPLICATE"]
    from biofloc_review.dedup.deterministic import _completeness_score
    rank = {s: (_completeness_score(new_by_id[s]),) for s in survivors}
    l3 = layer3_resolve(survivors, dup, rank)
    titles = {**{k: r.title for k, r in new_by_id.items()}, **{k: v.get("title") for k, v in corpus_by_id.items()}}
    status = {}
    for rid in new_by_id:
        s = l3.get(rid) or l1[rid]
        st = "new" if s["status"] == "survivor" else s["status"]
        best = max((p for p in judged if rid in (p["a_id"], p["b_id"])), key=lambda p: p["score"], default=None)
        status[rid] = {"record_id": rid, "source_id": new_by_id[rid].source_id, "title": new_by_id[rid].title,
                       "doi": new_by_id[rid].doi, "year": new_by_id[rid].year, "status": st,
                       "match_record_id": s.get("match"), "match_title": titles.get(s.get("match") or ""),
                       "layer1_via": s.get("via"),
                       "top_pair_score": best["score"] if best else None,
                       "top_pair_verdict": best["judge_verdict"] if best else None}
    dd = pd.DataFrame(status.values())
    _write_parquet(P["dedup"], dd)
    counts = Counter(dd.status)
    log.info("dedup: %s", dict(counts))
    unclear_new = int(((dd.status == "new") & (dd.top_pair_verdict == "UNCLEAR")).sum())

    summary = {
        "run_date": date.today().isoformat(),
        "search": {"database": DATABASE, "search_date": SEARCH_DATE, "filter": "peer reviewed",
                   "export": args.xls.name},
        "models": {"embedding": EMBED_MODEL, "llm": None, "screening_mode": "specificity"},
        "n_retrieved": len(recs),
        "document_types": dict(Counter(r.document_type or "(none)" for r in recs).most_common()),
        "n_with_doi": sum(1 for r in recs if r.doi),
        "n_with_abstract": sum(1 for r in recs if r.abstract),
        "dedup": {
            "corpus_reference": "biofloc_final_full.parquet (22,056 records = 17,075 primaries + merged siblings); "
                                "Qdrant papers_biofloc (read-only)",
            "layer1_deterministic": {"already_in_corpus": counts["L1_corpus"],
                                     "internal_duplicates": counts["L1_internal"],
                                     "already_in_corpus_by": dict(Counter(dd[dd.status == "L1_corpus"].layer1_via))},
            "layer2_embedding": {"threshold": THRESHOLD, "top_k": TOP_K, "n_survivors_embedded": len(survivors),
                                 "candidate_pairs_vs_corpus": sum(p["kind"] == "corpus" for p in pairs),
                                 "candidate_pairs_internal": sum(p["kind"] == "internal" for p in pairs),
                                 "records_with_candidate": len({p["a_id"] for p in pairs} | {p["b_id"] for p in pairs
                                                                if p["kind"] == "internal"})},
            "layer3_llm_judge": {"verdicts": dict(Counter(p["judge_verdict"] for p in judged)),
                                 "already_in_corpus": counts["L3_corpus"],
                                 "internal_duplicates": counts["L3_internal"],
                                 "new_with_unclear_top_pair": unclear_new},
            "already_in_corpus_total": counts["L1_corpus"] + counts["L3_corpus"],
            "internal_duplicates_total": counts["L1_internal"] + counts["L3_internal"],
        },
        "n_new": counts["new"],
    }
    if judged:
        summary["models"]["llm"] = judged[0].get("judge_model")
    _write_json(P["summary"], summary)
    if args.dedup_only:
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        return

    from biofloc_review.dedup.llm_judge import default_llm_model
    summary["models"]["llm"] = default_llm_model()
    new = [new_by_id[r] for r in dd[dd.status == "new"].record_id]
    dec, sens = screen(new, args.with_sensitivity)
    elig = dec[dec.final_decision.isin(ELIGIBLE)].copy()
    summary["screening"] = {
        "layer1_decisions": dict(Counter(dec.final_decision_primary)),
        "layer2_processed": int(dec.l2_processed.sum()),
        "final_decisions": dict(Counter(dec.final_decision)),
        "n_llm_errors_unresolved": int(sum((dec[c] == "(error LLM)").sum()
                                           for c in dec.columns if c.endswith("_evidence"))),
    }
    if sens:
        summary["screening"]["sensitivity_mode"] = sens
    summary["n_new_eligible"] = int(len(elig))
    summary["n_new_eligible_by_decision"] = dict(Counter(elig.final_decision))
    _write_json(P["summary"], summary)

    ca = assign_clusters(elig, vecs)
    cc = Counter(ca.cluster_label)
    summary["clusters"] = {
        "method": "assign_ieee_clusters.fit/predict: UMAP-10 + HDBSCAN refit on archived 4,682 vectors "
                  "(ARI 1.0 vs stored labels), umap.transform + hdbscan.approximate_predict",
        "counts": {"C03": cc.get("C03", 0), "C15": cc.get("C15", 0), "noise": cc.get("noise", 0),
                   "other": int(len(ca) - cc.get("C03", 0) - cc.get("C15", 0) - cc.get("noise", 0))},
        "by_cluster": dict(sorted(cc.items())),
        "titles": {lab: ca[ca.cluster_label == lab].sort_values("strength", ascending=False)
                   [["title", "final_decision", "strength"]].to_dict(orient="records")
                   for lab in ("C03", "C15")},
    }
    summary["elapsed_seconds"] = round(time.time() - t0, 1)
    _write_json(P["summary"], summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
