"""AI deduplication and thematic clusters with an embedded Qdrant (no Docker).

Same method and parameters as the published pipeline:
- embeddings of title + abstract with Ollama (``qwen3-embedding:8b``), cosine;
- near-duplicate candidates: top-5 neighbours with similarity >= 0.93, outside
  the record's deterministic cluster;
- each candidate pair judged by the LLM; DUPLICATE verdicts merged transitively;
- clusters: UMAP (10-D, 30 neighbours, cosine, seed 42) + HDBSCAN (min cluster
  size 40, min samples 10) on the eligible records, plus a 2-D UMAP for plots.
Projects with fewer than 400 eligible records use smaller cluster sizes
(recorded in ``results/latest/clusters.json``).

Vectors live in ``data/qdrant/`` inside the project (Qdrant local mode).
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from biofloc_review.dedup import embeddings as emb
from biofloc_review.dedup.merge_confirmed import merge_confirmed_duplicates
from biofloc_review.dedup.ollama_embed import default_embed_model

from floccus_app.textsafe import redact
from floccus_app.pipeline import (
    PipelineError,
    atomic_write,
    _read_records,
    _record_id,
    _write_records,
    latest_dir,
    processed_dir,
    read_counts,
    write_counts,
    write_progress,
)

COLLECTION = "papers"
THRESHOLD = 0.93
TOP_K = 5
PUBLISHED_MIN_RECORDS = 400


def _client(project: Path):
    from qdrant_client import QdrantClient

    path = project / "data" / "qdrant"
    path.mkdir(parents=True, exist_ok=True)
    try:
        return QdrantClient(path=str(path))
    except Exception as exc:
        raise PipelineError(
            "the vector store in data/qdrant could not be opened (another program may be using it, or it "
            "is damaged): close other runs, or delete the data/qdrant folder and run AI deduplication "
            f"again ({type(exc).__name__})") from None


def _current_collection(project: Path) -> str:
    marker = processed_dir(project) / "qdrant_collection.txt"
    return marker.read_text(encoding="utf-8").strip() if marker.is_file() else COLLECTION


def _default_judge():
    from biofloc_review.dedup.llm_judge import judge_pair

    return judge_pair


def dedup_similar(project: Path, judge=None, threshold: float = THRESHOLD, top_k: int = TOP_K) -> dict:
    full_path = processed_dir(project) / "dedup_full.parquet"
    if not full_path.is_file():
        raise PipelineError("deduplicate first (exact DOI/title step)")
    judge = judge or _default_judge()
    full = _read_records(full_path)
    primaries = [r for r in full if r.is_primary]
    write_progress(project, step="dedup-ai", status="starting", done=0, total=len(primaries), errors=0)
    import time

    # Index into a fresh collection; the previous one is dropped only after success.
    new = f"{COLLECTION}_{int(time.time() * 1000)}"
    old = _current_collection(project)
    client = _client(project)
    try:
        emb.ensure_collection(client, new, recreate=True)
        indexed = emb.index_records(client, new, primaries, model=default_embed_model())
        if indexed < len(primaries):
            client.delete_collection(new)
            raise PipelineError(
                f"embeddings failed for {len(primaries) - indexed} of {len(primaries)} records; check that "
                "Ollama is running with the embedding model (OLLAMA_EMBED_MODEL) and run the step again")
        pairs = emb.find_near_duplicate_pairs(client, new, primaries, threshold=threshold, top_k=top_k)
        (processed_dir(project) / "qdrant_collection.txt").write_text(new, encoding="utf-8")
        if old != new and client.collection_exists(old):
            client.delete_collection(old)
    finally:
        client.close()
    by_id = {emb._record_to_point_id(r): r for r in primaries}
    errors = 0
    write_progress(project, status="running", done=0, total=len(pairs))
    for i, pair in enumerate(pairs, start=1):
        a, b = by_id[pair["a_id"]], by_id[pair["b_id"]]
        try:
            v = judge(a.model_dump(mode="json"), b.model_dump(mode="json"), pair["score"])
            pair.update(judge_verdict=v.verdict, judge_confidence=v.confidence, judge_reasoning=v.reasoning)
        except Exception as exc:  # counted, the pair is kept apart
            errors += 1
            pair.update(judge_verdict="UNCLEAR", judge_confidence=0,
                        judge_reasoning=f"LLM error: {type(exc).__name__}: {redact(str(exc))}")
        write_progress(project, done=i, errors=errors)
    new_primaries, new_full, _ = merge_confirmed_duplicates(full, pairs)
    _write_records(new_full, processed_dir(project) / "dedup_full.parquet")
    _write_records(new_primaries, processed_dir(project) / "unique.parquet")
    atomic_write(latest_dir(project) / "near_duplicate_pairs.json",
                 lambda tmp: tmp.write_text(json.dumps(pairs, indent=1, default=str), encoding="utf-8"))

    counts = read_counts(project)
    exact = counts.get("deduplication", {}).get("after_layer1_deterministic", len(primaries))
    counts["deduplication"] = {
        "method": "exact DOI/title + embeddings (Qdrant local) + LLM judge",
        "after_layer1_deterministic": exact,
        "after_layers2_3_and_merge": len(new_primaries),
        "after_round2_ieee_merges": len(new_primaries),
        "removed": len(full) - len(new_primaries),
        "near_duplicate_pairs": len(pairs),
    }
    counts["deduplicated_by_source"] = dict(Counter(str(r.source) for r in new_primaries))
    for key in ("screening_layer1", "screening_layer2", "final_specificity_mode"):
        counts.pop(key, None)
    write_counts(project, counts)
    write_progress(project, status="done")
    return {"input": len(primaries), "pairs": len(pairs),
            "duplicates": len(primaries) - len(new_primaries), "unique": len(new_primaries)}


def cluster_params(n: int) -> dict:
    if n >= PUBLISHED_MIN_RECORDS:
        mcs, ms = 40, 10
    else:
        mcs = max(5, n // 10)
        ms = min(10, mcs)
    return {
        "umap": {"n_components": min(10, max(2, n - 2)), "n_neighbors": min(30, max(2, n - 1)), "min_dist": 0.0,
                 "metric": "cosine", "random_state": 42, "compatibility_layout": True},
        "umap_2d": {"n_components": 2, "n_neighbors": min(15, max(2, n - 1)), "min_dist": 0.1,
                    "metric": "cosine", "random_state": 42, "compatibility_layout": True},
        "hdbscan": {"min_cluster_size": mcs, "min_samples": ms},
        "published": n >= PUBLISHED_MIN_RECORDS,
    }


def thematic_clusters(project: Path) -> dict:
    import hdbscan
    import umap

    dec_path = latest_dir(project) / "screening_decisions.csv"
    if not dec_path.is_file():
        raise PipelineError("run the screening first")
    dec = pd.read_csv(dec_path, low_memory=False)
    eligible = dec[dec["final_decision"].isin(["INCLUDE", "HUMAN_REVIEW"])]
    recs = {_record_id(r): r for r in _read_records(processed_dir(project) / "unique.parquet")}
    eligible = eligible[eligible["record_id"].isin(recs)].reset_index(drop=True)
    if len(eligible) < 10:
        raise PipelineError("clusters need at least 10 eligible records")
    write_progress(project, step="clusters", status="starting", done=0, total=len(eligible), errors=0)
    chosen = [recs[i] for i in eligible["record_id"]]
    point_ids = [emb._record_to_point_id(r) for r in chosen]
    collection = _current_collection(project)
    client = _client(project)
    try:
        if not client.collection_exists(collection):
            emb.ensure_collection(client, collection)
        have = {str(p.id) for p in client.retrieve(collection, point_ids, with_vectors=False)}
        missing = [r for r, pid in zip(chosen, point_ids) if pid not in have]
        if missing:
            emb.index_records(client, collection, missing, model=default_embed_model())
        points = {}
        for start in range(0, len(point_ids), 500):  # bounded batches instead of one huge call
            for pt in client.retrieve(collection, point_ids[start:start + 500], with_vectors=True):
                points[str(pt.id)] = pt.vector
    finally:
        client.close()
    keep = [i for i, pid in enumerate(point_ids) if pid in points]
    if len(keep) < len(point_ids):
        raise PipelineError(
            f"embeddings failed for {len(point_ids) - len(keep)} of {len(point_ids)} eligible records; check "
            "that Ollama is running with the embedding model (OLLAMA_EMBED_MODEL) and run the step again")
    eligible = eligible.iloc[keep].reset_index(drop=True)
    vectors = np.empty((len(keep), len(points[point_ids[keep[0]]])), dtype=np.float32)
    for row, i in enumerate(keep):
        vectors[row] = points.pop(point_ids[i])

    params = cluster_params(len(vectors))
    reduced = umap.UMAP(**params["umap"]).fit_transform(vectors)
    labels = hdbscan.HDBSCAN(metric="euclidean", **params["hdbscan"]).fit_predict(reduced)
    xy = umap.UMAP(**params["umap_2d"]).fit_transform(vectors)

    out = eligible[["record_id", "final_decision", "title", "year"]].copy()
    out["cluster"] = labels
    out["cluster_label"] = [f"C{k:02d}" if k >= 0 else "noise" for k in labels]
    out["umap_x"], out["umap_y"] = xy[:, 0], xy[:, 1]
    info = {"records": len(out), "clusters": int(len(set(labels) - {-1})),
            "noise": int((labels == -1).sum()), "params": params}
    atomic_write(latest_dir(project) / "clusters.json",
                 lambda tmp: tmp.write_text(json.dumps(info, indent=2), encoding="utf-8"))
    atomic_write(latest_dir(project) / "eligible_clusters.csv", lambda tmp: out.to_csv(tmp, index=False))
    write_progress(project, status="done", done=len(out))
    return info
