"""Assign the 201 eligible IEEE records without embeddings to the thematic clusters.

The 25 thematic clusters were fitted on the 4,682 eligible records that had a
vector in Qdrant (data/processed/eligible_vectors.npy). The other 201 eligible
records, all from IEEE Xplore, have no usable vector: in the June export the
IEEE "Document Identifier" was not unique per paper, so their Qdrant points
overwrote each other. This script embeds them again with the same text
construction and model as the dedup step (title + abstract, qwen3-embedding:8b),
refits UMAP-10 + HDBSCAN on the 4,682 archived vectors with the archived
parameters (checking ARI = 1.0 against the stored labels) and maps the IEEE
records into the fitted partition with umap.transform + hdbscan.approximate_predict.
The archived partition is not modified.

Before using the mapping, it is validated with a hold-out test: in 5 runs,
200 random records are removed, UMAP + HDBSCAN are refitted on the rest, the
held-out records are mapped back, and their predicted labels (aligned to the
archived numbering by majority vote on the training records) are compared with
their archived labels.

IEEE vectors are cached (as returned by the server) in
data/processed/ieee_eligible_vectors.npy + ieee_eligible_record_ids.txt; a
rerun does not call the embedding server. Load the env first (Ollama URL/token).

Run:  set -a && . ../.env && set +a && .venv/bin/python scripts/assign_ieee_clusters.py
Out:  data/processed/ieee_cluster_assignment.parquet
      data/processed/ieee_cluster_assignment.json
      data/processed/ieee_eligible_vectors.npy, ieee_eligible_record_ids.txt (cache)
"""
from __future__ import annotations

import json
import sys
import uuid
import warnings
from collections import Counter
from datetime import date
from pathlib import Path
from types import SimpleNamespace

import hdbscan
import numpy as np
import pandas as pd
import umap
from sklearn.metrics import adjusted_rand_score

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "processed"
SEED = 42
# a fixed random_state forces single-threaded UMAP; the notice is expected
warnings.filterwarnings("ignore", message="n_jobs value", category=UserWarning)
UMAP_PARAMS = {"n_components": 10, "n_neighbors": 30, "min_dist": 0.0, "metric": "cosine"}
HDBSCAN_PARAMS = {"min_cluster_size": 40, "min_samples": 10, "metric": "euclidean"}
ELIGIBLE = ("INCLUDE", "HUMAN_REVIEW")
N_HOLDOUT, HOLDOUT_SEEDS = 200, (1, 2, 3, 4, 5)
EMBED_BATCH = 32
FOCUS = (3, 15)
CACHE_VEC = DATA / "ieee_eligible_vectors.npy"
CACHE_IDS = DATA / "ieee_eligible_record_ids.txt"


# ---------------------------------------------------------------- pure helpers

def record_id(source: str, source_id: str | None, title_norm: str) -> str:
    """Point id used by the dedup embedding step (embeddings._record_to_point_id)."""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"{source}:{source_id or title_norm[:80]}"))


def align_labels(refit: np.ndarray, reference: np.ndarray) -> dict[int, int]:
    """Map each refit label to the reference label most frequent among its records.

    Noise (-1) always maps to noise. A refit cluster whose records are mostly
    reference noise maps to -1. Ties go to the smallest reference label.
    """
    mapping = {-1: -1}
    for k in np.unique(refit[refit >= 0]):
        counts = Counter(reference[refit == k].tolist())
        best = max(counts.values())
        mapping[int(k)] = min(lab for lab, c in counts.items() if c == best)
    return mapping


def agreement(pred: np.ndarray, orig: np.ndarray) -> dict:
    """Exact agreement (noise included), agreement on non-noise originals, noise rate."""
    clustered = orig >= 0
    return {
        "exact": float((pred == orig).mean()),
        "non_noise": float((pred[clustered] == orig[clustered]).mean()) if clustered.any() else float("nan"),
        "predicted_noise": float((pred == -1).mean()),
        "n": int(len(pred)), "n_non_noise": int(clustered.sum()),
    }


# ---------------------------------------------------------------- pipeline

def fit(X: np.ndarray) -> tuple[umap.UMAP, hdbscan.HDBSCAN]:
    um = umap.UMAP(**UMAP_PARAMS, random_state=SEED)
    emb = um.fit_transform(X)
    hd = hdbscan.HDBSCAN(**HDBSCAN_PARAMS, prediction_data=True).fit(emb)
    return um, hd


def predict(um: umap.UMAP, hd: hdbscan.HDBSCAN, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    labels, strengths = hdbscan.approximate_predict(hd, um.transform(X))
    return np.asarray(labels), np.asarray(strengths)


def missing_ieee(ids: list[str]) -> pd.DataFrame:
    dec = pd.read_parquet(DATA / "biofloc_final_decisions.parquet",
                          columns=["record_id", "source", "final_decision"])
    dec["record_id"] = dec.record_id.astype(str)
    el = dec[dec.final_decision.isin(ELIGIBLE)]
    miss = el[~el.record_id.isin(set(ids))]
    print(f"eligible={len(el)}  with vector={len(el) - len(miss)}  missing={len(miss)}  "
          f"sources={miss.source.value_counts().to_dict()}")
    if len(miss) != 201 or set(miss.source) != {"ieee"}:
        sys.exit("unexpected missing set (expected 201 IEEE records); stopping")
    corpus = pd.read_parquet(DATA / "biofloc_final_primary_v3.parquet",
                             columns=["source", "source_id", "title", "title_norm", "abstract"])
    corpus["record_id"] = [record_id(s, sid, tn) for s, sid, tn in
                           zip(corpus.source, corpus.source_id, corpus.title_norm)]
    corpus = corpus.drop_duplicates("record_id")
    out = miss[["record_id"]].merge(corpus, on="record_id", how="left")
    if out.title.isna().any():
        sys.exit(f"{int(out.title.isna().sum())} missing records not found in the corpus; stopping")
    return out.sort_values("record_id").reset_index(drop=True)


def embed(recs: pd.DataFrame) -> np.ndarray:
    if CACHE_VEC.exists() and CACHE_IDS.exists():
        cached = [r for r in CACHE_IDS.read_text().split("\n") if r]
        if cached == recs.record_id.tolist():
            print(f"using cached IEEE vectors ({CACHE_VEC.name})")
            return np.load(CACHE_VEC)
        print("cache ids differ from the missing set; re-embedding")
    from biofloc_review.dedup.embeddings import _text_for
    from biofloc_review.dedup.ollama_embed import DEFAULT_MODEL, chunked, embed_batch

    texts = [_text_for(SimpleNamespace(title=t, abstract=a)) for t, a in zip(recs.title, recs.abstract)]
    vecs: list[list[float]] = []
    try:
        for i, batch in enumerate(chunked(texts, EMBED_BATCH)):
            vecs.extend(embed_batch(batch, model=DEFAULT_MODEL))
            print(f"  embedded {len(vecs)}/{len(texts)}")
    except Exception as e:  # report the error class/status only, never the URL or token
        status = getattr(getattr(e, "response", None), "status_code", None)
        sys.exit(f"embedding failed: {type(e).__name__} (HTTP status {status}); stopping")
    V = np.asarray(vecs, dtype=np.float32)
    np.save(CACHE_VEC, V)
    CACHE_IDS.write_text("\n".join(recs.record_id))
    return V


def main() -> None:
    V = np.load(DATA / "eligible_vectors.npy")
    ids = [r for r in (DATA / "eligible_record_ids.txt").read_text().split("\n") if r]
    stored = np.load(DATA / "eligible_clusters.npy")

    recs = missing_ieee(ids)
    raw = embed(recs)
    stored_norm = np.linalg.norm(V, axis=1)
    raw_norm = np.linalg.norm(raw, axis=1)
    if raw.shape[1] != V.shape[1]:
        sys.exit(f"dimension mismatch: IEEE {raw.shape[1]} vs stored {V.shape[1]}; stopping")
    # stored vectors are unit-norm (Qdrant cosine collection); normalise the same way
    Q = (raw / raw_norm[:, None]).astype(np.float32)
    vec_check = {"dim": int(raw.shape[1]), "stored_dim": int(V.shape[1]),
                 "stored_norm_range": [float(stored_norm.min()), float(stored_norm.max())],
                 "ieee_raw_norm_range": [float(raw_norm.min()), float(raw_norm.max())],
                 "ieee_normalised": True}
    print("vector check:", vec_check)

    um, hd = fit(V)
    ari = float(adjusted_rand_score(stored, hd.labels_))
    print(f"full refit (prediction_data=True): ARI vs stored = {ari:.4f}")
    if ari != 1.0:
        sys.exit("prediction_data=True changed the partition (ARI != 1.0); stopping")
    full_map = align_labels(hd.labels_, stored)  # identity up to renumbering

    holdout = []
    for s in HOLDOUT_SEEDS:
        rng = np.random.default_rng(s)
        test = rng.choice(len(V), N_HOLDOUT, replace=False)
        train = np.setdiff1d(np.arange(len(V)), test)
        um_h, hd_h = fit(V[train])
        mapping = align_labels(hd_h.labels_, stored[train])
        pred, _ = predict(um_h, hd_h, V[test])
        pred = np.array([mapping.get(int(p), -1) for p in pred])
        res = {"seed": s, **agreement(pred, stored[test]),
               "train_ari_vs_stored": float(adjusted_rand_score(stored[train], hd_h.labels_)),
               "k_refit": int(hd_h.labels_.max() + 1)}
        holdout.append(res)
        print("hold-out:", {k: round(v, 3) if isinstance(v, float) else v for k, v in res.items()})
    summary = {m: {"mean": float(np.mean([h[m] for h in holdout])),
                   "min": float(np.min([h[m] for h in holdout])),
                   "max": float(np.max([h[m] for h in holdout]))}
               for m in ("exact", "non_noise", "predicted_noise", "train_ari_vs_stored")}

    labels, strength = predict(um, hd, Q)
    labels = np.array([full_map.get(int(p), -1) for p in labels])
    out = pd.DataFrame({"record_id": recs.record_id, "title": recs.title,
                        "cluster": labels.astype(int), "strength": strength.astype(float)})
    out.to_parquet(DATA / "ieee_cluster_assignment.parquet", index=False)

    counts = {f"C{k:02d}" if k >= 0 else "noise": int(v)
              for k, v in sorted(Counter(labels.tolist()).items())}
    sanity = {f"C{c:02d}": out[out.cluster == c].nlargest(5, "strength").title.tolist() for c in FOCUS}
    res = {
        "date": date.today().isoformat(),
        "model": "qwen3-embedding:8b",
        "n_eligible": int(len(ids) + len(recs)), "n_stored": int(len(ids)), "n_ieee": int(len(recs)),
        "vector_check": vec_check,
        "params": {"umap": {**UMAP_PARAMS, "random_state": SEED},
                   "hdbscan": {**HDBSCAN_PARAMS, "prediction_data": True},
                   "assignment": "umap.transform + hdbscan.approximate_predict"},
        "full_refit_ari_vs_stored": ari,
        "holdout": {"n_holdout": N_HOLDOUT, "seeds": list(HOLDOUT_SEEDS),
                    "alignment": "majority vote on training records", "runs": holdout, "summary": summary},
        "counts": counts,
        "focus": {f"C{c:02d}": int((labels == c).sum()) for c in FOCUS} | {"noise": int((labels == -1).sum())},
        "strength_median": float(np.median(strength)),
        "sanity_titles_top5_by_strength": sanity,
    }
    (DATA / "ieee_cluster_assignment.json").write_text(json.dumps(res, indent=2, ensure_ascii=False))
    print(json.dumps({k: res[k] for k in ("counts", "focus", "sanity_titles_top5_by_strength")},
                     indent=2, ensure_ascii=False))
    print("hold-out summary:", json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
