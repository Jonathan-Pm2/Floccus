"""Stability of the 25-cluster thematic partition across seeds and parameters.

cluster_reproducibility.py shows that the archived seed reproduces the
partition exactly (ARI = 1.0). That is determinism, not stability. This script
measures stability on the same 4,682 archived vectors with the published
parameters (UMAP: 10 components, n_neighbors=30, min_dist=0, cosine;
HDBSCAN: min_cluster_size=40, min_samples=10, Euclidean):

1. seed stability: UMAP random_state in SEEDS (published parameters);
2. min_cluster_size grid MCS_GRID (-40% ... +40% around 40) with
   min_samples=10, on the published-seed embedding and on every seed embedding;
3. n_neighbors in NEIGHBORS_EXTRA (seed 42, published HDBSCAN parameters).

Per run: number of clusters, noise fraction, ARI and AMI against the published
partition on all records (noise is one extra label) and on the records
clustered in both partitions, and, for each published cluster, the best
Jaccard index with any cluster of the run (best-match recovery; >= 0.5 is the
usual "recovered" threshold, Hennig 2007). No subsampling: every run uses all
4,682 records. UMAP embeddings are cached in data/processed/cluster_stability_umap/
so a rerun only repeats HDBSCAN.

Run:  .venv/bin/python scripts/cluster_stability.py
Out:  data/processed/cluster_stability.json
"""
from __future__ import annotations

import json
import time
import warnings
from itertools import combinations
from pathlib import Path

import hdbscan
import numpy as np
import umap
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "processed"
CACHE = DATA / "cluster_stability_umap"
warnings.filterwarnings("ignore", message="n_jobs value", category=UserWarning)

PUBLISHED_SEED = 42
UMAP_BASE = {"n_components": 10, "n_neighbors": 30, "min_dist": 0.0, "metric": "cosine"}
MCS, MIN_SAMPLES = 40, 10
SEEDS = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12)
MCS_GRID = (24, 28, 32, 36, 40, 44, 48, 52, 56)
NEIGHBORS_EXTRA = (15, 50)
FOCUS = (3, 15)
RECOVERED = 0.5


# ---------------------------------------------------------------- pure helpers

def best_jaccard(reference: np.ndarray, labels: np.ndarray) -> dict[int, float]:
    """For each non-noise reference cluster, the best Jaccard index with any non-noise run cluster."""
    out: dict[int, float] = {}
    run_sets = {int(k): set(np.flatnonzero(labels == k)) for k in np.unique(labels[labels >= 0])}
    for k in np.unique(reference[reference >= 0]):
        ref = set(np.flatnonzero(reference == k))
        best = 0.0
        for s in run_sets.values():
            inter = len(ref & s)
            if inter:
                best = max(best, inter / len(ref | s))
        out[int(k)] = best
    return out


def compare(reference: np.ndarray, labels: np.ndarray) -> dict:
    """ARI/AMI on all records (noise as one label) and on records clustered in both."""
    both = (reference >= 0) & (labels >= 0)
    return {
        "k": len(np.unique(labels[labels >= 0])),
        "noise_fraction": float((labels == -1).mean()),
        "ari_all": float(adjusted_rand_score(reference, labels)),
        "ami_all": float(adjusted_mutual_info_score(reference, labels)),
        "n_clustered_both": int(both.sum()),
        "ari_clustered_both": float(adjusted_rand_score(reference[both], labels[both])) if both.any() else float("nan"),
        "ami_clustered_both": float(adjusted_mutual_info_score(reference[both], labels[both])) if both.any() else float("nan"),
    }


def summarise(values: list[float]) -> dict:
    a = np.asarray(values, dtype=float)
    return {"median": float(np.median(a)), "min": float(a.min()), "max": float(a.max()),
            "mean": float(a.mean())}


# ---------------------------------------------------------------- pipeline

def embedding(V: np.ndarray, seed: int, n_neighbors: int) -> np.ndarray:
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / f"umap_nn{n_neighbors}_seed{seed}.npy"
    if path.exists():
        return np.load(path)
    t0 = time.time()
    emb = umap.UMAP(**{**UMAP_BASE, "n_neighbors": n_neighbors}, random_state=seed).fit_transform(V)
    np.save(path, emb)
    print(f"  UMAP nn={n_neighbors} seed={seed}: {time.time() - t0:.0f}s", flush=True)
    return emb


def cluster(emb: np.ndarray, mcs: int = MCS, ms: int = MIN_SAMPLES) -> np.ndarray:
    return hdbscan.HDBSCAN(min_cluster_size=mcs, min_samples=ms).fit_predict(emb)


def run_block(reference: np.ndarray, labels: np.ndarray, **meta) -> dict:
    return {**meta, **compare(reference, labels), "best_jaccard": best_jaccard(reference, labels)}


def per_cluster(runs: list[dict], reference: np.ndarray) -> dict:
    out = {}
    for k in np.unique(reference[reference >= 0]):
        vals = [r["best_jaccard"][int(k)] for r in runs]
        out[f"C{int(k):02d}"] = {"size": int((reference == k).sum()), **summarise(vals),
                                 "share_runs_recovered": float(np.mean(np.asarray(vals) >= RECOVERED))}
    return out


def overview(runs: list[dict]) -> dict:
    keys = ("k", "noise_fraction", "ari_all", "ami_all", "ari_clustered_both", "ami_clustered_both")
    return {k: summarise([r[k] for r in runs]) for k in keys}


def main() -> None:
    V = np.load(DATA / "eligible_vectors.npy")
    ref = np.load(DATA / "eligible_clusters.npy")
    print(f"{len(V)} records, {len(np.unique(ref[ref >= 0]))} published clusters", flush=True)

    base = cluster(embedding(V, PUBLISHED_SEED, UMAP_BASE["n_neighbors"]))
    check = float(adjusted_rand_score(ref, base))
    print(f"published seed reproduces: ARI = {check:.4f}", flush=True)

    seed_labels = {s: cluster(embedding(V, s, UMAP_BASE["n_neighbors"])) for s in SEEDS}
    seed_runs = [run_block(ref, lab, seed=s) for s, lab in seed_labels.items()]
    for r in seed_runs:
        print(f"seed {r['seed']:>2}: k={r['k']} noise={r['noise_fraction']:.3f} "
              f"ARI={r['ari_all']:.3f} AMI={r['ami_all']:.3f} ARI(both)={r['ari_clustered_both']:.3f}", flush=True)
    pairwise = [float(adjusted_rand_score(seed_labels[a], seed_labels[b])) for a, b in combinations(SEEDS, 2)]

    mcs_runs_published_seed = []
    mcs_runs_all_seeds = []
    for m in MCS_GRID:
        lab = base if m == MCS else cluster(embedding(V, PUBLISHED_SEED, UMAP_BASE["n_neighbors"]), m)
        mcs_runs_published_seed.append(run_block(ref, lab, min_cluster_size=m, seed=PUBLISHED_SEED))
        for s in SEEDS:
            lab_s = seed_labels[s] if m == MCS else cluster(embedding(V, s, UMAP_BASE["n_neighbors"]), m)
            mcs_runs_all_seeds.append(run_block(ref, lab_s, min_cluster_size=m, seed=s))
        r = mcs_runs_published_seed[-1]
        print(f"mcs {m}: k={r['k']} noise={r['noise_fraction']:.3f} ARI={r['ari_all']:.3f} "
              f"ARI(both)={r['ari_clustered_both']:.3f}", flush=True)

    nn_runs = []
    for nn in NEIGHBORS_EXTRA:
        lab = cluster(embedding(V, PUBLISHED_SEED, nn))
        nn_runs.append(run_block(ref, lab, n_neighbors=nn, seed=PUBLISHED_SEED))
        r = nn_runs[-1]
        print(f"nn {nn}: k={r['k']} noise={r['noise_fraction']:.3f} ARI={r['ari_all']:.3f}", flush=True)

    mcs_by_value = {
        str(m): {"overview_over_seeds": overview([r for r in mcs_runs_all_seeds if r["min_cluster_size"] == m])}
        for m in MCS_GRID
    }
    seed_pc = per_cluster(seed_runs, ref)
    mcs_pc = per_cluster(mcs_runs_published_seed, ref)
    res = {
        "n_records": len(V),
        "subsampled": False,
        "published": {"umap": {**UMAP_BASE, "random_state": PUBLISHED_SEED},
                      "hdbscan": {"min_cluster_size": MCS, "min_samples": MIN_SAMPLES, "metric": "euclidean"},
                      "k": len(np.unique(ref[ref >= 0])), "noise_fraction": float((ref == -1).mean())},
        "published_seed_ari": check,
        "recovered_threshold_jaccard": RECOVERED,
        "seeds": {"values": list(SEEDS), "runs": seed_runs, "overview": overview(seed_runs),
                  "pairwise_ari_between_seeds": summarise(pairwise),
                  "per_cluster_best_jaccard": seed_pc,
                  "n_clusters_recovered_in_all_seeds": int(sum(v["min"] >= RECOVERED for v in seed_pc.values())),
                  "n_clusters_recovered_median": int(sum(v["median"] >= RECOVERED for v in seed_pc.values()))},
        "min_cluster_size": {"values": list(MCS_GRID), "min_samples": MIN_SAMPLES,
                             "published_seed_runs": mcs_runs_published_seed,
                             "per_cluster_best_jaccard_published_seed": mcs_pc,
                             "by_value_over_seeds": mcs_by_value,
                             "all_seed_runs": mcs_runs_all_seeds},
        "n_neighbors": {"values": list(NEIGHBORS_EXTRA), "runs": nn_runs},
        "focus": {f"C{c:02d}": {"seeds": seed_pc[f"C{c:02d}"], "min_cluster_size_published_seed": mcs_pc[f"C{c:02d}"],
                                "min_cluster_size_all_seeds": summarise([r["best_jaccard"][c] for r in mcs_runs_all_seeds]),
                                "n_neighbors": {str(r["n_neighbors"]): r["best_jaccard"][c] for r in nn_runs}}
                  for c in FOCUS},
    }
    (DATA / "cluster_stability.json").write_text(json.dumps(res, indent=2))
    print(json.dumps({"seeds": res["seeds"]["overview"], "pairwise": res["seeds"]["pairwise_ari_between_seeds"],
                      "focus": res["focus"]}, indent=2))


if __name__ == "__main__":
    main()
