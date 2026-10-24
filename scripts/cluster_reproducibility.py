"""Reproducibility of the thematic clustering and of the C03/C15 sub-clustering.

The manuscript's 25 thematic clusters come from UMAP (10 components,
n_neighbors=30, min_dist=0, cosine, seed 42) followed by HDBSCAN
(min_cluster_size=40, min_samples=10) on the embeddings of the eligible
records (data/processed/eligible_vectors.npy, rows aligned with
eligible_record_ids.txt). The two core clusters were then sub-clustered with
UMAP (5 components, n_neighbors=15, min_dist=0, cosine, seed 42) + HDBSCAN
(8/3 for C03, 6/2 for C15).

This script re-runs both steps from the archived vectors, compares the labels
with the archived ones (adjusted Rand index; 1.0 means identical partitions),
runs a sensitivity grid over n_neighbors x n_components for the
sub-clustering, and profiles the archived sub-clusters (size, source shares,
median year and INCLUDE share).

ARI values depend on the UMAP/HDBSCAN/numba versions pinned in uv.lock.

Run:  .venv/bin/python scripts/cluster_reproducibility.py
Out:  data/processed/cluster_reproducibility.json
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

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
MAIN = {"n_components": 10, "n_neighbors": 30, "min_cluster_size": 40, "min_samples": 10}
SUB = {  # archived file, HDBSCAN (min_cluster_size, min_samples)
    3: ("c03_subclusters", 8, 3),
    15: ("c15_subclusters", 6, 2),
}
SUB_BASE = {"n_components": 5, "n_neighbors": 15}
GRID_NEIGHBORS = (10, 15, 30)
GRID_COMPONENTS = (5, 10)


def embed_and_cluster(X: np.ndarray, n_components: int, n_neighbors: int,
                      min_cluster_size: int, min_samples: int) -> np.ndarray:
    emb = umap.UMAP(n_components=n_components, n_neighbors=n_neighbors, min_dist=0.0,
                    metric="cosine", random_state=SEED).fit_transform(X)
    return hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples).fit_predict(emb)


def summarize(labels: np.ndarray) -> dict:
    sizes = np.bincount(labels[labels >= 0]) if (labels >= 0).any() else np.array([], int)
    return {"k": int(len(sizes)), "noise": int((labels == -1).sum()), "sizes": sorted(sizes.tolist())}


def profile(sub: pd.DataFrame) -> list[dict]:
    """Size, source shares, median year and INCLUDE share per archived sub-cluster."""
    out = []
    for k, g in sub.groupby("sub_cluster"):
        out.append({
            "sub_cluster": int(k), "n": int(len(g)),
            "source_share": {s: round(float(v), 3) for s, v in g.source.value_counts(normalize=True).items()},
            "median_year": float(g.year.median()),
            "include_share": round(float((g.final_decision == "INCLUDE").mean()), 3),
        })
    return out


def main() -> None:
    V = np.load(DATA / "eligible_vectors.npy")
    ids = [r for r in (DATA / "eligible_record_ids.txt").read_text().split("\n") if r]
    stored = np.load(DATA / "eligible_clusters.npy")
    row = {r: i for i, r in enumerate(ids)}
    decisions = pd.read_parquet(DATA / "eligible_with_citations.parquet", columns=["record_id", "final_decision"])

    labels = embed_and_cluster(V, **MAIN)
    main_res = {"params": {**MAIN, "min_dist": 0.0, "metric": "cosine", "seed": SEED},
                "n_records": int(len(V)), "archived": summarize(stored), "rerun": summarize(labels),
                "ari_vs_archived": round(float(adjusted_rand_score(stored, labels)), 3)}
    print("main clustering:", main_res)

    subs = {}
    for cluster, (fname, mcs, ms) in SUB.items():
        sub = pd.read_parquet(DATA / f"{fname}.parquet")
        if "final_decision" not in sub:
            sub = sub.merge(decisions, on="record_id", how="left")
        X = V[[row[r] for r in sub.record_id]]
        archived = sub.sub_cluster.to_numpy()
        base = embed_and_cluster(X, **SUB_BASE, min_cluster_size=mcs, min_samples=ms)
        grid = []
        for nn in GRID_NEIGHBORS:
            for nc in GRID_COMPONENTS:
                lab = base if (nn, nc) == (SUB_BASE["n_neighbors"], SUB_BASE["n_components"]) else \
                    embed_and_cluster(X, n_components=nc, n_neighbors=nn, min_cluster_size=mcs, min_samples=ms)
                grid.append({"n_neighbors": nn, "n_components": nc, **summarize(lab),
                             "ari_vs_archived": round(float(adjusted_rand_score(archived, lab)), 3)})
        off_base = [g["ari_vs_archived"] for g in grid
                    if (g["n_neighbors"], g["n_components"]) != (SUB_BASE["n_neighbors"], SUB_BASE["n_components"])]
        subs[f"C{cluster:02d}"] = {
            "params": {**SUB_BASE, "min_dist": 0.0, "metric": "cosine", "seed": SEED,
                       "min_cluster_size": mcs, "min_samples": ms},
            "n": int(len(sub)),
            "archived": summarize(archived),
            "covered": int((archived >= 0).sum()),
            "ari_base_vs_archived": round(float(adjusted_rand_score(archived, base)), 3),
            "sensitivity_grid": grid,
            "sensitivity_ari_range_off_base": [min(off_base), max(off_base)],
            "profile": profile(sub),
        }
        print(f"C{cluster:02d}:", subs[f"C{cluster:02d}"]["ari_base_vs_archived"],
              subs[f"C{cluster:02d}"]["sensitivity_ari_range_off_base"])

    res = {"main": main_res, "subclusters": subs}
    (DATA / "cluster_reproducibility.json").write_text(json.dumps(res, indent=2, ensure_ascii=False))
    print(json.dumps(res, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
