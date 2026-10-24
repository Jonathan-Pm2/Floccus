"""C03--C15 bridge score with a permutation null model.

The manuscript defines, for every eligible record d outside the two core
clusters,

    b(d) = min{ sim(e_d, c_C03), sim(e_d, c_C15) }          (Eq. eq:bridge)

with e_d the record embedding, c_k the mean member embedding of cluster k
and sim the cosine similarity. The original computation was not part of the
repository; this script reimplements it from that definition, checks it
against the persisted values (data/processed/c03_c15_bridges.parquet) and
adds the null model requested in review.

Null model. If the observed count of strong bridges (b >= 0.7) reflected
nothing more than the general proximity of any two clusters in this corpus,
pairs of random pseudo-clusters with the same sizes as C03 (141) and C15
(68), drawn from the other eligible records, would produce comparable counts.
We draw R such pairs, recompute the count for each, and report the empirical
distribution and a one-sided p-value for "fewer bridges than chance"
(the manuscript's claim is that the traffic is THIN).

Run:  .venv/bin/python scripts/bridge_score.py [--perm 1000]
Out:  data/processed/bridge_null_model.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "processed"
QDRANT = "http://localhost:6333/collections/papers_biofloc/points"
THRESH = (0.5, 0.7)
SEED = 20260603


def fetch_vectors(ids: list[str]) -> dict[str, np.ndarray]:
    out = {}
    for i in range(0, len(ids), 256):
        r = requests.post(QDRANT, json={"ids": ids[i:i + 256], "with_vector": True,
                                        "with_payload": False}, timeout=120)
        r.raise_for_status()
        for p in r.json()["result"]:
            out[str(p["id"])] = np.asarray(p["vector"], dtype=np.float32)
    return out


def unit(m: np.ndarray) -> np.ndarray:
    return m / np.linalg.norm(m, axis=-1, keepdims=True)


def bridge(E: np.ndarray, a_idx: np.ndarray, b_idx: np.ndarray, pool: np.ndarray) -> np.ndarray:
    ca, cb = unit(E[a_idx].mean(0)), unit(E[b_idx].mean(0))
    X = E[pool]
    return np.minimum(X @ ca, X @ cb)


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--perm", type=int, default=1000)
    args = ap.parse_args()
    assign = pd.read_parquet(DATA / "eligible_clusters_assignment.parquet")[["record_id", "cluster"]]
    vec = fetch_vectors(assign.record_id.tolist())
    assign = assign[assign.record_id.isin(vec)].reset_index(drop=True)
    print(f"eligible with embedding: {len(assign)} (vectors found {len(vec)})")
    E = unit(np.stack([vec[r] for r in assign.record_id]))
    c = assign.cluster.to_numpy()
    a_idx, b_idx = np.where(c == 3)[0], np.where(c == 15)[0]
    pool = np.where((c != 3) & (c != 15))[0]

    # --- 1. reproduce the persisted scores -----------------------------
    obs = bridge(E, a_idx, b_idx, pool)
    stored = pd.read_parquet(DATA / "c03_c15_bridges.parquet")[["record_id", "bridge_score"]]
    cmp = pd.DataFrame({"record_id": assign.record_id.iloc[pool].values, "b": obs}).merge(stored, on="record_id")
    err = np.abs(cmp.b - cmp.bridge_score)
    print(f"reproduction vs stored: n={len(cmp)}  max|Δ|={err.max():.2e}  mean|Δ|={err.mean():.2e}")
    counts_obs = {t: int((obs >= t).sum()) for t in THRESH}
    print(f"observed: b>=0.5 → {counts_obs[0.5]}   b>=0.7 → {counts_obs[0.7]}   (pool {len(pool)})")

    # --- 2. permutation null ------------------------------------------
    rng = np.random.default_rng(SEED)
    null = {t: [] for t in THRESH}
    for _ in range(args.perm):
        perm = rng.permutation(pool)
        a, b, rest = perm[:len(a_idx)], perm[len(a_idx):len(a_idx) + len(b_idx)], perm[len(a_idx) + len(b_idx):]
        s = bridge(E, a, b, rest)
        for t in THRESH:
            null[t].append(int((s >= t).sum()))
    res = {"n_pool": int(len(pool)), "n_perm": args.perm, "seed": SEED,
           "reproduction_max_abs_err": float(err.max()), "observed": counts_obs, "null": {}}
    for t in THRESH:
        arr = np.array(null[t])
        p_low = (np.sum(arr <= counts_obs[t]) + 1) / (len(arr) + 1)
        res["null"][str(t)] = {"mean": float(arr.mean()), "sd": float(arr.std()),
                               "q025": float(np.percentile(arr, 2.5)), "q975": float(np.percentile(arr, 97.5)),
                               "p_fewer_than_chance": float(p_low)}
        print(f"null b>={t}: mean {arr.mean():.1f} sd {arr.std():.1f} 95% [{np.percentile(arr,2.5):.0f}–{np.percentile(arr,97.5):.0f}]"
              f"   observed {counts_obs[t]}   p(≤obs) = {p_low:.4f}")
    (DATA / "bridge_null_model.json").write_text(json.dumps(res, indent=2))
    print("→", DATA / "bridge_null_model.json")


if __name__ == "__main__":
    main()


def real_pair_null() -> None:
    """Stronger null: the same statistic for every pair of REAL clusters.

    Random pseudo-clusters have centroids close to the corpus mean, so almost
    every record is near them and the permutation null is lenient. Comparing
    C03-C15 with the other pairs of thematically coherent clusters asks the
    sharper question: is the C03-C15 boundary thinner than a typical boundary
    between two real topics of this corpus?
    """
    assign = pd.read_parquet(DATA / "eligible_clusters_assignment.parquet")[["record_id", "cluster"]]
    vec = fetch_vectors(assign.record_id.tolist())
    assign = assign[assign.record_id.isin(vec)].reset_index(drop=True)
    E = unit(np.stack([vec[r] for r in assign.record_id])); c = assign.cluster.to_numpy()
    ks = sorted(k for k in set(c) if k >= 0)
    cent = {k: unit(E[c == k].mean(0)) for k in ks}
    rows = []
    for i, a in enumerate(ks):
        for b in ks[i + 1:]:
            pool = (c != a) & (c != b)
            s = np.minimum(E[pool] @ cent[a], E[pool] @ cent[b])
            rows.append({"a": a, "b": b, "n_pool": int(pool.sum()), "frac_07": float((s >= .7).mean()),
                         "n_07": int((s >= .7).sum()), "cos_centroids": float(cent[a] @ cent[b])})
    df = pd.DataFrame(rows); me = df[(df.a == 3) & (df.b == 15)].iloc[0]
    rank = int((df.frac_07 <= me.frac_07).sum())
    print(f"\nREAL-PAIR NULL over {len(df)} cluster pairs")
    print(f"  C03-C15: n(b>=0.7) = {me.n_07} ({me.frac_07:.2%} of pool), centroid cosine {me.cos_centroids:.3f}")
    print(f"  all pairs: median {df.frac_07.median():.2%}  IQR [{df.frac_07.quantile(.25):.2%}–{df.frac_07.quantile(.75):.2%}]")
    print(f"  rank of C03-C15 from the thinnest: {rank}/{len(df)}  (empirical percentile {rank/len(df):.1%})")
    # control for how far apart the two centroids are: compare with pairs of similar centroid distance
    near = df[(df.cos_centroids - me.cos_centroids).abs() <= 0.03]
    print(f"  pairs with similar centroid cosine (±0.03): {len(near)} · median frac {near.frac_07.median():.2%} · "
          f"C03-C15 rank {int((near.frac_07 <= me.frac_07).sum())}/{len(near)}")
    df.to_parquet(DATA / "bridge_real_pairs.parquet", index=False)
    j = json.loads((DATA / "bridge_null_model.json").read_text())
    j["real_pair_null"] = {"n_pairs": int(len(df)), "c03_c15_frac": float(me.frac_07), "c03_c15_n": int(me.n_07),
                           "median_frac": float(df.frac_07.median()), "rank_thinnest": rank,
                           "c03_c15_centroid_cos": float(me.cos_centroids),
                           "similar_distance_pairs": int(len(near)),
                           "rank_among_similar": int((near.frac_07 <= me.frac_07).sum())}
    (DATA / "bridge_null_model.json").write_text(json.dumps(j, indent=2))


if __name__ == "__main__":
    main()
    real_pair_null()
