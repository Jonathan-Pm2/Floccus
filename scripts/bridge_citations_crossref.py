"""Citation links between clusters from Crossref reference lists (C03--C15 check).

Same question as scripts/bridge_citations.py (do the applied biofloc
cluster C03 and the formal floc-modelling cluster C15 cite each other?),
answered with Crossref instead of OpenAlex, whose daily quota is exhausted
after about 400 title searches. Crossref exposes the reference list of a
work by DOI when its publisher deposits open references, so coverage is
partial and is reported per cluster: only records with a DOI can be
matched, and only those whose references are deposited can cite.

A link a -> b is counted when a record of cluster a lists the DOI of a
record of cluster b. Link density between clusters a and b is
    rho(a, b) = (links a->b + links b->a) / (n_cite_a n_doi_b + n_cite_b n_doi_a),
with n_cite the records whose references are deposited and n_doi the
records with a DOI, i.e. links over possible citing-cited pairs. C03--C15
is ranked against all other pairs of real clusters and against the pairs
whose centroids are about as similar (cosine within 0.03), the same
control as bridge_score.real_pair_null.

Run:  .venv/bin/python scripts/bridge_citations_crossref.py
Out:  data/processed/crossref_refs.parquet       (cache, re-used on rerun)
      data/processed/bridge_citations_crossref.json
      data/processed/bridge_citation_pairs_crossref.parquet
"""
from __future__ import annotations

import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "processed"
CACHE = DATA / "crossref_refs.parquet"
A, B = 3, 15
_EMAIL = os.environ.get("OPENALEX_EMAIL", "")
HEADERS = {"User-Agent": f"Floccus/1.0 (mailto:{_EMAIL})" if _EMAIL else "Floccus/1.0"}


def clean_doi(d) -> str | None:
    if not isinstance(d, str) or not d.strip():
        return None
    d = re.sub(r"^https?://(dx\.)?doi\.org/", "", d.strip().lower())
    return d if d.startswith("10.") else None


def reference_dois(message: dict) -> list[str]:
    """DOIs listed in a Crossref work's reference list (references without a DOI are skipped)."""
    return sorted({c for c in (clean_doi(r.get("DOI")) for r in message.get("reference", [])) if c})


def fetch(doi: str) -> dict:
    for attempt in range(5):
        try:
            r = requests.get(f"https://api.crossref.org/works/{doi}", headers=HEADERS, timeout=30)
        except requests.RequestException:
            time.sleep(2 ** attempt)
            continue
        if r.status_code == 200:
            m = r.json()["message"]
            return {"doi": doi, "status": 200, "deposited": "reference" in m,
                    "refs": reference_dois(m)}
        if r.status_code == 404:
            return {"doi": doi, "status": 404, "deposited": False, "refs": []}
        time.sleep(2 ** attempt)
    return {"doi": doi, "status": -1, "deposited": False, "refs": []}


def resolve(dois: list[str], workers: int = 4) -> pd.DataFrame:
    done = pd.read_parquet(CACHE) if CACHE.exists() else pd.DataFrame(columns=["doi", "status", "deposited", "refs"])
    ok = set(done.doi[done.status.isin([200, 404])])
    todo = [d for d in dois if d not in ok]
    rows = done[done.doi.isin(ok)].to_dict("records")
    print(f"crossref: {len(ok)} cached, {len(todo)} to fetch", flush=True)
    with ThreadPoolExecutor(workers) as ex:
        futs = [ex.submit(fetch, d) for d in todo]
        for k, f in enumerate(as_completed(futs), 1):
            rows.append(f.result())
            if k % 250 == 0 or k == len(todo):
                pd.DataFrame(rows).to_parquet(CACHE, index=False)
                print(f"  {k}/{len(todo)}", flush=True)
    return pd.DataFrame(rows)


def link_matrix(clusters_of_citer: list[int], refs: list[list[str]], cluster_of_doi: dict[str, int],
                clusters: list[int]) -> np.ndarray:
    """L[i, j] = number of references from records of clusters[i] to records of clusters[j]."""
    idx = {c: i for i, c in enumerate(clusters)}
    L = np.zeros((len(clusters), len(clusters)), dtype=int)
    for c, rs in zip(clusters_of_citer, refs):
        for r in rs:
            if r in cluster_of_doi:
                L[idx[c], idx[cluster_of_doi[r]]] += 1
    return L


def pair_density(L: np.ndarray, i: int, j: int, n_cite: np.ndarray, n_doi: np.ndarray) -> float:
    possible = n_cite[i] * n_doi[j] + n_cite[j] * n_doi[i]
    return float((L[i, j] + L[j, i]) / possible) if possible else float("nan")


def main() -> None:
    assign = pd.read_parquet(DATA / "eligible_clusters_assignment.parquet")[["record_id", "cluster"]]
    dec = pd.read_parquet(DATA / "biofloc_final_decisions.parquet", columns=["record_id", "doi"])
    recs = assign.merge(dec.drop_duplicates("record_id"), on="record_id", how="left")
    recs["doi"] = recs.doi.map(clean_doi)
    recs = recs[recs.cluster >= 0]                                   # drop HDBSCAN noise
    cr = resolve(sorted(recs.doi.dropna().unique()))
    d = recs.merge(cr, on="doi", how="left")
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

    rows = [{"a": a, "b": b, "links": int(L[idx[a], idx[b]] + L[idx[b], idx[a]]),
             "density": pair_density(L, idx[a], idx[b], n_cite, n_doi)}
            for a, b in combinations(clusters, 2)]
    df = pd.DataFrame(rows)
    pairs_file = DATA / "bridge_real_pairs.parquet"
    if pairs_file.exists():
        df = df.merge(pd.read_parquet(pairs_file)[["a", "b", "cos_centroids"]], on=["a", "b"], how="left")
    me = df[(df.a == A) & (df.b == B)].iloc[0]

    def cov(c):
        g = d[d.cluster == c]
        return {"records": int(len(g)), "with_doi": int(g.doi.notna().sum()),
                "references_deposited": int(g.deposited.sum())}

    res = {
        "coverage": {"all_clustered": {"records": int(len(d)), "with_doi": int(d.doi.notna().sum()),
                                       "references_deposited": int(d.deposited.sum())},
                     "C03": cov(A), "C15": cov(B)},
        "C03_to_C15": int(L[idx[A], idx[B]]), "C15_to_C03": int(L[idx[B], idx[A]]),
        "within_C03": int(L[idx[A], idx[A]]), "within_C15": int(L[idx[B], idx[B]]),
        "C03_C15_density": float(me.density),
        "median_pair_density": float(df.density.median()),
        "share_pairs_with_zero_links": float((df.links == 0).mean()),
        "rank_from_sparsest": f"{int((df.density <= me.density).sum())}/{len(df)}",
    }
    if "cos_centroids" in df and df.cos_centroids.notna().any():
        near = df[(df.cos_centroids - me.cos_centroids).abs() <= 0.03]
        res["distance_matched"] = {"n_pairs": int(len(near)),
                                   "rank_from_sparsest": int((near.density <= me.density).sum()),
                                   "median_density": float(near.density.median()),
                                   "share_with_zero_links": float((near.links == 0).mean())}
    df.to_parquet(DATA / "bridge_citation_pairs_crossref.parquet", index=False)
    (DATA / "bridge_citations_crossref.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
