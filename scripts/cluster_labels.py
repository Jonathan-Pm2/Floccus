"""Label evidence for the 25 thematic clusters: top title terms and sample titles.

The manuscript names each UMAP+HDBSCAN cluster from the ten most frequent
title terms of its members and three representative titles. Both were
archived in data/processed/eligible_clusters_info.parquet (columns `tokens`
and `samples`) but the code that produced them was not. This script
reimplements the term extraction and reports how closely it reproduces the
archived terms, instead of claiming exactness:

- terms: lower-cased alphanumeric runs of length >= 3, digits-only runs and a
  short list of English function words and generic research vocabulary
  removed, ranked by total frequency over the cluster titles, ties broken by
  first appearance in the assignment order;
- samples: the first three member titles in assignment order (this rule
  reproduces all 25 archived sample lists).

Run:  .venv/bin/python scripts/cluster_labels.py
Out:  data/processed/cluster_label_evidence.json
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "processed"
TOP_K = 10
N_SAMPLES = 3
MIN_LEN = 3
TOKEN = re.compile(r"[a-z0-9]+")
# function words and generic research vocabulary that carry no topic
STOP = frozenset({
    "the", "and", "for", "with", "from", "its", "via", "under", "during",
    "using", "based", "study", "new", "effect", "effects", "analysis", "case",
    "model", "models", "modeling", "application", "performance", "different",
    "system", "systems", "evaluation", "review", "role", "approach",
    "full", "scale", "lab", "pilot", "high", "low",
})


# ---------------------------------------------------------------- pure helpers
def tokens(title) -> list[str]:
    """Content tokens of one title (stop words, digits and short words dropped)."""
    if not isinstance(title, str):
        return []
    return [w for w in TOKEN.findall(title.lower())
            if len(w) >= MIN_LEN and w not in STOP and not w.isdigit()]


def title_terms(titles: Iterable, k: int = TOP_K) -> list[str]:
    """Top-k terms by frequency; ties keep the order of first appearance."""
    count: dict[str, int] = {}
    for t in titles:
        for w in tokens(t):
            count[w] = count.get(w, 0) + 1
    # dicts keep insertion order and sorted() is stable -> first-appearance tie-break
    return [w for w, _ in sorted(count.items(), key=lambda kv: -kv[1])[:k]]


def label_match(mine: Sequence[str], archived: Sequence[str]) -> dict:
    """Shared terms (any order) and whether the ranked lists are identical."""
    return {"overlap": len(set(mine) & set(archived)), "exact": list(mine) == list(archived)}


def main() -> None:
    assign = pd.read_parquet(DATA / "eligible_clusters_assignment.parquet")
    info = pd.read_parquet(DATA / "eligible_clusters_info.parquet").sort_values("cluster")

    clusters, overlap, exact, samples_ok = [], 0, 0, 0
    for _, row in info.iterrows():
        titles = assign.loc[assign.cluster == row.cluster, "title"].tolist()
        terms = title_terms(titles)
        archived = [str(t) for t in row.tokens]
        m = label_match(terms, archived)
        samples = titles[:N_SAMPLES]
        same_samples = samples == [str(s) for s in row.samples]
        overlap += m["overlap"]
        exact += int(m["exact"])
        samples_ok += int(same_samples)
        clusters.append({
            "cluster": int(row.cluster), "n": len(titles),
            "top_terms": terms, "archived_terms": archived,
            "term_overlap": m["overlap"], "exact_term_list": m["exact"],
            "representative_titles": samples, "matches_archived_samples": same_samples,
        })

    n = len(info)
    res = {
        "n_clusters": n,
        "match_vs_archived": {
            "terms_shared": overlap, "terms_total": n * TOP_K,
            "terms_shared_pct": round(100 * overlap / (n * TOP_K), 1),
            "exact_ranked_lists": exact,
            "sample_lists_identical": samples_ok,
        },
        "method": {"min_len": MIN_LEN, "top_k": TOP_K, "stop_words": sorted(STOP),
                   "tie_break": "first appearance in assignment order",
                   "samples": "first three member titles in assignment order"},
        "clusters": clusters,
    }
    (DATA / "cluster_label_evidence.json").write_text(json.dumps(res, indent=2, ensure_ascii=False))
    print(json.dumps(res, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
