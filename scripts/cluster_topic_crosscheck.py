"""External cross-check of the thematic clusters against OpenAlex topics.

The 25 UMAP+HDBSCAN clusters are built from embeddings only. As an
independent check, each clustered eligible record is matched to the topics
OpenAlex assigns to it (biofloc_enriched.oa_topics, topics in OpenAlex rank
order, the first one being the primary topic). The script reports:

- per cluster, the most and second most frequent topic over all assigned
  topics (checked against the archived clusters_vs_oa_topics.parquet), and
  whether the top topic is specific or the generic
  "Wastewater Treatment and Nitrogen Removal";
- topic coverage of the clustered records, overall and by source;
- agreement between cluster and primary topic: NMI, AMI with a permutation
  null (cluster labels shuffled, fixed seed) and purity.

Run:  .venv/bin/python scripts/cluster_topic_crosscheck.py [--perm 200]
Out:  data/processed/cluster_topic_crosscheck.json
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_mutual_info_score, normalized_mutual_info_score

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "processed"
GENERIC = "Wastewater Treatment and Nitrogen Removal"
SEED = 0


# ---------------------------------------------------------------- pure helpers
def topic_list(value) -> list[str]:
    """OpenAlex topics of one record as display names (empty when missing)."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    out = []
    for t in list(value):
        name = (t.get("display_name") or t.get("name")) if isinstance(t, dict) else str(t)
        if name:
            out.append(name)
    return out


def purity(clusters: Sequence, classes: Sequence) -> float:
    """Share of records whose class is the majority class of their cluster."""
    tab = pd.crosstab(pd.Series(list(clusters), name="c"), pd.Series(list(classes), name="t"))
    return float(tab.max(axis=1).sum() / tab.values.sum())


def top_two(topic_lists: Sequence[list[str]]) -> list[tuple[str, int]]:
    """Two most frequent topics over all topics of the records (first-seen tie-break)."""
    return Counter(t for lst in topic_lists for t in lst).most_common(2)


# ---------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--perm", type=int, default=200, help="permutations of the AMI null")
    args = ap.parse_args()

    enr = pd.read_parquet(DATA / "biofloc_enriched.parquet", columns=["record_id", "oa_topics"])
    assign = pd.read_parquet(DATA / "eligible_clusters_assignment.parquet")
    ref = pd.read_parquet(DATA / "clusters_vs_oa_topics.parquet").set_index("cluster")
    src = pd.read_parquet(DATA / "eligible_with_citations.parquet", columns=["record_id", "source"])

    m = assign.merge(enr.drop_duplicates("record_id"), on="record_id", how="left")
    m["topics"] = m.oa_topics.apply(topic_list)
    m["primary"] = m.topics.apply(lambda lst: lst[0] if lst else None)
    clustered = m[m.cluster >= 0].merge(src, on="record_id", how="left")

    # per-cluster top / runner-up topics vs the archived table
    per_cluster, count_matches = [], 0
    for c in sorted(clustered.cluster.unique()):
        s = clustered[clustered.cluster == c]
        (t1, n1), (t2, n2) = top_two(s.topics.tolist())
        r = ref.loc[c]
        hits = int((t1, int(n1)) == (r.top1_oa_topic, int(r.top1_count))) \
            + int((t2, int(n2)) == (r.top2_oa_topic, int(r.top2_count)))
        match = hits == 2
        count_matches += hits
        per_cluster.append({
            "cluster": int(c), "n": int(len(s)), "n_with_topics": int((s.topics.str.len() > 0).sum()),
            "top_topic": t1, "top_count": int(n1), "runner_up_topic": t2, "runner_up_count": int(n2),
            "top_is_generic": t1 == GENERIC, "matches_archived": match,
        })

    has = clustered.primary.notna()
    by_source = clustered.assign(has=has).groupby("source").has.agg(["sum", "size"])
    x = clustered[has]
    nmi = normalized_mutual_info_score(x.cluster, x.primary)
    ami = adjusted_mutual_info_score(x.cluster, x.primary)
    rng = np.random.default_rng(SEED)
    null = [adjusted_mutual_info_score(rng.permutation(x.cluster.values), x.primary)
            for _ in range(args.perm)]

    res = {
        "clustered_records": int(len(clustered)),
        "with_topic": int(has.sum()),
        "coverage_pct": round(100 * has.mean(), 1),
        "coverage_by_source": {s: {"with_topic": int(r["sum"]), "n": int(r["size"]),
                                   "pct": round(100 * r["sum"] / r["size"], 1)}
                               for s, r in by_source.iterrows()},
        "distinct_primary_topics": int(x.primary.nunique()),
        "nmi": round(float(nmi), 3),
        "ami": round(float(ami), 3),
        "ami_null": {"permutations": args.perm, "seed": SEED,
                     "max": round(float(max(null)), 4), "mean": round(float(np.mean(null)), 4)},
        "purity": round(purity(x.cluster, x.primary), 3),
        "generic_topic": GENERIC,
        "generic_primary_share": round(float((x.primary == GENERIC).mean()), 3),
        "clusters_with_specific_top_topic": [f"C{c['cluster']}" for c in per_cluster if not c["top_is_generic"]],
        "archived_counts_matched": f"{count_matches}/{2 * len(per_cluster)}",
        "clusters": per_cluster,
    }
    (DATA / "cluster_topic_crosscheck.json").write_text(json.dumps(res, indent=2, ensure_ascii=False))
    print(json.dumps(res, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
