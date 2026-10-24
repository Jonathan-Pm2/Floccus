"""Deduplication audit: Layer-1/2/3 statistics reported in the manuscript.

Recomputes, from the persisted dedup artefacts of the biofloc corpus, every
number the manuscript reports for the three deduplication layers:

- Layer 1 (deterministic DOI / normalised title + year): identified records,
  primaries, removed share, and how many multi-record clusters span more
  than one source.
- Layer 2 (embedding near-duplicates): number of candidate pairs and their
  median cosine similarity, with the duplicate share per cosine band.
- Layer 3 (LLM judge): verdict counts and confidence by verdict; DUPLICATE
  verdicts are closed transitively with a disjoint-set union (union-find),
  giving the components and net merges that turn the Layer-1 primaries into
  the final corpus; confirmed duplicates whose two records carry different
  DOIs.
- The 12 SciELO records repeated with the same SciELO identifier, collapsed
  by record_id at screening, which takes the final corpus to 17,063.

Run:  .venv/bin/python scripts/dedup_audit.py
Out:  data/processed/dedup_audit.json
"""
from __future__ import annotations

import json
import re
import statistics
import uuid
from collections import Counter
from pathlib import Path
from typing import Hashable, Iterable, Sequence

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "processed"
CORPUS = "biofloc"
BAND_LOW, BAND_HIGH = 0.97, 0.98
DOI_PREFIX = re.compile(r"^(https?://(dx\.)?doi\.org/|doi:)")


# ---------------------------------------------------------------- pure helpers
def merge_components(pairs: Iterable[tuple[Hashable, Hashable]]) -> list[set]:
    """Connected components of the graph whose edges are the given pairs.

    Disjoint-set union with path halving; only nodes that appear in a pair
    are returned (singletons are not components of a merge).
    """
    parent: dict = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in pairs:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb
    comps: dict = {}
    for node in list(parent):
        comps.setdefault(find(node), set()).add(node)
    return list(comps.values())


def net_merges(components: Iterable[set]) -> int:
    """Records absorbed when every component collapses to one record."""
    return sum(len(c) - 1 for c in components)


def cosine_bands(scores: Sequence[float], verdicts: Sequence[str],
                 low: float = BAND_LOW, high: float = BAND_HIGH) -> dict:
    """Pairs and DUPLICATE verdicts below `low`, in [low, high) and >= `high`."""
    names = (f"below_{low}", f"{low}_to_{high}", f"at_least_{high}")
    out = {n: {"n": 0, "duplicate": 0} for n in names}
    for s, v in zip(scores, verdicts):
        band = names[0] if s < low else names[1] if s < high else names[2]
        out[band]["n"] += 1
        out[band]["duplicate"] += int(v == "DUPLICATE")
    return out


def norm_doi(doi) -> str:
    """Lower-case DOI without resolver prefix; empty string when missing."""
    if doi is None or (isinstance(doi, float) and doi != doi):  # None or NaN
        return ""
    return DOI_PREFIX.sub("", str(doi).lower().strip())


def record_id(source: str, source_id, title_norm) -> str:
    """Screening record identifier (same rule as _record_to_point_id in src/biofloc_review/dedup/embeddings.py)."""
    sid = None if pd.isna(source_id) else source_id
    tn = "" if pd.isna(title_norm) else title_norm
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"{source}:{sid or tn[:80]}"))


# ---------------------------------------------------------------- sections
def layer1(full: pd.DataFrame, prim: pd.DataFrame) -> dict:
    g = full.groupby("cluster_id")
    size, n_src = g.size(), g["source"].nunique()
    multi = size > 1
    removed = len(full) - len(prim)
    return {
        "identified": int(len(full)),
        "primaries": int(len(prim)),
        "removed": int(removed),
        "removed_pct": round(100 * removed / len(full), 1),
        "cluster_size_distribution": {int(k): int(v) for k, v in size.value_counts().sort_index().items()},
        "multi_record_clusters": int(multi.sum()),
        "multi_record_clusters_multi_source": int((multi & (n_src > 1)).sum()),
    }


def layer2_3(pairs: list[dict], prim: pd.DataFrame) -> dict:
    verdicts = [p["judge_verdict"] for p in pairs]
    scores = [p["score"] for p in pairs]
    conf = {v: dict(sorted(Counter(p["judge_confidence"] for p in pairs
                                   if p["judge_verdict"] == v).items()))
            for v in sorted(set(verdicts))}
    dup = [p for p in pairs if p["judge_verdict"] == "DUPLICATE"]
    comps = merge_components((p["a_cluster"], p["b_cluster"]) for p in dup)
    merges = net_merges(comps)

    # confirmed duplicates whose two records carry two different DOIs
    doi = {record_id(s, sid, tn): d for s, sid, tn, d in
           zip(prim.source, prim.source_id, prim.title_norm, prim.doi)}
    two_dois = 0
    for p in dup:
        da, db = norm_doi(doi.get(p["a_id"])), norm_doi(doi.get(p["b_id"]))
        two_dois += int(bool(da and db and da != db))

    return {
        "layer2_pairs": len(pairs),
        "layer2_median_cosine": round(statistics.median(scores), 3),
        "cosine_bands": cosine_bands(scores, verdicts),
        "layer3_verdicts": dict(Counter(verdicts)),
        "layer3_confidence_by_verdict": {v: {int(k): n for k, n in c.items()} for v, c in conf.items()},
        "duplicate_pairs_unresolved_ids": sum(1 for p in dup if p["a_id"] not in doi or p["b_id"] not in doi),
        "duplicates_with_two_different_dois": two_dois,
        "duplicates_with_two_different_dois_pct": round(100 * two_dois / len(dup), 1),
        "transitive_components": len(comps),
        "component_sizes": {int(k): int(v) for k, v in sorted(Counter(len(c) for c in comps).items())},
        "net_merges": merges,
        "after_layer3": int(len(prim) - merges),
    }


def scielo_repeats(final_primary: pd.DataFrame, decisions: pd.DataFrame) -> dict:
    """SciELO primaries that repeat the same SciELO identifier (collapsed by record_id)."""
    sci = final_primary[final_primary.source == "scielo"]
    repeats = int(len(sci) - sci.source_id.nunique())
    return {
        "final_primaries": int(len(final_primary)),
        "scielo_same_identifier_repeats": repeats,
        "after_scielo_repeats": int(len(final_primary) - repeats),
        "screened_records": int(len(decisions)),
    }


def main() -> None:
    full = pd.read_parquet(DATA / f"{CORPUS}_dedup_full.parquet", columns=["cluster_id", "source"])
    prim = pd.read_parquet(DATA / f"{CORPUS}_dedup_primary.parquet",
                           columns=["source", "source_id", "title_norm", "doi"])
    final_primary = pd.read_parquet(DATA / f"{CORPUS}_final_primary.parquet", columns=["source", "source_id"])
    decisions = pd.read_parquet(DATA / f"{CORPUS}_final_decisions.parquet", columns=["record_id"])
    pairs = json.loads((DATA / f"{CORPUS}_dedup_primary_near_dup_pairs_judged.json").read_text())

    res = {
        "corpus": CORPUS,
        "layer1": layer1(full, prim),
        "layer2_layer3": layer2_3(pairs, prim),
        "final": scielo_repeats(final_primary, decisions),
    }
    (DATA / "dedup_audit.json").write_text(json.dumps(res, indent=2, ensure_ascii=False))
    print(json.dumps(res, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
