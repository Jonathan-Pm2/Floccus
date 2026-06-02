from __future__ import annotations

import logging
from collections import defaultdict
from collections.abc import Iterable

from biofloc_review.dedup.deterministic import _merge_into_primary, _select_primary
from biofloc_review.schema import PaperRecord

log = logging.getLogger(__name__)


class _DSU:
    """Disjoint-set union over cluster_id strings."""
    def __init__(self) -> None:
        self.parent: dict[str, str] = {}

    def find(self, x: str) -> str:
        if x not in self.parent:
            self.parent[x] = x
            return x
        # path compression
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        cur = x
        while self.parent[cur] != root:
            nxt = self.parent[cur]
            self.parent[cur] = root
            cur = nxt
        return root

    def union(self, x: str, y: str) -> None:
        rx, ry = self.find(x), self.find(y)
        if rx != ry:
            self.parent[rx] = ry


def merge_confirmed_duplicates(
    records: Iterable[PaperRecord],
    judged_pairs: Iterable[dict],
) -> tuple[list[PaperRecord], list[PaperRecord], dict]:
    """Apply DUPLICATE verdicts to merge clusters across Capa 1 boundaries.

    For each judged pair with verdict==DUPLICATE, union the two cluster_ids.
    Then re-derive primary records and siblings per merged cluster.

    Returns: (primary_records, full_marked, stats)
    """
    records = list(records)
    pairs = list(judged_pairs)

    dsu = _DSU()
    dup_pairs = [p for p in pairs if p.get("judge_verdict") == "DUPLICATE"]
    n_unclear = sum(1 for p in pairs if p.get("judge_verdict") == "UNCLEAR")
    n_not_dup = sum(1 for p in pairs if p.get("judge_verdict") == "NOT_DUPLICATE")
    for p in dup_pairs:
        a = p.get("a_cluster")
        b = p.get("b_cluster")
        if a and b:
            dsu.union(a, b)

    by_new_cluster: dict[str, list[PaperRecord]] = defaultdict(list)
    for r in records:
        if r.cluster_id:
            new_cid = dsu.find(r.cluster_id)
        else:
            new_cid = r.cluster_id or "unclustered"
        by_new_cluster[new_cid].append(r)

    primaries: list[PaperRecord] = []
    full_marked: list[PaperRecord] = []
    n_clusters_merged = 0
    merged_pairs_applied = 0

    for new_cid, group in by_new_cluster.items():
        original_clusters = {r.cluster_id for r in group if r.cluster_id}
        if len(original_clusters) > 1:
            n_clusters_merged += 1
            merged_pairs_applied += len(original_clusters) - 1

        winner_idx = _select_primary(group)
        winner = group[winner_idx]
        siblings = [c for i, c in enumerate(group) if i != winner_idx]
        merged_winner = _merge_into_primary(winner, siblings) if siblings else winner

        prim_data = merged_winner.model_dump()
        prim_data["cluster_id"] = new_cid
        prim_data["is_primary"] = True
        prim_final = PaperRecord.model_validate(prim_data)
        primaries.append(prim_final)
        full_marked.append(prim_final)

        for s in siblings:
            sd = s.model_dump()
            sd["cluster_id"] = new_cid
            sd["is_primary"] = False
            full_marked.append(PaperRecord.model_validate(sd))

    stats = {
        "input_records": len(records),
        "input_pairs_judged": len(pairs),
        "verdicts": {
            "DUPLICATE": len(dup_pairs),
            "NOT_DUPLICATE": n_not_dup,
            "UNCLEAR": n_unclear,
        },
        "clusters_merged_by_capa3": n_clusters_merged,
        "primary_records_before": sum(1 for r in records if r.is_primary),
        "primary_records_after": len(primaries),
        "total_records": len(full_marked),
    }
    return primaries, full_marked, stats
