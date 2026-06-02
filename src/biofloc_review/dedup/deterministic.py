from __future__ import annotations

import hashlib
import logging
from collections import defaultdict
from collections.abc import Iterable

from biofloc_review.schema import PaperRecord, Source

log = logging.getLogger(__name__)


_SOURCE_PRIORITY: dict[Source, int] = {
    Source.SCOPUS: 0,
    Source.WOS: 1,
    Source.OPENALEX: 2,
    Source.IEEE: 3,
    Source.SCIELO: 4,
}


def _cluster_id_for(key: str) -> str:
    return hashlib.sha1(key.encode("utf-8")).hexdigest()[:16]


def _doi_key(rec: PaperRecord) -> str | None:
    return rec.doi if rec.doi else None


def _title_key(rec: PaperRecord) -> str:
    year = str(rec.year) if rec.year is not None else "any"
    return f"{rec.title_norm}::{year}"


def _completeness_score(rec: PaperRecord) -> int:
    score = 0
    if rec.doi:
        score += 8
    if rec.abstract:
        score += 16
    if rec.keywords:
        score += 4
    if rec.authors:
        score += 2
    if rec.year:
        score += 1
    if rec.venue:
        score += 2
    if rec.open_access is not None:
        score += 1
    if rec.url:
        score += 1
    return score


def _select_primary(cluster: list[PaperRecord]) -> int:
    best_idx = 0
    best_key: tuple = (-1, 99, "")
    for i, r in enumerate(cluster):
        sp = _SOURCE_PRIORITY.get(r.source, 99)
        key = (_completeness_score(r), -sp, r.source_id or "")
        if key > best_key:
            best_key = key
            best_idx = i
    return best_idx


def _merge_into_primary(primary: PaperRecord, siblings: list[PaperRecord]) -> PaperRecord:
    data = primary.model_dump()
    merged_from: set[str] = set(data.get("merged_from") or [])
    for s in siblings:
        if s.source_id:
            merged_from.add(f"{s.source.value}:{s.source_id}")
        else:
            merged_from.add(f"{s.source.value}:title:{s.title_norm[:60]}")
        if not data.get("doi") and s.doi:
            data["doi"] = s.doi
        if not data.get("abstract") and s.abstract:
            data["abstract"] = s.abstract
        if not data.get("keywords") and s.keywords:
            data["keywords"] = list(s.keywords)
        if data.get("open_access") is None and s.open_access is not None:
            data["open_access"] = s.open_access
        if not data.get("url") and s.url:
            data["url"] = s.url
        if not data.get("venue") and s.venue:
            data["venue"] = s.venue
        if not data.get("document_type") and s.document_type:
            data["document_type"] = s.document_type
        if not data.get("language") and s.language:
            data["language"] = s.language
    data["merged_from"] = sorted(merged_from)
    data["needs_enrichment"] = data.get("abstract") is None
    return PaperRecord.model_validate(data)


def cluster_deterministic(records: Iterable[PaperRecord]) -> dict[str, list[PaperRecord]]:
    records = list(records)
    clusters: dict[str, list[PaperRecord]] = defaultdict(list)
    title_buckets: dict[str, str] = {}

    for r in records:
        doi = _doi_key(r)
        if doi:
            cid = _cluster_id_for(f"doi::{doi}")
        else:
            tk = _title_key(r)
            if tk in title_buckets:
                cid = title_buckets[tk]
            else:
                cid = _cluster_id_for(f"title::{tk}")
                title_buckets[tk] = cid
        clusters[cid].append(r)
    return clusters


def dedup_deterministic(records: Iterable[PaperRecord]) -> tuple[list[PaperRecord], list[PaperRecord], dict]:
    records = list(records)
    clusters = cluster_deterministic(records)

    primaries: list[PaperRecord] = []
    full_marked: list[PaperRecord] = []
    cross_source_clusters = 0
    sizes: dict[int, int] = defaultdict(int)

    for cid, cluster in clusters.items():
        primary_idx = _select_primary(cluster)
        primary = cluster[primary_idx]
        siblings = [c for i, c in enumerate(cluster) if i != primary_idx]
        merged_primary = _merge_into_primary(primary, siblings) if siblings else primary

        primary_data = merged_primary.model_dump()
        primary_data["cluster_id"] = cid
        primary_data["is_primary"] = True
        primary_final = PaperRecord.model_validate(primary_data)
        primaries.append(primary_final)
        full_marked.append(primary_final)

        for s in siblings:
            d = s.model_dump()
            d["cluster_id"] = cid
            d["is_primary"] = False
            full_marked.append(PaperRecord.model_validate(d))

        sources_in_cluster = {c.source for c in cluster}
        if len(sources_in_cluster) > 1:
            cross_source_clusters += 1
        sizes[len(cluster)] += 1

    stats = {
        "input_records": len(records),
        "unique_clusters": len(clusters),
        "duplicates_removed": len(records) - len(clusters),
        "reduction_pct": round(100 * (1 - len(clusters) / max(len(records), 1)), 2),
        "cross_source_clusters": cross_source_clusters,
        "cluster_size_histogram": dict(sorted(sizes.items())),
    }
    return primaries, full_marked, stats
