from __future__ import annotations

import logging
import os
import time
import uuid
from collections import defaultdict
from collections.abc import Iterable

from qdrant_client import QdrantClient
from qdrant_client.http import models as qm

from biofloc_review.dedup.ollama_embed import (
    DEFAULT_MODEL,
    EMBEDDING_DIM,
    chunked,
    embed_batch,
)
from biofloc_review.schema import PaperRecord

log = logging.getLogger(__name__)


DEFAULT_QDRANT_URL = "http://localhost:6333"


def qdrant_client(url: str | None = None) -> QdrantClient:
    return QdrantClient(url=url or os.environ.get("QDRANT_URL", DEFAULT_QDRANT_URL))


def collection_name_for(corpus: str) -> str:
    return f"papers_{corpus}"


def ensure_collection(client: QdrantClient, name: str, recreate: bool = False) -> None:
    exists = client.collection_exists(name)
    if exists and recreate:
        client.delete_collection(name)
        exists = False
    if not exists:
        client.create_collection(
            collection_name=name,
            vectors_config=qm.VectorParams(size=EMBEDDING_DIM, distance=qm.Distance.COSINE),
        )
        client.create_payload_index(name, "cluster_id", qm.PayloadSchemaType.KEYWORD)
        client.create_payload_index(name, "source", qm.PayloadSchemaType.KEYWORD)
        client.create_payload_index(name, "year", qm.PayloadSchemaType.INTEGER)
        log.info("Created collection %r (dim=%d, cosine)", name, EMBEDDING_DIM)


def _text_for(rec: PaperRecord) -> str:
    parts = [rec.title.strip()]
    if rec.abstract:
        parts.append(rec.abstract.strip())
    return " ".join(parts)[:6000]


def _record_to_point_id(rec: PaperRecord) -> str:
    base = f"{rec.source.value}:{rec.source_id or rec.title_norm[:80]}"
    return str(uuid.uuid5(uuid.NAMESPACE_URL, base))


def _payload_for(rec: PaperRecord) -> dict:
    return {
        "cluster_id": rec.cluster_id,
        "source": rec.source.value,
        "source_id": rec.source_id,
        "title": rec.title,
        "doi": rec.doi,
        "year": rec.year,
        "venue": rec.venue,
        "has_abstract": rec.abstract is not None,
        "corpus": rec.corpus.value,
    }


MAX_CONSECUTIVE_FAILURES = 5


def index_records(
    client: QdrantClient,
    collection: str,
    records: Iterable[PaperRecord],
    model: str = DEFAULT_MODEL,
    batch_size: int = 32,
) -> int:
    records = list(records)
    total = len(records)
    log.info("Indexing %d records into %r (batch=%d, model=%s)", total, collection, batch_size, model)

    n_done = 0
    consecutive_failures = 0
    t0 = time.time()
    for batch in chunked(records, batch_size):
        texts = [_text_for(r) for r in batch]
        try:
            embeddings = embed_batch(texts, model=model)
        except Exception as e:
            consecutive_failures += 1
            log.warning("embed_batch failed for batch starting %r (%d consecutive): %s",
                        batch[0].title[:60], consecutive_failures, e)
            if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                log.error("Circuit breaker tripped after %d consecutive failures. Aborting indexing.",
                          consecutive_failures)
                break
            continue
        consecutive_failures = 0
        points = [
            qm.PointStruct(
                id=_record_to_point_id(r),
                vector=vec,
                payload=_payload_for(r),
            )
            for r, vec in zip(batch, embeddings)
        ]
        client.upsert(collection_name=collection, points=points)
        n_done += len(batch)
        if n_done % (batch_size * 4) == 0 or n_done == total:
            elapsed = time.time() - t0
            rate = n_done / max(elapsed, 1e-6)
            eta = (total - n_done) / max(rate, 1e-6)
            log.info("  %d / %d (%.1f rec/s, eta %.0fs)", n_done, total, rate, eta)
    return n_done


def find_near_duplicate_pairs(
    client: QdrantClient,
    collection: str,
    records: Iterable[PaperRecord],
    threshold: float = 0.93,
    top_k: int = 5,
) -> list[dict]:
    """For each record, use its stored vector as query (no re-embed) and emit cross-cluster pairs."""
    records = list(records)
    by_id: dict[str, PaperRecord] = {_record_to_point_id(r): r for r in records}
    pairs: dict[tuple[str, str], dict] = {}

    n_done = 0
    total = len(records)
    t0 = time.time()
    for r in records:
        r_id = _record_to_point_id(r)
        try:
            resp = client.query_points(
                collection_name=collection,
                query=r_id,
                limit=top_k + 1,
                score_threshold=threshold,
                with_payload=True,
            )
        except Exception as e:
            log.warning("query_points failed for %r: %s", r.title[:60], e)
            n_done += 1
            continue

        for h in resp.points:
            h_id = str(h.id)
            if h_id == r_id:
                continue
            h_cluster = h.payload.get("cluster_id") if h.payload else None
            if r.cluster_id and h_cluster == r.cluster_id:
                continue
            key = tuple(sorted([r_id, h_id]))
            if key in pairs and pairs[key]["score"] >= h.score:
                continue
            other_title = h.payload.get("title") if h.payload else None
            pairs[key] = {
                "a_id": key[0],
                "b_id": key[1],
                "a_title": r.title if key[0] == r_id else other_title,
                "b_title": other_title if key[0] == r_id else r.title,
                "a_cluster": r.cluster_id if key[0] == r_id else h_cluster,
                "b_cluster": h_cluster if key[0] == r_id else r.cluster_id,
                "score": float(h.score),
            }
        n_done += 1
        if n_done % 100 == 0 or n_done == total:
            elapsed = time.time() - t0
            log.info("  searched %d / %d (%.1f rec/s)", n_done, total, n_done / max(elapsed, 1e-6))

    out = sorted(pairs.values(), key=lambda p: -p["score"])
    log.info("Found %d candidate near-duplicate pairs (threshold=%.2f)", len(out), threshold)
    return out
