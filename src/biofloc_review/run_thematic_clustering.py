"""Layer 4 — thematic clustering of the eligible corpus.

Retrieves the dense embeddings of the records that advanced to the
eligibility phase (final_decision ∈ {INCLUDE, HUMAN_REVIEW}) from the
Qdrant collection used during deduplication, projects them to a
lower-dimensional space with UMAP, and clusters them with HDBSCAN.

The output is a thematic map of the corpus useful for:

  - Identifying sub-communities of papers that the screening did not
    separate (e.g. wastewater treatment vs. biofloc aquaculture).
  - Detecting outlier records that may need closer human review.
  - Quantifying the share of the corpus that aligns with the research
    question vs. adjacent literature captured by the search.

Authentication for Qdrant is not required (local container). The
script does not require an LLM call — clusters are labeled manually
by inspecting top tokens and representative titles after the run.

Outputs:
  - ``data/processed/eligible_vectors.npy``: (n, dim) matrix of the
    eligible records' embeddings.
  - ``data/processed/eligible_record_ids.txt``: parallel list of UUIDs.
  - ``data/processed/eligible_umap2d.npy``: (n, 2) projection for
    visualization.
  - ``data/processed/eligible_clusters.npy``: (n,) integer labels;
    ``-1`` means HDBSCAN noise.
  - ``data/processed/eligible_clusters_assignment.parquet``:
    record_id, cluster, title, year (joined with the canonical parquet).

Usage:
  uv run python -m biofloc_review.run_thematic_clustering \\
    data/processed/biofloc_final_decisions.parquet \\
    data/processed/biofloc_final_primary_v3.parquet \\
    --collection papers_biofloc --min-cluster-size 40
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import requests

log = logging.getLogger(__name__)


QDRANT_URL = "http://localhost:6333"


def _scroll_all_points(collection: str, with_vector: bool = True,
                        batch: int = 500) -> list[dict]:
    """Iterate every point in a Qdrant collection via the scroll endpoint."""
    out: list[dict] = []
    offset = None
    while True:
        body = {"limit": batch, "with_vector": with_vector,
                "with_payload": True}
        if offset is not None:
            body["offset"] = offset
        r = requests.post(
            f"{QDRANT_URL}/collections/{collection}/points/scroll",
            json=body, timeout=120,
        ).json()
        pts = r["result"]["points"]
        out.extend(pts)
        offset = r["result"].get("next_page_offset")
        if offset is None:
            break
    return out


def _build_title_index(canonical_parquet: Path) -> tuple[dict, dict]:
    """Map record_id (UUID5) → (title, year) from the canonical parquet."""
    from biofloc_review.dedup.embeddings import _record_to_point_id
    from biofloc_review.schema import PaperRecord
    df = pd.read_parquet(canonical_parquet)
    rid_to_title: dict[str, str] = {}
    rid_to_year: dict[str, int | None] = {}
    for _, row in df.iterrows():
        d = {k: (None if isinstance(v, float) and pd.isna(v) else v)
             for k, v in row.to_dict().items()}
        try:
            rec = PaperRecord.model_validate(d)
        except Exception:
            continue
        rid = _record_to_point_id(rec)
        rid_to_title[rid] = (rec.title or "")[:300]
        rid_to_year[rid] = rec.year
    return rid_to_title, rid_to_year


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Thematic clustering of the eligible corpus "
                    "(UMAP + HDBSCAN over Qdrant embeddings)"
    )
    parser.add_argument(
        "decisions_parquet", type=Path,
        help="Consolidated decisions parquet (biofloc_final_decisions.parquet)",
    )
    parser.add_argument(
        "canonical_parquet", type=Path,
        help="Canonical parquet for title/year join "
             "(biofloc_final_primary_v3.parquet)",
    )
    parser.add_argument(
        "--collection", default="papers_biofloc",
        help="Qdrant collection name (default: papers_biofloc)",
    )
    parser.add_argument(
        "--decisions", nargs="+",
        default=["INCLUDE", "HUMAN_REVIEW"],
        help="Final decisions whose records enter the analysis",
    )
    parser.add_argument(
        "--umap-clust-dim", type=int, default=10,
        help="Number of UMAP components for clustering (default: 10)",
    )
    parser.add_argument(
        "--umap-clust-neighbors", type=int, default=30,
        help="UMAP n_neighbors for clustering (default: 30)",
    )
    parser.add_argument(
        "--umap-2d-neighbors", type=int, default=15,
        help="UMAP n_neighbors for 2D visualization (default: 15)",
    )
    parser.add_argument(
        "--min-cluster-size", type=int, default=40,
        help="HDBSCAN min_cluster_size (default: 40)",
    )
    parser.add_argument(
        "--min-samples", type=int, default=10,
        help="HDBSCAN min_samples (default: 10)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for UMAP (default: 42)",
    )
    parser.add_argument(
        "--out-prefix", type=Path,
        default=Path("data/processed/eligible"),
        help="Output file prefix (default: data/processed/eligible)",
    )
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    log.info("Reading consolidated decisions: %s", args.decisions_parquet)
    dec = pd.read_parquet(args.decisions_parquet)
    dec["record_id"] = dec.record_id.astype(str)
    advancing = dec[dec.final_decision.isin(args.decisions)][
        ["record_id", "final_decision"]
    ].copy()
    log.info("  %d records advancing (%s)", len(advancing),
             "+".join(args.decisions))

    log.info("Retrieving embeddings from Qdrant collection %s...",
             args.collection)
    points = _scroll_all_points(args.collection)
    log.info("  %d points in collection", len(points))

    advancing_ids = set(advancing.record_id.tolist())
    filtered = [p for p in points if str(p["id"]) in advancing_ids]
    log.info("  %d points intersect with the advancing set", len(filtered))

    vectors = np.array([p["vector"] for p in filtered], dtype=np.float32)
    rids = [str(p["id"]) for p in filtered]
    log.info("  Matrix shape: %s", vectors.shape)

    # Persist raw vectors so downstream re-runs of UMAP/HDBSCAN can
    # skip the Qdrant scroll.
    args.out_prefix.parent.mkdir(parents=True, exist_ok=True)
    np.save(f"{args.out_prefix}_vectors.npy", vectors)
    Path(f"{args.out_prefix}_record_ids.txt").write_text("\n".join(rids))
    log.info("  saved → %s_vectors.npy + %s_record_ids.txt",
             args.out_prefix, args.out_prefix)

    # Lazy-import UMAP/HDBSCAN: optional dependencies, may not be in
    # a minimal install.
    import umap
    import hdbscan

    log.info("UMAP %dD for clustering (n_neighbors=%d, min_dist=0)...",
             args.umap_clust_dim, args.umap_clust_neighbors)
    um_clust = umap.UMAP(
        n_components=args.umap_clust_dim,
        n_neighbors=args.umap_clust_neighbors,
        min_dist=0.0,
        metric="cosine",
        random_state=args.seed,
    )
    emb_clust = um_clust.fit_transform(vectors)

    log.info("HDBSCAN (min_cluster_size=%d, min_samples=%d)...",
             args.min_cluster_size, args.min_samples)
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=args.min_cluster_size,
        min_samples=args.min_samples,
        metric="euclidean",
    )
    labels = clusterer.fit_predict(emb_clust)
    n_clusters = int(labels.max() + 1)
    n_noise = int((labels == -1).sum())
    log.info("  clusters=%d  noise=%d (%.1f%%)",
             n_clusters, n_noise, n_noise / len(labels) * 100)

    log.info("UMAP 2D for visualization (n_neighbors=%d)...",
             args.umap_2d_neighbors)
    um_2d = umap.UMAP(
        n_components=2,
        n_neighbors=args.umap_2d_neighbors,
        min_dist=0.1,
        metric="cosine",
        random_state=args.seed,
    )
    emb2d = um_2d.fit_transform(vectors)

    np.save(f"{args.out_prefix}_umap2d.npy", emb2d)
    np.save(f"{args.out_prefix}_clusters.npy", labels)
    log.info("  saved → %s_umap2d.npy + %s_clusters.npy",
             args.out_prefix, args.out_prefix)

    log.info("Joining with canonical for title/year...")
    rid_to_title, rid_to_year = _build_title_index(args.canonical_parquet)
    out = pd.DataFrame({
        "record_id": rids,
        "cluster": labels,
        "title": [rid_to_title.get(r, "") for r in rids],
        "year": [rid_to_year.get(r) for r in rids],
    })
    out_assign = Path(f"{args.out_prefix}_clusters_assignment.parquet")
    out.to_parquet(out_assign, index=False)
    log.info("  saved → %s", out_assign)

    # Per-cluster summary: size + median year. Labels are left to the
    # user (manual inspection of top tokens + representative titles).
    rows = []
    for c in sorted(out.cluster.unique()):
        sub = out[out.cluster == c]
        rows.append({
            "cluster": int(c),
            "n": int(len(sub)),
            "pct": round(len(sub) / len(out) * 100, 2),
            "year_median": int(sub.year.median()) if sub.year.notna().any() else None,
        })
    summary = pd.DataFrame(rows)
    out_summary = Path(f"{args.out_prefix}_clusters_summary.parquet")
    summary.to_parquet(out_summary, index=False)
    summary.to_csv(f"{args.out_prefix}_clusters_summary.csv", index=False)
    log.info("  saved → %s + .csv", out_summary)

    print()
    print("=" * 60)
    print(f" THEMATIC CLUSTERING — {n_clusters} clusters, {n_noise} noise")
    print("=" * 60)
    print(summary.sort_values("n", ascending=False).to_string(index=False))


if __name__ == "__main__":
    main()
