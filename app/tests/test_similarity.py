"""T15: embedded Qdrant similarity dedup + LLM judge + thematic clusters (S13)."""

from __future__ import annotations

import hashlib
import json

import numpy as np
import pandas as pd
import pytest
from biofloc_review.dedup import embeddings as published_embeddings

from floccus_app import cli, pipeline, results, similarity
from floccus_app.projects import create_project

DIM = 4096


def fake_vector(text: str) -> list[float]:
    """Same abstract -> same vector; 'groupA'/'groupB' texts form two tight groups."""
    abstract = text[text.index("group"):]  # titles never contain "group"
    seed = int(hashlib.sha1(abstract.encode()).hexdigest()[:8], 16)
    rng = np.random.default_rng(seed)
    base = np.zeros(DIM)
    base[0 if "groupA" in text else 1] = 30.0
    v = base + rng.normal(0, 0.3, DIM)
    return list(v / np.linalg.norm(v))


@pytest.fixture(autouse=True)
def _fake_embeddings(monkeypatch):
    monkeypatch.setattr(published_embeddings, "embed_batch",
                        lambda texts, model=None, **kw: [fake_vector(t) for t in texts])


def judge_duplicate(a, b, score, model=None):
    from biofloc_review.dedup.llm_judge import Verdict

    return Verdict(verdict="DUPLICATE", confidence=9, reasoning="same study", raw="")


NEAR_DUP = pd.DataFrame({
    "Title": ["Microplastics in rivers", "Microplastics in rivers: a field study", "Ocean gyres"],
    "DOI": ["10.1/a", "10.2/a-preprint", "10.1/c"],
    "Abstract": ["groupA same abstract about river plastics " * 20,
                 "groupA same abstract about river plastics " * 20,
                 "groupB different abstract about oceans " * 20],
    "Year": [2020, 2019, 2021],
    "EID": ["s1", "s2", "s3"],
})


def _project(tmp_path, df):
    p = create_project(tmp_path, "demo")
    pipeline.add_export(p, "scopus", "s.csv", df.to_csv(index=False).encode())
    pipeline.import_records(p)
    pipeline.deduplicate(p)
    return p


def test_similarity_dedup_merges_judged_duplicates_in_embedded_qdrant(tmp_path):
    p = _project(tmp_path, NEAR_DUP)
    stats = similarity.dedup_similar(p, judge=judge_duplicate)
    assert stats == {"input": 3, "pairs": 1, "duplicates": 1, "unique": 2}
    assert (p / "data" / "qdrant").is_dir()
    assert len(pd.read_parquet(p / "data" / "processed" / "unique.parquet")) == 2
    counts = pipeline.read_counts(p)
    assert counts["deduplication"]["after_layer1_deterministic"] == 3
    assert counts["deduplication"]["after_round2_ieee_merges"] == 2
    assert counts["deduplication"]["removed"] == 1
    judged = json.loads((p / "results" / "latest" / "near_duplicate_pairs.json").read_text())
    assert judged[0]["judge_verdict"] == "DUPLICATE" and judged[0]["score"] >= 0.93


def test_not_duplicate_verdict_keeps_both_records(tmp_path):
    from biofloc_review.dedup.llm_judge import Verdict

    p = _project(tmp_path, NEAR_DUP)
    stats = similarity.dedup_similar(p, judge=lambda a, b, s, model=None: Verdict("NOT_DUPLICATE", 8, "", ""))
    assert stats["unique"] == 3 and stats["duplicates"] == 0


def test_similarity_dedup_requires_deterministic_dedup_first(tmp_path):
    p = create_project(tmp_path, "demo")
    with pytest.raises(pipeline.PipelineError, match="deduplicate first"):
        similarity.dedup_similar(p, judge=judge_duplicate)


def _screened_groups(tmp_path):
    n = 30
    df = pd.DataFrame({
        "Title": [f"A{i}" for i in range(n)] + [f"B{i}" for i in range(n)],
        "DOI": [f"10.1/a{i}" for i in range(n)] + [f"10.1/b{i}" for i in range(n)],
        "Abstract": [f"groupA river study number {i} " * 30 for i in range(n)]
        + [f"groupB ocean study number {i} " * 30 for i in range(n)],
        "Year": [2020] * (2 * n),
        "EID": [f"e{i}" for i in range(2 * n)],
    })
    p = _project(tmp_path, df)
    pipeline.screen(p, layer=1, model="m",
                    chat=lambda m, model: "ANSWER: YES" if "CRITERION I1" in m[0]["content"] else "ANSWER: NO")
    pipeline.consolidate(p)
    return p


def test_clusters_group_eligible_records_and_feed_the_charts(tmp_path):
    p = _screened_groups(tmp_path)
    info = similarity.thematic_clusters(p)
    assert info["records"] == 60
    assert info["params"]["umap"]["n_components"] == 10 and info["params"]["umap"]["random_state"] == 42
    out = pd.read_csv(p / "results" / "latest" / "eligible_clusters.csv")
    assert len(out) == 60 and {"record_id", "cluster", "cluster_label", "umap_x", "umap_y"} <= set(out.columns)
    a = out[out["title"].str.startswith("A")]["cluster"]
    b = out[out["title"].str.startswith("B")]["cluster"]
    assert a.mode()[0] != b.mode()[0]  # the two topics land in different clusters
    r = results.load_results(p)
    assert r.eligible is not None
    assert results.fig_clusters(r.eligible, "es").data[0].type == "bar"
    assert results.fig_umap(r.eligible, "es").data[0].type == "scattergl"


def test_small_projects_scale_cluster_sizes_and_large_use_published():
    assert similarity.cluster_params(5000)["hdbscan"] == {"min_cluster_size": 40, "min_samples": 10}
    small = similarity.cluster_params(60)["hdbscan"]
    assert small == {"min_cluster_size": 6, "min_samples": 6}


def test_cli_runs_the_new_steps(tmp_path, capsys, monkeypatch):
    p = _project(tmp_path, NEAR_DUP)
    monkeypatch.setattr(similarity, "_default_judge", lambda: judge_duplicate)
    assert cli.main(["--home", str(tmp_path), "run", "demo", "dedup-ai"]) == 0
    assert "2 unique records after AI deduplication (1 near duplicates merged)" in capsys.readouterr().out
