"""T5: import database exports, deduplicate, PRISMA counts per project."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from floccus_app import pipeline
from floccus_app.projects import create_project

SCOPUS = pd.DataFrame({
    "Title": ["River microplastics", "Lake sediments", "Fibres in streams"],
    "DOI": ["10.1/a", "10.1/b", None],
    "Abstract": ["abc", "def", "ghi"],
    "Year": [2020, 2021, 2022],
    "EID": ["s1", "s2", "s3"],
})
WOS = pd.DataFrame({
    "Article Title": ["River Microplastics", "Ocean plastics"],
    "DOI": ["10.1/A", "10.1/c"],
    "Abstract": ["abc", "xyz"],
    "Publication Year": [2020, 2019],
    "UT (Unique ID)": ["w1", "w2"],
})


@pytest.fixture
def project(tmp_path):
    p = create_project(tmp_path, "demo")
    pipeline.add_export(p, "scopus", "scopus_export.csv", SCOPUS.to_csv(index=False).encode())
    buf = tmp_path / "wos.xlsx"
    WOS.to_excel(buf, index=False)
    pipeline.add_export(p, "wos", "savedrecs.xlsx", buf.read_bytes())
    return p


def test_exports_are_stored_per_source(project):
    assert pipeline.list_exports(project) == {"scopus": ["scopus_export.csv"], "wos": ["savedrecs.xlsx"]}


@pytest.mark.parametrize(("source", "name", "message"), [
    ("nope", "a.csv", "unknown source 'nope'"),
    ("scopus", "../evil.csv", "invalid file name '../evil.csv'"),
    ("scopus", "data.exe", "scopus exports must be .csv"),
])
def test_bad_exports_are_rejected_and_nothing_is_written(tmp_path, source, name, message):
    p = create_project(tmp_path, "demo")
    with pytest.raises(pipeline.PipelineError, match=message):
        pipeline.add_export(p, source, name, b"x")
    assert list((p / "data" / "raw").rglob("*")) == []


def test_import_and_deduplicate_write_prisma_counts(project):
    imported = pipeline.import_records(project)
    assert imported == {"scopus": 3, "wos": 2}
    stats = pipeline.deduplicate(project)
    assert stats["unique"] == 4  # "River microplastics" is in both databases (same DOI)
    counts = json.loads((project / "results" / "latest" / "prisma_counts.json").read_text())
    assert counts["identification"] == {"by_source": {"scopus": 3, "wos": 2}, "total": 5}
    assert counts["deduplication"]["after_round2_ieee_merges"] == 4
    assert counts["deduplication"]["removed"] == 1
    assert sum(counts["deduplicated_by_source"].values()) == 4
    assert len(pd.read_parquet(project / "data" / "processed" / "unique.parquet")) == 4


def test_import_without_exports_explains_what_to_do(tmp_path):
    p = create_project(tmp_path, "demo")
    with pytest.raises(pipeline.PipelineError, match="no exports yet"):
        pipeline.import_records(p)


def test_deduplicate_before_import_explains_order(tmp_path):
    p = create_project(tmp_path, "demo")
    with pytest.raises(pipeline.PipelineError, match="import the exports first"):
        pipeline.deduplicate(p)
