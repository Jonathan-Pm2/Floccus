"""Pure helpers of scripts/supplementary_search_eaasd.py (no data files, no server)."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from supplementary_search_eaasd import (  # noqa: E402
    cluster_label,
    layer1_overlap,
    layer3_resolve,
    redact_error,
)

from biofloc_review.dedup.deterministic import cluster_deterministic  # noqa: E402
from biofloc_review.dedup.embeddings import _record_to_point_id  # noqa: E402
from biofloc_review.schema import Corpus, PaperRecord, Source  # noqa: E402


def _rec(src: Source, sid: str, title: str, doi: str | None = None, year: int = 2020, abstract=None):
    return PaperRecord(title=title, title_norm=title.lower(), doi=doi, year=year, source=src,
                       source_id=sid, corpus=Corpus.BIOFLOC, abstract=abstract)


def test_layer1_overlap_corpus_internal_and_survivor():
    corpus = [_rec(Source.SCOPUS, "s1", "biofloc a", doi="10.1/a"),
              _rec(Source.WOS, "w1", "biofloc b")]
    new = [_rec(Source.PROQUEST, "1", "Biofloc A other title", doi="10.1/a"),   # DOI match
           _rec(Source.PROQUEST, "2", "biofloc b"),                             # title+year match
           _rec(Source.PROQUEST, "3", "biofloc c", abstract="x"),               # internal pair,
           _rec(Source.PROQUEST, "4", "biofloc c"),                             # 3 is more complete
           _rec(Source.PROQUEST, "5", "biofloc b", year=2021)]                  # different year: survives
    res = layer1_overlap(cluster_deterministic(corpus + new))
    rid = [_record_to_point_id(r) for r in new]
    assert res[rid[0]]["status"] == "L1_corpus" and res[rid[0]]["via"] == "doi"
    assert res[rid[0]]["match"] == _record_to_point_id(corpus[0])
    assert res[rid[1]]["status"] == "L1_corpus" and res[rid[1]]["via"] == "title+year"
    assert res[rid[2]]["status"] == "survivor"
    assert res[rid[3]] == {"status": "L1_internal", "match": rid[2], "match_title": "biofloc c",
                           "via": "title+year"}
    assert res[rid[4]]["status"] == "survivor"
    assert len(res) == 5


def test_layer3_resolve_corpus_and_internal_components():
    survivors = ["a", "b", "c", "d", "e"]
    dup = [("a", "corpus:X"), ("b", "a"), ("c", "d")]
    res = layer3_resolve(survivors, dup, rank={"d": (5,), "c": (1,)})
    assert res["a"] == {"status": "L3_corpus", "match": "X"}
    assert res["b"] == {"status": "L3_corpus", "match": "X"}   # transitively through a
    assert res["c"] == {"status": "L3_internal", "match": "d"}  # d kept (higher rank)
    assert "d" not in res and "e" not in res


def test_redact_error_and_cluster_label():
    assert redact_error("LLM error: ConnectionError: HTTPConnectionPool(host='h', port=1)") == \
        "LLM error: ConnectionError (message redacted)"
    assert redact_error("fine reasoning") == "fine reasoning"
    assert redact_error(None) is None
    assert cluster_label(3) == "C03" and cluster_label(15) == "C15" and cluster_label(-1) == "noise"
