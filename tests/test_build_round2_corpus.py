"""Pure merge / retention / language helpers of scripts/build_round2_corpus.py (no data files)."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from build_round2_corpus import (  # noqa: E402
    language_table,
    language_tokens,
    merged_groups,
    retain,
    retention_key,
    verified_pairs,
)
from ieee_integrity_check import DSU  # noqa: E402


def rec(rid, source="wos", sample=False, clustered=False, doi=False):
    return {"record_id": rid, "source": source, "in_sample": sample, "in_clustered": clustered, "has_doi": doi}


def test_retention_priority_order():
    assert retention_key(rec("a", "ieee", True, True, True)) > retention_key(rec("b", "wos", False, True, True))
    # a sample member beats a clustered record with a DOI
    keep, drop = retain([rec("ieee1", "ieee", clustered=True, doi=True), rec("wos1", "wos", sample=True)])
    assert (keep, drop) == ("wos1", ["ieee1"])
    # both clustered: the DOI decides
    keep, _ = retain([rec("wos1", "wos", clustered=True), rec("ieee1", "ieee", clustered=True, doi=True)])
    assert keep == "ieee1"
    # both clustered, both with DOI: the IEEE record is kept
    keep, _ = retain([rec("scopus1", "scopus", clustered=True, doi=True), rec("ieee1", "ieee", clustered=True, doi=True)])
    assert keep == "ieee1"


def test_retain_tie_break_is_smallest_id_and_keeps_input_order_of_dropped():
    keep, drop = retain([rec("z"), rec("b"), rec("m")])
    assert keep == "b"
    assert drop == ["z", "m"]
    with pytest.raises(ValueError):
        retain([rec("only")])


def test_merged_groups_unions_pairs_on_top_of_archived_merges():
    dsu = DSU()
    for c in "abcdef":
        dsu.find(c)
    dsu.union("a", "b")                       # archived Layer-3 merge: one record survives (r_a)
    cluster_of = {"r_a": "a", "r_c": "c", "r_d": "d", "r_e": "e", "r_f": "f"}
    groups = merged_groups(cluster_of, dsu, [("b", "c"), ("d", "e")])
    assert groups == [["r_a", "r_c"], ["r_d", "r_e"]]
    assert dsu.find("c") != dsu.find("b"), "the archived DSU is not modified"


def test_verified_pairs_keep_only_duplicate_with_identical_title_or_doi():
    pairs = [{"a_cluster": "a", "b_cluster": "b", "same_doi_or_title": "title"},
             {"a_cluster": "c", "b_cluster": "d", "same_doi_or_title": "no"},
             {"a_cluster": "e", "b_cluster": "f", "same_doi_or_title": "doi"},
             {"a_cluster": "g", "b_cluster": "h", "same_doi_or_title": "title"}]
    verdicts = {"a|b": {"verdict": "DUPLICATE"}, "c|d": {"verdict": "DUPLICATE"},
                "e|f": {"verdict": "NOT_DUPLICATE"}, "g|h": {"verdict": "UNCLEAR"}}
    out = verified_pairs(pairs, verdicts)
    assert [(p["a_cluster"], p["verification"]) for p in out] == [("a", "title")]


def test_language_table_counts_bilingual_records_in_each_language():
    assert language_tokens("Spanish, English") == ["Spanish", "English"]
    assert language_tokens("undefined") == [] and language_tokens(float("nan")) == [] and language_tokens(None) == []
    t = language_table(pd.Series(["English", "Spanish, English", "nan", None, "undefined", "en", "German"]))
    assert t == {"English": 2, "Spanish": 1, "Portuguese": 0, "Chinese": 0, "Other": 2, "Not declared": 3}
