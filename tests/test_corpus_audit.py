"""Pure helpers of the corpus-audit scripts (dedup, cluster labels, topics, SciELO).

The scripts themselves read data/processed; these tests exercise only the
small functions they are built from, on synthetic inputs.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import cluster_labels  # noqa: E402
import cluster_topic_crosscheck  # noqa: E402
import dedup_audit  # noqa: E402
import scielo_language_audit  # noqa: E402


# ---------------------------------------------------------------- dedup: union-find
def test_merge_components_triad_and_two_pairs():
    # a-b, b-c form one triad (two pairs, transitively closed); d-e and f-g are pairs
    pairs = [("a", "b"), ("b", "c"), ("d", "e"), ("f", "g")]
    comps = dedup_audit.merge_components(pairs)
    assert sorted(sorted(c) for c in comps) == [["a", "b", "c"], ["d", "e"], ["f", "g"]]
    assert dedup_audit.net_merges(comps) == 4


def test_merge_components_redundant_edge_does_not_add_merges():
    # the third edge closes a triangle: still one triad, two merges
    comps = dedup_audit.merge_components([("a", "b"), ("b", "c"), ("a", "c")])
    assert len(comps) == 1
    assert dedup_audit.net_merges(comps) == 2


def test_merge_components_empty():
    assert dedup_audit.merge_components([]) == []
    assert dedup_audit.net_merges([]) == 0


# ---------------------------------------------------------------- dedup: cosine bands
def test_cosine_band_split():
    scores = [0.93, 0.96, 0.969, 0.97, 0.975, 0.98, 0.999, 1.000001]
    verdicts = ["DUPLICATE", "NOT_DUPLICATE", "DUPLICATE", "DUPLICATE",
                "UNCLEAR", "DUPLICATE", "NOT_DUPLICATE", "DUPLICATE"]
    bands = dedup_audit.cosine_bands(scores, verdicts, low=0.97, high=0.98)
    assert bands["below_0.97"] == {"n": 3, "duplicate": 2}
    assert bands["0.97_to_0.98"] == {"n": 2, "duplicate": 1}
    assert bands["at_least_0.98"] == {"n": 3, "duplicate": 2}


@pytest.mark.parametrize("raw, expected", [
    ("https://doi.org/10.1/ABC", "10.1/abc"),
    ("http://dx.doi.org/10.1/abc ", "10.1/abc"),
    ("doi:10.1/abc", "10.1/abc"),
    (None, ""),
    (float("nan"), ""),
])
def test_norm_doi(raw, expected):
    assert dedup_audit.norm_doi(raw) == expected


# ---------------------------------------------------------------- cluster labels
def test_title_terms_drops_stop_words_generic_vocabulary_and_short_words():
    titles = ["The effect of biofloc on shrimp growth using a new model",
              "Biofloc in aquaculture: a pilot scale study of N2 and pH in 2020"]
    terms = cluster_labels.title_terms(titles, k=10)
    for dropped in ("the", "effect", "using", "new", "model", "pilot", "scale", "study",
                    "and", "of", "on", "in", "a", "pH".lower(), "n2", "2020"):
        assert dropped not in terms
    assert terms[0] == "biofloc"
    assert set(terms) == {"biofloc", "shrimp", "growth", "aquaculture"}


def test_title_terms_frequency_order_with_first_occurrence_tie_break():
    titles = ["zeta alpha", "alpha beta", "zeta"]
    # zeta=2, alpha=2, beta=1: tie between zeta and alpha resolved by first appearance
    assert cluster_labels.title_terms(titles, k=3) == ["zeta", "alpha", "beta"]
    assert cluster_labels.title_terms(titles, k=1) == ["zeta"]


def test_title_terms_handles_missing_titles():
    assert cluster_labels.title_terms([None, "", "Sludge sludge"], k=5) == ["sludge"]


def test_label_match_counts_overlap_and_exact_lists():
    m = cluster_labels.label_match(["a", "b", "c"], ["a", "c", "b"])
    assert m == {"overlap": 3, "exact": False}
    assert cluster_labels.label_match(["a", "b"], ["a", "b"])["exact"] is True


# ---------------------------------------------------------------- topic cross-check
def test_purity_tiny_contingency():
    # cluster 0: x,x,y -> 2 ; cluster 1: z,z -> 2 ; cluster 2: y -> 1  => 5/6
    clusters = [0, 0, 0, 1, 1, 2]
    topics = ["x", "x", "y", "z", "z", "y"]
    assert cluster_topic_crosscheck.purity(clusters, topics) == pytest.approx(5 / 6)


def test_purity_perfect_and_single_cluster():
    assert cluster_topic_crosscheck.purity([0, 0, 1], ["a", "a", "b"]) == 1.0
    assert cluster_topic_crosscheck.purity([0, 0, 0, 0], ["a", "b", "b", "c"]) == 0.5


def test_topic_list_accepts_arrays_dicts_and_missing():
    assert cluster_topic_crosscheck.topic_list(None) == []
    assert cluster_topic_crosscheck.topic_list(float("nan")) == []
    assert cluster_topic_crosscheck.topic_list(["A", {"display_name": "B"}, ""]) == ["A", "B"]


# ---------------------------------------------------------------- SciELO / language
@pytest.mark.parametrize("field, lang, expected", [
    ("Spanish", "Spanish", True),
    ("Spanish, English", "Spanish", True),
    ("English, Spanish", "Spanish", True),
    ("Portuguese, English", "Spanish", False),
    ("English", "English", True),
    (None, "Spanish", False),
    ("nan", "Spanish", False),
])
def test_declares_language(field, lang, expected):
    assert scielo_language_audit.declares(field, lang) is expected


def test_count_language_includes_bilingual_records():
    langs = ["Spanish", "Spanish, English", "English, Spanish", "English", None, "Portuguese, English"]
    assert scielo_language_audit.count_language(langs, "Spanish") == 3
    assert scielo_language_audit.count_language(langs, "Portuguese") == 1
    assert scielo_language_audit.count_language(langs, "English") == 4
