"""Pure helpers of the Crossref citation-link analysis (no network, no data files)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from bridge_citations_crossref import clean_doi, link_matrix, pair_density, reference_dois  # noqa: E402


def test_clean_doi():
    assert clean_doi("https://doi.org/10.1016/J.AQUA.2020.1") == "10.1016/j.aqua.2020.1"
    assert clean_doi("  10.3390/w12010001 ") == "10.3390/w12010001"
    assert clean_doi(None) is None and clean_doi("") is None and clean_doi("not a doi") is None


def test_reference_dois_skips_references_without_doi_and_dedups():
    msg = {"reference": [{"DOI": "10.1/A"}, {"unstructured": "Smith 1990"}, {"DOI": "10.1/a"}, {"DOI": "10.2/b"}]}
    assert reference_dois(msg) == ["10.1/a", "10.2/b"]
    assert reference_dois({}) == []


def test_link_matrix_and_density():
    clusters = [3, 15]
    cluster_of_doi = {"d3a": 3, "d3b": 3, "d15": 15}
    # one C03 record cites a C15 record and a C03 record; the C15 record cites nothing in the corpus
    L = link_matrix([3, 15], [["d15", "d3b", "outside"], ["elsewhere"]], cluster_of_doi, clusters)
    assert L.tolist() == [[1, 1], [0, 0]]
    n_cite, n_doi = np.array([1, 1]), np.array([2, 1])
    # possible citing-cited pairs: 1*1 (C03 citers -> C15) + 1*2 (C15 citers -> C03) = 3
    assert pair_density(L, 0, 1, n_cite, n_doi) == 1 / 3
