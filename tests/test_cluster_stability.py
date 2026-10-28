"""Pure helpers of scripts/cluster_stability.py (no data files, no UMAP fit)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from cluster_stability import best_jaccard, compare, summarise  # noqa: E402


def test_best_jaccard_ignores_noise_and_relabelling():
    ref = np.array([0, 0, 0, 1, 1, -1])
    run = np.array([5, 5, -1, 7, 7, 7])
    bj = best_jaccard(ref, run)
    assert bj[0] == pytest.approx(2 / 3)
    assert bj[1] == pytest.approx(2 / 3)


def test_best_jaccard_zero_when_cluster_lost():
    assert best_jaccard(np.array([0, 0, 1]), np.array([-1, -1, 2])) == {0: 0.0, 1: 1.0}


def test_compare_identical_partition_up_to_labels():
    ref = np.array([0, 0, 1, 1, -1, -1])
    res = compare(ref, np.array([3, 3, 2, 2, -1, -1]))
    assert res["k"] == 2
    assert res["noise_fraction"] == pytest.approx(1 / 3)
    assert res["ari_all"] == pytest.approx(1.0)
    assert res["ari_clustered_both"] == pytest.approx(1.0)
    assert res["n_clustered_both"] == 4


def test_summarise():
    s = summarise([1.0, 0.0, 0.5])
    assert (s["median"], s["min"], s["max"]) == (0.5, 0.0, 1.0)
