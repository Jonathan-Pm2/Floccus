"""Pure helpers of scripts/assign_ieee_clusters.py (no data files, no server)."""
from __future__ import annotations

import sys
import uuid
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from assign_ieee_clusters import agreement, align_labels, record_id  # noqa: E402


def test_align_labels_majority_and_noise():
    refit = np.array([0, 0, 0, 1, 1, 2, 2, -1])
    ref = np.array([5, 5, 3, 0, 0, -1, -1, 4])
    assert align_labels(refit, ref) == {-1: -1, 0: 5, 1: 0, 2: -1}


def test_agreement_rates():
    pred = np.array([1, 2, -1, -1])
    orig = np.array([1, 3, -1, 2])
    res = agreement(pred, orig)
    assert res["exact"] == 0.5
    assert res["non_noise"] == 1 / 3
    assert res["predicted_noise"] == 0.5


def test_record_id_falls_back_to_title_norm():
    assert record_id("ieee", "123", "x") == str(uuid.uuid5(uuid.NAMESPACE_URL, "ieee:123"))
    assert record_id("ieee", None, "a" * 100) == str(uuid.uuid5(uuid.NAMESPACE_URL, "ieee:" + "a" * 80))
