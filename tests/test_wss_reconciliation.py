"""Pure helpers of scripts/wss_reconciliation.py (no data files)."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from wss_reconciliation import reconcile, wss  # noqa: E402


def test_wss_is_removed_minus_missed_share():
    assert wss(0.6, 0.9) == pytest.approx(0.5)


def test_reconcile_difference_is_the_workload_gap():
    rw = {"recall": 0.6, "burden_removed": 0.55, "wss": 0.55 - 0.4}
    ci = {"recall": [0.4, 0.8]}
    r = reconcile(rw, ci, n_advanced=40, n=100)  # exact share removed 0.60
    assert r["wss_exact_workload"] == pytest.approx(0.20)
    assert r["difference_exact_minus_reweighted"] == pytest.approx(0.05)
    assert r["wss_exact_workload_ci95"] == pytest.approx([0.0, 0.4])
    assert r["identity_check"]


def test_reconcile_agrees_when_workload_is_exact():
    rw = {"recall": 0.5, "burden_removed": 0.7, "wss": 0.2}
    r = reconcile(rw, {"recall": [0.3, 0.7]}, n_advanced=30, n=100)
    assert r["difference_exact_minus_reweighted"] == pytest.approx(0.0)
