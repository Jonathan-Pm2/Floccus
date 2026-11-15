"""T11: human review of uncertain records and agreement (kappa)."""

from __future__ import annotations

import pytest

from floccus_app import review
from test_results_latest import _finished_project  # noqa: E402


def test_cohens_kappa_known_values():
    assert review.cohens_kappa(["a", "b", "a", "b"], ["a", "b", "a", "b"]) == pytest.approx(1.0)
    # 2x2 table [[20,5],[10,15]] -> po=0.7, pe=0.5 -> kappa 0.4
    a = ["y"] * 25 + ["n"] * 25
    b = ["y"] * 20 + ["n"] * 5 + ["y"] * 10 + ["n"] * 15
    assert review.cohens_kappa(a, b) == pytest.approx(0.4)
    assert review.cohens_kappa([], []) is None


def test_uncertain_queue_lists_human_review_records(tmp_path):
    p = _finished_project(tmp_path)
    q = review.queue(p, "uncertain")
    assert len(q) == 1 and set(q["final_decision"]) == {"HUMAN_REVIEW"}


def test_validation_sample_is_reproducible_and_bounded(tmp_path):
    p = _finished_project(tmp_path)
    s1 = review.queue(p, "validation", size=5)
    s2 = review.queue(p, "validation", size=5)
    assert list(s1["record_id"]) == list(s2["record_id"]) and len(s1) == 5
    assert len(review.queue(p, "validation", size=500)) == 7


def test_judgements_are_saved_updated_and_validated(tmp_path):
    p = _finished_project(tmp_path)
    rid = review.queue(p, "uncertain")["record_id"].iloc[0]
    review.save_judgement(p, rid, "include", "fits", "uncertain")
    review.save_judgement(p, rid, "exclude", "changed my mind", "uncertain")
    j = review.judgements(p)
    assert len(j) == 1 and j.iloc[0]["label"] == "exclude" and j.iloc[0]["note"] == "changed my mind"
    with pytest.raises(ValueError, match="label must be include or exclude"):
        review.save_judgement(p, rid, "maybe", "", "uncertain")
    assert len(review.judgements(p)) == 1


def test_agreement_on_validation_sample(tmp_path):
    p = _finished_project(tmp_path)
    sample = review.queue(p, "validation", size=7)
    for _, row in sample.iterrows():
        label = "exclude" if row["final_decision"] == "EXCLUDE" else "include"
        review.save_judgement(p, row["record_id"], label, "", "validation")
    a = review.agreement(p)
    assert a["n"] == 7 and a["percent_agreement"] == pytest.approx(100.0) and a["kappa"] == pytest.approx(1.0)


def test_review_tab_saves_a_decision_from_the_interface(tmp_path, monkeypatch):
    from pathlib import Path

    from streamlit.testing.v1 import AppTest

    from floccus_app.i18n import t

    p = _finished_project(tmp_path)
    monkeypatch.setenv("FLOCCUS_HOME", str(tmp_path))
    app = Path(__file__).resolve().parents[1] / "src" / "floccus_app" / "ui" / "app.py"
    at = AppTest.from_file(str(app), default_timeout=60).run()
    assert not at.exception
    next(b for b in at.button if b.label == t("rv_include", "es")).click().run()
    assert not at.exception
    j = review.judgements(p)
    assert list(j["label"]) == ["include"] and list(j["kind"]) == ["uncertain"]
