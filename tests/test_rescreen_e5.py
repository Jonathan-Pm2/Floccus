"""Pure parts of scripts/rescreen_e5.py: affected-set selection, Layer-1
recomputation with E5 dropped, consolidation and sensitivity-mode mapping.
No data files, no server."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from rescreen_e5 import (  # noqa: E402
    CRITERION_IDS,
    consolidate_affected,
    recompute_layer1,
    select_affected,
    select_e2_rescuable,
    sensitivity_decisions,
)

ALL_IDS = CRITERION_IDS + ("E5",)


def _row(**over) -> dict:
    base = {f"{c}_answer": "YES" for c in ("I1", "I2", "I3", "I4")}
    base |= {f"{c}_answer": "NO" for c in ("E1", "E2", "E3", "E4", "E5", "D1", "D2", "D3")}
    base |= {f"{c}_confidence": 10 for c in ALL_IDS}
    base |= {f"{c}_evidence": "" for c in ALL_IDS}
    base |= {f"{c}_reasoning": "" for c in ALL_IDS}
    base |= {f"{c}_via": "llm" for c in ALL_IDS}
    base |= over
    return base


def _frame(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    df["record_id"] = [f"r{i}" for i in range(len(df))]
    return df


def test_select_affected_only_e5_without_other_exclusion():
    df = _frame([
        _row(E5_answer="YES"),                       # E5 only -> affected
        _row(E5_answer="YES", E4_answer="YES"),      # E5 + E4 -> not affected
        _row(E5_answer="YES", E2_answer="UNCLEAR"),  # E5 only (E2 unclear) -> affected
        _row(E5_answer="NO"),                        # not flagged
    ])
    affected = select_affected(df)
    assert affected.tolist() == [True, False, True, False]


def test_select_e2_rescuable_is_e5_plus_e2_only():
    df = _frame([
        _row(E5_answer="YES", E2_answer="YES"),                   # rescuable in sensitivity mode
        _row(E5_answer="YES", E2_answer="YES", E4_answer="YES"),  # another E fires -> no
        _row(E5_answer="YES"),                                    # E5 only -> handled as affected, not here
        _row(E5_answer="NO", E2_answer="YES"),                    # already covered by the published run
    ])
    assert select_e2_rescuable(df).tolist() == [True, False, False, False]


def test_recompute_layer1_drops_e5():
    include = recompute_layer1(pd.Series(_row(E5_answer="YES")))
    assert include.decision == "INCLUDE"

    hr_d2 = recompute_layer1(pd.Series(_row(E5_answer="YES", D2_answer="YES")))
    assert hr_d2.decision == "HUMAN_REVIEW" and hr_d2.triggered_criteria == ["D2"]

    hr_i2 = recompute_layer1(pd.Series(_row(E5_answer="YES", I2_answer="UNCLEAR")))
    assert hr_i2.decision == "HUMAN_REVIEW" and "I2" in hr_i2.triggered_criteria

    hr_e = recompute_layer1(pd.Series(_row(E5_answer="YES", E2_answer="UNCLEAR")))
    assert hr_e.decision == "HUMAN_REVIEW" and hr_e.triggered_criteria == ["E2"]

    excl = recompute_layer1(pd.Series(_row(E5_answer="YES", E4_answer="YES")))
    assert excl.decision == "EXCLUDE" and "E5" not in excl.triggered_criteria


def test_consolidate_affected_follows_pipeline_rule():
    l1 = pd.DataFrame({"record_id": ["a", "b", "c"],
                       "l1_new_decision": ["INCLUDE", "HUMAN_REVIEW", "HUMAN_REVIEW"]})
    l2 = {"b": ("EXCLUDE", "exclusión por E2")}
    out = consolidate_affected(l1, l2)
    assert out.set_index("record_id").final_decision_new.to_dict() == {
        "a": "INCLUDE", "b": "EXCLUDE", "c": "HUMAN_REVIEW"}
    assert out.set_index("record_id").final_decision_source_new["c"].endswith("L2 pendiente")


def test_sensitivity_decisions_mirror_published_rule():
    df = pd.DataFrame({
        "record_id": ["a", "b", "c", "d", "e"],
        "final_decision_new": ["INCLUDE", "HUMAN_REVIEW", "EXCLUDE", "EXCLUDE", "INCLUDE"],
        "e2_rescuable": [False, False, True, True, False],
        "E2_v2": ["YES", "NO", "NO", "YES", None],
    })
    sens = sensitivity_decisions(df)
    assert sens.tolist() == ["EXCLUDE", "HUMAN_REVIEW", "HUMAN_REVIEW", "EXCLUDE", "INCLUDE"]
