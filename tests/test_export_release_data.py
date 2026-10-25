"""Pure helpers of scripts/export_release_data.py (no data files read)."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from export_release_data import REASON_CODE, decision_reason, reason_code  # noqa: E402


@pytest.mark.parametrize("trace, expected", [
    ("Layer 1 INCLUDE (no re-evaluado en L2)", ("L1", "all_inclusion_met")),
    ("exclusión por E2, E4", ("L1", "excluded_by:E2,E4")),
    ("Layer 2: exclusión por E1", ("L2", "excluded_by:E1")),
    ("Layer 2: inclusión incompleta: I1, I2 ≠ YES", ("L2", "inclusion_not_met:I1,I2")),
    ("Layer 2: derivación por D1, D3", ("L2", "routed_by:D1,D3")),
    ("Layer 2: exclusión con incertidumbre: E2 = UNCLEAR", ("L2", "exclusion_unclear:E2")),
    ("Layer 2: todos I cumplidos, ningún E ni D fired", ("L2", "all_inclusion_met")),
])
def test_decision_reason_maps_every_trace_form(trace, expected):
    assert decision_reason(trace) == expected


def test_decision_reason_rejects_unknown_trace():
    with pytest.raises(SystemExit):
        decision_reason("something else")


def test_reason_code_maps_workbook_reasons_and_blanks():
    assert reason_code("No cumple I1 (objeto)") == "not_I1"
    assert reason_code("  E2 Sin modelación formal ") == "E2"
    assert reason_code(None) is None
    assert reason_code(float("nan")) is None
    assert reason_code("") is None
    assert len(set(REASON_CODE.values())) == len(REASON_CODE)


def test_reason_code_rejects_free_text():
    with pytest.raises(SystemExit):
        reason_code("a comment typed by the reviewer")
