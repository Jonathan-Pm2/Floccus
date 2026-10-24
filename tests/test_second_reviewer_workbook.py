"""Text clean-up for the second-reviewer workbook (no data files are read)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from make_second_reviewer_workbook import _join_keywords, fix_mojibake  # noqa: E402


@pytest.mark.parametrize("broken,clean", [
    ("Se evaluÃ³ el desempeÃ±o", "Se evaluó el desempeño"),
    ("MÃ©xico — conclusiÃ³n", "México — conclusión"),          # mixed with non-Latin-1 text
    ("Keywords:Â Biogas", "Keywords: Biogas"),                  # broken non-breaking space
    ("ya está bien: acción", "ya está bien: acción"),           # correct text is untouched
])
def test_fix_mojibake(broken, clean):
    assert fix_mojibake(broken) == clean


def test_fix_mojibake_passes_non_text_through():
    assert fix_mojibake(None) is None


@pytest.mark.parametrize("value,expected", [
    (np.array(["biofloc", "Monod"]), "biofloc; Monod"),
    (np.array([]), ""),
    (None, ""),
    ("already; joined", "already; joined"),
])
def test_join_keywords(value, expected):
    assert _join_keywords(value) == expected
