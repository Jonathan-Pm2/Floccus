"""Sensitivity-mode decision rule applied to stored answers (no data files read)."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from operating_modes_recall import sensitivity_decision  # noqa: E402


def frame(rows):
    return pd.DataFrame(rows, columns=["final_decision", "E1_answer", "E2_answer", "E3_answer",
                                       "E4_answer", "E5_answer", "E2_v2"])


def test_rule():
    d = frame([
        ("EXCLUDE", "NO", "YES", "NO", "NO", None, "NO"),        # excluded by E2 alone, rescued
        ("EXCLUDE", "NO", "YES", "NO", "NO", None, "UNCLEAR"),   # rescued: corrected E2 is not YES
        ("EXCLUDE", "NO", "YES", "NO", "NO", None, "YES"),       # corrected E2 confirms exclusion
        ("EXCLUDE", "YES", "YES", "NO", "NO", None, "NO"),       # another criterion also excludes
        ("INCLUDE", "NO", "NO", "NO", "NO", None, "YES"),        # advancing, corrected E2 excludes
        ("HUMAN_REVIEW", "NO", "NO", "NO", "NO", None, "NO"),    # advancing, unchanged
    ])
    assert sensitivity_decision(d).tolist() == [
        "HUMAN_REVIEW", "HUMAN_REVIEW", "EXCLUDE", "EXCLUDE", "EXCLUDE", "HUMAN_REVIEW"]
