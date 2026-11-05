"""Number formatting of scripts/render_sankey_pipeline.py (MDPI style)."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from render_sankey_pipeline import fmt  # noqa: E402


def test_fmt_leaves_four_digit_numbers_without_separator():
    assert fmt(545) == "545"
    assert fmt(4883) == "4883"
    assert fmt(9999) == "9999"


def test_fmt_uses_comma_from_five_digits():
    assert fmt(10000) == "10,000"
    assert fmt(16965) == "16,965"
    assert fmt(1242000) == "1,242,000"
