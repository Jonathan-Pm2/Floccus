"""The re-screen prompt must tolerate missing metadata fields.

After a left merge a missing abstract arrives as NaN, a truthy float, and
`(row.abstract or "(no abstract)")[:3000]` raised TypeError on it; 81
records of the sensitivity-mode re-screen failed that way.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import pilot_prompt_v2  # noqa: E402


class Row:
    title, abstract, keywords, venue = "A title", float("nan"), float("nan"), None


@pytest.fixture
def user_prompt(monkeypatch):
    seen = {}

    def fake_chat(msgs, model=None):
        seen["user"] = msgs[1]["content"]
        return "REASONING: has a model.\nANSWER: YES"

    monkeypatch.setattr(pilot_prompt_v2, "_ollama_chat", fake_chat)
    return seen


def test_nan_fields_do_not_crash_and_use_placeholder(user_prompt):
    answer, _ = pilot_prompt_v2.ask(Row(), "E2")
    assert answer == "NO"                       # positive YES is inverted
    assert "(no abstract)" in user_prompt["user"]
    assert "nan" not in user_prompt["user"].lower()


def test_present_abstract_is_passed_unchanged(user_prompt):
    row = Row()
    row.abstract = "We fit a Monod model."
    pilot_prompt_v2.ask(row, "E2")
    assert "We fit a Monod model." in user_prompt["user"]
