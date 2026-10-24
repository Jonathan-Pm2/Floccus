"""Operating modes of the screening classifier.

specificity (default) — the published configuration: answer-first output
format, criteria asked as written.
sensitivity — reasoning-first output format; criteria that define a
positive_question are asked in positive form and the answer is inverted, which
removes the polarity errors found on the negatively phrased E2.
"""
from __future__ import annotations

import pytest

from biofloc_review.screening import classifier
from biofloc_review.screening.classifier import invert_answer, parse_response, screening_mode
from biofloc_review.screening.criteria import CRITERIA
from biofloc_review.screening.prompts import build_system_prompt

CRIT = {c.id: c for c in CRITERIA}


# ---------------------------------------------------------------- parsing
def test_parse_answer_first_format_unchanged():
    text = ("ANSWER: YES\nCONFIDENCE: 8\nEVIDENCE: \"only measurements\"\n"
            "REASONING: purely descriptive study.")
    a = parse_response("E2", text)
    assert (a.answer, a.confidence) == ("YES", 8)
    assert a.evidence == '"only measurements"'
    assert a.reasoning == "purely descriptive study."


def test_parse_reasoning_first_format():
    text = ("REASONING: the study fits a kinetic model.\nEVIDENCE: \"Monod model\"\n"
            "CONFIDENCE: 9\nANSWER: NO")
    a = parse_response("E2", text)
    assert (a.answer, a.confidence) == ("NO", 9)
    assert a.evidence == '"Monod model"'
    assert a.reasoning == "the study fits a kinetic model."


def test_parse_takes_the_final_answer():
    # a model may restate the scale before deciding; the verdict is the last one
    text = "REASONING: YES would mean... ANSWER: YES is wrong here.\nCONFIDENCE: 7\nANSWER: NO"
    assert parse_response("E2", text).answer == "NO"


@pytest.mark.parametrize("a,b", [("YES", "NO"), ("NO", "YES"), ("UNCLEAR", "UNCLEAR")])
def test_invert_answer(a, b):
    assert invert_answer(a) == b


# ---------------------------------------------------------------- mode
def test_default_mode_is_specificity(monkeypatch):
    monkeypatch.delenv("FLOCCUS_SCREENING_MODE", raising=False)
    assert screening_mode() == "specificity"


def test_unknown_mode_is_rejected(monkeypatch):
    monkeypatch.setenv("FLOCCUS_SCREENING_MODE", "aggressive")
    with pytest.raises(ValueError):
        screening_mode()


def test_specificity_prompt_is_the_published_one():
    p = build_system_prompt(CRIT["E2"])
    assert p.index("ANSWER:") < p.index("REASONING:")
    assert CRIT["E2"].question in p


def test_sensitivity_prompt_puts_reasoning_first_and_asks_positively():
    p = build_system_prompt(CRIT["E2"], mode="sensitivity")
    assert p.index("REASONING:") < p.index("ANSWER:")
    assert CRIT["E2"].positive_question in p
    assert "SIN" not in CRIT["E2"].positive_question


def test_only_e2_has_a_positive_form():
    # E4's positive form was discarded in the pilot: it broadened the criterion
    assert [c.id for c in CRITERIA if c.positive_question] == ["E2"]


def test_sensitivity_inverts_positive_answer(monkeypatch):
    monkeypatch.setenv("FLOCCUS_SCREENING_MODE", "sensitivity")
    monkeypatch.setattr(classifier, "_ollama_chat",
                        lambda msgs, model=None: "REASONING: has a model.\nEVIDENCE: x\nCONFIDENCE: 9\nANSWER: YES")

    class Rec:
        title, abstract, keywords, venue = "t", "a", [], "v"

    a = classifier.ask_criterion(Rec(), CRIT["E2"])
    assert a.answer == "NO"          # "has modelling" → E2 (no modelling) does not fire
    assert a.via == "llm:positive"


def test_specificity_does_not_invert(monkeypatch):
    monkeypatch.setenv("FLOCCUS_SCREENING_MODE", "specificity")
    monkeypatch.setattr(classifier, "_ollama_chat",
                        lambda msgs, model=None: "ANSWER: YES\nCONFIDENCE: 9\nEVIDENCE: x\nREASONING: y")

    class Rec:
        title, abstract, keywords, venue = "t", "a", [], "v"

    a = classifier.ask_criterion(Rec(), CRIT["E2"])
    assert (a.answer, a.via) == ("YES", "llm")
