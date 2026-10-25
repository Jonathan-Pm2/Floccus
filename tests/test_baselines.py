"""Baselines for reviewer 2 comment 7: keyword matcher, answer parser, exact McNemar.
No data files or network."""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from baseline_keyword import (FLOC_RE, FORMAL_RE, MODELLING_RE, advance,  # noqa: E402
                              compile_terms, record_text)
from baseline_single_prompt import build_messages, parse_answer  # noqa: E402
from baselines_compare import cohen_kappa, mcnemar_exact  # noqa: E402


@pytest.mark.parametrize("text, hit", [
    ("a modelling study", True),
    ("models of growth", True),
    ("model", True),
    ("we remodel the plant", False),
    ("submodel coupling", False),
    ("a sludge-model coupling", True),    # a hyphen separates words, as in the database
])
def test_prefix_and_word_boundary(text, hit):
    assert bool(compile_terms(["model*"]).search(text)) is hit


def test_unstarred_term_needs_word_end_but_allows_plural():
    rx = compile_terms(["simulation", "floc size"])
    assert rx.search("numerical simulation")
    assert rx.search("several simulations")
    assert not rx.search("simulationist")
    assert rx.search("mean floc sizes")
    assert not rx.search("floc sized")


def test_phrase_spacing_and_hyphen():
    assert FLOC_RE.search("activated-sludge process")
    assert FLOC_RE.search("activated\n sludge")
    assert FLOC_RE.search("bio-floc systems")
    assert FLOC_RE.search("bioflocs")
    assert not FLOC_RE.search("flocculation only")


def test_rule_requires_both_sets_and_strict_formal_term():
    both = "A model of biofloc ponds"
    assert advance(both) and not advance(both, strict=True)
    assert advance("Monod kinetics of biofloc growth; a model", strict=True)
    assert not advance("biofloc shrimp culture")                 # no modelling term
    assert not advance("a mathematical model of soil")            # no floc term
    assert MODELLING_RE.search("growth kinetics") and FORMAL_RE.search("asm2d")
    assert not FORMAL_RE.search("asmall")


def test_record_text_handles_nan_and_none():
    assert record_text("Title", float("nan"), None) == "title  "


def test_parser_takes_last_answer_after_reasoning():
    raw = ("REASONING: An earlier draft said ANSWER: EXCLUDE but the abstract "
           "reports an ASM model.\nANSWER: INCLUDE")
    assert parse_answer(raw) == "INCLUDE"
    assert parse_answer("ANSWER: include\n...\nANSWER: **unclear**") == "UNCLEAR"
    assert parse_answer("REASONING: truncated output") == "UNCLEAR"
    assert parse_answer(float("nan")) == "UNCLEAR"


def test_messages_tolerate_missing_fields():
    msgs = build_messages("T", float("nan"), None, "")
    user = msgs[1]["content"]
    assert "(no abstract)" in user and "nan" not in user


def test_mcnemar_exact_known_values():
    # b=1, c=5: 2 * (C(6,0)+C(6,1)) / 2^6 = 14/64
    assert math.isclose(mcnemar_exact(1, 5), 14 / 64)
    assert math.isclose(mcnemar_exact(5, 1), 14 / 64)
    assert mcnemar_exact(3, 3) == 1.0
    assert mcnemar_exact(0, 0) == 1.0
    assert math.isclose(mcnemar_exact(0, 10), 2 / 1024)


def test_kappa():
    a = np.array([1, 1, 0, 0], bool)
    assert cohen_kappa(a, a) == 1.0
    assert cohen_kappa(a, ~a) == -1.0
