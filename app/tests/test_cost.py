"""T7: approximate cost of screening with a paid API (S5)."""

from __future__ import annotations

import pytest
from biofloc_review.protocol import builtin_protocol

from floccus_app import cost
from floccus_app.screening import system_prompt

BIOFLOC = builtin_protocol("biofloc")


def test_calls_follow_records_criteria_and_layer2_share():
    e = cost.estimate(BIOFLOC, n_records=1000, layer2_share=0.25, user_chars=1700,
                      price_in=5.0, price_out=25.0)
    llm = sum(1 for c in BIOFLOC.criteria if not c.deterministic)
    assert llm == 9
    assert e.calls == 1000 * 9 + 250 * 9


def test_tokens_use_the_real_prompts():
    e = cost.estimate(BIOFLOC, n_records=1, layer2_share=0.0, user_chars=1700, price_in=1, price_out=1)
    sys_chars = sum(len(system_prompt(BIOFLOC, c)) for c in BIOFLOC.criteria if not c.deterministic)
    expected_in = round((sys_chars + 9 * 1700) / cost.CHARS_PER_TOKEN)
    assert e.input_tokens == expected_in
    assert e.output_tokens == 9 * cost.OUTPUT_TOKENS_PER_CALL


def test_cost_is_tokens_times_price_per_million():
    e = cost.estimate(BIOFLOC, n_records=100, layer2_share=0.25, user_chars=1700, price_in=3.0, price_out=15.0)
    assert e.cost == pytest.approx(e.input_tokens / 1e6 * 3.0 + e.output_tokens / 1e6 * 15.0)


def test_local_ollama_costs_nothing():
    e = cost.estimate(BIOFLOC, n_records=100, layer2_share=0.25, user_chars=1700, price_in=0, price_out=0)
    assert e.cost == 0 and e.calls > 0


def test_average_user_chars_from_records():
    recs = [{"title": "abc", "abstract": "x" * 100}, {"title": "de", "abstract": None}]
    assert cost.average_user_chars(recs) == round((3 + 100 + 2) / 2 + cost.USER_TEMPLATE_CHARS)
    assert cost.average_user_chars([]) == cost.DEFAULT_USER_CHARS
