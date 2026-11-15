"""T4: generic screening engine driven by a protocol (S2, S4, S6)."""

from __future__ import annotations

import itertools
import random

import pytest
from biofloc_review.protocol import builtin_protocol, protocol_from_dict
from biofloc_review.schema import PaperRecord
from biofloc_review.screening import decision as published_decision
from biofloc_review.screening import prompts as published_prompts
from biofloc_review.screening import prefilter
from biofloc_review.screening.classifier import CriterionAnswer

from floccus_app import screening

BIOFLOC = builtin_protocol("biofloc")
REC = {
    "title": "  A model of biofloc settling ",
    "abstract": "We model floc settling with Monod kinetics. " * 30,
    "keywords": ["biofloc", "model"],
    "venue": "Aquacultural Engineering",
    "document_type": "Article",
}


@pytest.mark.parametrize("mode", ["specificity", "sensitivity"])
def test_biofloc_prompts_are_byte_identical_to_published(mode):
    for c in BIOFLOC.criteria:
        assert screening.system_prompt(BIOFLOC, c, mode) == published_prompts.build_system_prompt(c, mode=mode)
        assert screening.user_prompt(BIOFLOC, c, REC) == published_prompts.build_user_prompt(
            c, REC["title"], REC["abstract"], REC["keywords"], REC["venue"])


def test_missing_fields_match_published_placeholders():
    c = BIOFLOC.criteria[0]
    rec = {"title": "T"}
    assert screening.user_prompt(BIOFLOC, c, rec) == published_prompts.build_user_prompt(c, "T", None, [], None)


EN = protocol_from_dict({
    "name": "micro", "language": "en", "topic": "microplastic transport in rivers",
    "criteria": [
        {"id": "I1", "type": "inclusion", "question": "Rivers?", "justification": "Domain."},
        {"id": "E1", "type": "exclusion", "question": "Toxicity only?", "justification": "Scope.",
         "positive_question": "Transport?"},
        {"id": "D1", "type": "derivation", "question": "Short abstract?", "justification": "Human.",
         "check": "abstract_too_short", "params": {"min_words": 5}},
    ],
})


def test_english_protocol_gets_english_prompts_with_its_topic():
    s = screening.system_prompt(EN, EN.criteria[0], "specificity")
    assert "microplastic transport in rivers" in s
    assert "CRITERION I1 (inclusion):\nRivers?" in s
    assert "ANSWER: YES | NO | UNCLEAR" in s
    assert "Eres" not in s
    u = screening.user_prompt(EN, EN.criteria[0], {"title": "T"})
    assert u.startswith("Article to assess:")
    assert "(no abstract)" in u and "Apply criterion I1" in u


def _rec(**kw) -> PaperRecord:
    base = dict(source="scopus", source_id="1", title="t", title_norm="t", corpus="biofloc")
    base.update(kw)
    return PaperRecord(**base)


@pytest.mark.parametrize("doctype", [None, "Article", "review", "Editorial", "Book Chapter", "weird"])
def test_document_type_check_matches_published_prefilter(doctype):
    check = BIOFLOC.checks["I4"]
    assert screening.evaluate_check(check, {"document_type": doctype}) == prefilter.is_eligible_doctype(
        _rec(document_type=doctype))


@pytest.mark.parametrize("abstract", [None, "short abstract", "word " * 100])
def test_abstract_check_matches_published_prefilter(abstract):
    check = BIOFLOC.checks["D1"]
    assert screening.evaluate_check(check, {"abstract": abstract}) == prefilter.is_abstract_too_short(
        _rec(abstract=abstract))


def test_aggregate_matches_published_rules_for_biofloc():
    ids = [c.id for c in BIOFLOC.criteria]
    rng = random.Random(0)
    combos = [dict(zip(ids, v)) for v in itertools.islice(
        (tuple(rng.choice(["YES", "NO", "UNCLEAR"]) for _ in ids) for _ in iter(int, 1)), 3000)]
    combos.append(dict.fromkeys(ids, "NO") | {c.id: "YES" for c in BIOFLOC.criteria if c.type == "inclusion"})
    for combo in combos:
        answers = {cid: CriterionAnswer(cid, a, 5, "", "", "llm") for cid, a in combo.items()}
        mine = screening.aggregate(BIOFLOC, answers)
        ref = published_decision.aggregate(answers)
        assert (mine.decision, mine.reason, mine.triggered_criteria) == (
            ref.decision, ref.reason, ref.triggered_criteria)


def test_screen_record_calls_llm_once_per_llm_criterion_and_inverts_positive():
    calls = []

    def chat(messages, model):
        calls.append(messages)
        q = messages[0]["content"]
        return "ANSWER: YES\nCONFIDENCE: 8\nEVIDENCE: x\nREASONING: y" if ("Rivers?" in q or "Transport?" in q) \
            else "ANSWER: NO\nCONFIDENCE: 8\nEVIDENCE: x\nREASONING: y"

    rec = {"title": "T", "abstract": "one two three four five six"}
    r = screening.screen_record(EN, rec, chat=chat, model="m", mode="sensitivity")
    assert len(calls) == 2
    assert r.answers["E1"].answer == "NO"  # positive question answered YES -> inverted
    assert r.answers["D1"].via == "deterministic"
    assert r.decision.decision == "INCLUDE"
    assert r.errors == 0


def test_llm_failure_is_counted_not_silent():
    def chat(messages, model):
        raise TimeoutError("down")

    r = screening.screen_record(EN, {"title": "T", "abstract": "a b c d e f"}, chat=chat, model="m")
    assert r.errors == 2
    assert r.answers["I1"].answer == "UNCLEAR"
    assert r.answers["I1"].evidence == "(LLM error)"
    assert r.decision.decision == "HUMAN_REVIEW"


def test_extra_context_is_appended_for_layer_two():
    seen = []

    def chat(messages, model):
        seen.append(messages[1]["content"])
        return "ANSWER: NO"

    screening.screen_record(EN, {"title": "T", "abstract": "a b c d e f"}, chat=chat, model="m",
                            extra_context="TOPICS: rivers")
    assert all(u.rstrip().endswith("TOPICS: rivers") for u in seen)
