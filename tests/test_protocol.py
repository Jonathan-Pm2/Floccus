"""Review protocol: user-defined topic and criteria loaded from YAML.

The shipped biofloc protocol must equal the published criteria exactly (S6).
"""

from __future__ import annotations

import pytest

from biofloc_review.protocol import (
    Check,
    Protocol,
    ProtocolError,
    builtin_protocol,
    dump_protocol,
    load_protocol,
)
from biofloc_review.screening.criteria import CRITERIA


def test_biofloc_protocol_reproduces_published_criteria():
    p = builtin_protocol("biofloc")
    assert p.name == "biofloc"
    assert p.language == "es"
    assert p.topic == "modelado de flóculos biológicos en sistemas acuáticos"
    assert list(p.criteria) == CRITERIA


def test_biofloc_protocol_deterministic_checks_match_prefilter():
    p = builtin_protocol("biofloc")
    assert p.checks == {
        "I4": Check(name="document_type", params={}),
        "D1": Check(name="abstract_too_short", params={"min_words": 100}),
    }


def test_dump_and_load_round_trip(tmp_path):
    p = builtin_protocol("biofloc")
    path = tmp_path / "p.yaml"
    path.write_text(dump_protocol(p), encoding="utf-8")
    assert load_protocol(path) == p


USER_YAML = """
name: microplastics
language: en
topic: microplastic transport modelling in rivers
criteria:
  - id: I1
    type: inclusion
    question: Does the article study microplastics in rivers?
    justification: Defines the domain.
  - id: E1
    type: exclusion
    question: Is it only a laboratory toxicity test?
    justification: Out of scope.
    positive_question: Does it study transport or fate?
  - id: D1
    type: derivation
    question: Is the abstract too short to decide?
    justification: Route to a human.
    check: abstract_too_short
    params: {min_words: 50}
"""


def test_user_protocol_for_another_topic(tmp_path):
    path = tmp_path / "micro.yaml"
    path.write_text(USER_YAML, encoding="utf-8")
    p = load_protocol(path)
    assert isinstance(p, Protocol)
    assert p.language == "en"
    assert [c.id for c in p.criteria] == ["I1", "E1", "D1"]
    assert p.criteria[1].positive_question == "Does it study transport or fate?"
    assert p.criteria[2].deterministic is True
    assert p.criteria[0].deterministic is False
    assert p.checks == {"D1": Check(name="abstract_too_short", params={"min_words": 50})}


@pytest.mark.parametrize(
    ("bad", "message"),
    [
        (USER_YAML.replace("id: E1", "id: I1"), "duplicate criterion id 'I1'"),
        (USER_YAML.replace("type: exclusion", "type: other"), "E1: type must be one of"),
        (USER_YAML.replace("check: abstract_too_short", "check: magic"), "D1: unknown check 'magic'"),
        (USER_YAML.replace("language: en", "language: fr"), "language must be one of"),
        (USER_YAML.replace("    question: Does the article study microplastics in rivers?\n", ""),
         "I1: missing 'question'"),
        ("name: x\nlanguage: en\ntopic: t\ncriteria: []\n", "at least one criterion"),
    ],
)
def test_invalid_protocols_are_rejected_with_clear_message(tmp_path, bad, message):
    path = tmp_path / "bad.yaml"
    path.write_text(bad, encoding="utf-8")
    with pytest.raises(ProtocolError, match=message):
        load_protocol(path)
