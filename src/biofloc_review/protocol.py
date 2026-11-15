"""Review protocol: the topic-specific part of a systematic review.

A protocol is a YAML file holding the review topic, the prompt language and
the screening criteria. Everything else (dedup, two-layer screening,
clustering) is topic independent, so a new review only needs a new protocol.

Example::

    name: microplastics
    language: en            # es | en
    topic: microplastic transport modelling in rivers
    criteria:
      - id: I1
        type: inclusion     # inclusion | exclusion | derivation
        question: Does the article study microplastics in rivers?
        justification: Defines the domain.
      - id: D1
        type: derivation
        question: Is the abstract too short to decide?
        justification: Route to a human.
        check: abstract_too_short     # deterministic, no LLM call
        params: {min_words: 100}

A criterion with ``check`` is evaluated deterministically; the available
checks are listed in ``CHECKS``. ``positive_question`` is optional and only
used by the sensitivity operating mode.

The shipped ``biofloc`` protocol reproduces the published criteria exactly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path
from typing import Any

import yaml

from biofloc_review.screening.criteria import Criterion

LANGUAGES = ("es", "en")
CRITERION_TYPES = ("inclusion", "exclusion", "derivation")
# Deterministic checks and their allowed parameters.
CHECKS: dict[str, tuple[str, ...]] = {
    "document_type": ("included", "excluded"),
    "abstract_too_short": ("min_words",),
}


class ProtocolError(ValueError):
    """The protocol file is invalid; the message says what to fix."""


@dataclass(frozen=True)
class Check:
    name: str
    params: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Protocol:
    name: str
    language: str
    topic: str
    criteria: tuple[Criterion, ...]
    checks: dict[str, Check] = field(default_factory=dict)


def _require_text(entry: dict, key: str, where: str) -> str:
    value = entry.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ProtocolError(f"{where}: missing '{key}'")
    return value


def _validate_params(cid: str, params: dict) -> None:
    for key in ("included", "excluded"):
        if key in params:
            value = params[key]
            if not isinstance(value, list) or not all(isinstance(v, str) and v.strip() for v in value):
                raise ProtocolError(f"{cid}: '{key}' must be a list of document types, e.g. [article, review]")
    if "min_words" in params:
        value = params["min_words"]
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ProtocolError(f"{cid}: 'min_words' must be a positive whole number")


def protocol_from_dict(data: Any) -> Protocol:
    if not isinstance(data, dict):
        raise ProtocolError("protocol must be a mapping with name, language, topic and criteria")
    name = _require_text(data, "name", "protocol")
    topic = _require_text(data, "topic", "protocol")
    language = data.get("language")
    if language not in LANGUAGES:
        raise ProtocolError(f"language must be one of {', '.join(LANGUAGES)} (got {language!r})")
    entries = data.get("criteria")
    if not isinstance(entries, list) or not entries:
        raise ProtocolError("protocol needs at least one criterion")

    criteria: list[Criterion] = []
    checks: dict[str, Check] = {}
    seen: set[str] = set()
    for i, entry in enumerate(entries, start=1):
        if not isinstance(entry, dict):
            raise ProtocolError(f"criterion #{i} must be a mapping")
        cid = _require_text(entry, "id", f"criterion #{i}")
        if cid in seen:
            raise ProtocolError(f"duplicate criterion id '{cid}'")
        seen.add(cid)
        ctype = entry.get("type")
        if ctype not in CRITERION_TYPES:
            raise ProtocolError(f"{cid}: type must be one of {', '.join(CRITERION_TYPES)} (got {ctype!r})")
        question = _require_text(entry, "question", cid)
        justification = _require_text(entry, "justification", cid)
        positive = entry.get("positive_question")
        if positive is not None and (not isinstance(positive, str) or not positive.strip()):
            raise ProtocolError(f"{cid}: 'positive_question' must be text")

        check_name = entry.get("check")
        if check_name is not None:
            if check_name not in CHECKS:
                raise ProtocolError(
                    f"{cid}: unknown check '{check_name}' (available: {', '.join(CHECKS)})"
                )
            params = entry.get("params") or {}
            if not isinstance(params, dict):
                raise ProtocolError(f"{cid}: 'params' must be a mapping")
            unknown = sorted(set(params) - set(CHECKS[check_name]))
            if unknown:
                raise ProtocolError(f"{cid}: unknown params for {check_name}: {', '.join(unknown)}")
            _validate_params(cid, params)
            checks[cid] = Check(name=check_name, params=dict(params))
        elif "params" in entry:
            raise ProtocolError(f"{cid}: 'params' requires 'check'")

        criteria.append(
            Criterion(
                id=cid,
                type=ctype,
                question=question,
                justification=justification,
                deterministic=check_name is not None,
                positive_question=positive,
            )
        )
    return Protocol(name=name, language=language, topic=topic, criteria=tuple(criteria), checks=checks)


def protocol_to_dict(p: Protocol) -> dict[str, Any]:
    entries = []
    for c in p.criteria:
        entry: dict[str, Any] = {
            "id": c.id,
            "type": c.type,
            "question": c.question,
            "justification": c.justification,
        }
        if c.positive_question is not None:
            entry["positive_question"] = c.positive_question
        check = p.checks.get(c.id)
        if check is not None:
            entry["check"] = check.name
            if check.params:
                entry["params"] = dict(check.params)
        entries.append(entry)
    return {"name": p.name, "language": p.language, "topic": p.topic, "criteria": entries}


def load_protocol(path: str | Path) -> Protocol:
    path = Path(path)
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ProtocolError(f"{path.name}: invalid YAML: {exc}") from exc
    return protocol_from_dict(data)


def dump_protocol(p: Protocol) -> str:
    return yaml.safe_dump(protocol_to_dict(p), allow_unicode=True, sort_keys=False, width=100)


def builtin_protocol(name: str) -> Protocol:
    """Load a protocol shipped with the package (``protocols/<name>.yaml``)."""
    ref = resources.files("biofloc_review").joinpath("protocols", f"{name}.yaml")
    if not ref.is_file():
        raise ProtocolError(f"no built-in protocol named '{name}'")
    return protocol_from_dict(yaml.safe_load(ref.read_text(encoding="utf-8")))
