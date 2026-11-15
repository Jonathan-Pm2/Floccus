"""Protocol-driven screening engine (any topic, Spanish or English prompts).

One LLM call answers one criterion; deterministic checks need no call; the
answers are aggregated with the published Floccus rules. With the shipped
biofloc protocol the prompts and decisions are identical to the published
pipeline (tests pin this).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

from biofloc_review.protocol import Check, Protocol
from biofloc_review.screening import prefilter
from biofloc_review.screening import prompts as _published
from biofloc_review.screening.classifier import CriterionAnswer, invert_answer, parse_response
from biofloc_review.screening.criteria import Criterion
from biofloc_review.screening.decision import EXCLUDE, HUMAN_REVIEW, INCLUDE, FinalDecision

Chat = Callable[..., str]
MODES = ("specificity", "sensitivity")
_BIOFLOC_TOPIC = "modelado de flóculos biológicos en sistemas acuáticos"


def _topic_template(published: str) -> str:
    assert _BIOFLOC_TOPIC in published, "published prompt changed"
    return published.replace(_BIOFLOC_TOPIC, "{topic}")


SYSTEM = {
    ("es", "specificity"): _topic_template(_published.SYSTEM_PROMPT_TEMPLATE),
    ("es", "sensitivity"): _topic_template(_published.SYSTEM_PROMPT_TEMPLATE_REASONING_FIRST),
    ("en", "specificity"): """You are an expert systematic reviewer doing PRISMA reviews on {topic}. Your task is to answer ONE binary (Yes/No) question about an article, looking at its title, abstract, keywords and venue.

CRITERION {criterion_id} ({criterion_type}):
{question}

Why this criterion exists: {justification}

Answer rules:
- YES = the criterion IS met (the answer to the question is "Yes").
- NO = the criterion is NOT met (the answer is "No").
- UNCLEAR = the available information does NOT allow a decision with reasonable confidence.

Answer EXACTLY in this format (no text before or after):
ANSWER: YES | NO | UNCLEAR
CONFIDENCE: <0-10>
EVIDENCE: <exact fragment of the title/abstract/keywords/venue that supports the answer, or "(none)">
REASONING: <one short sentence>
""",
    ("en", "sensitivity"): """You are an expert systematic reviewer doing PRISMA reviews on {topic}. You answer ONE question about an article, looking at its title, abstract, keywords and venue.

CRITERION {criterion_id} ({criterion_type}):
{question}

Why this criterion exists: {justification}

First reason and quote the evidence; ONLY THEN decide. The final answer must agree with your reasoning.
- YES = the answer to the question is "Yes".
- NO = the answer to the question is "No".
- UNCLEAR = the available information does not allow a decision with reasonable confidence.

Answer EXACTLY in this format and order (no text before or after):
REASONING: <one or two sentences>
EVIDENCE: <exact fragment of the title/abstract/keywords/venue, or "(none)">
CONFIDENCE: <0-10>
ANSWER: YES | NO | UNCLEAR
""",
}
USER = {
    "es": _published.USER_TEMPLATE,
    "en": """Article to assess:

TITLE:
{title}

ABSTRACT:
{abstract}

KEYWORDS:
{keywords}

VENUE:
{venue}

Apply criterion {criterion_id} and answer in the required format.
""",
}
REASONS = {
    "es": {
        "exclude": "exclusión por {ids}",
        "derive": "derivación por {ids}",
        "incomplete": "inclusión incompleta: {ids} ≠ YES",
        "uncertain": "exclusión con incertidumbre: {ids} = UNCLEAR",
        "include": "todos I cumplidos, ningún E ni D fired",
    },
    "en": {
        "exclude": "excluded by {ids}",
        "derive": "sent to human review by {ids}",
        "incomplete": "inclusion not met: {ids} ≠ YES",
        "uncertain": "uncertain exclusion: {ids} = UNCLEAR",
        "include": "all inclusion criteria met, no exclusion or derivation fired",
    },
}


def system_prompt(p: Protocol, c: Criterion, mode: str = "specificity") -> str:
    question = (c.positive_question or c.question) if mode == "sensitivity" else c.question
    return SYSTEM[(p.language, mode)].format(
        topic=p.topic, criterion_id=c.id, criterion_type=c.type,
        question=question, justification=c.justification,
    )


def _keywords(value: Any) -> list[str]:
    if not value:
        return []
    if isinstance(value, str):
        return [k.strip() for k in value.split(";") if k.strip()]
    return [str(k) for k in value]


def user_prompt(p: Protocol, c: Criterion, rec: Mapping[str, Any]) -> str:
    abstract = rec.get("abstract")
    kws = _keywords(rec.get("keywords"))
    return USER[p.language].format(
        criterion_id=c.id,
        title=str(rec.get("title") or "").strip(),
        abstract=(abstract.strip() if abstract else "(no abstract)")[:3000],
        keywords=", ".join(kws) if kws else "(none)",
        venue=rec.get("venue") or "(none)",
    )


def evaluate_check(check: Check, rec: Mapping[str, Any]) -> tuple[str, str]:
    """Deterministic criteria. Defaults reproduce the published prefilter."""
    if check.name == "document_type":
        dt_raw = rec.get("document_type")
        if not dt_raw:
            return ("YES", "document_type=(none); assumed Article")
        included = {s.lower() for s in check.params.get("included", prefilter._DOCTYPES_INCLUDED)}
        excluded = {s.lower() for s in check.params.get("excluded", prefilter._DOCTYPES_EXCLUDED)}
        dt = dt_raw.strip().lower()
        if dt in included:
            return ("YES", f"document_type={dt_raw!r}")
        if dt in excluded:
            return ("NO", f"document_type={dt_raw!r}")
        return ("YES", f"document_type={dt_raw!r} unrecognized; assumed eligible")
    if check.name == "abstract_too_short":
        min_words = int(check.params.get("min_words", prefilter._MIN_ABSTRACT_WORDS))
        abstract = rec.get("abstract")
        if not abstract:
            return ("YES", "abstract=(none)")
        n = len(abstract.split())
        if n < min_words:
            return ("YES", f"abstract={n} words (<{min_words})")
        return ("NO", f"abstract={n} words")
    raise ValueError(f"unknown check {check.name!r}")


def aggregate(p: Protocol, answers: Mapping[str, CriterionAnswer]) -> FinalDecision:
    """Published rule order: exclusion > derivation > inclusion > uncertain exclusion > include."""
    r = REASONS[p.language]

    def ids_of(ctype: str) -> list[str]:
        return [c.id for c in p.criteria if c.type == ctype]

    def ans(cid: str) -> str:
        a = answers.get(cid)
        return a.answer if a else "UNCLEAR"

    rules = (
        (EXCLUDE, "exclude", [i for i in ids_of("exclusion") if ans(i) == "YES"]),
        (HUMAN_REVIEW, "derive", [i for i in ids_of("derivation") if ans(i) == "YES"]),
        (HUMAN_REVIEW, "incomplete", [i for i in ids_of("inclusion") if ans(i) != "YES"]),
        (HUMAN_REVIEW, "uncertain", [i for i in ids_of("exclusion") if ans(i) == "UNCLEAR"]),
    )
    for decision, key, triggered in rules:
        if triggered:
            return FinalDecision(decision=decision, reason=r[key].format(ids=", ".join(triggered)),
                                 triggered_criteria=triggered)
    return FinalDecision(decision=INCLUDE, reason=r["include"], triggered_criteria=[])


@dataclass
class RecordResult:
    answers: dict[str, CriterionAnswer]
    decision: FinalDecision
    errors: int = 0
    error_messages: list[str] = field(default_factory=list)


def _ask(p: Protocol, c: Criterion, rec: Mapping[str, Any], chat: Chat, model: str, mode: str,
         extra_context: str | None) -> tuple[CriterionAnswer, str | None]:
    user = user_prompt(p, c, rec)
    if extra_context:
        user = f"{user.rstrip()}\n\n{extra_context.strip()}\n"
    messages = [{"role": "system", "content": system_prompt(p, c, mode)},
                {"role": "user", "content": user}]
    try:
        text = chat(messages, model=model)
    except Exception as exc:  # recorded and counted, never silent
        from floccus_app.textsafe import redact

        msg = f"{c.id}: {type(exc).__name__}: {redact(str(exc))}"
        return CriterionAnswer(c.id, "UNCLEAR", 0, "(LLM error)", f"LLM error: {msg}", "llm"), msg
    a = parse_response(c.id, text)
    if mode == "sensitivity" and c.positive_question:
        a.answer = invert_answer(a.answer)
        a.via = "llm:positive"
    return a, None


def screen_record(p: Protocol, rec: Mapping[str, Any], *, chat: Chat, model: str,
                  mode: str = "specificity", extra_context: str | None = None,
                  workers: int = 1) -> RecordResult:
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}")
    answers: dict[str, CriterionAnswer] = {}
    for cid, check in p.checks.items():
        answer, evidence = evaluate_check(check, rec)
        answers[cid] = CriterionAnswer(cid, answer, 10, evidence, "deterministic check", "deterministic")
    llm = [c for c in p.criteria if not c.deterministic]
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        results = list(pool.map(lambda c: _ask(p, c, rec, chat, model, mode, extra_context), llm))
    errors = []
    for c, (a, err) in zip(llm, results):
        answers[c.id] = a
        if err:
            errors.append(err)
    return RecordResult(answers=answers, decision=aggregate(p, answers), errors=len(errors),
                        error_messages=errors)
