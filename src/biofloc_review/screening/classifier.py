"""Per-criterion classifier: one LLM call answers one binary question.

Reuses the proven Ollama chat client from dedup/llm_judge.py (same model
defaults — qwen3:14b — same settings: temperature=0, think=False, retries
via tenacity).
"""
from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass

from biofloc_review.dedup.llm_judge import _ollama_chat, default_llm_model
from biofloc_review.schema import PaperRecord
from biofloc_review.screening.criteria import Criterion
from biofloc_review.screening.prompts import build_system_prompt, build_user_prompt

log = logging.getLogger(__name__)


@dataclass
class CriterionAnswer:
    criterion_id: str
    answer: str          # YES | NO | UNCLEAR
    confidence: int      # 0-10
    evidence: str
    reasoning: str
    via: str             # "deterministic" | "llm"
    raw: str = ""        # full LLM response (empty for deterministic)


_ANSWER_RE = re.compile(r"ANSWER\s*:\s*(YES|NO|UNCLEAR)", re.IGNORECASE)
_CONFIDENCE_RE = re.compile(r"CONFIDENCE\s*:\s*(\d+)", re.IGNORECASE)
_NEXT_FIELD = r"(?:\n\s*(?:REASONING|EVIDENCE|CONFIDENCE|ANSWER)\s*:|\n\n|\Z)"
_EVIDENCE_RE = re.compile(r"EVIDENCE\s*:\s*(.+?)" + _NEXT_FIELD, re.IGNORECASE | re.DOTALL)
_REASONING_RE = re.compile(r"REASONING\s*:\s*(.+?)" + _NEXT_FIELD, re.IGNORECASE | re.DOTALL)

MODES = ("specificity", "sensitivity")
_INVERT = {"YES": "NO", "NO": "YES", "UNCLEAR": "UNCLEAR"}


def screening_mode() -> str:
    """Operating mode from FLOCCUS_SCREENING_MODE (default: specificity).

    specificity — published configuration; fewer records reach human review.
    sensitivity — reasoning-first prompt and positive E2; higher recall at the
    cost of more human review (see README, "Operating modes")."""
    mode = os.environ.get("FLOCCUS_SCREENING_MODE", "specificity").strip().lower()
    if mode not in MODES:
        raise ValueError(f"FLOCCUS_SCREENING_MODE must be one of {MODES}, got {mode!r}")
    return mode


def invert_answer(answer: str) -> str:
    return _INVERT[answer]


def parse_response(criterion_id: str, text: str) -> CriterionAnswer:
    answers = _ANSWER_RE.findall(text)   # the final ANSWER is the verdict
    c_match = _CONFIDENCE_RE.search(text)
    e_match = _EVIDENCE_RE.search(text)
    r_match = _REASONING_RE.search(text)

    answer = answers[-1].upper() if answers else "UNCLEAR"
    try:
        confidence = int(c_match.group(1)) if c_match else 0
    except ValueError:
        confidence = 0
    confidence = max(0, min(10, confidence))
    evidence = (e_match.group(1).strip() if e_match else "(ninguna)")[:500]
    reasoning = (r_match.group(1).strip() if r_match else text.strip())[:500]

    return CriterionAnswer(
        criterion_id=criterion_id,
        answer=answer,
        confidence=confidence,
        evidence=evidence,
        reasoning=reasoning,
        via="llm",
        raw=text,
    )


def ask_criterion(
    rec: PaperRecord,
    criterion: Criterion,
    model: str | None = None,
) -> CriterionAnswer:
    """Evaluate one criterion against a record. Returns a CriterionAnswer.

    Should only be called for LLM-evaluated criteria. Deterministic criteria
    are handled in prefilter.py.
    """
    mode = screening_mode()
    positive = mode == "sensitivity" and criterion.positive_question is not None
    system = build_system_prompt(criterion, mode=mode)
    user = build_user_prompt(
        criterion=criterion,
        title=rec.title,
        abstract=rec.abstract,
        keywords=list(rec.keywords or []),
        venue=rec.venue,
    )
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    try:
        text = _ollama_chat(messages, model=model or default_llm_model())
    except Exception as e:
        log.warning("Ollama call failed for criterion %s on %r: %s",
                    criterion.id, rec.title[:60], e)
        return CriterionAnswer(
            criterion_id=criterion.id,
            answer="UNCLEAR",
            confidence=0,
            evidence="(error LLM)",
            reasoning=f"LLM error: {type(e).__name__}: {e}",
            via="llm",
            raw="",
        )
    ans = parse_response(criterion.id, text)
    if positive:
        ans.answer = invert_answer(ans.answer)
        ans.via = "llm:positive"
    return ans
