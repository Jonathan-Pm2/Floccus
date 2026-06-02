"""PRISMA aggregation rule + sensitivity analysis variants.

Primary rule (the one pre-registered in the manuscript):
  EXCLUDE    if ANY exclusion criterion (E1..E5) = YES
  HUMAN_REVIEW  if ANY derivation criterion (D1..D3) = YES
                or if ANY inclusion criterion (I1..I4) ∈ {NO, UNCLEAR}
                or if ANY exclusion criterion = UNCLEAR
  INCLUDE    if ALL inclusion criteria (I1..I4) = YES
             AND ALL exclusion criteria (E1..E5) = NO
             AND ALL derivation criteria (D1..D3) = NO

UNCLEAR is treated conservatively: it never auto-excludes, and never
auto-includes — it routes to human review. This preserves the PRISMA
sensitivity-over-specificity stance.

For Camino 3 (sensitivity analysis), the raw answers are preserved so
alternative rules can be applied post-hoc without re-running the LLM.
"""
from __future__ import annotations

from dataclasses import dataclass

from biofloc_review.screening.classifier import CriterionAnswer
from biofloc_review.screening.criteria import (
    DERIVATION_IDS,
    EXCLUSION_IDS,
    INCLUSION_IDS,
)

INCLUDE = "INCLUDE"
EXCLUDE = "EXCLUDE"
HUMAN_REVIEW = "HUMAN_REVIEW"


@dataclass
class FinalDecision:
    decision: str          # INCLUDE | EXCLUDE | HUMAN_REVIEW
    reason: str            # short explanation (which criterion triggered)
    triggered_criteria: list[str]  # criterion ids that drove the decision


def _answer_of(answers: dict[str, CriterionAnswer], cid: str) -> str:
    a = answers.get(cid)
    return a.answer if a else "UNCLEAR"


def aggregate(answers: dict[str, CriterionAnswer]) -> FinalDecision:
    """Apply the primary PRISMA rule over a dict of {criterion_id → answer}.

    Args:
        answers: dict mapping criterion id (e.g. "I1") to its CriterionAnswer.
            Must include all 12 criteria (I1-I4, E1-E5, D1-D3).
    """
    # 1. Any exclusion criterion firing → EXCLUDE
    excl_triggers = [cid for cid in EXCLUSION_IDS if _answer_of(answers, cid) == "YES"]
    if excl_triggers:
        return FinalDecision(
            decision=EXCLUDE,
            reason=f"exclusión por {', '.join(excl_triggers)}",
            triggered_criteria=excl_triggers,
        )

    # 2. Any derivation criterion firing → HUMAN_REVIEW
    deriv_triggers = [cid for cid in DERIVATION_IDS if _answer_of(answers, cid) == "YES"]
    if deriv_triggers:
        return FinalDecision(
            decision=HUMAN_REVIEW,
            reason=f"derivación por {', '.join(deriv_triggers)}",
            triggered_criteria=deriv_triggers,
        )

    # 3. Any inclusion criterion not satisfied (NO or UNCLEAR) → HUMAN_REVIEW
    incl_failures = [cid for cid in INCLUSION_IDS if _answer_of(answers, cid) != "YES"]
    if incl_failures:
        return FinalDecision(
            decision=HUMAN_REVIEW,
            reason=f"inclusión incompleta: {', '.join(incl_failures)} ≠ YES",
            triggered_criteria=incl_failures,
        )

    # 4. Any exclusion criterion UNCLEAR (conservative) → HUMAN_REVIEW
    excl_uncertain = [cid for cid in EXCLUSION_IDS if _answer_of(answers, cid) == "UNCLEAR"]
    if excl_uncertain:
        return FinalDecision(
            decision=HUMAN_REVIEW,
            reason=f"exclusión con incertidumbre: {', '.join(excl_uncertain)} = UNCLEAR",
            triggered_criteria=excl_uncertain,
        )

    # 5. All I = YES, all E = NO, all D = NO → INCLUDE
    return FinalDecision(
        decision=INCLUDE,
        reason="todos I cumplidos, ningún E ni D fired",
        triggered_criteria=[],
    )


# ---------- Sensitivity-analysis variants (Camino 3 in the manuscript) ----------


def aggregate_aggressive_exclude(answers: dict[str, CriterionAnswer]) -> FinalDecision:
    """Variant: treat UNCLEAR on exclusion criteria as YES (more aggressive
    exclusion). Useful to bound how much the conservative rule preserves
    versus how much an aggressive rule would discard.
    """
    excl_triggers = [
        cid for cid in EXCLUSION_IDS
        if _answer_of(answers, cid) in ("YES", "UNCLEAR")
    ]
    if excl_triggers:
        return FinalDecision(
            decision=EXCLUDE,
            reason=f"exclusión agresiva por {', '.join(excl_triggers)}",
            triggered_criteria=excl_triggers,
        )

    deriv_triggers = [cid for cid in DERIVATION_IDS if _answer_of(answers, cid) == "YES"]
    if deriv_triggers:
        return FinalDecision(
            decision=HUMAN_REVIEW,
            reason=f"derivación por {', '.join(deriv_triggers)}",
            triggered_criteria=deriv_triggers,
        )

    incl_failures = [cid for cid in INCLUSION_IDS if _answer_of(answers, cid) != "YES"]
    if incl_failures:
        return FinalDecision(
            decision=HUMAN_REVIEW,
            reason=f"inclusión incompleta: {', '.join(incl_failures)}",
            triggered_criteria=incl_failures,
        )
    return FinalDecision(INCLUDE, "todos I=YES, ningún E ni D fired", [])


def aggregate_permissive_include(answers: dict[str, CriterionAnswer]) -> FinalDecision:
    """Variant: treat UNCLEAR on inclusion criteria as YES (more permissive
    inclusion). Bounds how much extra would be included if UNCLEAR is
    interpreted as 'probably satisfied'.
    """
    excl_triggers = [cid for cid in EXCLUSION_IDS if _answer_of(answers, cid) == "YES"]
    if excl_triggers:
        return FinalDecision(EXCLUDE, f"exclusión por {', '.join(excl_triggers)}", excl_triggers)

    deriv_triggers = [cid for cid in DERIVATION_IDS if _answer_of(answers, cid) == "YES"]
    if deriv_triggers:
        return FinalDecision(HUMAN_REVIEW, f"derivación por {', '.join(deriv_triggers)}", deriv_triggers)

    incl_failures = [
        cid for cid in INCLUSION_IDS
        if _answer_of(answers, cid) == "NO"   # only hard NO counts as failure
    ]
    if incl_failures:
        return FinalDecision(HUMAN_REVIEW, f"inclusión incompleta: {', '.join(incl_failures)}", incl_failures)
    return FinalDecision(INCLUDE, "todos I∈{YES,UNCLEAR}, ningún E ni D fired", [])
