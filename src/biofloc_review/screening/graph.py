"""LangGraph StateGraph for criterion-based PRISMA screening.

Topology:

  START
    ↓
  prefilter    (deterministic: E5 language, I4 doctype, D1 abstract length)
    ↓
  llm_evaluator   (fan-out: I1, I2, I3, E1, E2, E3, E4, D2, D3 in parallel
                   via ThreadPoolExecutor inside the node — fast and
                   compatible with sync invoke())
    ↓
  supervisor   (apply primary aggregation rule)
    ↓
  END

We collapse the 9 LLM criteria into a single LangGraph node that fans out
internally via ThreadPoolExecutor. This guarantees true concurrency under
the sync `invoke()` API, which the existing pipeline (run_judge.py) uses.
The graph is still meaningfully a state graph with a deterministic prefilter
node, an evaluator node, and a supervisor — matching the manuscript's
'orquestador supervisor + clasificadores especializados' framing while
honoring the binary-criterion PRISMA design.
"""
from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from biofloc_review.schema import PaperRecord
from biofloc_review.screening.classifier import CriterionAnswer, ask_criterion
from biofloc_review.screening.criteria import (
    CRITERIA_BY_ID,
    LLM_CRITERIA_IDS,
)
from biofloc_review.screening.decision import FinalDecision, aggregate
from biofloc_review.screening.prefilter import (
    is_abstract_too_short,
    is_eligible_doctype,
)

log = logging.getLogger(__name__)


class ScreeningState(TypedDict, total=False):
    record: PaperRecord
    model: str | None
    answers: dict[str, CriterionAnswer]
    final: FinalDecision


# ---------- Nodes ----------


def node_prefilter(state: ScreeningState) -> dict:
    """Evaluate the two deterministic criteria. Store as CriterionAnswer.

    Note: criterion E5 (language) was removed from the criteria set after
    a corpus review showed it added no informational value (Qwen3 reads
    Chinese/German/Korean abstracts natively, so a hard language exclusion
    is unjustified for a multilingual judge).
    """
    rec = state["record"]
    answers: dict[str, CriterionAnswer] = dict(state.get("answers") or {})

    # I4
    i4_ans, i4_ev = is_eligible_doctype(rec)
    answers["I4"] = CriterionAnswer(
        criterion_id="I4",
        answer=i4_ans,
        confidence=10,
        evidence=i4_ev,
        reasoning="determinista sobre rec.document_type",
        via="deterministic",
    )
    # D1
    d1_ans, d1_ev = is_abstract_too_short(rec)
    answers["D1"] = CriterionAnswer(
        criterion_id="D1",
        answer=d1_ans,
        confidence=10,
        evidence=d1_ev,
        reasoning="determinista sobre longitud de abstract",
        via="deterministic",
    )
    return {"answers": answers}


def node_llm_evaluator(state: ScreeningState) -> dict:
    """Run the 9 LLM-evaluated criteria in parallel via ThreadPoolExecutor.

    If a deterministic criterion already fires EXCLUDE (E5=YES) or I4=NO,
    we still run the LLM agents for audit-trail completeness. This guarantees
    each record has all 12 answers for downstream sensitivity analysis.
    """
    rec = state["record"]
    model = state.get("model")
    answers: dict[str, CriterionAnswer] = dict(state.get("answers") or {})

    def _run(cid: str) -> CriterionAnswer:
        return ask_criterion(rec, CRITERIA_BY_ID[cid], model=model)

    with ThreadPoolExecutor(max_workers=len(LLM_CRITERIA_IDS)) as ex:
        results = list(ex.map(_run, LLM_CRITERIA_IDS))

    for cid, ans in zip(LLM_CRITERIA_IDS, results):
        answers[cid] = ans
    return {"answers": answers}


def node_supervisor(state: ScreeningState) -> dict:
    answers = state.get("answers") or {}
    return {"final": aggregate(answers)}


# ---------- Compilation ----------


def build_graph():
    g = StateGraph(ScreeningState)
    g.add_node("prefilter", node_prefilter)
    g.add_node("llm_evaluator", node_llm_evaluator)
    g.add_node("supervisor", node_supervisor)
    g.add_edge(START, "prefilter")
    g.add_edge("prefilter", "llm_evaluator")
    g.add_edge("llm_evaluator", "supervisor")
    g.add_edge("supervisor", END)
    return g.compile()


_GRAPH = None


def _get_graph():
    global _GRAPH
    if _GRAPH is None:
        _GRAPH = build_graph()
    return _GRAPH


def screen_record(rec: PaperRecord, model: str | None = None) -> dict:
    """Run the screening graph on a single PaperRecord.

    Returns the final state dict (record, answers, final).
    """
    initial: ScreeningState = {"record": rec, "model": model}
    return _get_graph().invoke(initial)
