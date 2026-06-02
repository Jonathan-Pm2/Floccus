"""Layer 2 LLM screening — segundo pase BLIND sobre records que Layer 1
derivó a HUMAN_REVIEW, con input enriquecido vía OpenAlex + Europe PMC.

Diseño metodológico:
- Aplica los MISMOS 12 criterios PRISMA que Layer 1 (consistencia metodológica).
- Recibe input ENRIQUECIDO: title + abstract + keywords + venue +
  OpenAlex topics curados + Europe PMC MeSH terms + OpenAlex concepts.
- **NO ve la decisión de Layer 1** (blind dual review estilo Cochrane).
- Reusa la misma regla AND/OR de `decision.py`.

Funciones públicas:
- `build_user_prompt_layer2(criterion, paper, enriched)` — prompt extendido.
- `ask_criterion_layer2(paper, enriched, criterion, model)` — una llamada LLM.
- `screen_record_layer2(paper, enriched, model)` — pase completo (12 criterios).
"""
from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor

from biofloc_review.dedup.llm_judge import _ollama_chat, default_llm_model
from biofloc_review.schema import PaperRecord
from biofloc_review.screening.classifier import CriterionAnswer, parse_response
from biofloc_review.screening.criteria import (
    CRITERIA_BY_ID,
    LLM_CRITERIA_IDS,
    Criterion,
)
from biofloc_review.screening.decision import (
    FinalDecision,
    aggregate,
)

# Verdict constant para responses parseadas; coincide con classifier.parse_response.
ABSTAIN = "ABSTAIN"
from biofloc_review.screening.enricher import EnrichedRecord
from biofloc_review.screening.prefilter import (
    is_abstract_too_short,
    is_eligible_doctype,
    is_excluded_by_language,
)
from biofloc_review.screening.prompts import build_system_prompt

log = logging.getLogger(__name__)


USER_TEMPLATE_LAYER2 = """Información del artículo a evaluar:

TÍTULO:
{title}

ABSTRACT:
{abstract}

KEYWORDS (originales del export):
{keywords}

VENUE:
{venue}

---

METADATA ENRIQUECIDA (curated):
{enriched_block}

Aplica el criterio {criterion_id} y responde con el formato indicado.
"""


def _format_enriched_block(enriched: EnrichedRecord | None) -> str:
    """Format the enriched metadata into a structured block.

    If enriched is None or empty, returns "(metadata enriquecida no disponible)".
    """
    if enriched is None or not (enriched.oa_ok or enriched.pmc_ok):
        return "(metadata enriquecida no disponible para este record)"

    lines: list[str] = []
    if enriched.oa_topics:
        lines.append("OpenAlex topics curados:")
        for t in enriched.oa_topics[:8]:
            lines.append(f"  - {t}")
    if enriched.pmc_mesh_terms:
        lines.append("Europe PMC MeSH terms:")
        for m in enriched.pmc_mesh_terms[:12]:
            lines.append(f"  - {m}")
    if enriched.oa_concepts:
        # Concepts son más granulares y a veces ruidosos. Limitar a top-10.
        lines.append("OpenAlex concepts (granulares):")
        for c in enriched.oa_concepts[:10]:
            lines.append(f"  - {c}")
    if enriched.oa_abstract_full and len(enriched.oa_abstract_full) > 50:
        lines.append("OpenAlex abstract reconstruido (extracto):")
        lines.append(f"  {enriched.oa_abstract_full[:600]}...")
    if not lines:
        return "(metadata enriquecida vacía: OA/PMC respondieron pero sin contenido útil)"
    return "\n".join(lines)


def build_user_prompt_layer2(
    criterion: Criterion,
    paper: PaperRecord,
    enriched: EnrichedRecord | None,
) -> str:
    """Build the Layer 2 user prompt with enriched metadata block."""
    abstract = (paper.abstract.strip() if paper.abstract else "(no abstract)")[:3000]
    keywords = ", ".join(paper.keywords) if paper.keywords else "(none)"
    venue = paper.venue or "(none)"
    return USER_TEMPLATE_LAYER2.format(
        criterion_id=criterion.id,
        title=paper.title.strip(),
        abstract=abstract,
        keywords=keywords,
        venue=venue,
        enriched_block=_format_enriched_block(enriched),
    )


def ask_criterion_layer2(
    paper: PaperRecord,
    enriched: EnrichedRecord | None,
    criterion: Criterion,
    model: str | None = None,
) -> CriterionAnswer:
    """Evaluate one criterion against an enriched record (Layer 2)."""
    system = build_system_prompt(criterion)
    user = build_user_prompt_layer2(criterion, paper, enriched)
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    try:
        text = _ollama_chat(messages, model=model or default_llm_model())
    except Exception as e:
        log.warning("Ollama call failed (L2) for criterion %s on %r: %s",
                    criterion.id, paper.title[:60], e)
        return CriterionAnswer(
            criterion_id=criterion.id,
            answer=ABSTAIN,
            confidence=0,
            evidence="(error LLM)",
            reasoning=f"LLM error: {type(e).__name__}: {e}",
            via="llm",
            raw="",
        )
    return parse_response(criterion.id, text)


def screen_record_layer2(
    paper: PaperRecord,
    enriched: EnrichedRecord | None,
    model: str | None = None,
) -> dict:
    """Full Layer 2 screening: 3 deterministic + 9 LLM criteria → decision.

    Returns a state dict with `answers` (dict of criterion_id → CriterionAnswer)
    and `final` (FinalDecision).
    """
    answers: dict[str, CriterionAnswer] = {}

    # 1. Deterministic prefilter (same as Layer 1)
    e5_ans, e5_ev = is_excluded_by_language(paper)
    answers["E5"] = CriterionAnswer(
        criterion_id="E5", answer=e5_ans, confidence=10,
        evidence=e5_ev, reasoning="determinista sobre rec.language",
        via="deterministic",
    )
    i4_ans, i4_ev = is_eligible_doctype(paper)
    answers["I4"] = CriterionAnswer(
        criterion_id="I4", answer=i4_ans, confidence=10,
        evidence=i4_ev, reasoning="determinista sobre rec.document_type",
        via="deterministic",
    )
    d1_ans, d1_ev = is_abstract_too_short(paper)
    answers["D1"] = CriterionAnswer(
        criterion_id="D1", answer=d1_ans, confidence=10,
        evidence=d1_ev, reasoning="determinista sobre longitud de abstract",
        via="deterministic",
    )

    # 2. 9 LLM criteria in parallel
    def _run(cid: str) -> CriterionAnswer:
        return ask_criterion_layer2(paper, enriched, CRITERIA_BY_ID[cid], model=model)

    with ThreadPoolExecutor(max_workers=len(LLM_CRITERIA_IDS)) as ex:
        results = list(ex.map(_run, LLM_CRITERIA_IDS))
    for cid, ans in zip(LLM_CRITERIA_IDS, results):
        answers[cid] = ans

    # 3. Apply the same AND/OR aggregation rule
    final: FinalDecision = aggregate(answers)

    return {"answers": answers, "final": final}
