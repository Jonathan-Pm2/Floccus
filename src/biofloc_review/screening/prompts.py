"""System prompt template for per-criterion binary evaluation.

Each LLM call answers ONE criterion question over (title + abstract + keywords
+ venue). Output is strictly YES | NO | UNCLEAR with evidence + reasoning,
matching the audit-trail format from dedup/llm_judge.py.

The template is parameterized by the criterion (question, justification, type)
so that prompts remain consistent across criteria and the difference is only
the question being asked. This makes per-criterion calibration easier to
defend in the manuscript.
"""
from __future__ import annotations

from biofloc_review.screening.criteria import Criterion


SYSTEM_PROMPT_TEMPLATE = """Eres un revisor sistemático experto en revisiones PRISMA sobre modelado de flóculos biológicos en sistemas acuáticos. Tu tarea es responder UNA pregunta binaria (Sí/No) sobre un artículo, mirando su título, abstract, palabras clave y venue.

CRITERIO {criterion_id} ({criterion_type}):
{question}

Justificación del criterio: {justification}

Reglas de respuesta:
- YES = el criterio SE CUMPLE (la respuesta a la pregunta es "Sí").
- NO = el criterio NO se cumple (la respuesta es "No").
- UNCLEAR = la información disponible NO permite decidir con confianza razonable.

Responde EXACTAMENTE en este formato (sin texto antes ni después):
ANSWER: YES | NO | UNCLEAR
CONFIDENCE: <0-10>
EVIDENCE: <fragmento exacto del título/abstract/keywords/venue que justifica la respuesta, o "(ninguna)">
REASONING: <una frase breve>
"""


USER_TEMPLATE = """Información del artículo a evaluar:

TÍTULO:
{title}

ABSTRACT:
{abstract}

KEYWORDS:
{keywords}

VENUE:
{venue}

Aplica el criterio {criterion_id} y responde con el formato indicado.
"""


def build_system_prompt(criterion: Criterion) -> str:
    return SYSTEM_PROMPT_TEMPLATE.format(
        criterion_id=criterion.id,
        criterion_type=criterion.type,
        question=criterion.question,
        justification=criterion.justification,
    )


def build_user_prompt(criterion: Criterion, title: str, abstract: str | None,
                       keywords: list[str], venue: str | None) -> str:
    abstract_str = (abstract.strip() if abstract else "(no abstract)")[:3000]
    keywords_str = ", ".join(keywords) if keywords else "(none)"
    venue_str = venue if venue else "(none)"
    return USER_TEMPLATE.format(
        criterion_id=criterion.id,
        title=title.strip(),
        abstract=abstract_str,
        keywords=keywords_str,
        venue=venue_str,
    )
