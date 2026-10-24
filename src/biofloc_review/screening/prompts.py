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


# Sensitivity operating mode: the model reasons and quotes evidence BEFORE
# committing to a verdict. With the answer-first template above, the verdict
# was sometimes the opposite of the reasoning on negatively phrased criteria
# (E2), excluding relevant records (validation, 2026-10-24).
SYSTEM_PROMPT_TEMPLATE_REASONING_FIRST = """Eres un revisor sistemático experto en revisiones PRISMA sobre modelado de flóculos biológicos en sistemas acuáticos. Respondes UNA pregunta sobre un artículo mirando su título, abstract, palabras clave y venue.

CRITERIO {criterion_id} ({criterion_type}):
{question}

Justificación del criterio: {justification}

Primero razona y cita la evidencia; SOLO DESPUÉS decide. La respuesta final debe ser coherente con tu razonamiento.
- YES = la respuesta a la pregunta es "Sí".
- NO = la respuesta a la pregunta es "No".
- UNCLEAR = la información disponible no permite decidir con confianza razonable.

Responde EXACTAMENTE en este formato y en este orden (sin texto antes ni después):
REASONING: <una o dos frases>
EVIDENCE: <fragmento exacto del título/abstract/keywords/venue, o "(ninguna)">
CONFIDENCE: <0-10>
ANSWER: YES | NO | UNCLEAR
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


def build_system_prompt(criterion: Criterion, mode: str = "specificity") -> str:
    """Specificity mode reproduces the published prompt exactly. Sensitivity
    mode uses the reasoning-first template and, if the criterion defines one,
    its positive question (the classifier inverts the answer)."""
    if mode == "sensitivity":
        return SYSTEM_PROMPT_TEMPLATE_REASONING_FIRST.format(
            criterion_id=criterion.id,
            criterion_type=criterion.type,
            question=criterion.positive_question or criterion.question,
            justification=criterion.justification,
        )
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
