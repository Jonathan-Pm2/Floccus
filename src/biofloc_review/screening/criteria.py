"""PRISMA inclusion / exclusion / human-routing criteria.

Each criterion is a binary Sí/No question evaluated over (title + abstract +
keywords + venue). The aggregator (decision.py) applies AND for inclusion
criteria, OR for exclusion criteria, and routes to human review for any
derivation criterion.

Three criteria are deterministic (evaluated without an LLM call): I4 from
document_type, E5 from language, D1 from abstract length. The remaining nine
are LLM-evaluated.

This list is the **published methodology** for the manuscript. Reformatted
into a table for §3 Cribado.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


CriterionType = Literal["inclusion", "exclusion", "derivation"]


@dataclass(frozen=True)
class Criterion:
    id: str                  # e.g. "I1", "E1", "D1"
    type: CriterionType
    question: str            # Spanish yes/no question for the LLM
    justification: str       # Why this criterion exists (manuscript text)
    deterministic: bool      # True if evaluated without LLM
    # Positive phrasing used only in the "sensitivity" operating mode. When
    # set, the classifier asks this question and inverts the answer, so the
    # criterion keeps its meaning while the model never answers a negation.
    positive_question: str | None = None


CRITERIA: list[Criterion] = [
    # ---------- INCLUSION (must ALL pass) ----------
    Criterion(
        id="I1",
        type="inclusion",
        question=(
            "¿El artículo aborda flóculos microbianos, biofloc, lodos activados, "
            "perifiton o agregados bacterianos como objeto de estudio?"
        ),
        justification="Define el dominio biológico de la revisión.",
        deterministic=False,
    ),
    Criterion(
        id="I2",
        type="inclusion",
        question=(
            "¿El artículo presenta, aplica, valida o discute al menos un modelo "
            "matemático, ecuación, simulación numérica o expresión cuantitativa "
            "formal para describir el comportamiento del sistema biológico? "
            "(La palabra \"modelo\" sola no basta: tiene que haber matemática "
            "real — ecuaciones diferenciales, cinéticas tipo Monod, balances de "
            "masa, simulaciones CFD, modelos poblacionales, ASM, etc.)"
        ),
        justification="Criterio clave: filtra papers donde 'modelo' es retórico.",
        deterministic=False,
    ),
    Criterion(
        id="I3",
        type="inclusion",
        question=(
            "¿El contexto es acuático/biológico aplicado: acuicultura, "
            "tratamiento de aguas residuales, tratamiento de agua, sistemas de "
            "cultivo, biorreactores o calidad del agua?"
        ),
        justification="Filtra fuera contextos no aplicables (suelo, atmósfera, clínico).",
        deterministic=False,
    ),
    Criterion(
        id="I4",
        type="inclusion",
        question="¿Es un artículo original de revista (research article) o una revisión (review)?",
        justification="Coherente con PRISMA: excluye editoriales, cartas, actas.",
        deterministic=True,
    ),

    # ---------- EXCLUSION (any one fires excludes) ----------
    Criterion(
        id="E1",
        type="exclusion",
        question=(
            "¿El término 'modelo' se refiere únicamente a un organismo modelo "
            "(ej. 'rat model', 'mouse model', 'fish as model organism') y NO a "
            "modelación matemática?"
        ),
        justification="Captura falso positivo más probable: animal modelo vs modelo matemático.",
        deterministic=False,
    ),
    Criterion(
        id="E2",
        type="exclusion",
        question=(
            "¿El estudio es puramente experimental/empírico SIN componente "
            "cuantitativo formal de modelación (solo reporta mediciones, sin "
            "ecuaciones ni simulaciones)?"
        ),
        justification="Descarta estudios solo descriptivos.",
        deterministic=False,
        positive_question=(
            "¿El estudio incluye algún componente cuantitativo formal de "
            "modelación (ecuaciones, modelos matemáticos o simulaciones), y no "
            "solo el reporte de mediciones?"
        ),
    ),
    Criterion(
        id="E3",
        type="exclusion",
        question=(
            "¿El SISTEMA PRINCIPAL bajo estudio se encuentra FUERA de un medio "
            "acuoso?\n\n"
            "Responder YES (excluir) solo si el estudio se realiza sobre material "
            "o sistema NO acuoso: suelo, atmósfera abierta, alimentos sólidos, "
            "tejido clínico humano, fármacos sólidos o procesos industriales "
            "secos.\n\n"
            "Responder NO si el sistema principal es una planta de tratamiento "
            "de aguas residuales (PTAR/WWTP/STP), un sistema acuícola o un "
            "biorreactor de suspensión líquida — incluso cuando el artículo:\n"
            "  - reporte emisiones de gases (GHG, CO2, CH4, N2O) generadas POR la planta acuosa,\n"
            "  - describa tratamiento de lixiviados (landfill leachate) que ingresan a la planta,\n"
            "  - mencione contexto 'industrial' referido a aguas o efluentes industriales.\n\n"
            "Lo decisivo es DÓNDE ocurre el sistema biológico bajo estudio."
        ),
        justification=(
            "Refuerza el dominio acuático. Calibrado tras pilot n=100 (2026-05-30): "
            "el prompt original llevaba al modelo a confundir menciones colaterales "
            "de GHG/landfill/industrial con el contexto primario del estudio "
            "(5/100 falsos positivos identificados). El prompt actual explicita "
            "que lo decisivo es la localización del sistema biológico estudiado, "
            "no las palabras que aparecen en el abstract."
        ),
        deterministic=False,
    ),
    Criterion(
        id="E4",
        type="exclusion",
        question=(
            "¿El estudio trata de coagulación/floculación química pura (con "
            "coagulantes inorgánicos como alumbre, cloruro férrico) SIN "
            "componente microbiológico/biológico?"
        ),
        justification="Interés es flóculos BIOLÓGICOS, no químicos de potabilización.",
        deterministic=False,
    ),
    # ---------- DERIVATION (any one fires → HUMAN_REVIEW; never EXCLUDE) ----------
    Criterion(
        id="D1",
        type="derivation",
        question="¿El abstract es muy breve (<100 palabras) y no permite decidir con confianza?",
        justification="Falta de información suficiente — preservar para revisión humana.",
        deterministic=True,
    ),
    Criterion(
        id="D2",
        type="derivation",
        question=(
            "¿El artículo menciona modelación pero el tipo de modelo NO es "
            "claro (¿matemático? ¿conceptual? ¿in silico estadístico?)?"
        ),
        justification="Ambigüedad sobre I2: preservar para revisión humana.",
        deterministic=False,
    ),
    Criterion(
        id="D3",
        type="derivation",
        question=(
            "¿El artículo toca biofloc/lodos pero en un contexto mixto (ej. "
            "parte agua + parte suelo)?"
        ),
        justification="Ambigüedad sobre I3: preservar para revisión humana.",
        deterministic=False,
    ),
]


# Convenience views
CRITERIA_BY_ID: dict[str, Criterion] = {c.id: c for c in CRITERIA}
INCLUSION_IDS: tuple[str, ...] = tuple(c.id for c in CRITERIA if c.type == "inclusion")
EXCLUSION_IDS: tuple[str, ...] = tuple(c.id for c in CRITERIA if c.type == "exclusion")
DERIVATION_IDS: tuple[str, ...] = tuple(c.id for c in CRITERIA if c.type == "derivation")
LLM_CRITERIA_IDS: tuple[str, ...] = tuple(c.id for c in CRITERIA if not c.deterministic)
DETERMINISTIC_CRITERIA_IDS: tuple[str, ...] = tuple(c.id for c in CRITERIA if c.deterministic)
