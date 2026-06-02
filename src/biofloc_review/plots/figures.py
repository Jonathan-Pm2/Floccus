"""
Funciones de figura para la sección de resultados.

Cada función:
  * recibe los datos ya procesados (DataFrame, dict, JSON cargado),
  * crea una matplotlib.Figure,
  * la devuelve sin guardarla. El guardado lo hace ``save_fig``.

Eso desacopla "qué plotear" de "dónde guardar", y deja flexibilidad para
embeber las figuras en notebooks o tests sin tocar disco.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from biofloc_review.plots.style import (
    PALETTE_SOURCE,
    PALETTE_VERDICT,
)


# --------------------------------------------------------------------------
# Fig 1. Distribución de registros por fuente y corpus.
# --------------------------------------------------------------------------
def fig_source_distribution(
    biofloc_df: pd.DataFrame,
    biplot_df: pd.DataFrame,
) -> plt.Figure:
    """
    Bar chart agrupado: número de registros por fuente, en cada corpus.

    Decisiones de diseño:
      * Barras horizontales — mejor lectura para etiquetas largas.
      * Anotación numérica al final de cada barra (evita que el lector
        adivine el valor exacto desde la escala).
      * Eje x en log con ticks solo en potencias de 10 (sin 2, 3, ..., 9
        intermedias que ensucian la lectura).
    """
    sources = ["scopus", "wos", "ieee", "scielo"]
    labels = ["Scopus", "WoS", "IEEE", "SciELO"]
    biofloc_counts = biofloc_df["source"].value_counts()
    biplot_counts = biplot_df["source"].value_counts()

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(7.5, 3.2), sharey=True)

    for ax, counts, title in [
        (ax_a, biofloc_counts, "Corpus biofloc"),
        (ax_b, biplot_counts, "Corpus biplot"),
    ]:
        vals = [counts.get(s, 0) for s in sources]
        colors = [PALETTE_SOURCE[s] for s in sources]
        bars = ax.barh(labels, vals, color=colors, edgecolor="white", linewidth=0.8)
        ax.set_xscale("log")
        ax.set_xlabel("Registros (escala logarítmica)")
        ax.set_title(title, loc="left", pad=8)
        # solo ticks en potencias de 10
        from matplotlib.ticker import LogLocator, NullLocator
        ax.xaxis.set_major_locator(LogLocator(base=10, numticks=10))
        ax.xaxis.set_minor_locator(NullLocator())
        # anotación al final de cada barra
        for bar, v in zip(bars, vals):
            ax.text(v * 1.08, bar.get_y() + bar.get_height() / 2,
                    f"{v:,}".replace(",", "."), va="center", fontsize=8.5)
        ax.grid(axis="x", alpha=0.3)
        ax.grid(axis="y", alpha=0)
        ax.margins(x=0.15)  # más espacio a la derecha para que la anotación no se corte

    fig.suptitle("Identificación: registros por fuente y corpus", y=1.02)
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------
# Fig 2. Reducción del corpus por capa (waterfall).
# --------------------------------------------------------------------------
def fig_dedup_waterfall(
    stages: dict[str, dict[str, int]],
) -> plt.Figure:
    """
    Waterfall mostrando cuánto reduce cada capa el corpus.

    Argumentos:
      stages = {
        "biofloc": {"identif": 22056, "capa1": 17326, "capa3": 17075},
        "biplot":  {"identif": 1401,  "capa1": 1235,  "capa3": 1222},
      }

    Decisiones:
      * Una columna por corpus.
      * Barras en azul Tableau oscuro; anotación de delta dentro de la barra
        en azul medio (no rojo, consistente con la paleta monocromática).
    """
    from biofloc_review.plots.style import TABLEAU_BLUE

    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.8), sharey=False)

    for ax, corpus in zip(axes, stages.keys()):
        s = stages[corpus]
        labels = ["Identif.", "Capa 1", "Capa 3 (final)"]
        values = [s["identif"], s["capa1"], s["capa3"]]
        diffs = [None, values[1] - values[0], values[2] - values[1]]

        x = np.arange(len(labels))
        bars = ax.bar(x, values, color=TABLEAU_BLUE["darkest"],
                      edgecolor="white", linewidth=0.8, width=0.62)

        # anotación de valor absoluto encima de cada barra
        for bar, v in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + values[0] * 0.015,
                    f"{v:,}".replace(",", "."), ha="center", fontsize=9)

        # anotación de delta DENTRO de cada barra (a media altura),
        # color azul claro para que contraste con el azul oscuro de fondo
        for i in (1, 2):
            d = diffs[i]
            pct = 100.0 * d / values[i - 1]
            ax.text(x[i], values[i] * 0.45,
                    f"{d:+,}".replace(",", ".") + f"\n({pct:+.1f}%)",
                    ha="center", va="center", fontsize=8.5,
                    color="white", alpha=0.95)

        ax.set_xticks(x, labels)
        ax.set_title(f"Corpus {corpus}", loc="left", pad=6)
        ax.set_ylabel("Registros")
        ax.set_ylim(0, values[0] * 1.12)
        ax.grid(axis="x", alpha=0)

    fig.suptitle("Reducción del corpus por capa de deduplicación", y=1.02)
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------
# Fig 3. Distribución de scores Capa 2 coloreado por veredicto Capa 3.
# --------------------------------------------------------------------------
def fig_capa2_scores_by_verdict(
    judged_pairs: list[dict],
    title: str,
) -> plt.Figure:
    """
    Histograma apilado: scores coseno de la Capa 2, coloreados por el
    veredicto que les dio la Capa 3.

    Esta es la figura clave metodológica: visualiza que el umbral coseno
    NO separa duplicados de no-duplicados sin la corrección del LLM.

    Layout: la leyenda va FUERA del área de datos (encima del plot) para
    evitar overlap con las barras altas que aparecen cerca del umbral.
    """
    scores_by = {"DUPLICATE": [], "NOT_DUPLICATE": [], "UNCLEAR": []}
    for p in judged_pairs:
        v = p.get("judge_verdict", "UNCLEAR")
        scores_by.setdefault(v, []).append(float(p["score"]))

    fig, ax = plt.subplots(figsize=(7.0, 3.8))

    bins = np.linspace(0.92, 1.001, 30)
    ax.hist(
        [scores_by["NOT_DUPLICATE"], scores_by["UNCLEAR"], scores_by["DUPLICATE"]],
        bins=bins,
        stacked=True,
        label=["NOT_DUPLICATE", "UNCLEAR", "DUPLICATE"],
        color=[PALETTE_VERDICT["NOT_DUPLICATE"], PALETTE_VERDICT["UNCLEAR"], PALETTE_VERDICT["DUPLICATE"]],
        edgecolor="white",
        linewidth=0.5,
    )
    ax.set_xlabel("Similitud coseno (Capa 2)")
    ax.set_ylabel("Pares candidatos")
    ax.set_title(title, loc="left", pad=24)

    # leyenda horizontal fuera del área de datos, justo debajo del título
    ax.legend(frameon=False, loc="upper right",
              bbox_to_anchor=(1.0, 1.08), ncol=3, fontsize=9,
              handletextpad=0.5, columnspacing=1.2)

    # línea vertical en el umbral
    ax.axvline(0.93, color="black", ls=":", lw=0.8, alpha=0.6)
    ax.text(0.931, ax.get_ylim()[1] * 0.96, r"$\tau = 0{,}93$",
            ha="left", va="top", fontsize=9, color="black", alpha=0.75)

    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------
# Fig 4. Distribución de confianza Capa 3.
# --------------------------------------------------------------------------
def fig_capa3_confidence(
    judged_pairs: list[dict],
    title: str,
) -> plt.Figure:
    """
    Confianza (0–10) reportada por el LLM, separada por veredicto.

    Lectura esperada: DUPLICATE y NOT_DUPLICATE tienen confianza alta
    (8–10); UNCLEAR concentrada en 5–6.
    """
    conf_by = {"DUPLICATE": [], "NOT_DUPLICATE": [], "UNCLEAR": []}
    for p in judged_pairs:
        v = p.get("judge_verdict", "UNCLEAR")
        c = int(p.get("judge_confidence", 0))
        conf_by.setdefault(v, []).append(c)

    fig, ax = plt.subplots(figsize=(7.0, 3.6))
    width = 0.27
    bins = np.arange(0, 12) - 0.5
    centers = (bins[:-1] + bins[1:]) / 2

    for i, v in enumerate(["NOT_DUPLICATE", "UNCLEAR", "DUPLICATE"]):
        counts, _ = np.histogram(conf_by[v], bins=bins)
        offset = (i - 1) * width
        ax.bar(centers + offset, counts, width=width, color=PALETTE_VERDICT[v],
               edgecolor="white", linewidth=0.4, label=v)

    ax.set_xlabel(r"Confianza del LLM (Capa 3)")
    ax.set_ylabel("Pares")
    ax.set_xticks(range(0, 11))
    ax.set_title(title, loc="left", pad=24)
    ax.legend(frameon=False, loc="upper right",
              bbox_to_anchor=(1.0, 1.08), ncol=3, fontsize=9,
              handletextpad=0.5, columnspacing=1.2)
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------
# Fig 5. Histograma de tamaños de cluster (Capa 1).
# --------------------------------------------------------------------------
def fig_cluster_size_histogram(
    full_df: pd.DataFrame,
    title: str,
) -> plt.Figure:
    """
    Distribución de tamaños de cluster después de la Capa 1.
    Cluster size = cuántos registros (de distintas fuentes) terminaron
    en un mismo cluster.
    """
    sizes = full_df.groupby("cluster_id").size().tolist()
    counts = Counter(sizes)
    keys = sorted(counts.keys())
    vals = [counts[k] for k in keys]

    from biofloc_review.plots.style import TABLEAU_BLUE
    from matplotlib.ticker import LogLocator, NullLocator
    fig, ax = plt.subplots(figsize=(6.8, 3.6))
    bars = ax.bar(keys, vals, color=TABLEAU_BLUE["darkest"],
                  edgecolor="white", linewidth=0.7, width=0.7)
    ax.set_yscale("log")
    ax.set_xlabel("Tamaño de cluster (registros agrupados)")
    ax.set_ylabel("Número de clusters (escala log)")
    ax.set_title(title, loc="left", pad=8)
    ax.set_xticks(keys)
    # solo potencias de 10 en el eje y
    ax.yaxis.set_major_locator(LogLocator(base=10, numticks=10))
    ax.yaxis.set_minor_locator(NullLocator())

    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.25,
                f"{v:,}".replace(",", "."), ha="center", fontsize=8.5)

    fig.tight_layout()
    return fig
