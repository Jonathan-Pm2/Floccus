"""
Estilo de figuras tipo "publicación seria" (Nature/Science/Phys Rev).

Concepto general
----------------
matplotlib por defecto trae una estética utilitaria pensada para
exploración interactiva, no para impresión. Para llegar a calidad
publicación hay tres ejes que tunear:

  1. **Tipografía**: que las fuentes del gráfico matcheen las del paper
     (serifa, similar a Computer Modern de LaTeX). Si la figura tiene
     Helvetica y el paper Times, se nota inmediatamente.
  2. **Densidad y blanco**: figuras académicas favorecen menos tinta,
     menos cuadrícula, spines selectivos (sacar el top y el right es la
     convención Tufte/Cleveland).
  3. **Color**: paletas seguras para impresión y daltonismo. La
     viridis-family de matplotlib y la cubehelix son las apuestas
     seguras; tab10 para categóricos sirve y es accesible.

Lo que hace `apply_paper_style()` es aplicar todo eso vía
``matplotlib.rcParams``. No imprime nada; solo configura el backend
global.
"""
from __future__ import annotations

import logging
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt

log = logging.getLogger(__name__)


# Paleta Tableau "Blue Light" — gradiente monocromático azul.
#
# Por qué monocromático: revistas científicas (Nature, Science, JAMA)
# favorecen escalas de un solo hue porque:
#   1. Imprimen sin pérdida de información en B/N (la luminancia ordena los
#      datos de forma natural).
#   2. Evitan implicaciones semánticas no deseadas (rojo = malo, verde = bueno).
#   3. Son accesibles para cualquier tipo de daltonismo (todos los azules
#      se distinguen entre sí por luminancia).
TABLEAU_BLUE = {
    "darkest":   "#08306B",
    "dark":      "#08519C",
    "mediumD":   "#2171B5",
    "medium":    "#4292C6",
    "mediumL":   "#6BAED6",
    "light":     "#9ECAE1",
    "lighter":   "#C6DBEF",
    "lightest":  "#DEEBF7",
    "neutral":   "#8C8C8C",
}

# Categórico genérico: puntos discretos del gradiente, con diferencia de
# luminancia suficiente para distinguir en color y en B/N.
PALETTE_CATEGORICAL = [
    TABLEAU_BLUE["darkest"],
    TABLEAU_BLUE["mediumD"],
    TABLEAU_BLUE["mediumL"],
    TABLEAU_BLUE["light"],
    TABLEAU_BLUE["neutral"],
    TABLEAU_BLUE["dark"],
    TABLEAU_BLUE["medium"],
    TABLEAU_BLUE["lighter"],
]

# Veredictos: el más relevante (DUPLICATE) en el azul más oscuro (mayor
# peso visual); UNCLEAR en gris neutral (fuera del eje azul, refuerza la
# semántica de "indeciso").
PALETTE_VERDICT = {
    "DUPLICATE":     TABLEAU_BLUE["darkest"],
    "NOT_DUPLICATE": TABLEAU_BLUE["mediumL"],
    "UNCLEAR":       TABLEAU_BLUE["neutral"],
}

# Fuentes: gradiente por volumen (WoS domina → más oscuro; IEEE menor → más claro).
PALETTE_SOURCE = {
    "wos":      TABLEAU_BLUE["darkest"],
    "scopus":   TABLEAU_BLUE["dark"],
    "openalex": TABLEAU_BLUE["mediumD"],
    "scielo":   TABLEAU_BLUE["mediumL"],
    "ieee":     TABLEAU_BLUE["light"],
}


def apply_paper_style(font_family: str = "serif") -> None:
    """
    Aplica el estilo global. Llamar UNA VEZ al arrancar el script de plots.

    Decisiones documentadas:

    * ``style.use("seaborn-v0_8-paper")`` — base seaborn con tamaños
      reducidos pensados para columna de revista (~3.5 in de ancho).
    * ``font.family = 'serif'`` — para machear LaTeX. Si el sistema no tiene
      Computer Modern instalado, matplotlib cae a una serifa similar (STIX).
    * ``mathtext.fontset = 'stix'`` — la matemática en línea (p. ej.
      ``$\\tau$``) usa STIX, que es virtualmente idéntica a CM en pantalla.
    * ``axes.spines.{top,right} = False`` — convención Tufte: quitar tinta
      no informativa. Solo dejamos ejes izquierdo e inferior.
    * ``axes.grid = True`` con ``alpha = 0.3`` — grilla tenue, solo
      horizontal, ayuda a leer escalas sin competir con los datos.
    * ``savefig.bbox = 'tight'`` — recorta espacio blanco sobrante al
      exportar; importante para que la figura ocupe lo justo en el .tex.
    * ``savefig.dpi = 300`` — relevante solo para PNG. El PDF es vectorial.
    """
    plt.style.use("seaborn-v0_8-paper")

    mpl.rcParams.update({
        # Tipografía
        "font.family": font_family,
        "font.serif": ["STIX", "DejaVu Serif", "Times New Roman"],
        "mathtext.fontset": "stix",
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "figure.titlesize": 12,
        # Spines: la regla Tufte de quitar tinta no informativa
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.8,
        # Grilla
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linewidth": 0.5,
        "axes.axisbelow": True,
        # Ticks
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.size": 3,
        "ytick.major.size": 3,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        # Líneas y markers
        "lines.linewidth": 1.4,
        "lines.markersize": 4,
        # Figura
        "figure.dpi": 100,           # display
        "figure.figsize": (6.5, 4),  # single-column journal width
        # Export
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
        "pdf.fonttype": 42,          # embeber TrueType en PDF (no curvas) — importante para revistas
        "ps.fonttype": 42,
        # Paleta categórica por defecto
        "axes.prop_cycle": mpl.cycler(color=PALETTE_CATEGORICAL),
    })


def save_fig(fig: plt.Figure, out_dir: Path, stem: str,
             formats: tuple[str, ...] = ("pdf", "png")) -> list[Path]:
    """
    Guarda la figura en múltiples formatos.

    * PDF: vectorial, lo que el `.tex` carga vía ``\\includegraphics``.
      Sin pérdida, escala infinita, se puede editar en Inkscape.
    * PNG: por si querés previsualizar en chat, Slack, README. 300 DPI.

    Devuelve la lista de paths escritos.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for fmt in formats:
        p = out_dir / f"{stem}.{fmt}"
        fig.savefig(p, format=fmt)
        paths.append(p)
        log.info("Wrote %s", p)
    plt.close(fig)
    return paths
