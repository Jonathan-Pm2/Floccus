"""Pure template helpers of scripts/render_round2_figures.py (no data files)."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from render_round2_figures import LABELS_EN, LABELS_ES, cluster_rows, cluster_tex, prisma_tex, tex_int  # noqa: E402

COUNTS = {"identified": 22056, "by_source": {"scopus": 6380, "wos": 10685, "ieee": 545, "scielo": 4446},
          "unique": 17063, "l1_include": 1205, "l1_exclude": 11633, "l1_human_review": 4225,
          "l2_include": 180, "l2_exclude": 547, "include": 1385, "human_review": 3498}


def test_tex_int_uses_manuscript_thousands_separator():
    assert tex_int(17063) == "17{,}063"
    assert tex_int(545) == "545"
    assert tex_int(1242000) == "1{,}242{,}000"


def test_prisma_tex_derives_every_quantity_from_the_counts():
    tex = prisma_tex(COUNTS, "en", header="%% h\n")
    assert tex.startswith("%% h\n%% Input:")
    for s in ("$n=22{,}056$", "$n=17{,}063$", "$n=4{,}993$", "$n=11{,}633$", "$n=5{,}430$",
              "$n=4{,}225$", "$n=547$", "$n=180$", "$n=4{,}883$", "($82.8\\%$)", "($17.2\\%$"):
        assert s in tex, s
    es = prisma_tex(COUNTS, "es", header="", dups_note="Capas 1+2+3 + 98 fusiones IEEE")
    assert "Registros identificados" in es and "98 fusiones IEEE" in es


def test_cluster_rows_sorted_by_size_then_id_with_focus_in_bold():
    sizes = {3: 10, 15: 5, 1: 10, 20: 7, -1: 4}
    med = {3: 2019, 15: 2014, 1: 2010, 20: 2006}
    rows = cluster_rows(sizes, med, LABELS_EN).split("\n")
    assert [r.split(" &")[0].strip() for r in rows] == ["C01", "C03$^{\\bigstar}$", "C20", "C15$^{\\bigstar}$"]
    assert "\\textbf{Biofloc in aquaculture} & $\\mathbf{10}$ & $\\mathbf{27.8}$ & $2019$" in rows[1]
    assert "C20 & Nitrification / denitrification & $7$ & $19.4$ & $2006$" in rows[2]
    assert set(LABELS_ES) == set(LABELS_EN) == set(range(25))


def test_cluster_tex_totals_include_noise():
    tex = cluster_tex({3: 10, 15: 5, -1: 5}, {3: 2019, 15: 2014}, "es")
    assert "sobre los $20$ embeddings" in tex
    assert "\\emph{No agrupados (ruido)} & $5$ & $25.0$" in tex
    assert "$\\mathbf{20}$ & $\\mathbf{100.0}$" in tex
