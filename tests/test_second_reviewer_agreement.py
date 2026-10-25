"""Kappa statistics and workbook validation of second_reviewer_agreement.py (no data files read)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from second_reviewer_agreement import (confusion, has_problems, kappa,  # noqa: E402
                                       kappa_from_table, pabak, validate_decisions)

# 3x3 example, rows/cols ordered low -> high; margins rows 12,10,8 cols 14,9,7, n=30
T3 = np.array([[10, 2, 0], [3, 5, 2], [1, 2, 5]])


def expand(t):
    """Two label vectors whose cross-table is t."""
    a, b = [], []
    for i in range(t.shape[0]):
        for j in range(t.shape[1]):
            a += [i] * int(t[i, j])
            b += [j] * int(t[i, j])
    return np.array(a), np.array(b)


def test_binary_kappa_and_pabak():
    t = np.array([[20, 5], [10, 15]])       # p_o = 0.7, p_e = 0.5
    assert kappa_from_table(t) == pytest.approx(0.4)
    assert pabak(t) == pytest.approx(0.4)   # 2 * 0.7 - 1


def test_three_level_kappas_by_hand():
    assert kappa_from_table(T3) == pytest.approx(286 / 586)                    # (20/30 - 314/900) / (1 - 314/900)
    assert kappa_from_table(T3, "linear") == pytest.approx(226 / 391)          # 1 - 165/391
    assert kappa_from_table(T3, "quadratic") == pytest.approx(196 / 293.5)     # 1 - 97.5/293.5


def test_perfect_and_two_level_weights_coincide():
    assert kappa_from_table(np.diag([4, 3, 5]), "quadratic") == pytest.approx(1.0)
    t = np.array([[7, 3], [2, 8]])
    assert kappa_from_table(t, "linear") == pytest.approx(kappa_from_table(t))
    assert kappa_from_table(t, "quadratic") == pytest.approx(kappa_from_table(t))


def test_confusion_and_label_kappa():
    a, b = expand(T3)
    assert (confusion(a, b, [0, 1, 2]) == T3).all()
    assert kappa(a, b, [0, 1, 2], "linear") == pytest.approx(226 / 391)
    with pytest.raises(ValueError):
        confusion(["x"], ["Excluir"], ["Excluir", "Dudoso", "Incluir"])


def test_against_sklearn():
    skm = pytest.importorskip("sklearn.metrics")
    rng = np.random.default_rng(0)
    a = rng.integers(0, 3, 200)
    b = np.where(rng.random(200) < 0.6, a, rng.integers(0, 3, 200))
    for w in (None, "linear", "quadratic"):
        assert kappa(a, b, [0, 1, 2], w) == pytest.approx(skm.cohen_kappa_score(a, b, weights=w))


def sheet(decisions):
    return pd.DataFrame({"N°": range(1, len(decisions) + 1), "Título": [f"t{i}" for i in range(len(decisions))],
                         "Decisión": decisions,
                         "Motivo principal de exclusión": [None] * len(decisions),
                         "Comentario": [None] * len(decisions)})


def test_validate_complete_sheet_maps_levels():
    out, p = validate_decisions(sheet(["Incluir", " dudoso ", "EXCLUIR"]), expected_n=[1, 2, 3])
    assert not has_problems(p)
    assert out["r2_level"].tolist() == ["Incluir", "Dudoso", "Excluir"]
    assert out["r2_binary"].tolist() == ["cumple", "cumple", "no cumple"]


def test_validate_reports_blank_invalid_and_missing():
    out, p = validate_decisions(sheet(["Incluir", None, "Quizás", "", "Excluir"]),
                                expected_n=[1, 2, 3, 4, 5, 6])
    assert p["blank"] == [2, 4]
    assert p["invalid"] == [[3, "Quizás"]]
    assert p["missing_n"] == [6]
    assert has_problems(p)
    assert out["n"].tolist() == [1, 5]
