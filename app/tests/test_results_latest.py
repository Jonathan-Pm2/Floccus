"""T10: results pages work for the user's own projects (latest run)."""

from __future__ import annotations

import pandas as pd

from floccus_app import pipeline, results
from floccus_app.projects import create_project
from floccus_app.ui_logic import load_project_protocol

from test_pipeline_screen import SCOPUS, fake_chat  # noqa: E402

WOS = pd.DataFrame({"Article Title": ["Paper 0", "Other"], "DOI": ["10.1/0", "10.9/9"],
                    "Abstract": ["microplastics in rivers " * 40] * 2, "Publication Year": [2020, 2021],
                    "UT (Unique ID)": ["w1", "w2"]})


def _finished_project(tmp_path):
    p = create_project(tmp_path, "demo")
    pipeline.add_export(p, "scopus", "s.csv", SCOPUS.to_csv(index=False).encode())
    xls = tmp_path / "w.xlsx"
    WOS.to_excel(xls, index=False)
    pipeline.add_export(p, "wos", "w.xlsx", xls.read_bytes())
    pipeline.import_records(p)
    pipeline.deduplicate(p)
    pipeline.screen(p, layer=1, chat=fake_chat, model="m")
    pipeline.consolidate(p)
    return p


def test_partial_project_has_no_results(tmp_path):
    p = create_project(tmp_path, "demo")
    pipeline.add_export(p, "scopus", "s.csv", SCOPUS.to_csv(index=False).encode())
    pipeline.import_records(p)
    assert results.load_results(p) is None


def test_latest_results_load_and_plot_with_any_sources(tmp_path):
    p = _finished_project(tmp_path)
    r = results.load_results(p)
    assert r.kind == "own"
    assert r.counts["identification"]["by_source"] == {"scopus": 6, "wos": 2}
    flow = results.fig_flow(r.counts, "en")
    src = flow.data[0].link.source
    assert sum(v for s, v in zip(src, flow.data[0].link.value) if s < 2) == 8
    assert list(results.fig_sources(r.counts, "en").data[0].x) == ["Scopus", "Web of Science"]
    ids = [c.id for c in load_project_protocol(p).criteria]
    crit = results.fig_criteria(r.decisions, "en", ids)
    assert list(crit.data[0].x) == ids
    assert results.pipeline_steps(r.counts, "en")[1]["out"] == 7
    assert r.validation is None and r.eligible is None


def test_published_example_still_takes_precedence(tmp_path):
    p = create_project(tmp_path, "ex", "biofloc")
    assert results.load_results(p).kind == "example"


def test_app_shows_own_project_results(tmp_path, monkeypatch):
    from pathlib import Path

    from streamlit.testing.v1 import AppTest

    from floccus_app.i18n import t

    _finished_project(tmp_path)
    monkeypatch.setenv("FLOCCUS_HOME", str(tmp_path))
    app = Path(__file__).resolve().parents[1] / "src" / "floccus_app" / "ui" / "app.py"
    at = AppTest.from_file(str(app), default_timeout=60).run()
    assert not at.exception
    assert t("own_results_note", "es") in [c.value for c in at.caption]
    assert any(m.label == t("m_include", "es") and m.value == "4" for m in at.metric)
