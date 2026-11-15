"""T13: the biofloc example project ships the published results (S8, S9)."""

from __future__ import annotations

import json
from pathlib import Path

from floccus_app import results
from floccus_app.i18n import t
from floccus_app.projects import create_project

APP = Path(__file__).resolve().parents[1] / "src" / "floccus_app" / "ui" / "app.py"
RELEASE = Path(__file__).resolve().parents[2] / "release_data"


def _published_counts() -> dict:
    return json.loads((RELEASE / "prisma_counts.json").read_text(encoding="utf-8"))


def test_biofloc_example_copies_published_results(tmp_path):
    project = create_project(tmp_path, "demo", "biofloc")
    out = project / "results" / "published"
    for name in results.EXAMPLE_FILES:
        assert (out / name).read_bytes() == (RELEASE / name).read_bytes()
    info = json.loads((out / "run_info.json").read_text(encoding="utf-8"))
    assert info["provider"] == "ollama"
    assert info["generation_model"] == "qwen3:14b"
    assert info["embedding_model"] == "qwen3-embedding:8b"
    assert info["temperature"] == 0


def test_blank_project_has_no_results(tmp_path):
    project = create_project(tmp_path, "mine")
    assert results.load_results(project) is None


def test_pipeline_steps_use_published_counts(tmp_path):
    c = _published_counts()
    r = results.load_results(create_project(tmp_path, "demo", "biofloc"))
    steps = results.pipeline_steps(r.counts, "es")
    assert [s["out"] for s in steps] == [
        c["identification"]["total"],
        c["deduplication"]["after_round2_ieee_merges"],
        c["screening_layer1"]["include"] + c["screening_layer1"]["human_review"],
        c["final_specificity_mode"]["to_eligibility"],
    ]
    assert len(r.decisions) == 17063


def test_figures_are_built_from_the_results(tmp_path):
    c = _published_counts()
    r = results.load_results(create_project(tmp_path, "demo", "biofloc"))
    sankey = results.fig_flow(r.counts, "en")
    assert sankey.data[0].type == "sankey"
    assert sum(v for s, v in zip(sankey.data[0].link.source, sankey.data[0].link.value) if s < 4) == (
        c["identification"]["total"]
    )
    final = results.fig_final_decisions(r.counts, "en")
    assert list(final.data[0].y) == [
        c["final_specificity_mode"]["include"],
        c["final_specificity_mode"]["human_review"],
        c["final_specificity_mode"]["exclude"],
    ]
    assert results.fig_sources(r.counts, "en").data[0].type == "bar"
    assert results.fig_criteria(r.decisions, "en").data[0].type == "bar"
    assert results.fig_clusters(r.eligible, "en").data[0].type == "bar"


def test_validation_crosstab_counts_every_sampled_record(tmp_path):
    r = results.load_results(create_project(tmp_path, "demo", "biofloc"))
    tab = results.validation_crosstab(r.validation, "reviewer_1")
    assert int(tab.to_numpy().sum()) == 465


def test_model_tab_text_shows_ollama_used_and_anthropic_option_without_urls_or_keys():
    for lang in ("es", "en"):
        text = t("model_used", lang) + t("model_alternatives", lang)
        assert "qwen3:14b" in text and "qwen3-embedding:8b" in text
        assert "Anthropic" in text and "Ollama" in text
        assert "http" not in text and "KEY=" not in text


def test_example_app_shows_results_without_errors(tmp_path, monkeypatch):
    from streamlit.testing.v1 import AppTest

    create_project(tmp_path, "demo", "biofloc")
    monkeypatch.setenv("FLOCCUS_HOME", str(tmp_path))
    monkeypatch.setenv("OLLAMA_URL", "http://secret-host:11434")
    at = AppTest.from_file(str(APP), default_timeout=60).run()
    assert not at.exception
    labels = [m.label for m in at.metric]
    assert t("m_include", "es") in labels
    page = " ".join(str(m.value) for m in at.markdown)
    assert "secret-host" not in page
