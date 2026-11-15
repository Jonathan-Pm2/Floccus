"""T8: Streamlit skeleton, i18n and the testable UI helpers."""

from __future__ import annotations

from pathlib import Path

import pytest
from biofloc_review.protocol import builtin_protocol

from floccus_app import cli, i18n, ui_logic
from floccus_app.projects import create_project

APP = Path(__file__).resolve().parents[1] / "src" / "floccus_app" / "ui" / "app.py"


def test_every_ui_string_has_spanish_and_english():
    for key, texts in i18n.STRINGS.items():
        assert set(texts) == {"es", "en"}, key
        assert texts["es"].strip() and texts["en"].strip(), key
    assert i18n.t("tab_protocol", "en") == "Protocol"
    assert i18n.t("tab_protocol", "es") == "Protocolo"


def test_rows_round_trip_reproduces_biofloc_protocol():
    p = builtin_protocol("biofloc")
    rows = ui_logic.protocol_to_rows(p)
    assert ui_logic.rows_to_protocol(p.name, p.language, p.topic, rows) == p


def test_invalid_rows_report_error_and_leave_file_untouched(tmp_path):
    project = create_project(tmp_path, "demo", "biofloc")
    before = (project / "protocol.yaml").read_bytes()
    p = builtin_protocol("biofloc")
    rows = ui_logic.protocol_to_rows(p)
    rows[1]["id"] = rows[0]["id"]
    error = ui_logic.save_rows(project, p.name, p.language, p.topic, rows)
    assert error == "duplicate criterion id 'I1'"
    assert (project / "protocol.yaml").read_bytes() == before


def test_valid_rows_are_saved(tmp_path):
    project = create_project(tmp_path, "demo", "biofloc")
    p = builtin_protocol("biofloc")
    rows = ui_logic.protocol_to_rows(p)
    assert ui_logic.save_rows(project, p.name, p.language, "new topic", rows) is None
    assert ui_logic.load_project_protocol(project).topic == "new topic"


def test_llm_settings_mask_the_api_key(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-secret-123")
    monkeypatch.setenv("ANTHROPIC_MODEL", "claude-x")
    s = ui_logic.llm_settings()
    assert s["provider"] == "anthropic"
    assert s["model"] == "claude-x"
    assert s["api_key"] == "set"
    assert "sk-secret-123" not in repr(s)


def _ui_argv(monkeypatch, tmp_path, *extra):
    seen = {}

    def fake_call(argv):
        seen["argv"] = argv
        return 7

    monkeypatch.setattr(cli.subprocess, "call", fake_call)
    assert cli.main(["--home", str(tmp_path), "ui", "--port", "8600", "--headless", *extra]) == 7
    return seen["argv"]


def test_ui_command_launches_streamlit_with_light_and_dark_themes(monkeypatch, tmp_path):
    argv = _ui_argv(monkeypatch, tmp_path)
    assert argv[1:4] == ["-m", "streamlit", "run"]
    assert argv[4] == str(APP)
    assert argv[5:] == [
        "--server.port", "8600", "--server.address", "localhost", "--server.headless", "true",
        "--browser.gatherUsageStats", "false", "--client.toolbarMode", "viewer",
        "--theme.light.primaryColor", "#2e8b7a", "--theme.dark.primaryColor", "#2e8b7a",
        "--", "--home", str(tmp_path),
    ]


def test_ui_command_can_force_dark_or_light(monkeypatch, tmp_path):
    for mode in ("dark", "light"):
        argv = _ui_argv(monkeypatch, tmp_path, "--theme", mode)
        i = argv.index("--theme.base")
        assert argv[i + 1] == mode
        assert argv[argv.index("--theme.primaryColor") + 1] == "#2e8b7a"
        assert "--theme.dark.primaryColor" not in argv


def test_theme_hint_is_translated():
    assert "⋮" in i18n.t("theme_hint", "es") and "⋮" in i18n.t("theme_hint", "en")


def test_app_boots_and_switches_language(tmp_path, monkeypatch):
    from streamlit.testing.v1 import AppTest

    create_project(tmp_path, "demo", "biofloc")
    monkeypatch.setenv("FLOCCUS_HOME", str(tmp_path))
    at = AppTest.from_file(str(APP), default_timeout=30).run()
    assert not at.exception
    assert any(t.label == "Protocolo" for t in at.tabs)
    at.sidebar.radio(key="lang").set_value("en").run()
    assert not at.exception
    assert any(t.label == "Protocol" for t in at.tabs)


def test_creating_a_project_from_the_form_selects_it(tmp_path, monkeypatch):
    from streamlit.testing.v1 import AppTest

    create_project(tmp_path, "demo", "biofloc")
    monkeypatch.setenv("FLOCCUS_HOME", str(tmp_path))
    at = AppTest.from_file(str(APP), default_timeout=30).run()
    at.sidebar.text_input[0].input("nueva-revision")
    at.sidebar.button[0].click().run()
    assert not at.exception
    assert (tmp_path / "nueva-revision" / "protocol.yaml").is_file()
    assert at.sidebar.selectbox(key="project").value == "nueva-revision"


def test_step_command_runs_the_cli_in_a_subprocess(tmp_path):
    argv = ui_logic.step_command(tmp_path, "demo", "screen-l1", limit=20)
    assert argv[1:3] == ["-m", "floccus_app.cli"]
    assert argv[3:] == ["--home", str(tmp_path), "run", "demo", "screen-l1", "--limit", "20"]
    assert ui_logic.step_command(tmp_path, "demo", "import")[-1] == "import"


def test_run_tab_for_own_project_offers_uploads_and_steps(tmp_path, monkeypatch):
    from streamlit.testing.v1 import AppTest

    create_project(tmp_path, "mine")
    monkeypatch.setenv("FLOCCUS_HOME", str(tmp_path))
    at = AppTest.from_file(str(APP), default_timeout=60).run()
    assert not at.exception
    labels = [b.label for b in at.button]
    for key in ("btn_import", "btn_dedup", "btn_l1", "btn_l2"):
        assert i18n.t(key, "es") in labels
