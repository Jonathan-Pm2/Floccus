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


def test_ui_command_launches_streamlit(monkeypatch, tmp_path):
    seen = {}

    def fake_call(argv):
        seen["argv"] = argv
        return 7

    monkeypatch.setattr(cli.subprocess, "call", fake_call)
    assert cli.main(["--home", str(tmp_path), "ui", "--port", "8600", "--headless"]) == 7
    argv = seen["argv"]
    assert argv[1:4] == ["-m", "streamlit", "run"]
    assert argv[4] == str(APP)
    assert argv[5:] == [
        "--server.port", "8600", "--server.headless", "true",
        "--browser.gatherUsageStats", "false", "--client.toolbarMode", "minimal",
        "--theme.primaryColor", "#2e8b7a", "--", "--home", str(tmp_path),
    ]


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
    at.sidebar.text_input[0].input("Autismo")
    at.sidebar.button[0].click().run()
    assert not at.exception
    assert (tmp_path / "Autismo" / "protocol.yaml").is_file()
    assert at.sidebar.selectbox(key="project").value == "Autismo"
