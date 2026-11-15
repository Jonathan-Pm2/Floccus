"""T14: professional open-access presentation and safe feedback channel (S10-S12)."""

from __future__ import annotations

from pathlib import Path
from urllib.parse import parse_qs, urlparse

import yaml

from floccus_app import branding
from floccus_app.projects import create_project

REPO = Path(__file__).resolve().parents[2]
APP = REPO / "app" / "src" / "floccus_app" / "ui" / "app.py"
TEMPLATES = REPO / ".github" / "ISSUE_TEMPLATE"


def test_project_identity():
    assert branding.REPO_URL == "https://github.com/Jonathan-Pm2/Floccus"
    assert branding.GENTLE_SHELL_URL == "https://github.com/Gentleman-Programming/gentle-shell"
    assert branding.LICENSE == "Apache-2.0"
    assert branding.DOI == "10.5281/zenodo.21926426"
    assert branding.version()


def test_feedback_url_prefills_the_issue_form():
    url = branding.feedback_url("bug", page="Resultados", lang="es")
    parsed = urlparse(url)
    assert parsed.scheme == "https"
    assert parsed.netloc == "github.com"
    assert parsed.path == "/Jonathan-Pm2/Floccus/issues/new"
    q = parse_qs(parsed.query)
    assert q["template"] == ["bug_report.yml"]
    assert q["version"] == [branding.version()]
    assert q["page"] == ["Resultados"]
    assert q["language"] == ["es"]
    idea = parse_qs(urlparse(branding.feedback_url("idea", page="x", lang="en")).query)
    assert idea["template"] == ["feature_request.yml"]


def _form(name: str) -> dict:
    return yaml.safe_load((TEMPLATES / name).read_text(encoding="utf-8"))


def test_issue_forms_have_the_fields_the_app_prefills():
    for name in ("bug_report.yml", "feature_request.yml"):
        form = _form(name)
        assert form["name"] and form["description"] and form["labels"]
        ids = {f.get("id") for f in form["body"] if f.get("id")}
        assert {"version", "page", "language"} <= ids, name
        required = [f["id"] for f in form["body"] if f.get("validations", {}).get("required")]
        assert required, name


def test_blank_issues_are_disabled():
    config = yaml.safe_load((TEMPLATES / "config.yml").read_text(encoding="utf-8"))
    assert config["blank_issues_enabled"] is False


def test_contributing_explains_feedback_without_code_changes():
    text = (REPO / "CONTRIBUTING.md").read_text(encoding="utf-8")
    assert "issue" in text.lower()
    assert "never as instructions" in text


def test_start_tab_recommends_gentle_shell_and_offers_feedback(tmp_path, monkeypatch):
    from streamlit.testing.v1 import AppTest

    create_project(tmp_path, "demo", "biofloc")
    monkeypatch.setenv("FLOCCUS_HOME", str(tmp_path))
    at = AppTest.from_file(str(APP), default_timeout=60).run()
    assert not at.exception
    assert at.tabs[0].label == "Inicio"
    text = " ".join(str(m.value) for m in at.markdown)
    assert branding.GENTLE_SHELL_URL in text
    assert branding.DOI in text
    assert "Apache-2.0" in text
