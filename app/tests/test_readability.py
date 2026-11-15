"""G5: user-facing texts, labels and names (readability review of v0.4.0)."""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

from floccus_app import i18n, results
from floccus_app.i18n import STRINGS, t

SRC = Path(__file__).resolve().parents[1] / "src" / "floccus_app"
REPO = Path(__file__).resolve().parents[2]
DYNAMIC = ("tab_", "btn_", "rv_", "m_", "step_", "status_", "dec_", "inclusion", "exclusion", "derivation")


def test_every_key_has_es_and_en():
    for k, v in STRINGS.items():
        assert v.get("es") and v.get("en"), k


def test_no_unused_keys():  # READ-7
    code = "".join(p.read_text(encoding="utf-8") for p in SRC.rglob("*.py") if p.name != "i18n.py")
    unused = [k for k in STRINGS if not k.startswith(DYNAMIC) and not re.search(rf"[\"']{k}[\"']", code)]
    assert unused == []


def test_start_steps_do_not_name_a_project_that_is_not_created():  # READ-1
    for lang in ("es", "en"):
        assert "biofloc-ejemplo" not in t("start_steps", lang)


def test_internal_codes_have_translations():  # READ-5
    for code in ("INCLUDE", "EXCLUDE", "HUMAN_REVIEW"):
        assert i18n.decision_label(code, "es") != code
    assert i18n.decision_label("HUMAN_REVIEW", "es") == "Revisión humana"
    for status in ("starting", "running", "done", "failed"):
        assert i18n.status_label(status, "es") != status
    assert i18n.step_label("screen-l1", "es") == "Cribado capa 1"
    assert i18n.step_label("dedup-ai", "en") == "AI deduplication"


def test_proquest_has_a_label_and_order():  # READ-4
    assert results.SOURCE_LABELS["proquest"] == "ProQuest"
    assert "proquest" in results.SOURCE_ORDER


def test_result_kinds_say_what_they_are(tmp_path):  # READ-10
    from floccus_app.projects import create_project

    assert results.load_results(create_project(tmp_path, "ex", "biofloc")).kind == "example"


def test_unique_count_is_public():  # READ-8
    assert results.unique_count({"deduplication": {"after_round2_ieee_merges": 7}}) == 7


def test_public_llm_chat_name():  # READ-9
    from biofloc_review.dedup import llm_judge

    assert callable(llm_judge.llm_chat)


def test_cli_help_explains_steps_and_mode():  # READ-12
    r = subprocess.run([sys.executable, "-m", "floccus_app.cli", "run", "--help"], capture_output=True, text=True)
    assert r.returncode == 0
    assert "consolidate" in r.stdout and "rebuilds" in r.stdout
    assert "sensitivity" in r.stdout and "fewer missed records" in r.stdout


def test_env_example_works_for_a_local_install():  # READ-6
    text = (REPO / "env.example").read_text(encoding="utf-8")
    assert "\nOLLAMA_URL=http://localhost:11434" in text
    assert "\n# OLLAMA_TOKEN=" in text
    assert "OPENALEX_EMAIL" in text


def test_docs_name_real_buttons_and_scope_resume_to_screening():  # READ-3 / READ-13
    contributing = (REPO / "CONTRIBUTING.md").read_text(encoding="utf-8")
    assert "Send feedback" not in contributing and "Enviar comentario" not in contributing
    assert "Report a problem" in contributing and "Reportar un error" in contributing
    assert "An interrupted screening run resumes" in (REPO / "app/docs/GUIDE.md").read_text(encoding="utf-8")
    assert "Un cribado interrumpido se retoma" in (REPO / "app/docs/GUIA.md").read_text(encoding="utf-8")


def test_name_has_descriptor_and_disambiguation():  # L21
    from floccus_app import branding

    assert branding.FULL_NAME == "Floccus – LLM-assisted systematic review screening"
    for lang in ("es", "en"):
        note = t("not_affiliated", lang)
        assert "floccus.org" in note and "floccus.es" in note
    for doc in ("README.md", "app/README.md", "app/docs/GUIA.md", "app/docs/GUIDE.md"):
        text = (REPO / doc).read_text(encoding="utf-8")
        assert "LLM-assisted systematic review screening" in text, doc
        assert "floccus.org" in text and "floccus.es" in text, doc


def test_header_shows_the_full_name(tmp_path, monkeypatch):  # L23
    from streamlit.testing.v1 import AppTest

    from floccus_app.projects import create_project

    create_project(tmp_path, "demo", "biofloc")
    monkeypatch.setenv("FLOCCUS_HOME", str(tmp_path))
    at = AppTest.from_file(str(SRC / "ui" / "app.py"), default_timeout=60).run()
    assert not at.exception
    header = " ".join(str(m.value) for m in at.markdown[:3])
    assert "LLM-assisted systematic review screening" in header
