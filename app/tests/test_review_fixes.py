"""Fixes for the native-review advisory findings on run control (v0.4.0)."""

from __future__ import annotations

import json
import os
import subprocess
import sys

import pandas as pd
import pytest

from floccus_app import cli, cost, pipeline, ui_logic
from floccus_app.projects import create_project

from test_pipeline_screen import SCOPUS, fake_chat  # noqa: E402


@pytest.fixture
def project(tmp_path):
    p = create_project(tmp_path, "demo")
    pipeline.add_export(p, "scopus", "s.csv", SCOPUS.to_csv(index=False).encode())
    pipeline.import_records(p)
    pipeline.deduplicate(p)
    return p


def test_records_with_failed_calls_are_retried_on_the_next_run(project):  # R4-failed-calls-never-retried
    with pytest.raises(pipeline.PipelineError):
        pipeline.screen(project, layer=1, chat=lambda m, model: (_ for _ in ()).throw(ConnectionError("down")),
                        model="m")
    out = pipeline.screen(project, layer=1, chat=fake_chat, model="m")
    assert out["errors"] == 0 and out["screened"] == 6
    df = pd.read_parquet(project / "data" / "processed" / "screening_l1.parquet")
    assert len(df) == 6 and int(df["l1_errors"].sum()) == 0


def test_rows_of_records_no_longer_in_the_corpus_are_dropped(project):  # R3-005
    pipeline.screen(project, layer=1, chat=fake_chat, model="m")
    smaller = SCOPUS.iloc[:2]
    for f in (project / "data" / "raw" / "scopus").iterdir():
        f.unlink()
    pipeline.add_export(project, "scopus", "s.csv", smaller.to_csv(index=False).encode())
    pipeline.import_records(project)
    pipeline.deduplicate(project)
    out = pipeline.screen(project, layer=1, chat=fake_chat, model="m")
    assert out["screened"] == 2
    assert len(pd.read_parquet(project / "data" / "processed" / "screening_l1.parquet")) == 2


def test_screening_an_empty_selection_explains_instead_of_crashing(tmp_path):  # R3-004
    p = create_project(tmp_path, "demo")
    pipeline.add_export(p, "scopus", "s.csv", SCOPUS.iloc[:0].to_csv(index=False).encode())
    pipeline.import_records(p)
    pipeline.deduplicate(p)
    with pytest.raises(pipeline.PipelineError, match="no records to screen"):
        pipeline.screen(p, layer=1, chat=fake_chat, model="m")


@pytest.mark.parametrize("bad", ["0", "-3"])
def test_cli_rejects_non_positive_limit(tmp_path, capsys, bad):  # R3-004
    create_project(tmp_path, "demo")
    with pytest.raises(SystemExit) as exc:
        cli.main(["--home", str(tmp_path), "run", "demo", "screen-l1", "--limit", bad])
    assert exc.value.code == 2
    assert "must be a positive integer" in capsys.readouterr().err


def test_cli_run_rejects_names_outside_the_projects_folder(tmp_path, capsys):  # R1-001
    outside = tmp_path / "outside"
    create_project(tmp_path, "x")
    (outside).mkdir()
    (outside / "protocol.yaml").write_text((tmp_path / "x" / "protocol.yaml").read_text())
    home = tmp_path / "home"
    home.mkdir()
    assert cli.main(["--home", str(home), "run", "../outside", "import"]) == 1
    assert capsys.readouterr().err == (
        "error: project name must use letters, digits, '-' or '_' (got '../outside')\n")
    assert not (outside / "runs").exists()


def test_lock_is_atomic_and_handed_to_the_launched_run(project):  # R3-002 / R4-launch-race
    pipeline.acquire_lock(project, "screen-l1")
    with pytest.raises(pipeline.PipelineError, match="another run is active"):
        pipeline.acquire_lock(project, "import")
    pipeline.release_lock(project)
    # the launcher holds the lock for the child; the child takes it over
    pipeline.acquire_lock(project, "screen-l1", pid=os.getpid(), handoff=True)
    pipeline.acquire_lock(project, "screen-l1", take_over=True)
    assert pipeline.active_run(project)["pid"] == os.getpid()
    pipeline.release_lock(project)
    assert pipeline.active_run(project) is None


def test_release_only_removes_own_lock(project):  # R4-launch-race
    lock = project / "runs" / "lock.json"
    lock.write_text(json.dumps({"pid": os.getppid(), "step": "x"}))
    pipeline.release_lock(project)
    assert lock.is_file()


def test_lock_of_unrelated_process_with_reused_pid_is_stale(project):  # R4-stale-lock-pid-reuse
    if not os.path.isdir("/proc"):
        pytest.skip("needs /proc")
    sleeper = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    try:
        (project / "runs" / "lock.json").write_text(json.dumps({"pid": sleeper.pid, "step": "x"}))
        assert pipeline.active_run(project) is None
    finally:
        sleeper.kill()


def test_launch_step_holds_the_lock_before_returning(project, monkeypatch):  # R3-002
    started = []

    class FakeProc:
        pid = os.getpid()

    def fake_popen(argv, **kw):
        started.append((argv, kw.get("env", {}).get("FLOCCUS_LOCK_HANDOFF")))
        return FakeProc()

    monkeypatch.setattr(ui_logic.subprocess, "Popen", fake_popen)
    ui_logic.launch_step(project.parent, project.name, "import")
    assert started[0][1] == "1"
    assert pipeline.active_run(project)["step"] == "import"
    with pytest.raises(pipeline.PipelineError, match="another run is active"):
        ui_logic.launch_step(project.parent, project.name, "dedup")
    assert len(started) == 1


def test_failed_import_is_reported_in_progress(tmp_path, capsys):  # R4-import-dedup-failures-invisible
    p = create_project(tmp_path, "demo")
    assert cli.main(["--home", str(tmp_path), "run", "demo", "import"]) == 1
    prog = pipeline.read_progress(p)
    assert (prog["step"], prog["status"]) == ("import", "failed")
    assert "no exports yet" in prog["message"]
    assert pipeline.active_run(p) is None


def test_cli_screen_consolidates_warns_and_releases_lock(project, capsys, monkeypatch):  # R3-007
    import biofloc_review.dedup.llm_judge as lj

    monkeypatch.setattr(lj, "_ollama_chat", lambda m, model: (_ for _ in ()).throw(TimeoutError("x")))
    monkeypatch.setattr(lj, "default_llm_model", lambda: "m")
    assert cli.main(["--home", str(project.parent), "run", "demo", "screen-l1", "--limit", "2"]) == 0
    out = capsys.readouterr()
    assert json.loads(out.out)["errors"] == 4  # 2 records x 2 criteria, below the 3-record stop
    assert "warning: 4 LLM calls failed" in out.err
    assert (project / "results" / "latest" / "screening_decisions.csv").is_file()
    assert pipeline.active_run(project) is None


def test_cli_module_entry_point_runs():  # R3-007
    r = subprocess.run([sys.executable, "-m", "floccus_app.cli", "--help"], capture_output=True, text=True)
    assert r.returncode == 0 and "run" in r.stdout


def test_cost_uses_the_projects_real_record_length(project):  # R3-008
    chars = cost.project_user_chars(project)
    assert chars != cost.DEFAULT_USER_CHARS and chars > cost.USER_TEMPLATE_CHARS


# ---------------------------------------------------------------- G1 security

def test_ui_listens_only_on_this_machine_by_default(tmp_path, monkeypatch):  # RISK-1
    seen = {}
    monkeypatch.setattr(cli.subprocess, "call", lambda argv: seen.setdefault("argv", argv) and 0)
    cli.main(["--home", str(tmp_path), "ui", "--headless"])
    argv = seen["argv"]
    assert argv[argv.index("--server.address") + 1] == "localhost"
    seen.clear()
    cli.main(["--home", str(tmp_path), "ui", "--headless", "--host", "0.0.0.0"])
    assert seen["argv"][seen["argv"].index("--server.address") + 1] == "0.0.0.0"


def test_redact_removes_urls_and_embedded_credentials():  # RISK-2
    msg = "401 for url: https://user:tok@ollama.example.com:11434/api/embed and http://h/x"
    out = ui_logic.redact(msg)
    assert "http" not in out and "tok" not in out and "ollama.example.com" not in out
    assert out.count("<endpoint>") == 2


def test_unexpected_errors_are_reported_without_urls(tmp_path, capsys, monkeypatch):  # RISK-2
    p = create_project(tmp_path, "demo")

    def boom(project):
        raise RuntimeError("Connection refused: https://user:tok@remote-ollama:11434/api/chat")

    monkeypatch.setattr(pipeline, "import_records", boom)
    assert cli.main(["--home", str(tmp_path), "run", "demo", "import"]) == 1
    err = capsys.readouterr().err
    assert err == "error: RuntimeError: Connection refused: <endpoint>\n"
    assert "remote-ollama" not in json.dumps(pipeline.read_progress(p))
    assert pipeline.active_run(p) is None


def test_log_tail_is_redacted(tmp_path):  # RISK-2
    create_project(tmp_path, "demo")
    (tmp_path / "demo" / "runs" / "import.log").write_text("Traceback\nurl: https://k@api.example.com/v1\n")
    assert "api.example.com" not in ui_logic.log_tail(tmp_path, "demo", "import")


def test_judge_errors_stored_without_urls(tmp_path, monkeypatch):  # RISK-2
    from test_similarity import NEAR_DUP, _project, fake_vector  # noqa: F401
    from biofloc_review.dedup import embeddings as emb
    from floccus_app import similarity

    monkeypatch.setattr(emb, "embed_batch", lambda texts, model=None, **kw: [fake_vector(t) for t in texts])
    p = _project(tmp_path, NEAR_DUP)

    def failing_judge(a, b, s, model=None):
        raise ConnectionError("HTTPConnectionPool(host='10.0.0.5'): url: http://10.0.0.5:11434/api/chat")

    similarity.dedup_similar(p, judge=failing_judge)
    text = (p / "results" / "latest" / "near_duplicate_pairs.json").read_text()
    assert "10.0.0.5" not in text and "UNCLEAR" in text


@pytest.mark.parametrize(("params", "message"), [
    ({"excluded": "Review"}, "I4: 'excluded' must be a list of document types"),
    ({"included": ["article", ""]}, "I4: 'included' must be a list of document types"),
])
def test_document_type_params_must_be_lists(params, message):  # RISK-3 / REL-2
    from biofloc_review.protocol import ProtocolError, protocol_from_dict

    data = {"name": "x", "language": "en", "topic": "t", "criteria": [
        {"id": "I4", "type": "inclusion", "question": "q", "justification": "j",
         "check": "document_type", "params": params}]}
    with pytest.raises(ProtocolError, match=message):
        protocol_from_dict(data)


@pytest.mark.parametrize("value", ["abc", 0, -5, True, 2.5])
def test_min_words_must_be_a_positive_integer(value):  # RISK-3 / REL-2
    from biofloc_review.protocol import ProtocolError, protocol_from_dict

    data = {"name": "x", "language": "en", "topic": "t", "criteria": [
        {"id": "D1", "type": "derivation", "question": "q", "justification": "j",
         "check": "abstract_too_short", "params": {"min_words": value}}]}
    with pytest.raises(ProtocolError, match="D1: 'min_words' must be a positive whole number"):
        protocol_from_dict(data)


def test_record_text_is_escaped_for_markdown_and_doi_quoted():  # RISK-4
    assert ui_logic.md_escape("![x](https://tracker/p.png) *bold*") == \
        "\\!\\[x\\]\\(https://tracker/p.png\\) \\*bold\\*"
    assert ui_logic.doi_url("10.1/a b)c") == "https://doi.org/10.1/a%20b%29c"
