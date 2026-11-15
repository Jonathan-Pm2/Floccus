"""T6: screening runs with checkpoints, resume, progress file and CLI."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from floccus_app import cli, pipeline
from floccus_app.projects import create_project

SCOPUS = pd.DataFrame({
    "Title": [f"Paper {i}" for i in range(6)],
    "DOI": [f"10.1/{i}" for i in range(6)],
    "Abstract": ["microplastics in rivers " * 40] * 3 + ["toxicity test only " * 30] * 2 + [None],
    "Year": [2020] * 6,
    "EID": [f"s{i}" for i in range(6)],
})


def fake_chat(messages, model):
    """I1 (rivers) YES unless the abstract is about toxicity; E1 (toxicity) YES for toxicity."""
    system, user = messages[0]["content"], messages[1]["content"]
    toxic = "toxicity test only" in user
    if "CRITERION I1" in system:
        return "ANSWER: NO" if toxic else "ANSWER: YES"
    if "CRITERION E1" in system:
        return "ANSWER: YES" if toxic else "ANSWER: NO"
    return "ANSWER: NO"


@pytest.fixture
def project(tmp_path):
    p = create_project(tmp_path, "demo")  # blank protocol: I1, E1 (LLM) + D1 abstract_too_short
    pipeline.add_export(p, "scopus", "s.csv", SCOPUS.to_csv(index=False).encode())
    pipeline.import_records(p)
    pipeline.deduplicate(p)
    return p


def _progress(p):
    return json.loads((p / "runs" / "progress.json").read_text())


def test_layer1_screens_every_record_and_reports_progress(project):
    out = pipeline.screen(project, layer=1, chat=fake_chat, model="m", checkpoint_every=2)
    assert out == {"screened": 6, "errors": 0, "INCLUDE": 3, "EXCLUDE": 2, "HUMAN_REVIEW": 1}
    df = pd.read_parquet(project / "data" / "processed" / "screening_l1.parquet")
    assert len(df) == 6 and set(df["l1_decision"]) == {"INCLUDE", "EXCLUDE", "HUMAN_REVIEW"}
    prog = _progress(project)
    assert (prog["step"], prog["status"], prog["done"], prog["total"], prog["errors"]) == (
        "screen-l1", "done", 6, 6, 0)


def test_resume_skips_records_already_screened(project):
    pipeline.screen(project, layer=1, chat=fake_chat, model="m", limit=4)
    calls = []

    def counting_chat(messages, model):
        calls.append(1)
        return fake_chat(messages, model)

    pipeline.screen(project, layer=1, chat=counting_chat, model="m")
    assert len(calls) == 2 * 2  # 2 remaining records x 2 LLM criteria
    assert len(pd.read_parquet(project / "data" / "processed" / "screening_l1.parquet")) == 6


def test_unresponsive_model_stops_the_run_and_reports_it(project):
    def broken(messages, model):
        raise ConnectionError("no server")

    with pytest.raises(pipeline.PipelineError, match="the AI model is not responding"):
        pipeline.screen(project, layer=1, chat=broken, model="m")
    prog = _progress(project)
    assert (prog["status"], prog["errors"]) == ("failed", 6)  # 3 records x 2 LLM criteria
    assert "not responding" in prog["message"]
    df = pd.read_parquet(project / "data" / "processed" / "screening_l1.parquet")
    assert len(df) == 3 and int(df["l1_errors"].sum()) == 6


def test_layer2_rescreens_uncertain_records_with_extra_context(project):
    pipeline.screen(project, layer=1, chat=fake_chat, model="m")
    seen = []

    def chat(messages, model):
        seen.append(messages[1]["content"])
        return "ANSWER: YES" if "CRITERION I1" in messages[0]["content"] else "ANSWER: NO"

    out = pipeline.screen(project, layer=2, chat=chat, model="m", enrich=lambda rec: "EXTRA METADATA")
    assert out["screened"] == 1  # only the record routed to human review
    assert all("EXTRA METADATA" in u for u in seen)


def test_consolidate_writes_decisions_and_counts(project):
    pipeline.screen(project, layer=1, chat=fake_chat, model="m")
    pipeline.screen(project, layer=2, chat=lambda m, model: "ANSWER: YES" if "CRITERION I1" in m[0]["content"]
                    else "ANSWER: NO", model="m", enrich=lambda rec: "")
    counts = pipeline.consolidate(project)
    final = counts["final_specificity_mode"]
    assert final == {"include": 3, "human_review": 1, "exclude": 2, "to_eligibility": 4}
    assert counts["screening_layer1"] == {"input": 6, "include": 3, "exclude": 2, "human_review": 1}
    assert counts["screening_layer2"] == {"input": 1, "include": 0, "exclude": 0, "human_review": 1}
    dec = pd.read_csv(project / "results" / "latest" / "screening_decisions.csv")
    assert len(dec) == 6 and dec["final_decision"].value_counts().to_dict() == {
        "INCLUDE": 3, "EXCLUDE": 2, "HUMAN_REVIEW": 1}
    assert {"l1_I1", "l1_E1", "l1_D1"} <= set(dec.columns)


def test_layer2_before_layer1_explains_order(tmp_path):
    p = create_project(tmp_path, "demo")
    with pytest.raises(pipeline.PipelineError, match="run layer 1 first"):
        pipeline.screen(p, layer=2, chat=fake_chat, model="m")


def test_cli_run_import_and_dedup(tmp_path, capsys):
    p = create_project(tmp_path, "demo")
    pipeline.add_export(p, "scopus", "s.csv", SCOPUS.to_csv(index=False).encode())
    assert cli.main(["--home", str(tmp_path), "run", "demo", "import"]) == 0
    assert cli.main(["--home", str(tmp_path), "run", "demo", "dedup"]) == 0
    out = capsys.readouterr().out
    assert "imported 6 records (scopus: 6)" in out
    assert "6 unique records (0 duplicates removed)" in out


def test_cli_run_refuses_while_another_run_is_active(tmp_path, capsys):
    import os

    p = create_project(tmp_path, "demo")
    (p / "runs" / "lock.json").write_text(json.dumps(pipeline.lock_payload(os.getpid(), "screen-l1")))
    assert cli.main(["--home", str(tmp_path), "run", "demo", "import"]) == 1
    assert capsys.readouterr().err == "error: another run is active in this project (screen-l1)\n"


def test_stale_lock_from_dead_process_is_ignored(tmp_path):
    p = create_project(tmp_path, "demo")
    (p / "runs" / "lock.json").write_text(json.dumps({"pid": 999999999, "step": "x"}))
    assert pipeline.active_run(p) is None
