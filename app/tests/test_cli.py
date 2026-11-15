"""`floccus` command: create and list review projects (biofloc is the example)."""

from __future__ import annotations

from biofloc_review.protocol import builtin_protocol, load_protocol

from floccus_app.cli import main


def test_new_project_from_biofloc_example(tmp_path, capsys):
    code = main(["--home", str(tmp_path), "new", "demo", "--example", "biofloc"])
    out = capsys.readouterr()
    assert code == 0
    project = tmp_path / "demo"
    assert out.out == f"Created project 'demo' in {project}\n"
    assert out.err == ""
    assert load_protocol(project / "protocol.yaml") == builtin_protocol("biofloc")
    for sub in ("data/raw", "data/processed", "runs"):
        assert (project / sub).is_dir()


def test_new_project_from_blank_template_is_valid(tmp_path, capsys):
    assert main(["--home", str(tmp_path), "new", "mine"]) == 0
    p = load_protocol(tmp_path / "mine" / "protocol.yaml")
    assert p.language == "en"
    assert [c.id for c in p.criteria] == ["I1", "E1", "D1"]


def test_new_refuses_existing_project_and_leaves_it_untouched(tmp_path, capsys):
    main(["--home", str(tmp_path), "new", "demo", "--example", "biofloc"])
    protocol = tmp_path / "demo" / "protocol.yaml"
    before = protocol.read_bytes()
    capsys.readouterr()
    code = main(["--home", str(tmp_path), "new", "demo"])
    out = capsys.readouterr()
    assert code == 1
    assert out.out == ""
    assert out.err == f"error: project 'demo' already exists in {tmp_path / 'demo'}\n"
    assert protocol.read_bytes() == before


def test_new_rejects_unknown_example(tmp_path, capsys):
    code = main(["--home", str(tmp_path), "new", "demo", "--example", "nope"])
    out = capsys.readouterr()
    assert code == 1
    assert out.err == "error: no built-in protocol named 'nope'\n"
    assert not (tmp_path / "demo").exists()


def test_new_rejects_unsafe_name(tmp_path, capsys):
    code = main(["--home", str(tmp_path), "new", "../evil"])
    out = capsys.readouterr()
    assert code == 1
    assert out.err == (
        "error: project name must use letters, digits, '-' or '_' (got '../evil')\n"
    )
    assert list(tmp_path.iterdir()) == []


def test_list_projects(tmp_path, capsys):
    assert main(["--home", str(tmp_path), "list"]) == 0
    assert capsys.readouterr().out == "No projects yet.\n"
    main(["--home", str(tmp_path), "new", "b-proj", "--example", "biofloc"])
    main(["--home", str(tmp_path), "new", "a-proj"])
    capsys.readouterr()
    assert main(["--home", str(tmp_path), "list"]) == 0
    assert capsys.readouterr().out == "a-proj\tmicroplastics-example\nb-proj\tbiofloc\n"
