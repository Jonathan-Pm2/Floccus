"""``floccus`` command line: manage review projects."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from biofloc_review.protocol import ProtocolError

from floccus_app.branding import PRIMARY_COLOR
from floccus_app.pipeline import PipelineError
from floccus_app.projects import ProjectError, create_project, list_projects, projects_home


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="floccus", description="Floccus systematic-review screening.")
    p.add_argument("--home", help="projects folder (default: $FLOCCUS_HOME or ./floccus-projects)")
    sub = p.add_subparsers(dest="command", required=True)
    new = sub.add_parser("new", help="create a project with a protocol to edit")
    new.add_argument("name", help="project folder name (letters, digits, '-', '_')")
    new.add_argument("--example", help="start from a built-in protocol, e.g. 'biofloc'")
    sub.add_parser("list", help="list projects")
    run = sub.add_parser("run", help="run one pipeline step for a project")
    run.add_argument("project", help="project folder name")
    run.add_argument("step", choices=("import", "dedup", "screen-l1", "screen-l2", "consolidate"))
    run.add_argument("--limit", type=int, help="screen at most N more records (try a small sample first)")
    run.add_argument("--workers", type=int, default=2, help="parallel LLM calls per record (default 2)")
    run.add_argument("--mode", choices=("specificity", "sensitivity"), default="specificity")
    ui = sub.add_parser("ui", help="open the graphical interface in the browser")
    ui.add_argument("--port", type=int, default=8501, help="local port (default 8501)")
    ui.add_argument("--theme", choices=("auto", "dark", "light"), default="auto",
                    help="auto follows the system and lets users switch in the ⋮ menu (default)")
    ui.add_argument("--headless", action="store_true", help="do not open a browser window")
    return p


def _theme_args(mode: str) -> list[str]:
    if mode == "auto":  # both themes branded -> Streamlit offers the light/dark switch
        return ["--theme.light.primaryColor", PRIMARY_COLOR, "--theme.dark.primaryColor", PRIMARY_COLOR]
    return ["--theme.base", mode, "--theme.primaryColor", PRIMARY_COLOR]


def _run(home: Path, args) -> int:
    from floccus_app import pipeline

    project = home / args.project
    if not (project / "protocol.yaml").is_file():
        raise ProjectError(f"no project named '{args.project}' in {home}")
    pipeline.acquire_lock(project, args.step)
    try:
        if args.step == "import":
            by = pipeline.import_records(project)
            detail = ", ".join(f"{k}: {v}" for k, v in by.items())
            print(f"imported {sum(by.values())} records ({detail})")
        elif args.step == "dedup":
            s = pipeline.deduplicate(project)
            print(f"{s['unique']} unique records ({s['removed']} duplicates removed)")
        elif args.step in ("screen-l1", "screen-l2"):
            out = pipeline.screen(project, layer=int(args.step[-1]), limit=args.limit,
                                  workers=args.workers, mode=args.mode)
            print(json.dumps(out))
            if out["errors"]:
                print(f"warning: {out['errors']} LLM calls failed; check the AI model settings",
                      file=sys.stderr)
        if args.step in ("screen-l1", "screen-l2", "consolidate"):
            pipeline.consolidate(project)
    finally:
        pipeline.release_lock(project)
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    home = projects_home(args.home)
    if args.command == "ui":
        app = Path(__file__).resolve().parent / "ui" / "app.py"
        return subprocess.call([
            sys.executable, "-m", "streamlit", "run", str(app),
            "--server.port", str(args.port),
            "--server.headless", "true" if args.headless or not sys.stdin.isatty() else "false",
            "--browser.gatherUsageStats", "false",
            "--client.toolbarMode", "viewer",  # hides Deploy, keeps the theme menu
            *_theme_args(args.theme),
            "--", "--home", str(home),
        ])
    try:
        if args.command == "new":
            project = create_project(home, args.name, args.example)
            print(f"Created project '{args.name}' in {project}")
        elif args.command == "run":
            return _run(home, args)
        elif args.command == "list":
            rows = list_projects(home)
            if not rows:
                print("No projects yet.")
            for folder, name in rows:
                print(f"{folder}\t{name}")
    except (ProjectError, ProtocolError, PipelineError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
