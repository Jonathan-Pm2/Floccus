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
from floccus_app.textsafe import redact
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
    run.add_argument("step", choices=("import", "dedup", "dedup-ai", "screen-l1", "screen-l2", "consolidate", "clusters"))
    run.add_argument("--limit", type=_positive_int, help="screen at most N more records (try a small sample first)")
    run.add_argument("--workers", type=int, default=2, help="parallel LLM calls per record (default 2)")
    run.add_argument("--mode", choices=("specificity", "sensitivity"), default="specificity")
    ui = sub.add_parser("ui", help="open the graphical interface in the browser")
    ui.add_argument("--port", type=int, default=8501, help="local port (default 8501)")
    ui.add_argument("--theme", choices=("auto", "dark", "light"), default="auto",
                    help="auto follows the system and lets users switch in the ⋮ menu (default)")
    ui.add_argument("--host", default="localhost",
                    help="network address to listen on (default localhost: only this computer). "
                         "Use 0.0.0.0 only on a trusted network: anyone who reaches it can run steps and read data.")
    ui.add_argument("--headless", action="store_true", help="do not open a browser window")
    return p


def _theme_args(mode: str) -> list[str]:
    if mode == "auto":  # both themes branded -> Streamlit offers the light/dark switch
        return ["--theme.light.primaryColor", PRIMARY_COLOR, "--theme.dark.primaryColor", PRIMARY_COLOR]
    return ["--theme.base", mode, "--theme.primaryColor", PRIMARY_COLOR]


def _positive_int(value: str) -> int:
    try:
        n = int(value)
    except ValueError:
        n = 0
    if n < 1:
        raise argparse.ArgumentTypeError(f"must be a positive integer (got {value!r})")
    return n


def _run(home: Path, args) -> int:
    import os

    from floccus_app import pipeline
    from floccus_app.projects import validate_name

    validate_name(args.project)
    project = home / args.project
    if not (project / "protocol.yaml").is_file():
        raise ProjectError(f"no project named '{args.project}' in {home}")
    pipeline.acquire_lock(project, args.step, take_over=os.environ.get("FLOCCUS_LOCK_HANDOFF") == "1")
    if not args.step.startswith("screen"):
        pipeline.write_progress(project, step=args.step, status="starting", done=0, total=0, errors=0)
    try:
        if args.step == "import":
            by = pipeline.import_records(project)
            detail = ", ".join(f"{k}: {v}" for k, v in by.items())
            print(f"imported {sum(by.values())} records ({detail})")
        elif args.step == "dedup":
            s = pipeline.deduplicate(project)
            print(f"{s['unique']} unique records ({s['removed']} duplicates removed)")
        elif args.step == "dedup-ai":
            from floccus_app import similarity

            s = similarity.dedup_similar(project, judge=similarity._default_judge())
            print(f"{s['unique']} unique records after AI deduplication ({s['duplicates']} near duplicates merged)")
        elif args.step == "clusters":
            from floccus_app import similarity

            info = similarity.thematic_clusters(project)
            print(f"{info['clusters']} thematic clusters over {info['records']} eligible records "
                  f"({info['noise']} unclustered)")
        elif args.step in ("screen-l1", "screen-l2"):
            out = pipeline.screen(project, layer=int(args.step[-1]), limit=args.limit,
                                  workers=args.workers, mode=args.mode)
            print(json.dumps(out))
            if out["errors"]:
                print(f"warning: {out['errors']} LLM calls failed; check the AI model settings",
                      file=sys.stderr)
        if args.step in ("screen-l1", "screen-l2", "consolidate"):
            pipeline.consolidate(project)
        if not args.step.startswith("screen"):
            pipeline.write_progress(project, status="done")
    except Exception as exc:
        pipeline.write_progress(project, step=args.step, status="failed", message=redact(str(exc)))
        raise
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
            "--server.address", args.host,
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
        print(f"error: {redact(str(exc))}", file=sys.stderr)
        return 1
    except Exception as exc:  # never show endpoints or tracebacks to the user
        print(f"error: {type(exc).__name__}: {redact(str(exc))}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
