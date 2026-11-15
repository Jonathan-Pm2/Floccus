"""``floccus`` command line: manage review projects."""

from __future__ import annotations

import argparse
import sys

from biofloc_review.protocol import ProtocolError

from floccus_app.projects import ProjectError, create_project, list_projects, projects_home


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="floccus", description="Floccus systematic-review screening.")
    p.add_argument("--home", help="projects folder (default: $FLOCCUS_HOME or ./floccus-projects)")
    sub = p.add_subparsers(dest="command", required=True)
    new = sub.add_parser("new", help="create a project with a protocol to edit")
    new.add_argument("name", help="project folder name (letters, digits, '-', '_')")
    new.add_argument("--example", help="start from a built-in protocol, e.g. 'biofloc'")
    sub.add_parser("list", help="list projects")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    home = projects_home(args.home)
    try:
        if args.command == "new":
            project = create_project(home, args.name, args.example)
            print(f"Created project '{args.name}' in {project}")
        elif args.command == "list":
            rows = list_projects(home)
            if not rows:
                print("No projects yet.")
            for folder, name in rows:
                print(f"{folder}\t{name}")
    except (ProjectError, ProtocolError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
