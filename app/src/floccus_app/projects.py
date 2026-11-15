"""Review projects on disk: ``<home>/<name>/`` with protocol and data folders."""

from __future__ import annotations

import os
import re
from importlib import resources
from pathlib import Path

from biofloc_review.protocol import builtin_protocol, dump_protocol, load_protocol

DEFAULT_HOME = Path("floccus-projects")
SUBDIRS = ("data/raw", "data/processed", "runs")
_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")


class ProjectError(ValueError):
    pass


def projects_home(home: str | Path | None = None) -> Path:
    return Path(home or os.environ.get("FLOCCUS_HOME") or DEFAULT_HOME)


def create_project(home: Path, name: str, example: str | None = None) -> Path:
    if not _NAME.match(name):
        raise ProjectError(f"project name must use letters, digits, '-' or '_' (got '{name}')")
    project = home / name
    if project.exists():
        raise ProjectError(f"project '{name}' already exists in {project}")
    if example:
        text = dump_protocol(builtin_protocol(example))  # raises before touching disk
    else:
        text = resources.files("floccus_app").joinpath("templates", "blank.yaml").read_text(encoding="utf-8")
    for sub in SUBDIRS:
        (project / sub).mkdir(parents=True)
    (project / "protocol.yaml").write_text(text, encoding="utf-8")
    return project


def list_projects(home: Path) -> list[tuple[str, str]]:
    """(folder, protocol name) for every project folder with a protocol."""
    if not home.is_dir():
        return []
    out = []
    for d in sorted(home.iterdir()):
        if (d / "protocol.yaml").is_file():
            out.append((d.name, load_protocol(d / "protocol.yaml").name))
    return out
