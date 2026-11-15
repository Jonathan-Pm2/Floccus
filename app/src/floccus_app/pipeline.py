"""Project pipeline steps: import exports, deduplicate, screen.

Every step reads and writes inside the project folder:
``data/raw/<source>/`` (exports) → ``data/processed/records.parquet`` →
``data/processed/unique.parquet`` → ``results/latest/`` (counts, decisions).
Deduplication here is the deterministic layer (exact DOI or normalised
title); the embedding and LLM-judge layers of the published pipeline need a
Qdrant server and are not part of the app yet.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

import pandas as pd
from biofloc_review.dedup.deterministic import dedup_deterministic
from biofloc_review.schema import Corpus, PaperRecord
from biofloc_review.sources.ieee import load_ieee
from biofloc_review.sources.proquest import load_proquest
from biofloc_review.sources.scielo import load_scielo
from biofloc_review.sources.scopus import load_scopus
from biofloc_review.sources.wos import load_wos

# source -> (loader, allowed extensions)
SOURCES = {
    "scopus": (load_scopus, (".csv",)),
    "wos": (load_wos, (".xls", ".xlsx")),
    "ieee": (load_ieee, (".csv",)),
    "scielo": (load_scielo, (".csv",)),
    "proquest": (load_proquest, (".xls", ".xlsx")),
}
_SAFE_NAME = re.compile(r"^[\w.\- ]+$")
# The published loaders require a corpus tag; it has no effect on app projects.
_CORPUS = Corpus.BIOFLOC


class PipelineError(RuntimeError):
    """A step cannot run; the message says what the user should do."""


def raw_dir(project: Path) -> Path:
    return project / "data" / "raw"


def processed_dir(project: Path) -> Path:
    return project / "data" / "processed"


def latest_dir(project: Path) -> Path:
    return project / "results" / "latest"


def add_export(project: Path, source: str, filename: str, content: bytes) -> Path:
    if source not in SOURCES:
        raise PipelineError(f"unknown source '{source}' (available: {', '.join(SOURCES)})")
    name = Path(filename).name
    if name != filename or not _SAFE_NAME.match(name) or name.startswith("."):
        raise PipelineError(f"invalid file name '{filename}'")
    allowed = SOURCES[source][1]
    if Path(name).suffix.lower() not in allowed:
        raise PipelineError(f"{source} exports must be {' or '.join(allowed)}")
    target = raw_dir(project) / source / name
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(content)
    return target


def list_exports(project: Path) -> dict[str, list[str]]:
    out = {}
    for source in SOURCES:
        d = raw_dir(project) / source
        files = sorted(f.name for f in d.iterdir() if f.is_file()) if d.is_dir() else []
        if files:
            out[source] = files
    return out


def _write_records(records: list[PaperRecord], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame([r.model_dump(mode="json") for r in records])
    tmp = path.with_suffix(".tmp")
    df.to_parquet(tmp, index=False)
    tmp.replace(path)


def _read_records(path: Path) -> list[PaperRecord]:
    df = pd.read_parquet(path)
    out = []
    for row in df.to_dict("records"):
        row = {k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in row.items()}
        row = {k: (None if isinstance(v, float) and v != v else v) for k, v in row.items()}
        out.append(PaperRecord.model_validate(row))
    return out


def read_counts(project: Path) -> dict:
    path = latest_dir(project) / "prisma_counts.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def write_counts(project: Path, counts: dict) -> None:
    path = latest_dir(project) / "prisma_counts.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(counts, indent=2), encoding="utf-8")
    tmp.replace(path)


def import_records(project: Path) -> dict[str, int]:
    exports = list_exports(project)
    if not exports:
        raise PipelineError("no exports yet: add Scopus, Web of Science, IEEE, SciELO or ProQuest files first")
    records: list[PaperRecord] = []
    by_source: dict[str, int] = {}
    for source, files in exports.items():
        loader = SOURCES[source][0]
        n = 0
        for name in files:
            try:
                loaded = loader(raw_dir(project) / source / name, _CORPUS)
            except Exception as exc:
                raise PipelineError(f"could not read {source}/{name}: {exc}") from exc
            records.extend(loaded)
            n += len(loaded)
        by_source[source] = n
    _write_records(records, processed_dir(project) / "records.parquet")
    write_counts(project, {"identification": {"by_source": by_source, "total": len(records)}})
    return by_source


def deduplicate(project: Path) -> dict[str, int]:
    src = processed_dir(project) / "records.parquet"
    if not src.is_file():
        raise PipelineError("import the exports first")
    records = _read_records(src)
    primaries, _, stats = dedup_deterministic(records)
    _write_records(primaries, processed_dir(project) / "unique.parquet")
    counts = read_counts(project)
    counts["deduplication"] = {
        "method": "deterministic (exact DOI or normalised title)",
        "after_layers2_3_and_merge": len(primaries),
        "after_round2_ieee_merges": len(primaries),
        "removed": stats["duplicates_removed"],
    }
    counts["deduplicated_by_source"] = dict(Counter(str(r.source) for r in primaries))
    for key in ("screening_layer1", "screening_layer2", "final_specificity_mode"):
        counts.pop(key, None)  # earlier screening no longer matches the corpus
    write_counts(project, counts)
    return {"input": len(records), "unique": len(primaries), "removed": stats["duplicates_removed"]}
