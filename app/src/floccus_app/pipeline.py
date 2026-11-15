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


class EnrichmentError(RuntimeError):
    """OpenAlex / Europe PMC returned nothing for a record that has a DOI."""


def atomic_write(path: Path, writer) -> None:
    """Write through a unique temp file in the same folder, then replace: a
    reader never sees a half-written file and a failure keeps the old one."""
    import os
    import uuid

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        writer(tmp)
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


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
    df = pd.DataFrame([r.model_dump(mode="json") for r in records])
    atomic_write(path, lambda tmp: df.to_parquet(tmp, index=False))


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
    atomic_write(path, lambda tmp: tmp.write_text(json.dumps(counts, indent=2), encoding="utf-8"))


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
    primaries, full, stats = dedup_deterministic(records)
    _write_records(primaries, processed_dir(project) / "unique.parquet")
    _write_records(full, processed_dir(project) / "dedup_full.parquet")
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


# ---------------------------------------------------------------- run control

def runs_dir(project: Path) -> Path:
    return project / "runs"


def _proc_start(pid: int) -> str | None:
    """Process start time on Linux (/proc), used to tell a reused PID apart."""
    try:
        stat = Path(f"/proc/{pid}/stat").read_text()
    except OSError:
        return None
    return stat.rsplit(")", 1)[1].split()[19]


def _pid_alive(pid: int) -> bool:
    import os

    if pid <= 0:
        return False
    if os.name == "nt":  # os.kill(pid, 0) would terminate the process on Windows
        import ctypes

        handle = ctypes.windll.kernel32.OpenProcess(0x1000, False, pid)  # QUERY_LIMITED_INFORMATION
        if not handle:
            return False
        code = ctypes.c_ulong()
        ctypes.windll.kernel32.GetExitCodeProcess(handle, ctypes.byref(code))
        ctypes.windll.kernel32.CloseHandle(handle)
        return code.value == 259  # STILL_ACTIVE
    try:
        os.kill(pid, 0)
    except (ProcessLookupError, ValueError, OverflowError):
        return False
    except PermissionError:
        return True
    return True


def lock_payload(pid: int, step: str, handoff: bool = False) -> dict:
    return {"pid": pid, "step": step, "start": _proc_start(pid), "handoff": handoff}


def _read_lock(project: Path) -> dict | None:
    lock = runs_dir(project) / "lock.json"
    try:
        return json.loads(lock.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def active_run(project: Path) -> dict | None:
    """The run holding the project lock, or None. A lock whose process is gone,
    or whose PID now belongs to another process (Linux start time differs), is stale."""
    import os

    info = _read_lock(project)
    if not info:
        return None
    pid = int(info.get("pid", -1))
    if not _pid_alive(pid):
        return None
    if os.path.isdir("/proc") and info.get("start") != _proc_start(pid):
        return None
    return info


def acquire_lock(project: Path, step: str, pid: int | None = None, handoff: bool = False,
                 take_over: bool = False) -> None:
    """Atomically create runs/lock.json. ``handoff`` marks a lock taken by the UI
    for a child process; the child calls with ``take_over`` to become its owner."""
    import os

    runs_dir(project).mkdir(parents=True, exist_ok=True)
    path = runs_dir(project) / "lock.json"
    if take_over:
        current = _read_lock(project)
        if current and current.get("handoff") and current.get("step") == step:
            tmp = path.with_suffix(f".{os.getpid()}.tmp")
            tmp.write_text(json.dumps(lock_payload(os.getpid(), step)), encoding="utf-8")
            tmp.replace(path)
            return
    payload = json.dumps(lock_payload(pid or os.getpid(), step, handoff))
    for _ in range(2):
        try:
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            current = active_run(project)
            if current:
                raise PipelineError(f"another run is active in this project ({current.get('step')})") from None
            path.unlink(missing_ok=True)  # stale lock
            continue
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(payload)
        return
    raise PipelineError("could not take the project lock; try again")


def set_lock_pid(project: Path, pid: int, step: str) -> None:
    import os

    path = runs_dir(project) / "lock.json"
    tmp = path.with_suffix(f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(lock_payload(pid, step, handoff=True)), encoding="utf-8")
    tmp.replace(path)


def release_lock(project: Path) -> None:
    """Remove the lock only if this process owns it."""
    import os

    info = _read_lock(project)
    if info is None or int(info.get("pid", -1)) == os.getpid():
        (runs_dir(project) / "lock.json").unlink(missing_ok=True)


def write_progress(project: Path, **fields) -> None:
    import time

    path = runs_dir(project) / "progress.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    current = read_progress(project) if fields.get("status") != "starting" else {}
    current.update(fields)
    current["updated_at"] = time.time()
    current.setdefault("started_at", current["updated_at"])
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(current), encoding="utf-8")
    tmp.replace(path)


def read_progress(project: Path) -> dict:
    path = runs_dir(project) / "progress.json"
    try:
        return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    except json.JSONDecodeError:
        return {}


# ---------------------------------------------------------------- screening

def _default_enrich(rec: PaperRecord) -> str:
    from biofloc_review.screening.enricher import enrich_record
    from biofloc_review.screening.layer2 import _format_enriched_block

    if not rec.doi:
        return "(metadata enriquecida no disponible: el registro no tiene DOI)"
    enriched = enrich_record(rec)
    if not (enriched.oa_ok or enriched.pmc_ok):
        raise EnrichmentError("OpenAlex and Europe PMC returned nothing")
    return _format_enriched_block(enriched)


def _record_id(rec: PaperRecord) -> str:
    return rec.cluster_id or f"{rec.source}:{rec.source_id}"


MAX_FAILING_RECORDS = 3  # stop when every LLM call failed for this many records in a row
MAX_ENRICH_FAILURES = 5  # stop layer 2 when enrichment failed for this many records in a row


def screen(project: Path, layer: int, *, chat=None, model: str | None = None, mode: str = "specificity",
           workers: int = 1, checkpoint_every: int = 25, limit: int | None = None, enrich=None) -> dict:
    """Screen layer 1 (all unique records) or layer 2 (layer-1 HUMAN_REVIEW, with
    enriched metadata). Resumes from the existing output; writes progress."""
    from biofloc_review.dedup.llm_judge import default_llm_model, llm_chat

    from floccus_app.screening import screen_record
    from floccus_app.ui_logic import load_project_protocol

    protocol = load_project_protocol(project)
    chat = chat or llm_chat
    model = model or default_llm_model()
    step = f"screen-l{layer}"
    unique_path = processed_dir(project) / "unique.parquet"
    if layer == 2 and not (processed_dir(project) / "screening_l1.parquet").is_file():
        raise PipelineError("run layer 1 first")
    if not unique_path.is_file():
        raise PipelineError("import and deduplicate the exports first")
    records = _read_records(unique_path)
    if layer == 2:
        l1 = processed_dir(project) / "screening_l1.parquet"
        if not l1.is_file():
            raise PipelineError("run layer 1 first")
        uncertain = set(pd.read_parquet(l1).query("l1_decision == 'HUMAN_REVIEW'")["record_id"])
        records = [r for r in records if _record_id(r) in uncertain]
        enrich = enrich or _default_enrich
    elif layer != 1:
        raise PipelineError("layer must be 1 or 2")

    out_path = processed_dir(project) / f"screening_l{layer}.parquet"
    if not records:
        raise PipelineError("no records to screen")
    current = {_record_id(r) for r in records}
    rows = pd.read_parquet(out_path).to_dict("records") if out_path.is_file() else []
    # keep only finished rows of records still in the corpus; failed ones are retried
    rows = [r for r in rows if r["record_id"] in current and int(r.get(f"l{layer}_errors", 0)) == 0]
    done = {r["record_id"] for r in rows}
    todo = [r for r in records if _record_id(r) not in done]
    if limit is not None:
        todo = todo[:limit]
    total = len(done) + len(todo)
    errors = 0
    write_progress(project, step=step, status="starting", done=len(done), total=total, errors=errors)

    def flush() -> None:
        frame = pd.DataFrame(rows)
        atomic_write(out_path, lambda tmp: frame.to_parquet(tmp, index=False))

    p = f"l{layer}_"
    n_llm = sum(1 for c in protocol.criteria if not c.deterministic)
    failing_in_a_row = 0
    enrich_errors = 0
    enrich_failing_in_a_row = 0
    try:
        for i, rec in enumerate(todo, start=1):
            extra, enrich_failed = None, False
            if enrich:
                try:
                    extra = enrich(rec)
                    enrich_failing_in_a_row = 0
                except EnrichmentError:
                    enrich_failed = True
                    enrich_errors += 1
                    enrich_failing_in_a_row += 1
                    if enrich_failing_in_a_row >= MAX_ENRICH_FAILURES:
                        if rows:
                            flush()
                        raise PipelineError(
                            f"OpenAlex and Europe PMC are not responding (no metadata for {MAX_ENRICH_FAILURES} "
                            "records in a row); layer 2 stopped. Check the internet connection and run it "
                            "again: finished records are kept") from None
            result = screen_record(protocol, rec.model_dump(mode="json"), chat=chat, model=model, mode=mode,
                                   extra_context=extra, workers=workers)
            row = {"record_id": _record_id(rec), "source": str(rec.source), "source_id": rec.source_id,
                   "doi": rec.doi, "year": rec.year, "title": rec.title, "document_type": rec.document_type}
            for cid, a in result.answers.items():
                row[p + cid] = a.answer
                row[p + cid + "_evidence"] = a.evidence
            row[p + "decision"] = result.decision.decision
            row[p + "reason"] = result.decision.reason
            row[p + "errors"] = result.errors
            if enrich:
                row[p + "enrich_error"] = enrich_failed
            rows.append(row)
            errors += result.errors
            failing_in_a_row = failing_in_a_row + 1 if n_llm and result.errors == n_llm else 0
            if failing_in_a_row >= MAX_FAILING_RECORDS:
                flush()
                last = result.error_messages[-1] if result.error_messages else "unknown error"
                raise PipelineError(
                    f"the AI model is not responding ({last}); stopped after {MAX_FAILING_RECORDS} records "
                    "in a row failed. Check the AI model settings and run the step again: finished "
                    "records are kept and failed ones are retried")
            if i % checkpoint_every == 0:
                flush()
            write_progress(project, status="running", done=len(done) + i, errors=errors,
                           enrich_errors=enrich_errors)
        flush()
    except BaseException as exc:
        try:
            if rows:
                flush()
        finally:
            from floccus_app.textsafe import redact

            write_progress(project, status="failed", errors=errors, message=redact(str(exc)))
        raise
    write_progress(project, status="done")
    decisions = Counter(r[p + "decision"] for r in rows)
    out = {"screened": len(rows), "errors": errors, **{k: decisions.get(k, 0)
                                                        for k in ("INCLUDE", "EXCLUDE", "HUMAN_REVIEW")}}
    if layer == 2:
        out["enrich_errors"] = enrich_errors
    return out


def consolidate(project: Path) -> dict:
    """Final decision = layer 2 when the record went there, else layer 1."""
    l1_path = processed_dir(project) / "screening_l1.parquet"
    if not l1_path.is_file():
        raise PipelineError("run layer 1 first")
    l1 = pd.read_parquet(l1_path)
    l2_path = processed_dir(project) / "screening_l2.parquet"
    l2 = pd.read_parquet(l2_path) if l2_path.is_file() else pd.DataFrame(columns=["record_id"])
    l2_cols = [c for c in l2.columns if c.startswith("l2_")]
    dec = l1.merge(l2[["record_id", *l2_cols]], on="record_id", how="left")
    final = dec["l2_decision"] if "l2_decision" in dec else pd.Series([None] * len(dec))
    dec["final_decision"] = final.fillna(dec["l1_decision"])
    dec["decision_layer"] = final.notna().map({True: "L2", False: "L1"})
    atomic_write(latest_dir(project) / "screening_decisions.csv", lambda tmp: dec.to_csv(tmp, index=False))

    def tally(series) -> dict:
        c = Counter(series.dropna())
        return {"include": c.get("INCLUDE", 0), "exclude": c.get("EXCLUDE", 0),
                "human_review": c.get("HUMAN_REVIEW", 0)}

    counts = read_counts(project)
    counts["screening_layer1"] = {"input": len(l1), **tally(l1["l1_decision"])}
    l2_in = l2[l2["record_id"].isin(l1["record_id"])] if len(l2) else l2
    counts["screening_layer2"] = {"input": len(l2_in),
                                  **tally(l2_in["l2_decision"] if "l2_decision" in l2_in else pd.Series([], dtype=object))}
    f = tally(dec["final_decision"])
    counts["final_specificity_mode"] = {**f, "to_eligibility": f["include"] + f["human_review"]}
    write_counts(project, counts)
    return counts
