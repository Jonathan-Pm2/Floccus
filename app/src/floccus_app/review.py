"""Human review: judge uncertain records or a validation sample; measure agreement.

Agreement compares the pipeline, read as "advances" (INCLUDE or
HUMAN_REVIEW → include) versus EXCLUDE, with the reviewer's include/exclude
on the validation sample, using Cohen's kappa.
"""

from __future__ import annotations

import threading
import time
from collections import Counter
from pathlib import Path

import pandas as pd

LABELS = ("include", "exclude")
_SAVE_LOCK = threading.Lock()  # Streamlit sessions are threads of one process
SEED = 20260101


def decisions_path(project: Path) -> Path:
    return project / "results" / "latest" / "screening_decisions.csv"


def judgements_path(project: Path) -> Path:
    return project / "results" / "latest" / "human_review.csv"


def _decisions(project: Path) -> pd.DataFrame:
    path = decisions_path(project)
    if not path.is_file():
        raise FileNotFoundError("no screening results yet")
    return pd.read_csv(path, low_memory=False)


def queue(project: Path, kind: str, size: int = 100) -> pd.DataFrame:
    d = _decisions(project)
    if kind == "uncertain":
        return d[d["final_decision"] == "HUMAN_REVIEW"].reset_index(drop=True)
    if kind == "validation":
        return d.sample(n=min(size, len(d)), random_state=SEED).reset_index(drop=True)
    raise ValueError("kind must be 'uncertain' or 'validation'")


def judgements(project: Path) -> pd.DataFrame:
    path = judgements_path(project)
    if not path.is_file():
        return pd.DataFrame(columns=["record_id", "label", "note", "kind", "saved_at"])
    return pd.read_csv(path, dtype={"note": str}, keep_default_na=False)


def save_judgement(project: Path, record_id: str, label: str, note: str, kind: str) -> None:
    if label not in LABELS:
        raise ValueError("label must be include or exclude")
    from floccus_app.pipeline import atomic_write

    with _SAVE_LOCK:
        j = judgements(project)
        j = j[~((j["record_id"] == record_id) & (j["kind"] == kind))]  # one label per record and kind
        row = pd.DataFrame([{"record_id": record_id, "label": label, "note": note or "", "kind": kind,
                             "saved_at": time.strftime("%Y-%m-%d %H:%M:%S")}])
        j = pd.concat([j, row], ignore_index=True)
        atomic_write(judgements_path(project), lambda tmp: j.to_csv(tmp, index=False))


def cohens_kappa(a: list[str], b: list[str]) -> float | None:
    n = len(a)
    if n == 0 or n != len(b):
        return None
    po = sum(x == y for x, y in zip(a, b)) / n
    ca, cb = Counter(a), Counter(b)
    pe = sum(ca[k] * cb[k] for k in set(ca) | set(cb)) / (n * n)
    return 1.0 if pe == 1 else (po - pe) / (1 - pe)


def agreement(project: Path) -> dict:
    j = judgements(project)
    j = j[j["kind"] == "validation"]
    d = _decisions(project)[["record_id", "final_decision"]]
    m = j.merge(d, on="record_id")
    pipeline = ["exclude" if x == "EXCLUDE" else "include" for x in m["final_decision"]]
    human = list(m["label"])
    n = len(m)
    agree = sum(x == y for x, y in zip(pipeline, human))
    return {
        "n": n,
        "percent_agreement": 100.0 * agree / n if n else None,
        "kappa": cohens_kappa(pipeline, human),
        "table": pd.crosstab(pd.Series(pipeline, name="pipeline"), pd.Series(human, name="human")) if n else None,
    }
