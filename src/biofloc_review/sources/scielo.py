from __future__ import annotations

from pathlib import Path

import pandas as pd

from biofloc_review.normalize import (
    coerce_year,
    normalize_title,
    split_authors,
)
from biofloc_review.schema import Corpus, PaperRecord, Source


def load_scielo(path: Path, corpus: Corpus) -> list[PaperRecord]:
    df = pd.read_csv(path, low_memory=False)
    df.columns = [c.strip() for c in df.columns]
    out: list[PaperRecord] = []
    for _, row in df.iterrows():
        title = str(row.get("Title") or "").strip()
        if not title:
            continue
        rec = PaperRecord(
            doi=None,
            title=title,
            title_norm=normalize_title(title),
            authors=split_authors(row.get("Author(s)")),
            year=coerce_year(row.get("Publication year")),
            venue=(str(row.get("Journal")).strip() or None) if row.get("Journal") is not None else None,
            document_type=None,
            language=(str(row.get("Language(s)")).strip() or None) if row.get("Language(s)") is not None else None,
            abstract=None,
            keywords=[],
            open_access=None,
            url=(str(row.get("Fulltext URL")).strip() or None) if row.get("Fulltext URL") is not None else None,
            source=Source.SCIELO,
            source_id=(str(row.get("ID")).strip() or None) if row.get("ID") is not None else None,
            corpus=corpus,
            needs_enrichment=True,
        )
        out.append(rec)
    return out
