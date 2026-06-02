from __future__ import annotations

from pathlib import Path

import pandas as pd

from biofloc_review.normalize import (
    coerce_year,
    normalize_doi,
    normalize_title,
    split_authors,
    split_keywords,
)
from biofloc_review.schema import Corpus, PaperRecord, Source


def load_ieee(path: Path, corpus: Corpus) -> list[PaperRecord]:
    df = pd.read_csv(path, low_memory=False)
    df = df.loc[:, ~df.columns.str.startswith("Unnamed:")]
    out: list[PaperRecord] = []
    for _, row in df.iterrows():
        title = str(row.get("Document Title") or "").strip()
        if not title:
            continue
        kw = (
            split_keywords(row.get("Author Keywords"))
            + split_keywords(row.get("IEEE Terms"))
            + split_keywords(row.get("Mesh_Terms"))
        )
        abstract = row.get("Abstract")
        abstract_s = str(abstract).strip() if abstract is not None and str(abstract).strip() not in {"", "nan"} else None
        rec = PaperRecord(
            doi=normalize_doi(row.get("DOI")),
            title=title,
            title_norm=normalize_title(title),
            authors=split_authors(row.get("Authors"), sep=";"),
            year=coerce_year(row.get("Publication Year")),
            venue=(str(row.get("Publication Title")).strip() or None) if row.get("Publication Title") is not None else None,
            document_type=None,
            language=None,
            abstract=abstract_s,
            keywords=list(dict.fromkeys(kw)),
            open_access=None,
            url=(str(row.get("PDF Link")).strip() or None) if row.get("PDF Link") is not None else None,
            source=Source.IEEE,
            source_id=(str(row.get("Document Identifier")).strip() or None) if row.get("Document Identifier") is not None else None,
            corpus=corpus,
            needs_enrichment=abstract_s is None,
        )
        out.append(rec)
    return out
