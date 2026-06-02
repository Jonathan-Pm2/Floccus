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


def load_wos(path: Path, corpus: Corpus) -> list[PaperRecord]:
    if path.suffix.lower() == ".xls":
        df = pd.read_excel(path, engine="xlrd")
    else:
        df = pd.read_excel(path)
    out: list[PaperRecord] = []
    for _, row in df.iterrows():
        title = str(row.get("Article Title") or "").strip()
        if not title:
            continue
        abstract = row.get("Abstract")
        abstract_s = str(abstract).strip() if abstract is not None and str(abstract).strip() not in {"", "nan"} else None
        rec = PaperRecord(
            doi=normalize_doi(row.get("DOI")),
            title=title,
            title_norm=normalize_title(title),
            authors=split_authors(row.get("Authors")),
            year=coerce_year(row.get("Publication Year")),
            venue=(str(row.get("Source Title")).strip() or None) if row.get("Source Title") is not None else None,
            document_type=(str(row.get("Document Type")).strip() or None) if row.get("Document Type") is not None else None,
            language=(str(row.get("Language")).strip() or None) if row.get("Language") is not None else None,
            abstract=abstract_s,
            keywords=split_keywords(row.get("Author Keywords")),
            open_access=None,
            url=None,
            source=Source.WOS,
            source_id=(str(row.get("UT (Unique ID)")).strip() or None) if row.get("UT (Unique ID)") is not None else None,
            corpus=corpus,
            needs_enrichment=abstract_s is None,
        )
        out.append(rec)
    return out
