from __future__ import annotations

from pathlib import Path

import pandas as pd

from biofloc_review.normalize import (
    coerce_bool_oa,
    coerce_year,
    normalize_doi,
    normalize_title,
    split_authors,
    split_keywords,
)
from biofloc_review.schema import Corpus, PaperRecord, Source


def load_scopus(path: Path, corpus: Corpus) -> list[PaperRecord]:
    df = pd.read_csv(path, low_memory=False)
    out: list[PaperRecord] = []
    for _, row in df.iterrows():
        title = str(row.get("Title") or "").strip()
        if not title:
            continue
        kw = split_keywords(row.get("Author Keywords")) + split_keywords(row.get("Index Keywords"))
        abstract = row.get("Abstract")
        abstract_s = str(abstract).strip() if abstract is not None and str(abstract).strip() not in {"", "nan", "[No abstract available]"} else None
        rec = PaperRecord(
            doi=normalize_doi(row.get("DOI")),
            title=title,
            title_norm=normalize_title(title),
            authors=split_authors(row.get("Authors")),
            year=coerce_year(row.get("Year")),
            venue=(str(row.get("Source title")).strip() or None) if row.get("Source title") is not None else None,
            document_type=(str(row.get("Document Type")).strip() or None) if row.get("Document Type") is not None else None,
            language=(str(row.get("Language of Original Document")).strip() or None) if row.get("Language of Original Document") is not None else None,
            abstract=abstract_s,
            keywords=list(dict.fromkeys(kw)),
            open_access=coerce_bool_oa(row.get("Open Access")),
            url=(str(row.get("Link")).strip() or None) if row.get("Link") is not None else None,
            source=Source.SCOPUS,
            source_id=(str(row.get("EID")).strip() or None) if row.get("EID") is not None else None,
            corpus=corpus,
            needs_enrichment=abstract_s is None,
        )
        out.append(rec)
    return out
