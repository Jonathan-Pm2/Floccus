"""ProQuest export adapter (Earth, Atmospheric & Aquatic Science Database).

Maps the columns of a ProQuest "Export → Excel" file to PaperRecord:

  Title -> title, Abstract -> abstract, StoreId -> source_id (unique per paper),
  Authors (";"-separated) -> authors, digitalObjectIdentifier -> doi,
  documentType -> document_type, language -> language, pubtitle -> venue,
  year -> year, DocumentURL -> url,
  identifierKeywords + subjectTerms (" , "-separated) -> keywords (deduplicated).

ProQuest pads most cells with a leading space and repeats subject terms; both
are cleaned here. The document type is kept verbatim (e.g. "Feature , Journal
Article"), as the other adapters keep their source vocabulary.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from biofloc_review.normalize import coerce_year, normalize_doi, normalize_title, split_authors
from biofloc_review.schema import Corpus, PaperRecord, Source


def _clean(v) -> str | None:
    if v is None:
        return None
    if isinstance(v, float) and pd.isna(v):
        return None
    s = str(v).strip()
    return None if not s or s.lower() in {"nan", "none", "null"} else s


def _clean_id(v) -> str | None:
    """StoreId is read as int (or float when the column has gaps)."""
    if isinstance(v, float) and not pd.isna(v) and v.is_integer():
        v = int(v)
    return _clean(v)


def split_proquest_terms(*raw) -> list[str]:
    """Split ProQuest " , "-separated term lists; drop blanks and repeats, keep order."""
    out: list[str] = []
    seen: set[str] = set()
    for r in raw:
        s = _clean(r)
        if not s:
            continue
        sep = " , " if " , " in s else ";" if ";" in s else ","
        for term in s.split(sep):
            t = term.strip()
            if t and t.lower() not in seen:
                seen.add(t.lower())
                out.append(t)
    return out


def records_from_frame(df: pd.DataFrame, corpus: Corpus = Corpus.BIOFLOC) -> list[PaperRecord]:
    out: list[PaperRecord] = []
    for _, row in df.iterrows():
        title = _clean(row.get("Title"))
        if not title:
            continue
        abstract = _clean(row.get("Abstract"))
        out.append(PaperRecord(
            doi=normalize_doi(_clean(row.get("digitalObjectIdentifier"))),
            title=title,
            title_norm=normalize_title(title),
            authors=split_authors(_clean(row.get("Authors")), sep=";"),
            year=coerce_year(row.get("year")),
            venue=_clean(row.get("pubtitle")),
            document_type=_clean(row.get("documentType")),
            language=_clean(row.get("language")),
            abstract=abstract,
            keywords=split_proquest_terms(row.get("identifierKeywords"), row.get("subjectTerms")),
            open_access=None,
            url=_clean(row.get("DocumentURL")),
            source=Source.PROQUEST,
            source_id=_clean_id(row.get("StoreId")),
            corpus=corpus,
            needs_enrichment=abstract is None,
        ))
    return out


def load_proquest(path: Path, corpus: Corpus = Corpus.BIOFLOC) -> list[PaperRecord]:
    engine = "xlrd" if path.suffix.lower() == ".xls" else None
    try:
        df = pd.read_excel(path, engine=engine)
    except Exception:
        # some ProQuest ".xls" exports are actually xlsx
        df = pd.read_excel(path)
    return records_from_frame(df, corpus)
