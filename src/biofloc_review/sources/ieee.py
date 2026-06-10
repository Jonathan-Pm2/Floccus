from __future__ import annotations

import hashlib
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


def _ieee_source_id(row: pd.Series) -> str | None:
    """Compute a per-paper unique identifier for IEEE records.

    The ``Document Identifier`` column in the IEEE Xplore CSV export is a
    session/batch-level value with massive collisions (only ~11 unique
    values across 545 papers in our corpus). Using it as ``source_id``
    causes record_id collapse downstream: dozens of distinct papers map
    to the same UUID and inherit each other's screening verdict.

    This helper builds a stable per-paper identifier by preferring the
    PDF Link (545/545 unique in practice), then the DOI, then a hash of
    the normalized title as final fallback. The chosen value is prefixed
    with the field name (``pdf:`` / ``doi:`` / ``titlehash:``) so the
    provenance of the identifier remains explicit.
    """
    for col, prefix in (("PDF Link", "pdf"), ("DOI", "doi")):
        v = row.get(col)
        if v is None:
            continue
        s = str(v).strip()
        if s and s.lower() not in {"nan", "none", ""}:
            return f"{prefix}:{s}"
    title = str(row.get("Document Title") or "").strip()
    if title:
        h = hashlib.sha1(normalize_title(title).encode("utf-8")).hexdigest()[:16]
        return f"titlehash:{h}"
    return None


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
            source_id=_ieee_source_id(row),
            corpus=corpus,
            needs_enrichment=abstract_s is None,
        )
        out.append(rec)
    return out
