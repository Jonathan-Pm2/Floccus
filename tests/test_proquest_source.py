"""Column mapping of the ProQuest (EAASD) export adapter, on an in-memory frame."""
from __future__ import annotations

import numpy as np
import pandas as pd

from biofloc_review.schema import Corpus, Source
from biofloc_review.sources.proquest import records_from_frame, split_proquest_terms


def _frame() -> pd.DataFrame:
    return pd.DataFrame([
        {
            "Title": "Biofloc technology in shrimp culture",
            "Abstract": "A study of floc.",
            "StoreId": 3365902793,
            "Authors": "Avnimelech, Yoram;Crab, Roselien",
            "digitalObjectIdentifier": "https://doi.org/10.1016/J.AQUACULTURE.2007.1",
            "documentType": " Feature , Journal Article",
            "language": " English",
            "pubtitle": "Aquaculture",
            "year": 2007,
            "DocumentURL": "https://example.org/doc/1",
            "identifierKeywords": " Biofloc , C/N ratio",
            "subjectTerms": " Aquaculture , biofloc , Aquaculture , Shrimp",
        },
        {
            "Title": "  ",  # dropped: empty title
            "Abstract": "x", "StoreId": 2, "Authors": np.nan, "digitalObjectIdentifier": np.nan,
            "documentType": np.nan, "language": np.nan, "pubtitle": np.nan, "year": np.nan,
            "DocumentURL": np.nan, "identifierKeywords": np.nan, "subjectTerms": np.nan,
        },
        {
            "Title": "Zero water exchange systems",
            "Abstract": np.nan, "StoreId": 17.0, "Authors": np.nan, "digitalObjectIdentifier": np.nan,
            "documentType": "Literature Review", "language": np.nan, "pubtitle": np.nan, "year": np.nan,
            "DocumentURL": np.nan, "identifierKeywords": np.nan, "subjectTerms": np.nan,
        },
    ])


def test_column_mapping():
    recs = records_from_frame(_frame())
    assert len(recs) == 2
    r = recs[0]
    assert r.source is Source.PROQUEST and r.corpus is Corpus.BIOFLOC
    assert r.source_id == "3365902793"
    assert r.doi == "10.1016/j.aquaculture.2007.1"
    assert r.title_norm == "biofloc technology in shrimp culture"
    assert r.authors == ["Avnimelech, Yoram", "Crab, Roselien"]
    assert r.year == 2007 and r.venue == "Aquaculture"
    assert r.document_type == "Feature , Journal Article"
    assert r.language == "English"
    assert r.url == "https://example.org/doc/1"
    assert r.keywords == ["Biofloc", "C/N ratio", "Aquaculture", "Shrimp"]
    assert r.needs_enrichment is False


def test_missing_fields_and_float_store_id():
    r = records_from_frame(_frame())[1]
    assert r.source_id == "17"
    assert r.abstract is None and r.needs_enrichment is True
    assert r.doi is None and r.year is None and r.authors == [] and r.keywords == []
    assert r.document_type == "Literature Review"


def test_split_proquest_terms():
    assert split_proquest_terms(None, " a , b , A ", "c;d") == ["a", "b", "c", "d"]
    assert split_proquest_terms(np.nan) == []
