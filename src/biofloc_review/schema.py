from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, Field, field_validator


class Source(StrEnum):
    SCOPUS = "scopus"
    WOS = "wos"
    IEEE = "ieee"
    SCIELO = "scielo"
    OPENALEX = "openalex"


class Corpus(StrEnum):
    BIOFLOC = "biofloc"
    BIPLOT = "biplot"


class PaperRecord(BaseModel):
    doi: str | None = None
    title: str
    title_norm: str

    authors: list[str] = Field(default_factory=list)
    year: int | None = None
    venue: str | None = None
    document_type: str | None = None
    language: str | None = None

    abstract: str | None = None
    keywords: list[str] = Field(default_factory=list)

    open_access: bool | None = None
    url: str | None = None

    source: Source
    source_id: str | None = None
    corpus: Corpus

    enriched_from: list[Source] = Field(default_factory=list)
    needs_enrichment: bool = False

    cluster_id: str | None = None
    is_primary: bool = True
    merged_from: list[str] = Field(default_factory=list)

    @field_validator("title")
    @classmethod
    def _title_nonempty(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("title cannot be empty")
        return v

    @field_validator("year")
    @classmethod
    def _year_plausible(cls, v: int | None) -> int | None:
        if v is None:
            return None
        if v < 1900 or v > 2100:
            return None
        return v

    @field_validator("authors", "keywords", "enriched_from", "merged_from", mode="before")
    @classmethod
    def _strip_list(cls, v) -> list:
        if v is None:
            return []
        try:
            items = list(v)
        except TypeError:
            return []
        if len(items) == 0:
            return []
        return [s for s in (str(x).strip() for x in items) if s]
