"""SciELO and declared-language audit of the screening outcome.

Reviewers asked whether the screening pipeline penalises the regional
(Ibero-American) literature. This script recomputes the SciELO figures the
manuscript reports:

- final screening decisions of the SciELO records and how many reach the
  eligible set (INCLUDE + HUMAN_REVIEW);
- citation profile of the eligible SciELO records (OpenAlex cited-by counts):
  uncited share and share of the eligible citation stock;
- SciELO weight in the biofloc-aquaculture cluster C03 and the share of its
  main venue, the Latin American Journal of Aquatic Research (LAJAR);
- the three SciELO records with a final INCLUDE;
- declared-language counts over the whole screened corpus, where a record
  counts for a language when its declared language field contains it (so
  bilingual records such as "Spanish, English" count for both).

Run:  .venv/bin/python scripts/scielo_language_audit.py
Out:  data/processed/scielo_language_audit.json
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "processed"
ELIGIBLE = ("INCLUDE", "HUMAN_REVIEW")
LANGUAGES = ("English", "Spanish", "Portuguese")
LAJAR = "LATIN AMERICAN JOURNAL OF AQUATIC RESEARCH"
CORE_CLUSTER = 3
MISSING = {"", "nan", "na", "none", "undefined"}


# ---------------------------------------------------------------- pure helpers
def declares(field, language: str) -> bool:
    """True when a declared language field (possibly a list "A, B") contains `language`."""
    if not isinstance(field, str) or field.strip().lower() in MISSING:
        return False
    return language.lower() in {p.strip().lower() for p in field.replace(";", ",").split(",")}


def count_language(fields: Iterable, language: str) -> int:
    """Records whose declared language field contains `language`."""
    return sum(declares(f, language) for f in fields)


def decision_counts(decisions: pd.Series) -> dict:
    return {k: int(v) for k, v in decisions.value_counts().items()}


# ---------------------------------------------------------------- sections
def screening(dec: pd.DataFrame) -> dict:
    sci = dec[dec.source == "scielo"]
    elig = dec.final_decision.isin(ELIGIBLE)
    return {
        "screened": int(len(sci)),
        "decisions": decision_counts(sci.final_decision),
        "eligible": int(sci.final_decision.isin(ELIGIBLE).sum()),
        "eligible_share_of_all_eligible_pct": round(100 * sci.final_decision.isin(ELIGIBLE).sum() / elig.sum(), 2),
    }


def citations(cit: pd.DataFrame) -> dict:
    sci = cit[cit.source == "scielo"]
    c = sci.oa_cited_by_count.fillna(0)
    total = cit.oa_cited_by_count.fillna(0).sum()
    return {
        "eligible_scielo": int(len(sci)),
        "uncited": int((c == 0).sum()),
        "uncited_pct": round(100 * float((c == 0).mean()), 1),
        "citations": int(c.sum()),
        "citation_stock_eligible": int(total),
        "share_of_citations_pct": round(100 * float(c.sum() / total), 2),
    }


def core_cluster(cit: pd.DataFrame, ven: pd.DataFrame) -> dict:
    c3 = cit[cit.cluster == CORE_CLUSTER]
    venues = ven.loc[ven.cluster == CORE_CLUSTER, "venue_norm"].value_counts()
    lajar = int(venues.get(LAJAR, 0))
    return {
        "cluster": f"C{CORE_CLUSTER:02d}",
        "n": int(len(c3)),
        "by_source": {k: int(v) for k, v in c3.source.value_counts().items()},
        "lajar": lajar,
        "lajar_pct": round(100 * lajar / len(c3), 1),
        "top_venues": {k: int(v) for k, v in venues.head(5).items()},
    }


def includes(dec: pd.DataFrame, cit: pd.DataFrame) -> list[dict]:
    inc = dec[(dec.source == "scielo") & (dec.final_decision == "INCLUDE")]
    inc = inc.merge(cit[["record_id", "cluster", "oa_cited_by_count"]], on="record_id", how="left")
    return [{"title": r.title, "year": int(r.year) if pd.notna(r.year) else None, "venue": r.venue,
             "language": r.language, "cluster": None if pd.isna(r.cluster) else int(r.cluster),
             "cited_by": None if pd.isna(r.oa_cited_by_count) else int(r.oa_cited_by_count),
             "decision_source": r.final_decision_source}
            for r in inc.itertuples()]


def languages(dec: pd.DataFrame) -> dict:
    elig = dec[dec.final_decision.isin(ELIGIBLE)]
    inc = dec[dec.final_decision == "INCLUDE"]
    sci = dec[dec.source == "scielo"]
    out = {}
    for lang in LANGUAGES:
        out[lang] = {
            "screened": count_language(dec.language, lang),
            "eligible": count_language(elig.language, lang),
            "include": count_language(inc.language, lang),
            "screened_scielo": count_language(sci.language, lang),
            "eligible_scielo": count_language(sci[sci.final_decision.isin(ELIGIBLE)].language, lang),
        }
    out["bilingual_records"] = int(dec.language.fillna("").str.contains(",").sum())
    return out


def main() -> None:
    dec = pd.read_parquet(DATA / "biofloc_final_decisions.parquet",
                          columns=["record_id", "title", "year", "venue", "source", "language",
                                   "final_decision", "final_decision_source"])
    cit = pd.read_parquet(DATA / "eligible_with_citations.parquet")
    ven = pd.read_parquet(DATA / "eligible_with_venues.parquet")

    res = {
        "screened_corpus": int(len(dec)),
        "scielo_screening": screening(dec),
        "scielo_citations": citations(cit),
        "scielo_in_core_cluster": core_cluster(cit, ven),
        "scielo_include_records": includes(dec, cit),
        "declared_language_contains": languages(dec),
    }
    (DATA / "scielo_language_audit.json").write_text(json.dumps(res, indent=2, ensure_ascii=False))
    print(json.dumps(res, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
