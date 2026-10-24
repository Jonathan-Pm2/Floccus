"""Database overlap per publisher, to justify the choice of bibliographic sources.

Publishers such as MDPI or Taylor & Francis were not queried separately
because their journals are indexed in Scopus and Web of Science; IEEE
Xplore was queried because it also indexes proceedings the other two miss.
This script measures both claims at the level of works (Layer-1 duplicate
clusters), not of surviving records: the source label of a merged record
is only the source of its primary copy, so counting surviving records
overstates how many works a database contributes alone.

Publishers are identified by DOI prefix. Works are Layer-1 clusters
(deterministic DOI / normalised-title matching).

Run:  .venv/bin/python scripts/source_coverage_audit.py
Out:  data/processed/source_coverage_audit.json
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "processed"
PUBLISHERS = {"10.1016": "Elsevier", "10.1080": "Taylor & Francis", "10.3390": "MDPI",
              "10.1007": "Springer", "10.1002": "Wiley", "10.1111": "Wiley",
              "10.2166": "IWA Publishing", "10.1021": "ACS", "10.1109": "IEEE"}


def overlap(cluster_sources: pd.Series) -> dict:
    """Share of works found in both, or in only one, of Scopus and WoS, and only in IEEE."""
    s = cluster_sources
    in_scopus = s.map(lambda x: "scopus" in x)
    in_wos = s.map(lambda x: "wos" in x)
    return {"works": int(len(s)),
            "scopus_and_wos": int((in_scopus & in_wos).sum()),
            "only_one_of_scopus_wos": int((in_scopus ^ in_wos).sum()),
            "only_ieee": int(s.map(lambda x: x == frozenset({"ieee"})).sum())}


def main() -> None:
    f = pd.read_parquet(DATA / "biofloc_dedup_full.parquet", columns=["doi", "source", "cluster_id"])
    f["prefix"] = f.doi.astype(str).str.lower().str.extract(r"^(10\.\d{4,5})")[0]
    f["publisher"] = f.prefix.map(PUBLISHERS)
    sources = f.groupby("cluster_id").source.apply(frozenset)
    works = f.dropna(subset=["publisher"]).drop_duplicates(["cluster_id", "publisher"])
    works = works.assign(cluster_sources=works.cluster_id.map(sources))
    res = {pub: overlap(g.cluster_sources) for pub, g in works.groupby("publisher")}
    # own coverage of each queried database: works found in that database alone
    res["by_database"] = {}
    for db in ("scopus", "wos", "ieee", "scielo"):
        has = sources[sources.map(lambda x: db in x)]
        alone = int(has.map(lambda x: x == frozenset({db})).sum())
        res["by_database"][db] = {"works": int(len(has)), "only_here": alone,
                                  "only_here_share": round(alone / len(has), 3)}
    for k, r in res.items():
        if k == "by_database":
            continue
        r["only_one_share"] = round(r["only_one_of_scopus_wos"] / r["works"], 3)
        r["only_ieee_share"] = round(r["only_ieee"] / r["works"], 3)
    (DATA / "source_coverage_audit.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
