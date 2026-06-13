"""Resolve authors + canonical citation metadata for the priority papers.

Reads data/processed/refs_priority.parquet (~111 papers with DOI) and
queries the OpenAlex /works endpoint to fetch authorships, year and
host venue display name. Saves a parquet with the resolved data ready
to generate `\bibitem` entries.

Rate-limited at 1 request per second to respect the OpenAlex polite
pool (~10 req/s nominal). Resumable: if the output exists, skips DOIs
already resolved.
"""
from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
import requests

OUT = Path("data/processed/refs_resolved.parquet")
SRC = Path("data/processed/refs_priority.parquet")


def lead_author(authorships) -> str:
    """Pick the first author's surname."""
    if not authorships:
        return ""
    a0 = authorships[0]
    name = (a0.get("author") or {}).get("display_name") or ""
    parts = name.split()
    return parts[-1] if parts else ""


def n_authors(authorships) -> int:
    return len(authorships or [])


def fetch_one(doi: str) -> dict:
    doi_clean = doi.strip()
    if doi_clean.startswith("http"):
        url = f"https://api.openalex.org/works/{doi_clean.split('doi.org/')[-1]}"
    else:
        url = f"https://api.openalex.org/works/doi:{doi_clean}"
    try:
        r = requests.get(url, timeout=20,
                         headers={"User-Agent": "Floccus/1.0 (mailto:REDACTED_EMAIL)"})
        if r.status_code != 200:
            return {"ok": False, "status": r.status_code}
        d = r.json()
        return {
            "ok": True,
            "lead_author": lead_author(d.get("authorships")),
            "n_authors": n_authors(d.get("authorships")),
            "year": d.get("publication_year"),
            "venue": ((d.get("host_venue") or {}).get("display_name")
                      or (d.get("primary_location") or {}).get("source", {}).get("display_name")
                      or ""),
            "type": d.get("type") or "",
        }
    except Exception as e:
        return {"ok": False, "error": str(e)}


def main() -> None:
    src = pd.read_parquet(SRC)
    print(f"Source: {len(src)} papers")

    done = pd.read_parquet(OUT) if OUT.exists() else pd.DataFrame()
    done_dois = set(done.doi.astype(str)) if len(done) else set()
    print(f"Already resolved: {len(done_dois)}")

    rows = list(done.to_dict("records")) if len(done) else []
    pending = src[~src.doi.astype(str).isin(done_dois)]
    print(f"Pending: {len(pending)}")

    for i, (_, r) in enumerate(pending.iterrows(), 1):
        doi = str(r.doi).strip()
        res = fetch_one(doi)
        out_row = {
            "record_id": r.record_id,
            "doi": doi,
            "orig_title": r.title,
            "orig_year": r.year,
            "orig_venue": r.venue if hasattr(r, "venue") else "",
            "orig_cit": r.cit if hasattr(r, "cit") else 0,
            **res,
        }
        rows.append(out_row)
        if i % 10 == 0:
            pd.DataFrame(rows).to_parquet(OUT, index=False)
            print(f"  {i}/{len(pending)} resolved (checkpoint)")
        time.sleep(1.0)

    pd.DataFrame(rows).to_parquet(OUT, index=False)
    print(f"\nFinal: {len(rows)} rows -> {OUT}")


if __name__ == "__main__":
    main()
