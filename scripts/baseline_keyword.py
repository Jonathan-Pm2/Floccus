"""Baseline 1 (reviewer 2, comment 7): a keyword rule with no LLM.

A record advances to human review when its title + abstract + keywords
(lowercased) contain at least one MODELLING descriptor AND at least one FLOC
descriptor, using the descriptor sets of the Scopus TITLE-ABS-KEY query
(main text, Search strategy). Matching follows the query syntax:

- a trailing ``*`` is a prefix match ("model*" matches "modelling", "models");
- every term starts at a word boundary ("remodel" does not match "model*");
- a term without ``*`` must end at a word boundary, tolerating a plural "s"
  (Scopus matches plurals of unstarred terms);
- the words of a phrase may be separated by whitespace or a hyphen.

Because every corpus record already matched that query, the plain rule is
expected to advance almost everything. The STRICT variant additionally
requires one formal-modelling term (equation, differential, ODE, PDE,
kinetic, Monod, ASM, population balance, CFD, simulation, neural network),
so that a bare "model" does not suffice.

Run:  .venv/bin/python scripts/baseline_keyword.py
Out:  data/processed/baseline_keyword_decisions.parquet
      (record_id, review_id, advance_keyword, advance_keyword_strict)
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "processed"
OUT = DATA / "baseline_keyword_decisions.parquet"

# Descriptor sets of the database query ("bio-floc" added as a spelling variant).
MODELLING_TERMS = ("mathematical model*", "numerical model*", "kinetic model*",
                   "growth kinetic*", "simulation", "model*")
FLOC_TERMS = ("bacterial floc*", "microbial floc*", "floc dynamic*", "floc size",
              "floc formation", "biofloc*", "bio-floc*", "biofloc technology",
              "periphyton", "activated sludge")
# Formal-modelling terms for the strict variant.
FORMAL_TERMS = ("equation*", "differential*", "ode", "pde", "kinetic*", "monod",
                "asm", "asm1", "asm2", "asm2d", "asm3", "population balance*", "cfd",
                "computational fluid dynamic*", "simulat*", "neural network*")


def term_regex(term: str) -> str:
    """Regex for one query term: word-boundary start, '*' as prefix match,
    phrase words separated by whitespace or hyphens."""
    star = term.endswith("*")
    words = re.split(r"[\s-]+", term.rstrip("*").strip().lower())
    body = r"[\s\-]+".join(re.escape(w) for w in words)
    tail = r"\w*" if star else r"s?\b"
    return r"(?<!\w)" + body + tail


def compile_terms(terms) -> re.Pattern:
    return re.compile("|".join(term_regex(t) for t in terms))


MODELLING_RE = compile_terms(MODELLING_TERMS)
FLOC_RE = compile_terms(FLOC_TERMS)
FORMAL_RE = compile_terms(FORMAL_TERMS)


def _text(v) -> str:
    return v if isinstance(v, str) else ""


def record_text(title, abstract, keywords) -> str:
    return " ".join(_text(x) for x in (title, abstract, keywords)).lower()


def advance(text: str, strict: bool = False) -> bool:
    text = text.lower()
    ok = bool(MODELLING_RE.search(text)) and bool(FLOC_RE.search(text))
    if strict:
        ok = ok and bool(FORMAL_RE.search(text))
    return ok


def decide(sample: pd.DataFrame) -> pd.DataFrame:
    text = [record_text(t, a, k) for t, a, k in
            zip(sample.title, sample.abstract, sample.keywords)]
    return pd.DataFrame({
        "record_id": sample.record_id.to_numpy(),
        "review_id": sample.review_id.to_numpy(),
        "advance_keyword": [advance(t) for t in text],
        "advance_keyword_strict": [advance(t, strict=True) for t in text],
    })


def main() -> None:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from make_second_reviewer_workbook import load_sample
    out = decide(load_sample())
    out.to_parquet(OUT, index=False)
    print(f"n={len(out)}  advance (plain) {int(out.advance_keyword.sum())}  "
          f"advance (strict) {int(out.advance_keyword_strict.sum())}")
    print("->", OUT)


if __name__ == "__main__":
    main()
