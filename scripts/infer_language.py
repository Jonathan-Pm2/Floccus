"""Language coverage of the corpus, completing the missing language field.

About 40% of the deduplicated records carry no language field in the source
exports, so the declared counts understate the non-English share (reviewer 2,
comment 5). This script keeps the declared language when present and infers
it from title + abstract otherwise (langdetect, fixed seed). It then checks
the detector against records whose language IS declared, so the inferred
counts come with a measured accuracy.

Caveat: the detector reads the indexed metadata, not the full text. Every
record declared Chinese, and ~40% of those declared Spanish, carry an English
(or bilingual) title and abstract, so they are detected as English. Inferred
counts are therefore a LOWER bound on non-English publications; the
declared-only counts are reported alongside.

Run:  .venv/bin/python scripts/infer_language.py
Out:  data/processed/language_coverage.json
"""
from __future__ import annotations

import json
import uuid
from pathlib import Path

import pandas as pd
from langdetect import DetectorFactory, LangDetectException, detect

DetectorFactory.seed = 0
ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "processed"
ISO = {"en": "English", "es": "Spanish", "pt": "Portuguese", "zh-cn": "Chinese", "zh-tw": "Chinese",
       "fr": "French", "de": "German", "it": "Italian", "ru": "Russian", "ja": "Japanese", "ko": "Korean"}
MISSING = {"", "nan", "na", "none", "undefined"}


def norm(lang) -> str | None:
    s = str(lang).strip()
    return None if s.lower() in MISSING else s


def bucket(langs: pd.Series) -> dict:
    return langs.where(langs.isin(ISO.values()), "Other").value_counts().to_dict()


def guess(title, abstract) -> str | None:
    text = " ".join(t for t in (title, abstract) if isinstance(t, str))
    if len(text) < 40:
        return None
    try:
        return ISO.get(detect(text), "Other")
    except LangDetectException:
        return None


def main() -> None:
    dec = pd.read_parquet(DATA / "biofloc_final_decisions.parquet",
                          columns=["record_id", "title", "language", "final_decision"])
    src = pd.read_parquet(DATA / "biofloc_final_primary_v3.parquet",
                          columns=["source", "source_id", "title_norm", "abstract"])
    src["record_id"] = [str(uuid.uuid5(uuid.NAMESPACE_URL, f"{s}:{sid or (tn or '')[:80]}"))
                        for s, sid, tn in zip(src.source, src.source_id, src.title_norm)]
    d = dec.merge(src[["record_id", "abstract"]].drop_duplicates("record_id"), on="record_id", how="left")
    d["declared"] = d.language.map(norm)
    d["inferred"] = [guess(t, a) for t, a in zip(d.title, d.abstract)]

    # accuracy of the detector where the language is declared
    chk = d[d.declared.notna() & d.inferred.notna()]
    decl = chk.declared.where(chk.declared.isin(ISO.values()), "Other")
    acc = float((decl == chk.inferred).mean())
    per = {l: float((chk.inferred[decl == l] == l).mean()) for l in ["English", "Spanish", "Portuguese", "Chinese"]
           if (decl == l).sum() >= 20}

    d["final_language"] = d.declared.fillna(d.inferred).fillna("Undetermined")
    d["final_language"] = d.final_language.where(
        d.final_language.isin(list(ISO.values()) + ["Undetermined"]), "Other")
    elig = d.final_decision.isin(["INCLUDE", "HUMAN_REVIEW"])
    res = {
        "n": int(len(d)),
        "declared_missing": int(d.declared.isna().sum()),
        "filled_by_inference": int((d.declared.isna() & d.inferred.notna()).sum()),
        "detector_accuracy_on_declared": acc, "detector_accuracy_by_language": per,
        "declared_only_deduplicated": bucket(d.declared.dropna()),
        "declared_only_eligible": bucket(d[elig].declared.dropna()),
        "deduplicated": d.final_language.value_counts().to_dict(),
        "eligible": d[elig].final_language.value_counts().to_dict(),
        "include": d[d.final_decision == "INCLUDE"].final_language.value_counts().to_dict(),
    }
    (DATA / "language_coverage.json").write_text(json.dumps(res, indent=2, ensure_ascii=False))
    print(json.dumps(res, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
