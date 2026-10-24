"""Pilot: does answer-last + positive phrasing fix the E2/E4 polarity errors?

Diagnosis (2026-10-24, reviewer 2 revision): 13 of the 15 false negatives in
the blind validation sample were excluded by E2 or E4 with a REASONING that
argued the opposite of the ANSWER (e.g. E2=YES "purely experimental, without
modelling" while the reasoning says "indicates a formal quantitative modelling
component"). Two causes compound: the output format asks for ANSWER before
REASONING, so the verdict is committed before the reasoning is produced, and
E2/E4 are negatively phrased ("... WITHOUT ...").

This pilot re-asks only E2 and E4 on the 465-record blind sample with
  (1) reasoning first, answer last, and
  (2) the question phrased positively, the answer then inverted so that the
      stored exclusion semantics are unchanged,
substitutes the two answers into the persisted response vector, re-applies
the pre-registered decision rule, and compares sensitivity/specificity
against the human labels. Production code is not modified.

Run:  .venv/bin/python scripts/pilot_prompt_v2.py [--workers 4]
Out:  data/processed/pilot_prompt_v2.parquet
"""
from __future__ import annotations

import argparse
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from scipy.stats import beta

from biofloc_review.dedup.llm_judge import _ollama_chat, default_llm_model

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "processed"

SYSTEM_V2 = """Eres un revisor sistemático experto en revisiones PRISMA sobre modelado de flóculos biológicos en sistemas acuáticos. Respondes UNA pregunta sobre un artículo mirando su título, abstract, palabras clave y venue.

PREGUNTA:
{question}

Primero razona y cita la evidencia; SOLO DESPUÉS decide. La respuesta final debe ser coherente con tu razonamiento.
- YES = la respuesta a la pregunta es "Sí".
- NO = la respuesta a la pregunta es "No".
- UNCLEAR = la información disponible no permite decidir con confianza razonable.

Responde EXACTAMENTE en este formato y en este orden (sin texto antes ni después):
REASONING: <una o dos frases>
EVIDENCE: <fragmento exacto del título/abstract/keywords/venue, o "(ninguna)">
CONFIDENCE: <0-10>
ANSWER: YES | NO | UNCLEAR
"""

USER = """TÍTULO:
{title}

ABSTRACT:
{abstract}

KEYWORDS:
{keywords}

VENUE:
{venue}
"""

# Positive phrasings. The stored criterion fires (YES) exactly when the
# positive question is answered NO, so the answer is inverted on the way back.
POSITIVE = {
    "E2": ("¿El estudio incluye algún componente cuantitativo formal de "
           "modelación (ecuaciones, modelos matemáticos o simulaciones), y no "
           "solo el reporte de mediciones?"),
    "E4": ("¿El estudio involucra algún componente microbiológico o biológico "
           "(flóculos biológicos, lodos activados, biomasa microbiana, biofloc, "
           "biopelícula), y no solo coagulación/floculación química con "
           "coagulantes inorgánicos?"),
}
INVERT = {"YES": "NO", "NO": "YES", "UNCLEAR": "UNCLEAR"}
ANS = re.compile(r"ANSWER\s*:\s*(YES|NO|UNCLEAR)", re.I)


def ask(row, cid: str) -> tuple[str, str]:
    msgs = [{"role": "system", "content": SYSTEM_V2.format(question=POSITIVE[cid])},
            {"role": "user", "content": USER.format(
                title=row.title or "", abstract=(row.abstract or "(no abstract)")[:3000],
                keywords=row.keywords or "", venue=row.venue or "")}]
    text = _ollama_chat(msgs, model=default_llm_model())
    m = ANS.findall(text)               # last ANSWER wins: it is the post-reasoning verdict
    pos = m[-1].upper() if m else "UNCLEAR"
    return INVERT[pos], text


def decide(r: pd.Series) -> str:
    """Pre-registered asymmetric rule (Eq. screening-rule)."""
    if any(r[f"{c}_answer"] == "YES" for c in ("E1", "E2", "E3", "E4", "E5")):
        return "EXCLUDE"
    if any(r[f"{c}_answer"] == "YES" for c in ("D1", "D2", "D3")):
        return "HUMAN_REVIEW"
    if any(r[f"{c}_answer"] != "YES" for c in ("I1", "I2", "I3", "I4")):
        return "HUMAN_REVIEW"
    return "INCLUDE"


def cp(k: int, n: int) -> str:
    lo = beta.ppf(.025, k, n - k + 1) if k else 0.0
    hi = beta.ppf(.975, k + 1, n - k) if k < n else 1.0
    return f"{k}/{n} = {k/n:.3f} [{lo:.3f}–{hi:.3f}]"


def metrics(df: pd.DataFrame, col: str) -> None:
    adv = df[col] != "EXCLUDE"; h = df.meets
    print(f"  sensitivity {cp(int((adv & h).sum()), int(h.sum()))}")
    print(f"  specificity {cp(int((~adv & ~h).sum()), int((~h).sum()))}")


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    blind = pd.read_csv(DATA / "validation_sample_blind.csv")
    truth = pd.read_parquet(DATA / "validation_sample_truth.parquet")[["review_id", "record_id"]]
    dec = pd.read_parquet(DATA / "biofloc_final_decisions.parquet")
    df = blind[["review_id", "human_decision", "abstract", "keywords"]].merge(truth, on="review_id") \
        .merge(dec, on="record_id")
    df["meets"] = df.human_decision.str.strip().str.lower().eq("cumple")
    # Layer-1 decisions are the ones the rule produced from these answers
    df["base"] = df.apply(decide, axis=1)

    new = {}
    with ThreadPoolExecutor(args.workers) as ex:
        futs = {ex.submit(ask, r, c): (r.record_id, c) for r in df.itertuples() for c in ("E2", "E4")}
        for i, f in enumerate(as_completed(futs), 1):
            rid, c = futs[f]
            try:
                new[(rid, c)] = f.result()
            except Exception as e:                  # keep the stored answer on failure
                new[(rid, c)] = (None, f"ERROR {e}")
            if i % 50 == 0:
                print(f"  {i}/{len(futs)} calls", flush=True)

    v2 = df.copy()
    for c in ("E2", "E4"):
        v2[f"{c}_answer_v1"] = v2[f"{c}_answer"]
        v2[f"{c}_answer"] = [new[(r, c)][0] or v1 for r, v1 in zip(v2.record_id, v2[f"{c}_answer_v1"])]
        v2[f"{c}_raw_v2"] = [new[(r, c)][1] for r in v2.record_id]
    v2["v2"] = v2.apply(decide, axis=1)

    print("\n=== Layer-1 rule, stored answers (v1) ===");  metrics(v2, "base")
    print("=== Layer-1 rule, E2/E4 re-asked (v2) ==="); metrics(v2, "v2")
    for c in ("E2", "E4"):
        ch = (v2[f"{c}_answer"] != v2[f"{c}_answer_v1"])
        print(f"  {c}: answers changed {ch.sum()}  (YES→ {((v2[f'{c}_answer_v1']=='YES') & ch).sum()})")
    errs = v2.E2_raw_v2.str.startswith("ERROR").sum() + v2.E4_raw_v2.str.startswith("ERROR").sum()
    print(f"  failed calls: {errs}")
    v2.drop(columns=[c for c in v2.columns if c.endswith("_evidence") or c.endswith("_reasoning")]) \
      .to_parquet(DATA / "pilot_prompt_v2.parquet", index=False)
    print("→", DATA / "pilot_prompt_v2.parquet")


if __name__ == "__main__":
    main()
