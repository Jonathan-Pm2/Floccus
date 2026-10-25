"""Baseline 2 (reviewer 2, comment 7): one LLM call per record, one question.

Instead of nine criterion-wise calls, the model receives the whole
eligibility definition (inclusion I1-I3 and exclusion E1-E4 of
biofloc_review.screening.criteria, condensed into one paragraph) and answers
INCLUDE, EXCLUDE or UNCLEAR. Same model, client and settings as the pipeline
(_ollama_chat: temperature 0, think=False, num_predict 400), same record
text (title, abstract truncated to 3000 characters, keywords, venue) and the
reasoning-first output format of the sensitivity mode. The LAST "ANSWER:"
line is the verdict; no parseable answer counts as UNCLEAR.

Resume-safe cache: data/processed/baseline_single_prompt_answers.parquet
(record_id, answer, raw, seconds); a rerun only calls records not yet cached.
The run stops early if the server fails repeatedly or a call takes longer
than MAX_SECONDS.

Run:  set -a; . ../.env; set +a
      .venv/bin/python scripts/baseline_single_prompt.py [--workers 4]
"""
from __future__ import annotations

import argparse
import re
import sys
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "processed"
CACHE = DATA / "baseline_single_prompt_answers.parquet"
MAX_SECONDS = 120.0
MAX_WORKERS = 4
MAX_CONSECUTIVE_FAILURES = 3
CHECKPOINT_EVERY = 25

SYSTEM = """Eres un revisor sistemático experto en revisiones PRISMA sobre modelado de flóculos biológicos en sistemas acuáticos. Decides si un artículo debe pasar a lectura de texto completo mirando su título, abstract, palabras clave y venue.

CRITERIO DE ELEGIBILIDAD:
Un artículo es elegible si cumple las tres condiciones siguientes: (1) estudia flóculos microbianos, biofloc, lodos activados, perifiton o agregados bacterianos; (2) presenta, aplica, valida o discute al menos un modelo matemático, ecuación, simulación numérica o expresión cuantitativa formal del sistema biológico (ecuaciones diferenciales, cinéticas tipo Monod, balances de masa, modelos ASM, simulaciones CFD, modelos poblacionales, etc.; la palabra "modelo" sola no basta); y (3) el contexto es acuático aplicado: acuicultura, tratamiento de aguas residuales o de agua, sistemas de cultivo, biorreactores o calidad del agua. No es elegible si ocurre cualquiera de estas situaciones: "modelo" se refiere solo a un organismo modelo y no a modelación matemática; el estudio es puramente experimental o empírico, sin componente cuantitativo formal de modelación; el sistema principal estudiado está fuera de un medio acuoso (suelo, atmósfera, alimentos sólidos, tejido clínico, procesos industriales secos); o trata de coagulación/floculación química pura, con coagulantes inorgánicos, sin componente microbiológico.

Primero razona y cita la evidencia; SOLO DESPUÉS decide. La respuesta final debe ser coherente con tu razonamiento.
- INCLUDE = el artículo es elegible.
- EXCLUDE = el artículo no es elegible.
- UNCLEAR = la información disponible no permite decidir con confianza razonable.

Responde EXACTAMENTE en este formato y en este orden (sin texto antes ni después):
REASONING: <dos a cuatro frases>
ANSWER: INCLUDE | EXCLUDE | UNCLEAR
"""

USER = """Información del artículo a evaluar:

TÍTULO:
{title}

ABSTRACT:
{abstract}

KEYWORDS:
{keywords}

VENUE:
{venue}

Aplica el criterio de elegibilidad y responde con el formato indicado.
"""

_ANSWER_RE = re.compile(r"ANSWER\s*\**\s*:\s*\**\s*(INCLUDE|EXCLUDE|UNCLEAR)", re.IGNORECASE)


def parse_answer(text) -> str:
    """Verdict = the LAST 'ANSWER:' line (the post-reasoning one); UNCLEAR if none."""
    found = _ANSWER_RE.findall(text) if isinstance(text, str) else []
    return found[-1].upper() if found else "UNCLEAR"


def _text(v, empty: str = "(none)") -> str:
    """Missing fields arrive as None or NaN; NaN is a truthy float."""
    return v.strip() if isinstance(v, str) and v.strip() else empty


def build_messages(title, abstract, keywords, venue) -> list[dict]:
    return [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": USER.format(
                title=_text(title), abstract=_text(abstract, "(no abstract)")[:3000],
                keywords=_text(keywords), venue=_text(venue))}]


def ask(row) -> tuple[str, str, float]:
    from biofloc_review.dedup.llm_judge import _ollama_chat, default_llm_model
    t0 = time.monotonic()
    raw = _ollama_chat(build_messages(row.title, row.abstract, row.keywords, row.venue),
                       model=default_llm_model(), timeout=MAX_SECONDS)
    return parse_answer(raw), raw, time.monotonic() - t0


def _save(done: pd.DataFrame, rows: list[dict]) -> pd.DataFrame:
    if rows:
        done = pd.concat([done, pd.DataFrame(rows)], ignore_index=True)
        done.to_parquet(CACHE, index=False)
    return done


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=MAX_WORKERS)
    ap.add_argument("--limit", type=int, default=None, help="call at most this many records")
    args = ap.parse_args()
    workers = max(1, min(args.workers, MAX_WORKERS))

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from make_second_reviewer_workbook import load_sample
    sample = load_sample()
    done = (pd.read_parquet(CACHE) if CACHE.exists()
            else pd.DataFrame(columns=["record_id", "answer", "raw", "seconds"]))
    pending = sample[~sample.record_id.isin(done.record_id)]
    if args.limit is not None:
        pending = pending.head(args.limit)
    print(f"sample {len(sample)} · cached {len(done)} · pending {len(pending)} · workers {workers}",
          flush=True)

    rows: list[dict] = []
    failures = 0
    stop_reason = None
    t_start = time.monotonic()
    it = iter(pending.itertuples())
    with ThreadPoolExecutor(workers) as ex:
        running = {}
        for _ in range(workers):
            r = next(it, None)
            if r is not None:
                running[ex.submit(ask, r)] = r.record_id
        while running:
            finished, _ = wait(running, return_when=FIRST_COMPLETED)
            for f in finished:
                rid = running.pop(f)
                try:
                    ans, raw, sec = f.result()
                except Exception as e:  # not cached: retried on the next run
                    failures += 1
                    print(f"  fail {rid}: {type(e).__name__}", flush=True)
                    if failures >= MAX_CONSECUTIVE_FAILURES:
                        stop_reason = f"{failures} consecutive failed calls"
                    continue
                failures = 0
                rows.append({"record_id": rid, "answer": ans, "raw": raw, "seconds": sec})
                if sec > MAX_SECONDS:
                    stop_reason = f"call took {sec:.0f} s (> {MAX_SECONDS:.0f} s)"
                if len(rows) >= CHECKPOINT_EVERY:
                    done = _save(done, rows); rows = []
                    print(f"  {len(done)}/{len(sample)}  "
                          f"elapsed {time.monotonic() - t_start:.0f} s", flush=True)
            if stop_reason is None:
                while len(running) < workers:
                    r = next(it, None)
                    if r is None:
                        break
                    running[ex.submit(ask, r)] = r.record_id
    done = _save(done, rows)
    print(f"cached {len(done)}/{len(sample)} · run time {time.monotonic() - t_start:.0f} s")
    print(done.answer.value_counts().to_string())
    if stop_reason:
        print(f"STOPPED EARLY: {stop_reason}")
        sys.exit(1)


if __name__ == "__main__":
    main()
