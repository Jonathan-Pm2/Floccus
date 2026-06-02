"""CLI for criterion-based PRISMA screening with LangGraph orchestrator.

Output parquet schema (one row per record):
  - record_id, cluster_id, title, doi, year, venue, source, language, document_type
  - For each criterion X in {I1..I4, E1..E5, D1..D3}:
      X_answer       (YES | NO | UNCLEAR)
      X_confidence   (0-10)
      X_evidence
      X_reasoning
      X_via          (deterministic | llm)
  - final_decision_primary, final_reason_primary, final_triggers_primary
  - final_decision_aggressive, final_decision_permissive
    (sensitivity-analysis variants for Camino 3)

Usage:
  uv run python -m biofloc_review.run_screening \\
    data/processed/biofloc_final_primary.parquet \\
    --workers 2 --limit 50 -v
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

from biofloc_review.dedup.embeddings import _record_to_point_id
from biofloc_review.schema import PaperRecord
from biofloc_review.screening.classifier import CriterionAnswer
from biofloc_review.screening.criteria import CRITERIA
from biofloc_review.screening.decision import (
    HUMAN_REVIEW,
    FinalDecision,
    aggregate,
    aggregate_aggressive_exclude,
    aggregate_permissive_include,
)
from biofloc_review.screening.graph import screen_record

log = logging.getLogger(__name__)


def _parquet_to_records(path: Path) -> list[PaperRecord]:
    df = pd.read_parquet(path)
    out: list[PaperRecord] = []
    for _, row in df.iterrows():
        d = {
            k: (None if isinstance(v, float) and pd.isna(v) else v)
            for k, v in row.to_dict().items()
        }
        out.append(PaperRecord.model_validate(d))
    return out


def _criterion_cols(answer: CriterionAnswer | None, cid: str) -> dict:
    if answer is None:
        return {
            f"{cid}_answer": None,
            f"{cid}_confidence": None,
            f"{cid}_evidence": None,
            f"{cid}_reasoning": None,
            f"{cid}_via": None,
        }
    return {
        f"{cid}_answer": answer.answer,
        f"{cid}_confidence": answer.confidence,
        f"{cid}_evidence": answer.evidence,
        f"{cid}_reasoning": answer.reasoning,
        f"{cid}_via": answer.via,
    }


def _row_for(rec: PaperRecord, state: dict) -> dict:
    answers: dict[str, CriterionAnswer] = state.get("answers") or {}
    final: FinalDecision = state.get("final")

    # Sensitivity-analysis variants computed from the same answers
    final_aggr = aggregate_aggressive_exclude(answers)
    final_perm = aggregate_permissive_include(answers)

    row = {
        "record_id": _record_to_point_id(rec),
        "cluster_id": rec.cluster_id,
        "title": rec.title,
        "doi": rec.doi,
        "year": rec.year,
        "venue": rec.venue,
        "source": rec.source.value,
        "language": rec.language,
        "document_type": rec.document_type,
        "has_abstract": rec.abstract is not None,
    }
    for crit in CRITERIA:
        row.update(_criterion_cols(answers.get(crit.id), crit.id))

    row.update({
        "final_decision_primary": final.decision if final else None,
        "final_reason_primary": final.reason if final else None,
        "final_triggers_primary": (",".join(final.triggered_criteria) if final else None),
        "final_decision_aggressive": final_aggr.decision,
        "final_decision_permissive": final_perm.decision,
    })
    return row


def _save_partial(rows: list[dict], out_path: Path) -> None:
    if not rows:
        return
    df = pd.DataFrame(rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)


def _safe_screen(rec: PaperRecord, model: str | None) -> tuple[PaperRecord, dict]:
    try:
        state = screen_record(rec, model=model)
        return (rec, state)
    except Exception as e:
        log.warning("screen_record failed for %r: %s", rec.title[:60], e)
        return (rec, {
            "answers": {},
            "final": FinalDecision(
                decision=HUMAN_REVIEW,
                reason=f"graph error: {type(e).__name__}: {e}",
                triggered_criteria=[],
            ),
        })


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Cribado PRISMA basado en criterios binarios (I1-I4, E1-E5, D1-D3) con LangGraph"
    )
    parser.add_argument("parquet", type=Path,
                        help="Parquet canónico (típicamente *_final_primary.parquet)")
    parser.add_argument("--workers", type=int, default=2,
                        help="Trabajadores concurrentes por records (default: 2). "
                             "Cada record dispara 9 llamadas LLM en paralelo internamente.")
    parser.add_argument("--model", default=None,
                        help="Override OLLAMA_LLM_MODEL")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limitar a los primeros N registros (para tests/calibración)")
    parser.add_argument("--out", type=Path, default=None,
                        help="Parquet de salida (default: <input>_screened.parquet)")
    parser.add_argument("--checkpoint-every", type=int, default=25)
    parser.add_argument("--resume", action="store_true",
                        help="Saltar record_id ya presentes en el parquet de salida")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    if not args.parquet.exists():
        raise SystemExit(f"Parquet no encontrado: {args.parquet}")

    out_path = args.out or args.parquet.with_name(
        args.parquet.stem.replace("_final_primary", "") + "_screened.parquet"
    )

    log.info("Leyendo %s", args.parquet)
    records = _parquet_to_records(args.parquet)
    log.info("  %d registros cargados", len(records))

    already_done: set[str] = set()
    existing_rows: list[dict] = []
    if args.resume and out_path.exists():
        ex_df = pd.read_parquet(out_path)
        already_done = set(ex_df["record_id"].astype(str).tolist())
        existing_rows = ex_df.to_dict(orient="records")
        log.info("  --resume: %d ya cribados en %s", len(already_done), out_path)

    pending = [r for r in records if _record_to_point_id(r) not in already_done]
    if args.limit is not None:
        pending = pending[: args.limit]
        log.info("  --limit: procesando solo %d registros", len(pending))

    log.info("Pendientes: %d (workers=%d, ~9 LLM calls/record paralelas)",
             len(pending), args.workers)

    rows: list[dict] = list(existing_rows)
    completed = 0
    last_checkpoint = 0
    t0 = time.time()

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(_safe_screen, r, args.model): r for r in pending}
        for fut in as_completed(futures):
            rec, state = fut.result()
            rows.append(_row_for(rec, state))
            completed += 1
            if completed % 5 == 0 or completed == len(pending):
                elapsed = time.time() - t0
                rate = completed / max(elapsed, 1e-6)
                eta = (len(pending) - completed) / max(rate, 1e-6)
                log.info("  cribados %d / %d (%.2f rec/s, eta %.0fs)",
                         completed, len(pending), rate, eta)
            if (completed - last_checkpoint) >= args.checkpoint_every:
                _save_partial(rows, out_path)
                last_checkpoint = completed

    _save_partial(rows, out_path)
    log.info("Escritos %d registros cribados → %s", len(rows), out_path)

    summary = {
        "total_screened": len(rows),
        "by_decision_primary": dict(Counter(r.get("final_decision_primary") for r in rows)),
        "by_decision_aggressive": dict(Counter(r.get("final_decision_aggressive") for r in rows)),
        "by_decision_permissive": dict(Counter(r.get("final_decision_permissive") for r in rows)),
        "elapsed_seconds": round(time.time() - t0, 1),
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
