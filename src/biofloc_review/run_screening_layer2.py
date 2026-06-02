"""CLI Layer 2 LLM screening — segundo pase BLIND con input enriquecido.

Filtra `biofloc_screened.parquet` a records con `final_decision_primary` ==
HUMAN_REVIEW (4.225 esperados), une con `biofloc_enriched.parquet` por
record_id, y aplica los 12 criterios con metadata enriquecida.

Output: `biofloc_layer2_screened.parquet` con columnas:
  - record_id, cluster_id, title, doi, source, language, document_type
  - Layer 1 reference: l1_decision, l1_reason
  - Layer 2: l2_<criterion>_answer, l2_<criterion>_confidence,
    l2_<criterion>_evidence, l2_<criterion>_reasoning, l2_<criterion>_via
  - l2_final_decision, l2_final_reason, l2_final_triggers
  - final_decision (consolidado: Layer 1 INCLUDE wins, sino Layer 2 wins)

Usage:
  uv run python -m biofloc_review.run_screening_layer2 \\
    data/processed/biofloc_screened.parquet \\
    data/processed/biofloc_enriched.parquet \\
    --workers 2
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

from biofloc_review.dedup.embeddings import _record_to_point_id
from biofloc_review.schema import PaperRecord
from biofloc_review.screening.classifier import CriterionAnswer
from biofloc_review.screening.criteria import CRITERIA
from biofloc_review.screening.decision import FinalDecision, HUMAN_REVIEW
from biofloc_review.screening.enricher import EnrichedRecord
from biofloc_review.screening.layer2 import screen_record_layer2

log = logging.getLogger(__name__)


def _build_paper_index(full_parquet: Path) -> dict[str, PaperRecord]:
    df = pd.read_parquet(full_parquet)
    out: dict[str, PaperRecord] = {}
    for _, row in df.iterrows():
        d = {k: (None if isinstance(v, float) and pd.isna(v) else v)
             for k, v in row.to_dict().items()}
        try:
            rec = PaperRecord.model_validate(d)
        except Exception:
            continue
        out[_record_to_point_id(rec)] = rec
    return out


def _coerce_list(v) -> list:
    """Safely turn a pandas cell into a list (handles numpy arrays)."""
    if v is None:
        return []
    try:
        if len(v) == 0:
            return []
        return list(v)
    except TypeError:
        return []


def _coerce_str(v) -> str | None:
    if v is None:
        return None
    s = str(v).strip()
    if not s or s.lower() in {"nan", "none", "null"}:
        return None
    return s


def _build_enriched_index(enriched_parquet: Path) -> dict[str, EnrichedRecord]:
    df = pd.read_parquet(enriched_parquet)
    out: dict[str, EnrichedRecord] = {}
    for _, row in df.iterrows():
        rid = str(row["record_id"])
        out[rid] = EnrichedRecord(
            doi=_coerce_str(row.get("doi")),
            oa_concepts=_coerce_list(row.get("oa_concepts")),
            oa_topics=_coerce_list(row.get("oa_topics")),
            oa_keywords=_coerce_list(row.get("oa_keywords")),
            oa_referenced_works_count=int(row.get("oa_referenced_works_count") or 0),
            oa_cited_by_count=int(row.get("oa_cited_by_count") or 0),
            oa_abstract_full=_coerce_str(row.get("oa_abstract_full")),
            oa_is_oa=bool(row.get("oa_is_oa")) if pd.notna(row.get("oa_is_oa")) else None,
            oa_url=_coerce_str(row.get("oa_url")),
            oa_pdf_url=_coerce_str(row.get("oa_pdf_url")),
            oa_publication_date=_coerce_str(row.get("oa_publication_date")),
            oa_ok=bool(row.get("oa_ok", False)),
            pmc_id=_coerce_str(row.get("pmc_id")),
            pmc_pmid=_coerce_str(row.get("pmc_pmid")),
            pmc_mesh_terms=_coerce_list(row.get("pmc_mesh_terms")),
            pmc_has_fulltext=bool(row.get("pmc_has_fulltext", False)),
            pmc_fulltext_url=_coerce_str(row.get("pmc_fulltext_url")),
            pmc_is_oa=bool(row.get("pmc_is_oa")) if pd.notna(row.get("pmc_is_oa")) else None,
            pmc_ok=bool(row.get("pmc_ok", False)),
        )
    return out


def _criterion_cols(answer: CriterionAnswer | None, cid: str) -> dict:
    if answer is None:
        return {f"l2_{cid}_{k}": None for k in ("answer", "confidence", "evidence", "reasoning", "via")}
    return {
        f"l2_{cid}_answer": answer.answer,
        f"l2_{cid}_confidence": answer.confidence,
        f"l2_{cid}_evidence": answer.evidence,
        f"l2_{cid}_reasoning": answer.reasoning,
        f"l2_{cid}_via": answer.via,
    }


def _row_for(rid: str, paper: PaperRecord, l1_row: dict, state: dict) -> dict:
    answers: dict[str, CriterionAnswer] = state.get("answers") or {}
    final: FinalDecision = state.get("final")

    row = {
        "record_id": rid,
        "cluster_id": paper.cluster_id,
        "title": paper.title,
        "doi": paper.doi,
        "source": paper.source.value,
        "language": paper.language,
        "document_type": paper.document_type,
        # Layer 1 reference
        "l1_decision": l1_row.get("final_decision_primary"),
        "l1_reason": l1_row.get("final_reason_primary"),
    }
    for crit in CRITERIA:
        row.update(_criterion_cols(answers.get(crit.id), crit.id))
    row.update({
        "l2_final_decision": final.decision if final else None,
        "l2_final_reason": final.reason if final else None,
        "l2_final_triggers": (",".join(final.triggered_criteria) if final else None),
        # Consolidated decision: Layer 1 INCLUDE wins (no se re-evalúan).
        # En este CLI solo procesamos HUMAN_REVIEW, así que final = L2.
        "final_decision": final.decision if final else l1_row.get("final_decision_primary"),
    })
    return row


def _save_partial(rows: list[dict], out_path: Path) -> None:
    if not rows:
        return
    df = pd.DataFrame(rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)


def _safe_screen(paper: PaperRecord, enriched: EnrichedRecord | None,
                  model: str | None) -> tuple[PaperRecord, dict]:
    try:
        state = screen_record_layer2(paper, enriched, model=model)
        return paper, state
    except Exception as e:
        log.warning("screen_record_layer2 failed for %r: %s", paper.title[:60], e)
        return paper, {
            "answers": {},
            "final": FinalDecision(
                decision=HUMAN_REVIEW,
                reason=f"layer2 error: {type(e).__name__}: {e}",
                triggered_criteria=[],
            ),
        }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Layer 2 LLM screening con metadata enriquecida sobre HUMAN_REVIEW"
    )
    parser.add_argument("screened_parquet", type=Path,
                        help="Parquet de Layer 1 (biofloc_screened.parquet)")
    parser.add_argument("enriched_parquet", type=Path,
                        help="Parquet de Layer 2 enricher (biofloc_enriched.parquet)")
    parser.add_argument("--full-parquet", type=Path,
                        default=Path("data/processed/biofloc_final_primary_v2.parquet"),
                        help="Parquet canónico (default: v2 con SciELO scraped)")
    parser.add_argument("--out", type=Path,
                        default=Path("data/processed/biofloc_layer2_screened.parquet"))
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--model", default=None,
                        help="Override OLLAMA_LLM_MODEL")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limitar a N records (testing)")
    parser.add_argument("--resume", action="store_true",
                        help="Saltar record_id ya presentes en el output")
    parser.add_argument("--checkpoint-every", type=int, default=25)
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    log.info("Leyendo Layer 1 screened %s", args.screened_parquet)
    screened = pd.read_parquet(args.screened_parquet)
    hr = screened[screened.final_decision_primary == "HUMAN_REVIEW"]
    log.info("  %d records HUMAN_REVIEW (a re-screenar)", len(hr))

    log.info("Indexando full parquet %s", args.full_parquet)
    paper_by_rid = _build_paper_index(args.full_parquet)
    log.info("  %d papers indexados", len(paper_by_rid))

    log.info("Indexando enriched parquet %s", args.enriched_parquet)
    enriched_by_rid = _build_enriched_index(args.enriched_parquet)
    log.info("  %d enriched indexados", len(enriched_by_rid))

    target_rids = list(hr.record_id.astype(str))
    l1_by_rid = {str(r): r.to_dict() for _, r in hr.iterrows()}
    l1_by_rid = {row["record_id"]: row for row in hr.to_dict(orient="records")}

    already_done: set[str] = set()
    existing_rows: list[dict] = []
    if args.resume and args.out.exists():
        ex = pd.read_parquet(args.out)
        already_done = set(ex.record_id.astype(str).tolist())
        existing_rows = ex.to_dict(orient="records")
        log.info("  --resume: %d ya screenead. Saltando.", len(already_done))

    pending_rids = [rid for rid in target_rids if rid not in already_done]
    if args.limit:
        pending_rids = pending_rids[: args.limit]
    log.info("Pendientes: %d (workers=%d)", len(pending_rids), args.workers)

    def _task(rid: str) -> tuple[str, dict, dict]:
        paper = paper_by_rid.get(rid)
        enriched = enriched_by_rid.get(rid)
        l1_row = l1_by_rid.get(rid, {})
        if not paper:
            log.warning("paper not found for record_id %s", rid)
            return rid, l1_row, {
                "answers": {},
                "final": FinalDecision(
                    decision=HUMAN_REVIEW,
                    reason="paper not in canonical parquet",
                    triggered_criteria=[],
                ),
            }
        _, state = _safe_screen(paper, enriched, args.model)
        return rid, l1_row, state

    rows: list[dict] = list(existing_rows)
    completed = 0
    last_checkpoint = 0
    t0 = time.time()

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(_task, rid): rid for rid in pending_rids}
        for fut in as_completed(futures):
            rid, l1_row, state = fut.result()
            paper = paper_by_rid.get(rid)
            if paper:
                rows.append(_row_for(rid, paper, l1_row, state))
            completed += 1
            if completed % 5 == 0 or completed == len(pending_rids):
                elapsed = time.time() - t0
                rate = completed / max(elapsed, 1e-6)
                eta = (len(pending_rids) - completed) / max(rate, 1e-6)
                log.info("  L2 %d/%d (%.2f rec/s, eta %.0fs)",
                         completed, len(pending_rids), rate, eta)
            if (completed - last_checkpoint) >= args.checkpoint_every:
                _save_partial(rows, args.out)
                last_checkpoint = completed

    _save_partial(rows, args.out)
    log.info("Escritos %d records L2 → %s", len(rows), args.out)

    res = pd.DataFrame(rows)
    from collections import Counter
    summary = {
        "total_l2_screened": len(res),
        "by_final_decision": dict(Counter(res.final_decision.fillna("(missing)"))),
        "elapsed_seconds": round(time.time() - t0, 1),
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
