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
from biofloc_review.dedup.llm_judge import judge_pair
from biofloc_review.schema import PaperRecord

log = logging.getLogger(__name__)


def _parquet_to_records(path: Path) -> list[PaperRecord]:
    df = pd.read_parquet(path)
    out: list[PaperRecord] = []
    for _, row in df.iterrows():
        d = {k: (None if isinstance(v, float) and pd.isna(v) else v) for k, v in row.to_dict().items()}
        out.append(PaperRecord.model_validate(d))
    return out


def _build_id_index(records: list[PaperRecord]) -> dict[str, dict]:
    return {_record_to_point_id(r): r.model_dump(mode="json") for r in records}


def _safe_judge(rec_a: dict | None, rec_b: dict | None, score: float, model: str | None) -> dict:
    if rec_a is None or rec_b is None:
        return {
            "verdict": "UNCLEAR",
            "confidence": 0,
            "reasoning": "ID no resoluble en el parquet (record absent).",
            "raw": "",
        }
    try:
        v = judge_pair(rec_a, rec_b, score, model=model)
        return {"verdict": v.verdict, "confidence": v.confidence, "reasoning": v.reasoning, "raw": v.raw}
    except Exception as e:
        log.warning("judge failed: %s", e)
        return {"verdict": "UNCLEAR", "confidence": 0, "reasoning": f"LLM error: {type(e).__name__}: {e}", "raw": ""}


def main() -> None:
    parser = argparse.ArgumentParser(description="Capa 3: LLM judge over Capa 2 candidate pairs")
    parser.add_argument("pairs", type=Path, help="JSON of candidate pairs from Capa 2")
    parser.add_argument("parquet", type=Path, help="Parquet with PaperRecords (to look up full context)")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--model", default=None, help="Override Ollama LLM model (default: env or qwen3:14b)")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of pairs to judge")
    parser.add_argument("--out", type=Path, default=None, help="Output JSON (default: <pairs>_judged.json)")
    parser.add_argument("--checkpoint-every", type=int, default=25)
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    pairs = json.loads(args.pairs.read_text())
    log.info("Loaded %d candidate pairs", len(pairs))
    if args.limit:
        pairs = pairs[: args.limit]
        log.info("Limiting to first %d pairs", len(pairs))

    log.info("Reading parquet %s", args.parquet)
    records = _parquet_to_records(args.parquet)
    by_id = _build_id_index(records)
    log.info("Indexed %d records by point_id", len(by_id))

    out_path = args.out or args.pairs.with_name(f"{args.pairs.stem}_judged.json")

    judged: list[dict] = []
    completed = 0
    last_checkpoint = 0
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {
            ex.submit(
                _safe_judge,
                by_id.get(p["a_id"]),
                by_id.get(p["b_id"]),
                float(p.get("score", 0.0)),
                args.model,
            ): i
            for i, p in enumerate(pairs)
        }
        results: dict[int, dict] = {}
        for fut in as_completed(futures):
            i = futures[fut]
            verdict_dict = fut.result()
            results[i] = verdict_dict
            completed += 1
            if completed % 25 == 0 or completed == len(pairs):
                elapsed = time.time() - t0
                rate = completed / max(elapsed, 1e-6)
                eta = (len(pairs) - completed) / max(rate, 1e-6)
                log.info("  judged %d / %d (%.1f pairs/s, eta %.0fs)",
                         completed, len(pairs), rate, eta)
            if (completed - last_checkpoint) >= args.checkpoint_every:
                _save_partial(pairs, results, out_path)
                last_checkpoint = completed

    judged = []
    for i, p in enumerate(pairs):
        verdict_dict = results.get(i, {"verdict": "UNCLEAR", "confidence": 0, "reasoning": "not judged", "raw": ""})
        judged.append({**p, **{f"judge_{k}": v for k, v in verdict_dict.items()}})

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(judged, indent=2, ensure_ascii=False))
    log.info("Wrote %d judged pairs → %s", len(judged), out_path)

    by_verdict = Counter(j["judge_verdict"] for j in judged)
    summary = {
        "total_pairs": len(judged),
        "by_verdict": dict(by_verdict),
        "avg_confidence": round(sum(j["judge_confidence"] for j in judged) / max(len(judged), 1), 2),
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def _save_partial(pairs: list[dict], results: dict[int, dict], out_path: Path) -> None:
    judged = []
    for i, p in enumerate(pairs):
        verdict_dict = results.get(i)
        if verdict_dict:
            judged.append({**p, **{f"judge_{k}": v for k, v in verdict_dict.items()}})
        else:
            judged.append({**p, "judge_verdict": "PENDING"})
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(judged, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
