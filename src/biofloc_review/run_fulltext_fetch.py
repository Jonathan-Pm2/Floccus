"""Layer 3 — Fulltext PDF fetching for the eligibility phase.

Reads ``biofloc_enriched.parquet`` (joined with consolidated decisions) and
attempts to download the fulltext PDF for every record that advanced past
Layer 2 (final decision ∈ {INCLUDE, HUMAN_REVIEW}). The fetcher tries three
tiers in order: OpenAlex OA PDF → Europe PMC fulltext URL → DOI via USAL
EZproxy.

Output:
  - ``data/fulltext/{record_id}.pdf`` for each successful download.
  - ``data/processed/fulltext_log.parquet`` with one row per record,
    capturing status (``ok`` / ``manual`` / ``error``), strategy, attempts,
    http codes, and bytes downloaded. Resumable via ``--resume``.

Authentication for Tier 3 (EZproxy) is **not automated**. The user must
log in via their browser to the USAL EZproxy, export the session cookies
to a Netscape-format ``cookies.txt`` file (e.g. with the "Get cookies.txt
LOCALLY" extension), and pass that file via ``--cookies``. Session
typically lasts 24 hours; on expiry, re-export and re-run with ``--resume``.

Usage:
  uv run python -m biofloc_review.run_fulltext_fetch \\
    data/processed/biofloc_enriched.parquet \\
    data/processed/biofloc_final_decisions.parquet \\
    --cookies /path/to/cookies.txt \\
    --workers 2 --delay 1.5
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

from biofloc_review.screening.fulltext_fetcher import (
    FetchResult,
    fetch_fulltext,
)
from biofloc_review.sources.ezproxy import (
    build_session,
    is_session_valid,
)

log = logging.getLogger(__name__)


def _row_for(result: FetchResult) -> dict:
    return {
        "record_id": result.record_id,
        "status": result.status,
        "strategy": result.strategy,
        "url_attempted": result.url_attempted,
        "http_status": result.http_status,
        "pdf_path": str(result.pdf_path) if result.pdf_path else None,
        "bytes_downloaded": result.bytes_downloaded,
        "error": result.error,
        "attempts_json": json.dumps(result.attempts, ensure_ascii=False),
    }


def _save_partial(rows: list[dict], out_path: Path) -> None:
    if not rows:
        return
    df = pd.DataFrame(rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)


def _coerce_str(v) -> str | None:
    if v is None:
        return None
    s = str(v).strip()
    if not s or s.lower() in {"nan", "none", "null"}:
        return None
    return s


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Layer 3: fetch fulltext PDFs for eligibility-bound records"
    )
    parser.add_argument(
        "enriched_parquet", type=Path,
        help="Layer 2 enricher output (biofloc_enriched.parquet)",
    )
    parser.add_argument(
        "decisions_parquet", type=Path,
        help="Consolidated decisions parquet (biofloc_final_decisions.parquet)",
    )
    parser.add_argument(
        "--cookies", type=Path,
        default=None,
        help="Netscape-format cookies.txt for EZproxy session "
             "(skip to disable Tier 3 / EZproxy attempts)",
    )
    parser.add_argument(
        "--out-dir", type=Path,
        default=Path("data/fulltext"),
        help="Directory where PDFs are saved (default: data/fulltext)",
    )
    parser.add_argument(
        "--log-parquet", type=Path,
        default=Path("data/processed/fulltext_log.parquet"),
    )
    parser.add_argument(
        "--decisions", nargs="+",
        default=["INCLUDE", "HUMAN_REVIEW"],
        help="Final decisions whose records are fetched (default: INCLUDE HUMAN_REVIEW)",
    )
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--delay", type=float, default=1.0,
                        help="Seconds between strategy attempts per record")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit to first N records (testing)")
    parser.add_argument("--resume", action="store_true",
                        help="Skip record_ids already in the log parquet")
    parser.add_argument("--checkpoint-every", type=int, default=25)
    parser.add_argument("--skip-session-check", action="store_true",
                        help="Skip the EZproxy session liveness check")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    log.info("Leyendo decisiones consolidadas: %s", args.decisions_parquet)
    decisions = pd.read_parquet(args.decisions_parquet)
    advancing_mask = decisions.final_decision.isin(args.decisions)
    advancing_rids = set(decisions[advancing_mask].record_id.astype(str))
    log.info("  %d records avanzan a Layer 3 (%s)", len(advancing_rids),
             "+".join(args.decisions))

    log.info("Leyendo enriched parquet: %s", args.enriched_parquet)
    enriched = pd.read_parquet(args.enriched_parquet)
    enriched = enriched[enriched.record_id.astype(str).isin(advancing_rids)]
    log.info("  %d enriched records cubren el subset advancing", len(enriched))

    # Resume support
    already_done: set[str] = set()
    existing_rows: list[dict] = []
    if args.resume and args.log_parquet.exists():
        ex = pd.read_parquet(args.log_parquet)
        already_done = set(ex.record_id.astype(str).tolist())
        existing_rows = ex.to_dict(orient="records")
        log.info("  --resume: %d ya intentados", len(already_done))

    pending = enriched[~enriched.record_id.astype(str).isin(already_done)]
    if args.limit:
        pending = pending.head(args.limit)
        log.info("  --limit: procesando solo %d", len(pending))
    log.info("Pendientes: %d (workers=%d, delay=%.1fs)",
             len(pending), args.workers, args.delay)

    # Build HTTP session
    session = build_session(cookies_file=args.cookies)
    if args.cookies and not args.skip_session_check:
        log.info("Probando session EZproxy...")
        if is_session_valid(session):
            log.info("  ✅ session EZproxy válida")
        else:
            log.warning("  ⚠️  session EZproxy inválida o expirada — Tier 3 puede fallar. "
                        "Re-exportá cookies y reanudá con --resume.")

    # Worker function
    def _do_fetch(row: pd.Series) -> dict:
        rid = str(row.record_id)
        oa = _coerce_str(row.get("oa_pdf_url"))
        pmc = _coerce_str(row.get("pmc_fulltext_url"))
        doi = _coerce_str(row.get("doi"))
        result = fetch_fulltext(
            record_id=rid,
            oa_pdf_url=oa,
            pmc_fulltext_url=pmc,
            doi=doi if args.cookies else None,  # only attempt Tier 3 if cookies provided
            session=session,
            out_dir=args.out_dir,
            delay=args.delay,
        )
        return _row_for(result)

    rows: list[dict] = list(existing_rows)
    completed = 0
    last_checkpoint = 0
    t0 = time.time()

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(_do_fetch, row): row for _, row in pending.iterrows()}
        for fut in as_completed(futures):
            try:
                rows.append(fut.result())
            except Exception as e:
                log.warning("worker error: %s", e)
            completed += 1
            if completed % 10 == 0 or completed == len(pending):
                elapsed = time.time() - t0
                rate = completed / max(elapsed, 1e-6)
                eta = (len(pending) - completed) / max(rate, 1e-6)
                ok_n = sum(1 for r in rows[-completed:] if r.get("status") == "ok")
                log.info("  fetch %d/%d (%.1f rec/s, ok últimos %d, eta %.0fs)",
                         completed, len(pending), rate, ok_n, eta)
            if (completed - last_checkpoint) >= args.checkpoint_every:
                _save_partial(rows, args.log_parquet)
                last_checkpoint = completed

    _save_partial(rows, args.log_parquet)
    log.info("Log guardado → %s", args.log_parquet)

    res = pd.DataFrame(rows)
    by_status = dict(Counter(res.status))
    by_strategy = dict(Counter(res.strategy))
    summary = {
        "total_attempted": int(len(res)),
        "by_status": by_status,
        "by_strategy": by_strategy,
        "ok_count": int((res.status == "ok").sum()),
        "manual_count": int((res.status == "manual").sum()),
        "error_count": int((res.status == "error").sum()),
        "out_dir": str(args.out_dir),
        "elapsed_seconds": round(time.time() - t0, 1),
    }
    print()
    print("=" * 60)
    print(" LAYER 3 — Fulltext Fetcher Summary")
    print("=" * 60)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
