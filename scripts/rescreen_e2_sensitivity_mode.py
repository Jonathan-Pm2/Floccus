"""Re-ask E2 with the answer-last, positively phrased prompt on every record
whose final decision depends on E2, to obtain exact corpus counts for the
sensitivity operating mode (reviewer 2 revision, 2026-10-24).

Scope (decision depends on E2):
  (a) records excluded by E2 alone (no other exclusion criterion fired) —
      they move to HUMAN_REVIEW if the corrected E2 is not YES;
  (b) records currently advancing (INCLUDE or HUMAN_REVIEW) — they move to
      EXCLUDE if the corrected E2 is YES.
E4 keeps its original answer (the positive E4 was discarded in the pilot).

Resumable: answers are checkpointed to data/processed/e2_v2_answers.parquet
every 200 calls; a restart skips records already answered.

Run:  set -a; . ../.env; set +a
      .venv/bin/python scripts/rescreen_e2_sensitivity_mode.py --workers 4
Out:  data/processed/e2_v2_answers.parquet
      data/processed/operating_modes.json
"""
from __future__ import annotations

import argparse
import json
import sys
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pilot_prompt_v2 import ask  # same prompt and parsing as the validated pilot

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "processed"
CKPT = DATA / "e2_v2_answers.parquet"


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    dec = pd.read_parquet(DATA / "biofloc_final_decisions.parquet")
    # The decision table carries no abstract; rebuild record_id exactly as the
    # pipeline does (uuid5 of source + source_id, else normalised title) from
    # the screening input. Coverage verified at 100%.
    src = pd.read_parquet(DATA / "biofloc_final_primary_v3.parquet",
                          columns=["source", "source_id", "title_norm", "abstract", "keywords"])
    src["record_id"] = [str(uuid.uuid5(uuid.NAMESPACE_URL, f"{s}:{sid or (tn or '')[:80]}"))
                        for s, sid, tn in zip(src.source, src.source_id, src.title_norm)]
    src["keywords"] = src.keywords.apply(lambda k: "; ".join(k) if k is not None and len(k) else "")
    text = src[["record_id", "abstract", "keywords"]].drop_duplicates("record_id")
    dec = dec.merge(text, on="record_id", how="left")
    other = dec[[f"{c}_answer" for c in ("E1", "E3", "E4", "E5")]].eq("YES").any(axis=1)
    only_e2 = (dec.E2_answer == "YES") & ~other
    advancing = dec.final_decision.isin(["INCLUDE", "HUMAN_REVIEW"])
    todo = dec[only_e2 | advancing].copy()

    done = pd.read_parquet(CKPT) if CKPT.exists() else pd.DataFrame(columns=["record_id", "E2_v2", "raw"])
    pending = todo[~todo.record_id.isin(done.record_id)]
    print(f"scope {len(todo)} (rescuable {only_e2.sum()}, advancing {advancing.sum()}) · done {len(done)} · pending {len(pending)}", flush=True)

    rows = []
    with ThreadPoolExecutor(args.workers) as ex:
        futs = {ex.submit(ask, r, "E2"): r.record_id for r in pending.itertuples()}
        for i, f in enumerate(as_completed(futs), 1):
            try:
                ans, raw = f.result()
            except Exception as e:  # keep going; failed ids are retried on restart
                print(f"  fail {futs[f]}: {e}", flush=True); continue
            rows.append({"record_id": futs[f], "E2_v2": ans, "raw": raw})
            if i % 200 == 0:
                done = pd.concat([done, pd.DataFrame(rows)], ignore_index=True); rows = []
                done.to_parquet(CKPT, index=False)
                print(f"  {len(done)}/{len(todo)}", flush=True)
    if rows:
        done = pd.concat([done, pd.DataFrame(rows)], ignore_index=True); done.to_parquet(CKPT, index=False)

    # ---- sensitivity-mode decisions ----
    m = dec.merge(done[["record_id", "E2_v2"]], on="record_id", how="left")
    sens = m.final_decision.copy()
    rescued = only_e2 & m.E2_v2.notna() & (m.E2_v2 != "YES")
    newly_excl = advancing & (m.E2_v2 == "YES")
    sens[rescued] = "HUMAN_REVIEW"
    sens[newly_excl] = "EXCLUDE"
    res = {"n": int(len(m)), "scope": int(len(todo)), "answered": int(m.E2_v2.notna().sum()),
           "specificity_mode": m.final_decision.value_counts().to_dict(),
           "sensitivity_mode": sens.value_counts().to_dict(),
           "rescued_to_human_review": int(rescued.sum()), "newly_excluded": int(newly_excl.sum())}
    for k in ("specificity_mode", "sensitivity_mode"):
        c = res[k]; res[k + "_human_load"] = int(c.get("INCLUDE", 0) + c.get("HUMAN_REVIEW", 0))
    (DATA / "operating_modes.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
