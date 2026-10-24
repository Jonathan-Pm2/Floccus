"""Recall, specificity and workload of the two screening operating modes.

The sensitivity mode re-asks criterion E2 in positive form (see
scripts/rescreen_e2_sensitivity_mode.py, which stored one corrected E2
answer per decision-dependent record). Here both modes are scored against
the blind human labels with the stratum reweighting and stratified
bootstrap of compute_recall_wss.py. The sampling strata are the design
cells (source x specificity-mode decision) and do not change; only each
record's decision changes in the sensitivity mode.

Caveat reported with the results: the sensitivity mode was designed after
inspecting errors on this same sample, so its recall here is optimistic.

Run:  .venv/bin/python scripts/operating_modes_recall.py
Out:  data/processed/operating_modes_recall.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from compute_recall_wss import ADVANCE, DATA, bootstrap, load, metrics  # noqa: E402

EXCLUSIONS = ("E1", "E3", "E4", "E5")


def sensitivity_decision(d: pd.DataFrame) -> pd.Series:
    """Same rule as rescreen_e2_sensitivity_mode.py: records excluded by E2 alone move to
    HUMAN_REVIEW unless the corrected E2 is YES; advancing records whose corrected E2 is YES
    move to EXCLUDE."""
    other = d[[f"{c}_answer" for c in EXCLUSIONS]].eq("YES").any(axis=1)
    only_e2 = (d.E2_answer == "YES") & ~other
    advancing = d.final_decision.isin(["INCLUDE", "HUMAN_REVIEW"])
    out = d.final_decision.copy()
    out[only_e2 & d.E2_v2.notna() & (d.E2_v2 != "YES")] = "HUMAN_REVIEW"
    out[advancing & (d.E2_v2 == "YES")] = "EXCLUDE"
    return out


def main() -> None:
    df, n_pop = load()
    truth = pd.read_parquet(DATA / "validation_sample_truth.parquet", columns=["review_id", "record_id"])
    dec = pd.read_parquet(DATA / "biofloc_final_decisions.parquet",
                          columns=["record_id"] + [f"{c}_answer" for c in ("E1", "E2", "E3", "E4", "E5")])
    e2 = pd.read_parquet(DATA / "e2_v2_answers.parquet", columns=["record_id", "E2_v2"])
    s = (df.merge(truth, on="review_id").merge(dec, on="record_id", how="left")
           .merge(e2, on="record_id", how="left"))
    s.index = df.index
    sens = df.copy()
    sens["final_decision"] = sensitivity_decision(s).to_numpy()

    modes = json.loads((DATA / "operating_modes.json").read_text())
    out = {"n_sample": int(len(df)), "n_population": n_pop,
           "caveat": "sensitivity mode designed on this sample: its recall is an optimistic upper bound"}
    for name, frame, load_key in (("specificity", df, "specificity_mode_human_load"),
                                  ("sensitivity", sens, "sensitivity_mode_human_load")):
        out[name] = {"reweighted": metrics(frame, frame["w"].to_numpy(), ADVANCE),
                     "reweighted_ci95": bootstrap(frame, ADVANCE),
                     "raw": metrics(frame, np.ones(len(frame)), ADVANCE),
                     "human_load_corpus": modes[load_key]}
    out["changed_in_sample"] = int((sens.final_decision != df.final_decision).sum())
    (DATA / "operating_modes_recall.json").write_text(json.dumps(out, indent=2))
    for name in ("specificity", "sensitivity"):
        r, ci = out[name]["reweighted"], out[name]["reweighted_ci95"]
        print(f"{name:12} recall {r['recall']:.3f} [{ci['recall'][0]:.3f}-{ci['recall'][1]:.3f}]  "
              f"spec {r['specificity']:.3f} [{ci['specificity'][0]:.3f}-{ci['specificity'][1]:.3f}]  "
              f"WSS {r['wss']:.3f}  raw recall {out[name]['raw']['recall']:.3f}  "
              f"load {out[name]['human_load_corpus']:,}")
    print("records whose decision changes in the sample:", out["changed_in_sample"])


if __name__ == "__main__":
    main()
