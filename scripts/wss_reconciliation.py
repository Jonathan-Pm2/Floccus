"""Reconcile the two WSS estimates of the sensitivity operating mode.

WSS (work saved over sampling) is

    WSS = (TN + FN) / N - (1 - recall) = r - (1 - R),

with r the share of the corpus removed from manual review and R the recall.
The manuscript's WSS values are fully re-weighted: both r and R are
stratum-weighted estimates from the blind sample (compute_recall_wss.metrics),

    r_hat = sum_c (N_c / n_c) * #{sampled records of c not advanced} / N.

The sampling strata are the specificity-mode decisions (source x decision), so
in the specificity mode every record of a stratum shares one decision and r_hat
equals the exact share removed. In the sensitivity mode decisions change inside
the strata, r_hat is only an estimate of the exact share (1 - 7,477 / 17,063),
and the two WSS values differ by r_hat - r_exact. This script reports both,
for every reference of the manuscript, with the stratified-bootstrap interval of
the exact-workload WSS (r_exact is fixed, so the interval is that of R shifted).

Run:  .venv/bin/python scripts/wss_reconciliation.py   (after second_reviewer_agreement.py)
Out:  data/processed/wss_reconciliation.json
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "processed"
REFERENCES = {"reviewer1": "First reviewer", "reviewer2_binary": "Second reviewer",
              "consensus_both": "Consensus", "union_either": "Union"}
LOAD_KEY = {"specificity": "specificity_mode_human_load", "sensitivity": "sensitivity_mode_human_load"}


def wss(removed: float, recall: float) -> float:
    return removed - (1.0 - recall)


def reconcile(rw: dict, ci: dict, n_advanced: int, n: int) -> dict:
    """Fully re-weighted WSS vs WSS with the exact corpus workload and re-weighted recall."""
    removed_exact = 1.0 - n_advanced / n
    exact = wss(removed_exact, rw["recall"])
    return {
        "recall_reweighted": rw["recall"],
        "share_advanced_reweighted": 1.0 - rw["burden_removed"],
        "share_advanced_exact": n_advanced / n,
        "wss_reweighted": rw["wss"],
        "wss_exact_workload": exact,
        "wss_exact_workload_ci95": [wss(removed_exact, ci["recall"][0]), wss(removed_exact, ci["recall"][1])],
        "difference_exact_minus_reweighted": exact - rw["wss"],
        "identity_check": abs((exact - rw["wss"]) - (removed_exact - rw["burden_removed"])) < 1e-12,
    }


def main() -> None:
    modes = json.loads((DATA / "operating_modes.json").read_text())
    agree = json.loads((DATA / "second_reviewer_agreement.json").read_text())
    n = modes["n"]
    out = {"n_population": n,
           "records_advanced_exact": {m: modes[k] for m, k in LOAD_KEY.items()},
           "formula": "WSS = (TN+FN)/N - (1 - recall); manuscript: both terms re-weighted by stratum",
           "references": {}}
    for ref, label in REFERENCES.items():
        block = agree["pipeline_vs_reference"][ref]
        out["references"][label] = {
            mode: reconcile(block[mode]["reweighted"], block[mode]["reweighted_ci95"], modes[LOAD_KEY[mode]], n)
            for mode in LOAD_KEY}
    (DATA / "wss_reconciliation.json").write_text(json.dumps(out, indent=2))
    print(f"{'reference':<17}{'mode':<13}{'recall':>7}{'adv rw':>8}{'adv ex':>8}{'WSS rw':>8}{'WSS ex':>8}")
    for label, block in out["references"].items():
        for mode, r in block.items():
            print(f"{label:<17}{mode:<13}{r['recall_reweighted']:>7.3f}{r['share_advanced_reweighted']:>8.3f}"
                  f"{r['share_advanced_exact']:>8.3f}{r['wss_reweighted']:>8.3f}{r['wss_exact_workload']:>8.3f}")


if __name__ == "__main__":
    main()
