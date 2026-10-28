"""Stratum-reweighted recall, specificity and WSS for the Floccus screening.

Why this script exists
----------------------
The manuscript validates the automated screening with Cohen's kappa, which
measures agreement between two *equivalent* raters. The Floccus screener is
not an equivalent rater: it is a deliberately conservative pre-filter whose
design goal is to remove obvious irrelevance while routing anything uncertain
to a human. The metrics that judge that design are the ones used in the
systematic-review automation literature: recall (sensitivity) and Work Saved
over Sampling (WSS).

    WSS = (TN + FN) / N - (1 - recall)

Both are computed here from the same blind validation sample already reported
in the manuscript, so no new labelling is involved.

Reweighting
-----------
The validation sample is NOT a simple random sample of the corpus: it was
drawn stratified by (source x final_decision), targeting 42 records per cell.
Rare cells are therefore over-represented and the raw sample rates are not
corpus-level estimates. Each sampled record is weighted by

    w_c = N_c / n_c

with N_c the population size of stratum c (from the full decision table) and
n_c the number of sampled records in that stratum. Every quantity below is
reported both raw (sample) and reweighted (corpus estimate).

Confidence intervals come from a stratified bootstrap: records are resampled
with replacement *within* each stratum, preserving the design.

Run:  .venv/bin/python scripts/compute_recall_wss.py
Out:  data/processed/recall_wss_reweighted.json
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "processed"

SEED = 20260603          # same seed used to draw the validation sample
N_BOOT = 10_000
ADVANCE = {"INCLUDE", "HUMAN_REVIEW"}    # Mapping A (PRISMA-permissive)
ADVANCE_STRICT = {"INCLUDE"}             # Mapping B (strict)


def load(population: pd.DataFrame | None = None) -> pd.DataFrame:
    """Blind sample with stratum weights w_c = N_c / n_c.

    population: the corpus the weights refer to (columns source, final_decision);
    defaults to the published decision table. The strata are always the
    sampling-design cells (source x specificity-mode decision) of the sample.
    """
    blind = pd.read_csv(DATA / "validation_sample_blind.csv")
    truth = pd.read_parquet(DATA / "validation_sample_truth.parquet")
    pop = (pd.read_parquet(DATA / "biofloc_final_decisions.parquet") if population is None
           else population[["source", "final_decision"]].copy())

    df = blind[["review_id", "human_decision"]].merge(
        truth[["review_id", "source", "final_decision"]], on="review_id", how="inner"
    )
    if len(df) != len(blind):
        raise SystemExit(f"join lost rows: {len(blind)} -> {len(df)}")

    df["relevant"] = df["human_decision"].str.strip().str.lower().eq("cumple")

    # Stratum = the sampling design cell.
    df["stratum"] = df["source"] + "|" + df["final_decision"]
    pop["stratum"] = pop["source"] + "|" + pop["final_decision"]

    n_c = df["stratum"].value_counts()
    N_c = pop["stratum"].value_counts()

    missing = set(n_c.index) - set(N_c.index)
    if missing:
        raise SystemExit(f"strata sampled but absent from the population: {missing}")

    df["w"] = df["stratum"].map(N_c / n_c)
    return df, int(len(pop))


def metrics(df: pd.DataFrame, w: np.ndarray, advance: set[str]) -> dict:
    rel = df["relevant"].to_numpy()
    adv = df["final_decision"].isin(advance).to_numpy()
    exc = ~adv

    W = w.sum()
    W_rel = w[rel].sum()
    W_irr = w[~rel].sum()

    recall = w[rel & adv].sum() / W_rel
    specificity = w[~rel & exc].sum() / W_irr
    removed = w[exc].sum() / W                      # screening burden removed
    wss = removed - (1.0 - recall)
    precision = w[rel & adv].sum() / w[adv].sum() if w[adv].sum() else float("nan")

    return {
        "prevalence": W_rel / W,
        "recall": recall,
        "specificity": specificity,
        "precision_advanced": precision,
        "burden_removed": removed,
        "wss": wss,
    }


def bootstrap(df: pd.DataFrame, advance: set[str]) -> dict:
    rng = np.random.default_rng(SEED)
    idx_by_stratum = [g.index.to_numpy() for _, g in df.groupby("stratum", sort=True)]
    base_w = df["w"].to_numpy()

    acc: dict[str, list[float]] = {}
    for _ in range(N_BOOT):
        pick = np.concatenate([rng.choice(ix, size=ix.size, replace=True)
                               for ix in idx_by_stratum])
        pos = df.index.get_indexer(pick)
        sub = df.iloc[pos]
        m = metrics(sub, base_w[pos], advance)
        for k, v in m.items():
            acc.setdefault(k, []).append(v)

    return {k: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
            for k, v in acc.items()}


def main() -> None:
    df, n_pop = load()
    ones = np.ones(len(df))

    out = {
        "n_sample": int(len(df)),
        "n_population": n_pop,
        "n_relevant_sample": int(df["relevant"].sum()),
        "n_strata": int(df["stratum"].nunique()),
        "seed": SEED,
        "n_bootstrap": N_BOOT,
    }

    for name, adv in (("mapping_A_permissive", ADVANCE),
                      ("mapping_B_strict", ADVANCE_STRICT)):
        out[name] = {
            "raw": metrics(df, ones, adv),
            "reweighted": metrics(df, df["w"].to_numpy(), adv),
            "reweighted_ci95": bootstrap(df, adv),
        }

    dest = DATA / "recall_wss_reweighted.json"
    dest.write_text(json.dumps(out, indent=2))

    print(f"sample n={out['n_sample']}  population N={n_pop:,}  "
          f"strata={out['n_strata']}  relevant(human)={out['n_relevant_sample']}\n")
    for name in ("mapping_A_permissive", "mapping_B_strict"):
        r, rw, ci = (out[name]["raw"], out[name]["reweighted"],
                     out[name]["reweighted_ci95"])
        print(f"--- {name} ---")
        print(f"{'metric':<20}{'raw':>9}{'reweighted':>13}   95% CI")
        for k in ("prevalence", "recall", "specificity", "precision_advanced",
                  "burden_removed", "wss"):
            lo, hi = ci[k]
            print(f"{k:<20}{r[k]:>9.3f}{rw[k]:>13.3f}   [{lo:.3f}, {hi:.3f}]")
        print()
    print("->", dest)


if __name__ == "__main__":
    main()
