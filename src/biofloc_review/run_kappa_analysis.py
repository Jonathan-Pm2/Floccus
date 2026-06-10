"""Compute Cohen's kappa between human reviewer and LLM agents.

Reads the human-completed CSV (`validation_sample_blind.csv` with
`human_decision` and optionally `human_I1..human_D3` filled in) and the
LLM truth parquet (`validation_sample_truth.parquet`). Joins by
`review_id` and computes:

  - Overall κ for the final decision (INCLUDE / EXCLUDE / HUMAN_REVIEW)
  - κ stratified by source (4 strata)
  - Per-criterion κ on the 12 binary criteria (YES / NO / UNCLEAR),
    skipping criteria the human left blank

Reports point estimates plus 95% bootstrap confidence intervals (1000
resamples). Output: a JSON file with all metrics and a stdout summary
table.

Usage:
  uv run python -m biofloc_review.run_kappa_analysis \\
    data/processed/validation_sample_blind.csv \\
    data/processed/validation_sample_truth.parquet
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from biofloc_review.screening.criteria import CRITERIA

log = logging.getLogger(__name__)


# Live criteria order — drops a removed criterion automatically.
CRITERIA_ORDER = tuple(c.id for c in CRITERIA)

DECISION_VALUES = ("cumple", "no cumple")
ANSWER_VALUES = ("YES", "NO", "UNCLEAR")


def _llm_to_binary(llm_decision: str) -> str:
    """Map the LLM's 3-class decision to the human's binary scale.

    INCLUDE and HUMAN_REVIEW both map to "cumple" because the PRISMA flow
    advances both to the next phase (eligibility-by-fulltext); EXCLUDE is
    the only verdict that stops a record. This collapse aligns the two
    raters on the same categorical scale so Cohen's kappa is well-defined.
    """
    dec = (llm_decision or "").strip().upper()
    if dec in ("INCLUDE", "HUMAN_REVIEW"):
        return "cumple"
    if dec == "EXCLUDE":
        return "no cumple"
    return ""  # unknown; will be filtered out


def cohen_kappa(rater_a: np.ndarray, rater_b: np.ndarray,
                 categories: tuple[str, ...]) -> float:
    """Compute Cohen's unweighted kappa for two categorical raters.

    Categories with zero presence in both raters contribute zero
    expected agreement. Returns 0.0 for trivially equal raters with one
    category present, and `nan` if the data is empty.
    """
    if len(rater_a) == 0:
        return float("nan")
    n = len(rater_a)
    p_o = float((rater_a == rater_b).sum()) / n
    p_e = 0.0
    for cat in categories:
        p_a = float((rater_a == cat).sum()) / n
        p_b = float((rater_b == cat).sum()) / n
        p_e += p_a * p_b
    if p_e >= 1.0:
        return 0.0
    return (p_o - p_e) / (1.0 - p_e)


def bootstrap_kappa_ci(rater_a: np.ndarray, rater_b: np.ndarray,
                        categories: tuple[str, ...],
                        n_resamples: int = 1000,
                        seed: int = 20260603) -> tuple[float, float, float]:
    """Return (point estimate, lower 2.5%, upper 97.5%) via bootstrap."""
    n = len(rater_a)
    if n == 0:
        return (float("nan"), float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    point = cohen_kappa(rater_a, rater_b, categories)
    samples = []
    for _ in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        try:
            k = cohen_kappa(rater_a[idx], rater_b[idx], categories)
        except Exception:
            continue
        if not np.isnan(k):
            samples.append(k)
    if not samples:
        return (point, float("nan"), float("nan"))
    arr = np.array(samples)
    return (point, float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5)))


def interpret_kappa(k: float) -> str:
    """Standard interpretation per Landis & Koch (1977)."""
    if np.isnan(k):
        return "n/a"
    if k < 0:
        return "no agreement (worse than chance)"
    if k < 0.20:
        return "slight"
    if k < 0.40:
        return "fair"
    if k < 0.60:
        return "moderate"
    if k < 0.80:
        return "substantial"
    return "almost perfect"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute Cohen's kappa between human and LLM agents"
    )
    parser.add_argument("blind_csv", type=Path,
                        help="Human-completed CSV (validation_sample_blind.csv)")
    parser.add_argument("truth_parquet", type=Path,
                        help="LLM truth parquet (validation_sample_truth.parquet)")
    parser.add_argument(
        "--out-json", type=Path,
        default=Path("data/processed/kappa_results.json"),
    )
    parser.add_argument("--bootstrap-resamples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260603)
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    log.info("Leyendo CSV humano: %s", args.blind_csv)
    human = pd.read_csv(args.blind_csv)
    log.info("  %d rows, %d con human_decision",
             len(human), human.human_decision.notna().sum())

    log.info("Leyendo truth parquet: %s", args.truth_parquet)
    truth = pd.read_parquet(args.truth_parquet)
    log.info("  %d rows", len(truth))

    # Join by review_id
    merged = human.merge(truth, on="review_id", how="inner", suffixes=("_h", ""))
    log.info("Después de join (inner por review_id): %d rows", len(merged))

    if len(merged) == 0:
        raise SystemExit("Sin overlap entre CSV humano y truth parquet. ¿Mismo seed/sample?")

    # Normalize human decision: lowercase, strip. Accept both "cumple" / "no cumple"
    # (Spanish) and legacy INCLUDE/EXCLUDE/HUMAN_REVIEW for backward compat.
    merged["human_decision_norm"] = (
        merged.human_decision.astype(str).str.strip().str.lower()
    )
    # Legacy 3-class → binary (same mapping as the LLM)
    legacy_map = {
        "include": "cumple",
        "human_review": "cumple",
        "exclude": "no cumple",
    }
    merged["human_decision_norm"] = merged.human_decision_norm.replace(legacy_map)

    valid_decision_mask = merged.human_decision_norm.isin(DECISION_VALUES)
    n_valid = int(valid_decision_mask.sum())
    log.info("  %d rows con human_decision válida en %s", n_valid, list(DECISION_VALUES))

    if n_valid == 0:
        raise SystemExit("Ningún human_decision válido. Llená la columna con "
                         "'cumple' o 'no cumple' primero.")

    valid = merged[valid_decision_mask].copy()
    valid["human_decision"] = valid["human_decision_norm"]
    # Collapse the LLM's 3-class verdict to the human's binary scale.
    valid["final_decision"] = valid.final_decision.apply(_llm_to_binary)
    # Drop any rows where the LLM verdict didn't map cleanly (shouldn't happen).
    valid = valid[valid.final_decision.isin(DECISION_VALUES)].copy()
    log.info("  Mapeado LLM 3-clases → binario {cumple, no cumple}: %d rows válidas", len(valid))

    results: dict = {
        "summary": {
            "n_total_sample": int(len(merged)),
            "n_with_human_decision": n_valid,
            "bootstrap_resamples": args.bootstrap_resamples,
            "seed": args.seed,
        },
        "decision_overall": {},
        "decision_by_source": {},
        "per_criterion": {},
    }

    # ---- Overall κ for final decision ----
    log.info("Computando κ overall sobre decisión final...")
    k, lo, hi = bootstrap_kappa_ci(
        valid.human_decision.values,
        valid.final_decision.values,
        DECISION_VALUES,
        n_resamples=args.bootstrap_resamples,
        seed=args.seed,
    )
    agreement_pct = float((valid.human_decision == valid.final_decision).sum()) / len(valid) * 100
    results["decision_overall"] = {
        "kappa": round(k, 4),
        "ci_95_low": round(lo, 4),
        "ci_95_high": round(hi, 4),
        "interpretation": interpret_kappa(k),
        "n": int(len(valid)),
        "agreement_pct": round(agreement_pct, 2),
    }
    log.info("  κ = %.3f [%.3f, %.3f] — %s",
             k, lo, hi, interpret_kappa(k))

    # ---- κ stratified by source ----
    log.info("Computando κ por fuente...")
    for src, sub in valid.groupby("source"):
        if len(sub) < 10:
            continue
        k, lo, hi = bootstrap_kappa_ci(
            sub.human_decision.values, sub.final_decision.values,
            DECISION_VALUES, args.bootstrap_resamples, args.seed,
        )
        agreement_pct = float((sub.human_decision == sub.final_decision).sum()) / len(sub) * 100
        results["decision_by_source"][src] = {
            "kappa": round(k, 4),
            "ci_95_low": round(lo, 4),
            "ci_95_high": round(hi, 4),
            "interpretation": interpret_kappa(k),
            "n": int(len(sub)),
            "agreement_pct": round(agreement_pct, 2),
        }
        log.info("  %-10s : κ = %.3f [%.3f, %.3f] (n=%d)",
                 src, k, lo, hi, len(sub))

    # ---- Per-criterion κ ----
    log.info("Computando κ por criterio (donde human filled in)...")
    for c in CRITERIA_ORDER:
        h_col = f"human_{c}"
        l_col = f"{c}_answer"
        if h_col not in valid.columns or l_col not in valid.columns:
            continue
        sub = valid[[h_col, l_col]].copy()
        sub[h_col] = sub[h_col].astype(str).str.strip().str.upper()
        sub[l_col] = sub[l_col].astype(str).str.strip().str.upper()
        sub = sub[sub[h_col].isin(ANSWER_VALUES) & sub[l_col].isin(ANSWER_VALUES)]
        if len(sub) < 10:
            continue
        k, lo, hi = bootstrap_kappa_ci(
            sub[h_col].values, sub[l_col].values,
            ANSWER_VALUES, args.bootstrap_resamples, args.seed,
        )
        agreement_pct = float((sub[h_col] == sub[l_col]).sum()) / len(sub) * 100
        results["per_criterion"][c] = {
            "kappa": round(k, 4),
            "ci_95_low": round(lo, 4),
            "ci_95_high": round(hi, 4),
            "interpretation": interpret_kappa(k),
            "n": int(len(sub)),
            "agreement_pct": round(agreement_pct, 2),
        }
        log.info("  %-4s : κ = %.3f [%.3f, %.3f] (n=%d)",
                 c, k, lo, hi, len(sub))

    # Output JSON
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    log.info("Resultados guardados → %s", args.out_json)

    print()
    print("=" * 70)
    print(" Resumen κ Cohen — Cribado Floccus vs revisor humano")
    print("=" * 70)
    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
