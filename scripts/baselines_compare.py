"""Criterion-wise pipeline vs two simple baselines (reviewer 2, comment 7).

All rows are scored on the SAME 465-record blind validation sample against
the first reviewer's labels, with the stratum reweighting (source x
specificity-mode decision) and stratified bootstrap of compute_recall_wss.py.
The strata and weights are those of the sampling design and do not depend
on the method being scored; only each record's advance/exclude call changes.

Rows
  pipeline_specificity   published configuration (INCLUDE + HUMAN_REVIEW advance)
  pipeline_sensitivity   positive E2 re-ask (operating_modes_recall.py)
  keyword                descriptor rule of the database query (baseline_keyword.py)
  keyword_strict         ... plus one formal-modelling term
  single_prompt          one LLM call per record; INCLUDE or UNCLEAR advance
  single_prompt_strict   ... only INCLUDE advances

Paired comparison: exact McNemar test (two-sided binomial on the discordant
pairs) between pipeline_specificity and each other row, on the raw sample,
using per-record correctness against the human label; also split into the
human-relevant records (recall) and the human-irrelevant ones (specificity).

Run:  .venv/bin/python scripts/baselines_compare.py
Needs data/processed/baseline_keyword_decisions.parquet and
      data/processed/baseline_single_prompt_answers.parquet
Out:  data/processed/baselines_compare.json
"""
from __future__ import annotations

import json
import sys
from math import comb
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from compute_recall_wss import ADVANCE, DATA, bootstrap, load, metrics  # noqa: E402

ROWS = ("pipeline_specificity", "pipeline_sensitivity", "keyword", "keyword_strict",
        "single_prompt", "single_prompt_strict")


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p: binomial(b + c, 1/2) on the discordant pairs."""
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(comb(n, k) for k in range(min(b, c) + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def cohen_kappa(a: np.ndarray, b: np.ndarray) -> float:
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    po = (a == b).mean()
    pe = a.mean() * b.mean() + (1 - a.mean()) * (1 - b.mean())
    return float((po - pe) / (1 - pe)) if pe < 1 else float("nan")


def paired(ref_ok: np.ndarray, other_ok: np.ndarray) -> dict:
    b = int((ref_ok & ~other_ok).sum())    # pipeline right, other wrong
    c = int((~ref_ok & other_ok).sum())    # pipeline wrong, other right
    return {"pipeline_right_other_wrong": b, "pipeline_wrong_other_right": c,
            "p_exact": mcnemar_exact(b, c)}


def advance_vectors(df: pd.DataFrame) -> dict[str, np.ndarray]:
    from operating_modes_recall import sensitivity_decision

    truth = pd.read_parquet(DATA / "validation_sample_truth.parquet",
                            columns=["review_id", "record_id"])
    rid = df[["review_id"]].merge(truth, on="review_id", how="left")["record_id"].to_numpy()

    dec = pd.read_parquet(DATA / "biofloc_final_decisions.parquet",
                          columns=["record_id"] + [f"E{i}_answer" for i in range(1, 6)])
    e2 = pd.read_parquet(DATA / "e2_v2_answers.parquet", columns=["record_id", "E2_v2"])
    s = (df.assign(record_id=rid).merge(dec, on="record_id", how="left")
           .merge(e2, on="record_id", how="left"))
    s.index = df.index

    kw = pd.read_parquet(DATA / "baseline_keyword_decisions.parquet").set_index("record_id")
    sp = pd.read_parquet(DATA / "baseline_single_prompt_answers.parquet").set_index("record_id")
    missing = set(rid) - set(sp.index)
    if missing:
        raise SystemExit(f"single-prompt answers missing for {len(missing)} records: "
                         "run scripts/baseline_single_prompt.py first")
    ans = sp.loc[rid, "answer"].to_numpy()
    return {
        "pipeline_specificity": df.final_decision.isin(ADVANCE).to_numpy(),
        "pipeline_sensitivity": sensitivity_decision(s).isin(ADVANCE).to_numpy(),
        "keyword": kw.loc[rid, "advance_keyword"].to_numpy(bool),
        "keyword_strict": kw.loc[rid, "advance_keyword_strict"].to_numpy(bool),
        "single_prompt": np.isin(ans, ["INCLUDE", "UNCLEAR"]),
        "single_prompt_strict": ans == "INCLUDE",
    }, pd.Series(ans).value_counts().to_dict()


def score(df: pd.DataFrame, adv: np.ndarray, n_pop: int) -> dict:
    frame = df.copy()
    frame["final_decision"] = np.where(adv, "INCLUDE", "EXCLUDE")
    rel = frame["relevant"].to_numpy()
    raw = metrics(frame, np.ones(len(frame)), {"INCLUDE"})
    rw = metrics(frame, frame["w"].to_numpy(), {"INCLUDE"})
    ci = bootstrap(frame, {"INCLUDE"})
    share = 1.0 - rw["burden_removed"]
    return {
        "confusion_raw": {"TP": int((rel & adv).sum()), "FN": int((rel & ~adv).sum()),
                          "FP": int((~rel & adv).sum()), "TN": int((~rel & ~adv).sum())},
        "n_advanced_sample": int(adv.sum()),
        "raw": raw,
        "reweighted": rw,
        "reweighted_ci95": ci,
        "share_advanced_corpus": share,
        "share_advanced_corpus_ci95": [1.0 - ci["burden_removed"][1], 1.0 - ci["burden_removed"][0]],
        "records_to_human_corpus_estimate": round(share * n_pop),
        "kappa_vs_human": cohen_kappa(adv, rel),
    }


def main() -> None:
    df, n_pop = load()
    vecs, sp_counts = advance_vectors(df)
    rel = df["relevant"].to_numpy()
    modes = json.loads((DATA / "operating_modes.json").read_text())

    out = {"n_sample": int(len(df)), "n_population": n_pop,
           "n_relevant_sample": int(rel.sum()),
           "single_prompt_answer_counts": {k: int(v) for k, v in sp_counts.items()},
           "human_load_corpus_exact": {
               "pipeline_specificity": modes["specificity_mode_human_load"],
               "pipeline_sensitivity": modes["sensitivity_mode_human_load"]},
           "note": ("pipeline_sensitivity was designed after inspecting errors on this "
                    "sample (optimistic); for baselines the corpus share is the "
                    "reweighted sample estimate"),
           "rows": {}, "mcnemar_vs_pipeline_specificity": {}}
    for name in ROWS:
        out["rows"][name] = score(df, vecs[name], n_pop)

    ref_ok = vecs["pipeline_specificity"] == rel
    for name in ROWS[1:]:
        ok = vecs[name] == rel
        out["mcnemar_vs_pipeline_specificity"][name] = {
            "all": paired(ref_ok, ok),
            "relevant_only": paired(ref_ok[rel], ok[rel]),
            "irrelevant_only": paired(ref_ok[~rel], ok[~rel]),
        }

    dest = DATA / "baselines_compare.json"
    dest.write_text(json.dumps(out, indent=2))

    print(f"sample n={out['n_sample']} (relevant {out['n_relevant_sample']})  "
          f"population N={n_pop:,}\n")
    hdr = (f"{'method':<22}{'sens':>6}{'spec':>6} | {'recall_rw [95% CI]':<22}"
           f"{'spec_rw [95% CI]':<22}{'WSS':>6}{'share_adv':>10}{'kappa':>7}")
    print(hdr); print("-" * len(hdr))
    for name in ROWS:
        r = out["rows"][name]; rw = r["reweighted"]; ci = r["reweighted_ci95"]
        print(f"{name:<22}{r['raw']['recall']:>6.3f}{r['raw']['specificity']:>6.3f} | "
              f"{rw['recall']:.3f} [{ci['recall'][0]:.3f},{ci['recall'][1]:.3f}]  "
              f"{rw['specificity']:.3f} [{ci['specificity'][0]:.3f},{ci['specificity'][1]:.3f}]  "
              f"{rw['wss']:>6.3f}{r['share_advanced_corpus']:>10.3f}{r['kappa_vs_human']:>7.3f}")
    print("\nMcNemar (exact) vs pipeline_specificity: b = pipeline right/other wrong, "
          "c = pipeline wrong/other right")
    for name, m in out["mcnemar_vs_pipeline_specificity"].items():
        parts = [f"{k}: b={v['pipeline_right_other_wrong']} c={v['pipeline_wrong_other_right']} "
                 f"p={v['p_exact']:.3g}" for k, v in m.items()]
        print(f"{name:<22}" + " | ".join(parts))
    print("->", dest)


if __name__ == "__main__":
    main()
