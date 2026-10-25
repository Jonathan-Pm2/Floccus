"""Precision of the deduplication layers from the manual pair audit.

Reads the workbook made by scripts/make_dedup_audit_workbook.py once the
human has decided every pair (Mismo trabajo / Trabajos distintos / No se
puede determinar), joins it with data/processed/dedup_audit_key.csv and
reports, per stratum, the share of pairs whose system verdict the human
confirms with an exact Clopper-Pearson 95% CI:

- Layer-1 precision: human "same" among deterministic merges;
- Layer-3 precision: human "same" among LLM DUPLICATE verdicts;
- Layer-3 negative predictive value: human "different" among NOT_DUPLICATE;
- the human's decisions on the UNCLEAR pairs (distribution only);
- on the Layer-2/3 pairs, the human x LLM table, raw agreement and Cohen's
  kappa on the pairs where both gave a decisive answer (pairs the human
  could not determine, and UNCLEAR verdicts, are excluded and counted).

Shares exclude "No se puede determinar"; a conservative variant counting
those pairs as not confirmed is also given. The Layer-2/3 strata are of
fixed size (30/30/10) whatever their size in the corpus, so the agreement
and kappa describe the audit sample, not the corpus.

Run:  .venv/bin/python scripts/dedup_audit_agreement.py --workbook <returned.xlsx>
          [--allow-incomplete] [--out-dir data/processed]
Out:  <out-dir>/dedup_audit_results.json
      <out-dir>/dedup_audit_disagreements.csv
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import beta

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_dedup_audit_workbook import DECISIONS  # noqa: E402
from second_reviewer_agreement import (DATA, confusion, has_problems,  # noqa: E402
                                       kappa_from_table, table_dict, validate_decisions)

SAME, DIFFERENT, UNDETERMINED = DECISIONS
HUMAN_TO_VERDICT = {SAME: "DUPLICATE", DIFFERENT: "NOT_DUPLICATE"}
# stratum -> (human decision that confirms the system verdict, statistic name)
CONFIRMS = {"L1_merge": (SAME, "layer1_precision"),
            "L3_DUPLICATE": (SAME, "layer3_precision"),
            "L3_NOT_DUPLICATE": (DIFFERENT, "layer3_npv")}
LLM_LEVELS = ["DUPLICATE", "NOT_DUPLICATE", "UNCLEAR"]


def clopper_pearson(k: int, n: int, alpha: float = 0.05) -> list[float]:
    """Exact (Clopper-Pearson) two-sided 1-alpha CI of a binomial proportion k/n."""
    if n == 0:
        return [float("nan"), float("nan")]
    lo = 0.0 if k == 0 else float(beta.ppf(alpha / 2, k, n - k + 1))
    hi = 1.0 if k == n else float(beta.ppf(1 - alpha / 2, k + 1, n - k))
    return [lo, hi]


def proportion(k: int, n: int) -> dict:
    return {"k": int(k), "n": int(n), "share": float(k / n) if n else float("nan"),
            "ci95": clopper_pearson(int(k), int(n))}


def stratum_summary(decisions: pd.Series, confirm: str) -> dict:
    """Share of decisive human answers equal to confirm, and the conservative
    share counting undetermined answers as not confirming."""
    counts = {d: int((decisions == d).sum()) for d in DECISIONS}
    k = counts[confirm]
    decisive = counts[SAME] + counts[DIFFERENT]
    return {"human_counts": counts, "n_undetermined": counts[UNDETERMINED],
            "decisive": proportion(k, decisive),
            "conservative": proportion(k, decisive + counts[UNDETERMINED])}


def human_vs_llm(df: pd.DataFrame) -> dict:
    """df: Layer-2/3 pairs with columns human (DECISIONS) and system_verdict (LLM_LEVELS)."""
    t = confusion(df.human, df.system_verdict, DECISIONS, LLM_LEVELS)
    both = df[df.human.isin([SAME, DIFFERENT]) & df.system_verdict.isin(["DUPLICATE", "NOT_DUPLICATE"])]
    h = both.human.map(HUMAN_TO_VERDICT)
    t2 = confusion(h, both.system_verdict, ["DUPLICATE", "NOT_DUPLICATE"])
    return {
        "table_human_rows_llm_cols": table_dict(t, DECISIONS, LLM_LEVELS),
        "n_pairs": int(len(df)),
        "n_excluded_human_undetermined": int((df.human == UNDETERMINED).sum()),
        "n_excluded_llm_unclear": int(((df.system_verdict == "UNCLEAR") & (df.human != UNDETERMINED)).sum()),
        "n_compared": int(len(both)),
        "raw_agreement": proportion(int(np.trace(t2)), int(t2.sum())),
        "kappa": kappa_from_table(t2) if t2.sum() else float("nan"),
        "table_binary_human_rows_llm_cols": table_dict(t2, ["DUPLICATE", "NOT_DUPLICATE"],
                                                       ["DUPLICATE", "NOT_DUPLICATE"]),
    }


def analyse(df: pd.DataFrame) -> dict:
    """df: one row per decided pair with columns stratum, system_verdict, human."""
    out: dict = {"strata": {}}
    for stratum, g in df.groupby("stratum", sort=True):
        if stratum in CONFIRMS:
            confirm, name = CONFIRMS[stratum]
            out["strata"][stratum] = {"statistic": name, "confirming_decision": confirm,
                                      **stratum_summary(g.human, confirm)}
        else:
            counts = {d: int((g.human == d).sum()) for d in DECISIONS}
            out["strata"][stratum] = {"human_counts": counts,
                                      "share_same_among_decisive": proportion(
                                          counts[SAME], counts[SAME] + counts[DIFFERENT])}
    l3 = df[df.stratum.str.startswith("L3_")]
    out["layer23_human_vs_llm"] = human_vs_llm(l3) if len(l3) else None
    return out


def disagreements(df: pd.DataFrame) -> pd.DataFrame:
    """Pairs where the human contradicts the system verdict, plus every pair the
    human could not determine and every UNCLEAR pair, for adjudication."""
    expected = df.stratum.map({s: c for s, (c, _) in CONFIRMS.items()})
    kind = np.select(
        [df.human == UNDETERMINED, df.stratum == "L3_UNCLEAR", df.human != expected],
        ["human_undetermined", "llm_unclear", "disagreement"], default="")
    d = df.assign(type=kind)
    d = d[d.type != ""]
    cols = ["n", "type", "stratum", "system_verdict", "human", "comment", "match_basis", "cosine",
            "judge_confidence", "a_ref", "b_ref", "a_title", "b_title"]
    return d[[c for c in cols if c in d.columns]].sort_values("n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workbook", type=Path, required=True, help="returned audit workbook")
    ap.add_argument("--allow-incomplete", action="store_true",
                    help="analyse only the decided pairs instead of stopping")
    ap.add_argument("--out-dir", type=Path, default=DATA)
    ap.add_argument("--key", type=Path, default=DATA / "dedup_audit_key.csv")
    args = ap.parse_args()

    key = pd.read_csv(args.key)
    reg = pd.read_excel(args.workbook, sheet_name="Pares")
    dec, problems = validate_decisions(reg, expected_n=key["n"].tolist(), levels=DECISIONS)
    if has_problems(problems):
        for k, v in problems.items():
            if v:
                print(f"[workbook] {k} ({len(v)}): {v}")
        if problems["duplicate_n"] or problems["missing_n"] or not args.allow_incomplete:
            raise SystemExit("workbook incomplete or invalid; fix it or pass --allow-incomplete")
    dec = dec.rename(columns={"r2_level": "human", "r2_comment": "comment"})[["n", "human", "comment"]]
    df = dec.merge(key, on="n", how="inner")
    if len(df) != len(dec):
        raise SystemExit(f"N° without a key entry: {sorted(set(dec.n) - set(key.n))}")

    out = {"workbook": str(args.workbook), "n_pairs": int(len(key)), "n_analysed": int(len(df)),
           "workbook_problems": problems,
           "sample_design": key.stratum.value_counts().sort_index().to_dict(),
           "ci_method": "exact Clopper-Pearson 95%",
           "caveat": "Layer-2/3 strata have fixed sizes (not proportional to the corpus): "
                     "agreement and kappa describe the audit sample. Records: fields shown as "
                     "exported by each database, before merging."}
    out.update(analyse(df))
    l1 = df[df.stratum == "L1_merge"]
    out["layer1_by_match_basis"] = {b: stratum_summary(g.human, SAME)
                                    for b, g in l1.groupby("match_basis", sort=True)}

    args.out_dir.mkdir(parents=True, exist_ok=True)
    dis = disagreements(df)
    dis_path = args.out_dir / "dedup_audit_disagreements.csv"
    dis.to_csv(dis_path, index=False)
    out["n_disagreements_csv_rows"] = int(len(dis))
    dest = args.out_dir / "dedup_audit_results.json"
    dest.write_text(json.dumps(out, indent=2, ensure_ascii=False))

    print(f"analysed {len(df)}/{len(key)} pairs")
    for s, r in out["strata"].items():
        if "decisive" in r:
            p = r["decisive"]
            print(f"{s:<17} {r['statistic']:<17} {p['k']}/{p['n']} = {p['share']:.3f} "
                  f"[{p['ci95'][0]:.3f}-{p['ci95'][1]:.3f}]  undetermined {r['n_undetermined']}")
        else:
            print(f"{s:<17} human counts {r['human_counts']}")
    hv = out["layer23_human_vs_llm"]
    if hv:
        print(f"L2/L3 human vs LLM: compared {hv['n_compared']} (excluded: human undetermined "
              f"{hv['n_excluded_human_undetermined']}, LLM unclear {hv['n_excluded_llm_unclear']})  "
              f"agreement {hv['raw_agreement']['share']:.3f}  kappa {hv['kappa']:.3f}")
    print(f"rows for adjudication: {len(dis)} -> {dis_path}")
    print("->", dest)


if __name__ == "__main__":
    main()
