"""Full manual screening of cluster C03 against the pipeline.

Reads the workbook made by scripts/make_c03_screening_workbook.py once the
human has screened the 141 records (Incluir / Dudoso / Excluir), joins it
with data/processed/c03_screening_key.csv and reports:

- the distribution of the human decisions;
- pipeline false positives within C03: every C03 record was advanced by the
  pipeline, so each record the human excludes is one; given overall and
  separately for INCLUDE and HUMAN_REVIEW records, with exact
  Clopper-Pearson 95% CIs;
- the human (3 levels) x pipeline table and unweighted, linear- and
  quadratic-weighted Cohen's kappa, with the pipeline mapped INCLUDE ->
  Incluir, HUMAN_REVIEW -> Dudoso (EXCLUDE -> Excluir for the sensitivity
  mode, which excludes a few C03 records);
- the records the human excludes, with the stated reason, and every other
  disagreement, in a CSV for adjudication.

Run:  .venv/bin/python scripts/c03_screening_agreement.py --workbook <returned.xlsx>
          [--allow-incomplete] [--out-dir data/processed]
Out:  <out-dir>/c03_screening_results.json
      <out-dir>/c03_screening_disagreements.csv
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dedup_audit_agreement import proportion  # noqa: E402
from second_reviewer_agreement import (DATA, R2_LEVELS, confusion, has_problems,  # noqa: E402
                                       kappa_from_table, table_dict, validate_decisions)

PIPE_TO_LEVEL = {"EXCLUDE": "Excluir", "HUMAN_REVIEW": "Dudoso", "INCLUDE": "Incluir"}


def kappas(human: pd.Series, pipeline_level: pd.Series) -> dict:
    """3 x 3 table (human rows, pipeline columns, Excluir < Dudoso < Incluir) and kappas."""
    t = confusion(human, pipeline_level, R2_LEVELS)
    return {"table_human_rows_pipeline_cols": table_dict(t, R2_LEVELS, R2_LEVELS),
            "raw_agreement": float(np.trace(t) / t.sum()),
            "kappa_unweighted": kappa_from_table(t),
            "kappa_linear": kappa_from_table(t, "linear"),
            "kappa_quadratic": kappa_from_table(t, "quadratic")}


def false_positives(df: pd.DataFrame, col: str) -> dict:
    """Records the human excludes among those the pipeline (column col) advances."""
    adv = df[df[col].isin(["INCLUDE", "HUMAN_REVIEW"])]
    out = {"all_advanced": proportion(int((adv.human == "Excluir").sum()), len(adv))}
    for level in ("INCLUDE", "HUMAN_REVIEW"):
        g = adv[adv[col] == level]
        out[level] = proportion(int((g.human == "Excluir").sum()), len(g))
    return out


def analyse(df: pd.DataFrame) -> dict:
    """df: one row per decided record with human (R2_LEVELS), human_reason,
    pipeline_decision and sensitivity_decision."""
    out = {"human_counts": {k: int((df.human == k).sum()) for k in R2_LEVELS},
           "human_by_pipeline": {p: {k: int(((df.pipeline_decision == p) & (df.human == k)).sum())
                                     for k in R2_LEVELS}
                                 for p in ("INCLUDE", "HUMAN_REVIEW")},
           "false_positives": {}, "agreement": {}}
    for mode, col in (("specificity", "pipeline_decision"), ("sensitivity", "sensitivity_decision")):
        if col not in df or df[col].isna().all():
            continue
        out["false_positives"][mode] = false_positives(df, col)
        out["agreement"][mode] = kappas(df.human, df[col].map(PIPE_TO_LEVEL))
    exc = df[df.human == "Excluir"]
    out["exclusion_reasons"] = {str(k): int(v) for k, v in
                                exc.human_reason.fillna("(sin motivo)").value_counts().items()}
    return out


def disagreements(df: pd.DataFrame) -> pd.DataFrame:
    d = df[df.human != df.pipeline_decision.map(PIPE_TO_LEVEL)]
    cols = ["n", "record_id", "human", "pipeline_decision", "sensitivity_decision",
            "human_reason", "human_comment", "title"]
    return d[[c for c in cols if c in d.columns]].sort_values(["human", "n"])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workbook", type=Path, required=True, help="returned C03 workbook")
    ap.add_argument("--allow-incomplete", action="store_true",
                    help="analyse only the decided records instead of stopping")
    ap.add_argument("--out-dir", type=Path, default=DATA)
    ap.add_argument("--key", type=Path, default=DATA / "c03_screening_key.csv")
    args = ap.parse_args()

    key = pd.read_csv(args.key)
    reg = pd.read_excel(args.workbook, sheet_name="Registros")
    dec, problems = validate_decisions(reg, expected_n=key["n"].tolist())
    if has_problems(problems):
        for k, v in problems.items():
            if v:
                print(f"[workbook] {k} ({len(v)}): {v}")
        if problems["duplicate_n"] or problems["missing_n"] or not args.allow_incomplete:
            raise SystemExit("workbook incomplete or invalid; fix it or pass --allow-incomplete")
    dec = dec.rename(columns={"r2_level": "human", "r2_reason": "human_reason",
                              "r2_comment": "human_comment"}).drop(columns=["r2_binary", "title"])
    df = dec.merge(key, on="n", how="inner")
    if len(df) != len(dec):
        raise SystemExit(f"N° without a key entry: {sorted(set(dec.n) - set(key.n))}")

    out = {"workbook": str(args.workbook), "n_records": int(len(key)), "n_analysed": int(len(df)),
           "workbook_problems": problems, "ci_method": "exact Clopper-Pearson 95%",
           "pipeline_mapping": PIPE_TO_LEVEL}
    out.update(analyse(df))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    dis = disagreements(df)
    dis_path = args.out_dir / "c03_screening_disagreements.csv"
    dis.to_csv(dis_path, index=False)
    out["n_disagreements"] = int(len(dis))
    dest = args.out_dir / "c03_screening_results.json"
    dest.write_text(json.dumps(out, indent=2, ensure_ascii=False))

    print(f"analysed {len(df)}/{len(key)}  human {out['human_counts']}")
    print(f"human by pipeline decision: {out['human_by_pipeline']}")
    for mode, fp in out["false_positives"].items():
        s = "  ".join(f"{k} {v['k']}/{v['n']} = {v['share']:.3f} [{v['ci95'][0]:.3f}-{v['ci95'][1]:.3f}]"
                      for k, v in fp.items())
        a = out["agreement"][mode]
        print(f"{mode}: human-excluded among advanced: {s}")
        print(f"  kappa unweighted {a['kappa_unweighted']:.3f}  linear {a['kappa_linear']:.3f}  "
              f"quadratic {a['kappa_quadratic']:.3f}  agreement {a['raw_agreement']:.3f}")
    print(f"exclusion reasons: {out['exclusion_reasons']}")
    print(f"disagreements: {len(dis)} -> {dis_path}")
    print("->", dest)


if __name__ == "__main__":
    main()
