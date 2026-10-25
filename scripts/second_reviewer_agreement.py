"""Agreement of the second human reviewer with the first reviewer and with the pipeline.

The second reviewer screened the 465-record blind validation sample in the
workbook made by scripts/make_second_reviewer_workbook.py, on an ordinal
three-level scale (Excluir < Dudoso < Incluir). This script:

1. reads and validates the returned workbook (every record decided, values
   in the allowed set); blanks and invalid values are listed by N° and the
   run stops unless --allow-incomplete is given;
2. inter-human agreement on the binary scale (Incluir/Dudoso -> cumple,
   Excluir -> no cumple) against reviewer 1: raw agreement, Cohen's kappa
   with bootstrap 95% CI, PABAK, prevalence per reviewer, 2x2 table, plus a
   3x2 table of reviewer 2's three levels against reviewer 1;
3. reviewer 2 (three levels) against the pipeline (EXCLUDE < HUMAN_REVIEW <
   INCLUDE): unweighted, linear-weighted and quadratic-weighted kappa with
   bootstrap 95% CIs and the 3x3 table, for the specificity and the
   sensitivity operating modes;
4. stratum-reweighted recall / specificity / WSS of the pipeline with the
   stratified bootstrap of compute_recall_wss.py, against reviewer 1,
   reviewer 2 (binary), the consensus (both say cumple) and the lenient
   union (either says cumple), for both operating modes;
5. a list of the inter-human disagreements for adjudication.

Agreement statistics are sample-level (unweighted) quantities; their CIs
come from the same stratified bootstrap design (resampling within the
source x decision strata) with the seed of compute_recall_wss.py.

Run:  .venv/bin/python scripts/second_reviewer_agreement.py --workbook <returned.xlsx>
          [--allow-incomplete] [--out-dir data/processed]
Out:  <out-dir>/second_reviewer_agreement.json
      <out-dir>/second_reviewer_disagreements.csv
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from compute_recall_wss import (ADVANCE, DATA, N_BOOT, SEED, bootstrap,  # noqa: E402
                                load, metrics)
from operating_modes_recall import sensitivity_decision  # noqa: E402

R2_LEVELS = ["Excluir", "Dudoso", "Incluir"]                 # ordinal, low -> high
PIPE_LEVELS = ["EXCLUDE", "HUMAN_REVIEW", "INCLUDE"]         # ordinal, low -> high
R2_BINARY = {"Incluir": "cumple", "Dudoso": "cumple", "Excluir": "no cumple"}
BIN_LEVELS = ["no cumple", "cumple"]
COL_N, COL_DEC = "N°", "Decisión"
COL_WHY, COL_COMMENT, COL_TITLE = "Motivo principal de exclusión", "Comentario", "Título"


# ----------------------------------------------------------------- kappa
def confusion(a, b, levels_a, levels_b=None) -> np.ndarray:
    """Counts table, rows = levels of a, columns = levels of b."""
    levels_b = levels_a if levels_b is None else levels_b
    pos_a = {v: i for i, v in enumerate(levels_a)}
    pos_b = {v: i for i, v in enumerate(levels_b)}
    try:
        ia = np.array([pos_a[v] for v in a], dtype=int)
        ib = np.array([pos_b[v] for v in b], dtype=int)
    except KeyError as e:
        raise ValueError(f"value outside the declared levels: {e}") from None
    t = np.zeros((len(levels_a), len(levels_b)), dtype=int)
    np.add.at(t, (ia, ib), 1)
    return t


def kappa_from_table(t: np.ndarray, weights: str | None = None) -> float:
    """Cohen's kappa of a square k x k table of counts.

    weights: None (unweighted), 'linear' or 'quadratic' disagreement weights
    |i-j|/(k-1) and ((i-j)/(k-1))^2.  kappa = 1 - sum(w O) / sum(w E)."""
    t = np.asarray(t, dtype=float)
    k = t.shape[0]
    if t.shape != (k, k) or k < 2:
        raise ValueError("need a square table with at least 2 levels")
    i, j = np.indices((k, k))
    if weights is None:
        w = (i != j).astype(float)
    elif weights == "linear":
        w = np.abs(i - j) / (k - 1)
    elif weights == "quadratic":
        w = ((i - j) / (k - 1)) ** 2
    else:
        raise ValueError(f"unknown weights {weights!r}")
    n = t.sum()
    o = t / n
    e = np.outer(t.sum(1), t.sum(0)) / n ** 2
    de = (w * e).sum()
    return float("nan") if de == 0 else float(1.0 - (w * o).sum() / de)


def kappa(a, b, levels, weights: str | None = None) -> float:
    return kappa_from_table(confusion(a, b, levels), weights)


def pabak(t: np.ndarray) -> float:
    """Prevalence- and bias-adjusted kappa (2 p_o - 1 for two categories;
    (k p_o - 1)/(k - 1) in general)."""
    t = np.asarray(t, dtype=float)
    k = t.shape[0]
    po = np.trace(t) / t.sum()
    return float((k * po - 1) / (k - 1))


# ------------------------------------------------------------- workbook
def validate_decisions(reg: pd.DataFrame, expected_n=None) -> tuple[pd.DataFrame, dict]:
    """Normalise the 'Registros' sheet. Returns (decided rows, problems).

    Decisions are matched case- and whitespace-insensitively to Incluir /
    Dudoso / Excluir. problems = {'blank': [N°...], 'invalid': [[N°, value]...],
    'missing_n': [N°...], 'duplicate_n': [N°...]}."""
    for col in (COL_N, COL_DEC):
        if col not in reg.columns:
            raise ValueError(f"column {col!r} missing from the sheet")
    d = reg.copy()
    d = d[d[COL_N].notna()]
    d["n"] = d[COL_N].astype(int)
    raw = d[COL_DEC].astype("string").str.strip()
    canon = {v.lower(): v for v in R2_LEVELS}
    d["r2_level"] = raw.str.lower().map(canon)
    blank = raw.isna() | raw.eq("")
    invalid = ~blank & d["r2_level"].isna()
    problems = {
        "blank": d.loc[blank, "n"].tolist(),
        "invalid": [[int(n), str(v)] for n, v in zip(d.loc[invalid, "n"], raw[invalid])],
        "duplicate_n": sorted(set(d.loc[d["n"].duplicated(), "n"].tolist())),
        "missing_n": sorted(set(expected_n) - set(d["n"])) if expected_n is not None else [],
    }
    for col in (COL_WHY, COL_COMMENT, COL_TITLE):
        if col not in d.columns:
            d[col] = None
    out = d.loc[~blank & ~invalid, ["n", "r2_level", COL_WHY, COL_COMMENT, COL_TITLE]].rename(
        columns={COL_WHY: "r2_reason", COL_COMMENT: "r2_comment", COL_TITLE: "title"})
    out["r2_binary"] = out["r2_level"].map(R2_BINARY)
    return out.reset_index(drop=True), problems


def has_problems(p: dict) -> bool:
    return any(p[k] for k in p)


# ------------------------------------------------------------ data join
def sensitivity_decisions(df: pd.DataFrame) -> pd.Series:
    """Sensitivity-mode decisions for the rows of load()'s frame (same join as
    operating_modes_recall.py)."""
    truth = pd.read_parquet(DATA / "validation_sample_truth.parquet", columns=["review_id", "record_id"])
    dec = pd.read_parquet(DATA / "biofloc_final_decisions.parquet",
                          columns=["record_id"] + [f"E{i}_answer" for i in range(1, 6)])
    e2 = pd.read_parquet(DATA / "e2_v2_answers.parquet", columns=["record_id", "E2_v2"])
    s = (df.merge(truth, on="review_id").merge(dec, on="record_id", how="left")
           .merge(e2, on="record_id", how="left"))
    s.index = df.index
    return pd.Series(sensitivity_decision(s).to_numpy(), index=df.index)


def reweight(df: pd.DataFrame, sub: pd.DataFrame) -> pd.DataFrame:
    """Recompute w = N_c / n_c on a subset of the sample (incomplete workbook)."""
    N_c = df.groupby("stratum")["w"].first() * df["stratum"].value_counts()
    sub = sub.copy()
    sub["w"] = sub["stratum"].map(N_c / sub["stratum"].value_counts())
    return sub


# ------------------------------------------------------------ bootstrap
def boot_ci(df: pd.DataFrame, stat) -> list[float]:
    """Stratified bootstrap 95% percentile CI of stat(frame) (same design and seed
    as compute_recall_wss.bootstrap)."""
    rng = np.random.default_rng(SEED)
    idx = [df.index.get_indexer(g.index) for _, g in df.groupby("stratum", sort=True)]
    vals = np.empty(N_BOOT)
    for b in range(N_BOOT):
        pos = np.concatenate([rng.choice(ix, size=ix.size, replace=True) for ix in idx])
        vals[b] = stat(df.iloc[pos])
    vals = vals[~np.isnan(vals)]
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def table_dict(t: np.ndarray, rows, cols) -> dict:
    return {r: {c: int(t[i, j]) for j, c in enumerate(cols)} for i, r in enumerate(rows)}


# ---------------------------------------------------------------- main
def analyse(df: pd.DataFrame) -> dict:
    """df: load() frame joined with reviewer 2 (r2_level, r2_binary) and the
    sensitivity decision (sens_decision)."""
    out: dict = {}
    r1 = df["human_decision"].str.strip().str.lower()
    df = df.assign(r1=r1)

    # 2. inter-human, binary
    t = confusion(df.r1, df.r2_binary, BIN_LEVELS)
    out["inter_human_binary"] = {
        "table_r1_rows_r2_cols": table_dict(t, BIN_LEVELS, BIN_LEVELS),
        "raw_agreement": float(np.trace(t) / t.sum()),
        "kappa": kappa_from_table(t),
        "kappa_ci95": boot_ci(df, lambda s: kappa(s.r1, s.r2_binary, BIN_LEVELS)),
        "pabak": pabak(t),
        "prevalence_cumple_r1": float((df.r1 == "cumple").mean()),
        "prevalence_cumple_r2": float((df.r2_binary == "cumple").mean()),
    }
    t32 = confusion(df.r2_level, df.r1, R2_LEVELS, BIN_LEVELS)
    out["r2_levels_vs_r1"] = table_dict(t32, R2_LEVELS, BIN_LEVELS)

    # 3. reviewer 2 (3 levels) vs pipeline, both modes
    r2_code = {v: i for i, v in enumerate(R2_LEVELS)}
    pipe_code = {v: i for i, v in enumerate(PIPE_LEVELS)}
    out["r2_vs_pipeline"] = {}
    for mode, col in (("specificity", "final_decision"), ("sensitivity", "sens_decision")):
        a = df.r2_level.map(r2_code)
        b = df[col].map(pipe_code)
        t = confusion(a, b, [0, 1, 2])
        res = {"table_r2_rows_pipeline_cols": table_dict(t, R2_LEVELS, PIPE_LEVELS),
               "raw_agreement": float(np.trace(t) / t.sum())}
        codes = df.assign(_a=a, _b=b)
        for wname, w in (("unweighted", None), ("linear", "linear"), ("quadratic", "quadratic")):
            res[f"kappa_{wname}"] = kappa_from_table(t, w)
            res[f"kappa_{wname}_ci95"] = boot_ci(
                codes, lambda s, w=w: kappa(s._a, s._b, [0, 1, 2], w))
        out["r2_vs_pipeline"][mode] = res

    # 4. pipeline recall / specificity / WSS against each reference
    refs = {
        "reviewer1": df.r1.eq("cumple"),
        "reviewer2_binary": df.r2_binary.eq("cumple"),
        "consensus_both": df.r1.eq("cumple") & df.r2_binary.eq("cumple"),
        "union_either": df.r1.eq("cumple") | df.r2_binary.eq("cumple"),
    }
    out["pipeline_vs_reference"] = {}
    for ref, rel in refs.items():
        out["pipeline_vs_reference"][ref] = {"n_relevant_sample": int(rel.sum())}
        for mode, col in (("specificity", "final_decision"), ("sensitivity", "sens_decision")):
            f = df.assign(relevant=rel.to_numpy(), final_decision=df[col].to_numpy())
            out["pipeline_vs_reference"][ref][mode] = {
                "raw": metrics(f, np.ones(len(f)), ADVANCE),
                "reweighted": metrics(f, f["w"].to_numpy(), ADVANCE),
                "reweighted_ci95": bootstrap(f, ADVANCE),
            }
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workbook", type=Path, required=True, help="workbook returned by reviewer 2")
    ap.add_argument("--allow-incomplete", action="store_true",
                    help="analyse only the decided rows instead of stopping")
    ap.add_argument("--out-dir", type=Path, default=DATA)
    args = ap.parse_args()

    key = pd.read_csv(DATA / "validation_reviewer2_key.csv")
    reg = pd.read_excel(args.workbook, sheet_name="Registros")
    r2, problems = validate_decisions(reg, expected_n=key["n"].tolist())
    if has_problems(problems):
        for k, v in problems.items():
            if v:
                print(f"[workbook] {k} ({len(v)}): {v}")
        if problems["duplicate_n"] or problems["missing_n"] or not args.allow_incomplete:
            raise SystemExit("workbook incomplete or invalid; fix it or pass --allow-incomplete")

    df, n_pop = load()
    df["sens_decision"] = sensitivity_decisions(df)
    r2 = r2.merge(key, on="n", how="left")
    if r2["review_id"].isna().any():
        raise SystemExit(f"N° without a key entry: {r2.loc[r2.review_id.isna(), 'n'].tolist()}")
    sub = df.reset_index().merge(r2, on="review_id", how="inner").set_index("index")
    if len(sub) != len(r2):
        raise SystemExit(f"join lost rows: {len(r2)} -> {len(sub)}")
    if len(sub) < len(df):
        sub = reweight(df, sub)
        print(f"[workbook] INCOMPLETE: analysing {len(sub)} of {len(df)} decided records")

    out = {"workbook": str(args.workbook), "n_sample": int(len(df)), "n_analysed": int(len(sub)),
           "n_population": n_pop, "seed": SEED, "n_bootstrap": N_BOOT,
           "workbook_problems": problems,
           "r2_level_counts": {k: int((sub.r2_level == k).sum()) for k in R2_LEVELS},
           "caveat": "sensitivity mode designed on this sample (reviewer 1 labels): "
                     "its recall is an optimistic upper bound"}
    out.update(analyse(sub))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    r1 = sub["human_decision"].str.strip().str.lower()
    dis = sub[r1 != sub.r2_binary]
    dis = pd.DataFrame({
        "n": dis["n"], "review_id": dis["review_id"], "title": dis["title"],
        "reviewer1": r1[dis.index], "reviewer2": dis["r2_level"],
        "pipeline_specificity": dis["final_decision"], "pipeline_sensitivity": dis["sens_decision"],
        "reviewer2_reason": dis["r2_reason"], "reviewer2_comment": dis["r2_comment"],
    }).sort_values("n")
    dis_path = args.out_dir / "second_reviewer_disagreements.csv"
    dis.to_csv(dis_path, index=False)
    out["n_disagreements_binary"] = int(len(dis))
    dest = args.out_dir / "second_reviewer_agreement.json"
    dest.write_text(json.dumps(out, indent=2, ensure_ascii=False))

    ih = out["inter_human_binary"]
    print(f"analysed {len(sub)}/{len(df)}  reviewer-2 levels {out['r2_level_counts']}")
    print(f"inter-human (binary): agreement {ih['raw_agreement']:.3f}  kappa {ih['kappa']:.3f} "
          f"[{ih['kappa_ci95'][0]:.3f}-{ih['kappa_ci95'][1]:.3f}]  PABAK {ih['pabak']:.3f}  "
          f"prev r1 {ih['prevalence_cumple_r1']:.3f} r2 {ih['prevalence_cumple_r2']:.3f}")
    print(f"  r1 rows x r2 cols: {ih['table_r1_rows_r2_cols']}")
    print(f"  r2 levels x r1:    {out['r2_levels_vs_r1']}")
    for mode, r in out["r2_vs_pipeline"].items():
        ks = "  ".join(f"{w} {r[f'kappa_{w}']:.3f} [{r[f'kappa_{w}_ci95'][0]:.3f}-"
                       f"{r[f'kappa_{w}_ci95'][1]:.3f}]" for w in ("unweighted", "linear", "quadratic"))
        print(f"r2 vs pipeline ({mode}): {ks}")
    print(f"{'reference':<18}{'mode':<13}{'recall [95% CI]':<24}{'specificity [95% CI]':<24}WSS")
    for ref, modes in out["pipeline_vs_reference"].items():
        for mode in ("specificity", "sensitivity"):
            rw, ci = modes[mode]["reweighted"], modes[mode]["reweighted_ci95"]
            print(f"{ref:<18}{mode:<13}"
                  f"{rw['recall']:.3f} [{ci['recall'][0]:.3f}-{ci['recall'][1]:.3f}]   "
                  f"{rw['specificity']:.3f} [{ci['specificity'][0]:.3f}-{ci['specificity'][1]:.3f}]   "
                  f"{rw['wss']:.3f}")
    print(f"disagreements (binary): {len(dis)} -> {dis_path}")
    print("->", dest)


if __name__ == "__main__":
    main()
