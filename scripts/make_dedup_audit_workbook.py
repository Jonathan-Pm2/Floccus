"""Workbook for the manual audit of the deduplication layers.

Reviewer 2 (comment 7) asked for a manual check of the deduplication. This
script draws a stratified sample of record PAIRS whose system verdict is
known, so that the precision of each layer can be estimated, and writes a
blind Excel workbook in which a human decides whether each pair is the same
work:

- Layer 1 (deterministic DOI / normalised title + year merges): 40 pairs,
  one per sampled multi-record cluster of data/processed/biofloc_dedup_full.parquet
  (the cluster's primary record against one randomly chosen merged record);
- Layers 2/3 (embedding candidates judged by the LLM, from
  biofloc_dedup_primary_near_dup_pairs_judged.json): 30 DUPLICATE, 30
  NOT_DUPLICATE and up to 10 UNCLEAR pairs.

The pairs are shuffled together with a fixed seed and the side (A/B) of each
record is randomised, so neither the row order nor the side reveals the
stratum. Every record is shown with its pre-merge fields (data/processed/
biofloc.parquet, as exported by its database), so neither Layer-1 enrichment
nor merging changes what the reviewer sees. Layer, verdict, cosine score and
LLM reasoning are left out of the
workbook and written to a separate key that the reviewer must not receive.
The decision definitions mirror the LLM judge prompt
(src/biofloc_review/dedup/llm_judge.py, SYSTEM_PROMPT).

Run:  .venv/bin/python scripts/make_dedup_audit_workbook.py --out <file.xlsx>
Out:  <file.xlsx> (to send) and data/processed/dedup_audit_key.csv
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_second_reviewer_workbook import (DATA, add_instructions_sheet,  # noqa: E402
                                           add_list_validation, counter_formula, fit_to_width,
                                           fix_mojibake)

SEED = 20261026
CORPUS = "biofloc"
N_LAYER1 = 40
QUOTAS = {"DUPLICATE": 30, "NOT_DUPLICATE": 30, "UNCLEAR": 10}
MAX_AUTHORS = 6
ABSTRACT_CHARS = 400

DECISIONS = ["Mismo trabajo", "Trabajos distintos", "No se puede determinar"]
SOURCE_LABELS = {"wos": "Web of Science", "scopus": "Scopus", "scielo": "SciELO", "ieee": "IEEE Xplore"}
FIELDS = ["title", "authors", "year", "venue", "doi", "source", "abstract"]
FIELD_LABELS = {"title": "Título", "authors": "Autores", "year": "Año", "venue": "Revista o fuente",
                "doi": "DOI", "source": "Base de datos", "abstract": "Resumen (primeros 400 caracteres)"}
FIELD_WIDTHS = {"title": 40, "authors": 28, "year": 7, "venue": 22, "doi": 24, "source": 13,
                "abstract": 60}

INSTRUCTIONS = [
    ("Auditoría manual de la deduplicación", "title"),
    ("", None),
    ("Objetivo", "head"),
    ("Decidir, para cada par de registros bibliográficos, si los dos registros corresponden a la misma "
     "publicación o a trabajos distintos. Con estas decisiones se estimará la precisión de cada etapa "
     "de la deduplicación.", None),
    ("", None),
    ("Condiciones (importante)", "head"),
    ("1. Trabajo ciego: la hoja no muestra cómo se obtuvo cada par ni qué decidió el procedimiento "
     "automático. No consulte esos resultados.", None),
    ("2. Decida solo con lo que ve en la hoja «Pares»: título, autores, año, revista o fuente, DOI, base "
     "de datos y el inicio del resumen de cada registro. No consulte fuentes externas.", None),
    ("3. El orden de los pares y el lado (A o B) en que aparece cada registro son aleatorios y no "
     "significan nada.", None),
    ("4. Los resúmenes están recortados a 400 caracteres; los textos recortados terminan en «…». Las "
     "listas de más de 6 autores muestran los 6 primeros seguidos de «et al.».", None),
    ("", None),
    ("Cómo decidir (columna «Decisión», obligatoria)", "head"),
    ("Las definiciones son las mismas que se dieron al juez automático de duplicados, para que ambas "
     "decisiones sean comparables.", None),
    ("Mismo trabajo: los dos registros describen la misma publicación. Cuenta como mismo trabajo: el "
     "mismo DOI; el mismo título salvo diferencias de formato, traducción o subtítulo; o claramente la "
     "misma investigación publicada en versiones distintas (por ejemplo, preprint y versión publicada, "
     "o una traducción). Incluye la misma publicación registrada en bases de datos distintas y las "
     "variaciones menores de metadatos (mayúsculas, abreviatura de autores o de revista, año "
     "desplazado por fecha en línea frente a fecha impresa).", None),
    ("Trabajos distintos: mismo tema o área pero publicaciones diferentes; mismos autores pero estudios "
     "diferentes; o mismo título genérico (por ejemplo, «Activated sludge») que corresponde a objetos "
     "distintos (capítulo de libro, entrada de enciclopedia, otro artículo con título corto).", None),
    ("No se puede determinar: la información disponible no permite decidir con una confianza razonable.",
     None),
    ("Caso no previsto en las definiciones: una ponencia de congreso y un artículo de revista posterior "
     "con el mismo título. Decida según la regla de «la misma investigación en versiones distintas» y "
     "escriba «congreso» en la columna «Comentario», para poder analizar ese caso por separado.", None),
    ("", None),
    ("Columna opcional", "head"),
    ("«Comentario»: cualquier observación útil (por ejemplo, qué dato le hizo dudar).", None),
    ("", None),
    ("Al terminar", "head"),
    ("Guarde el archivo con el mismo nombre y devuélvalo. Puede trabajar en varias sesiones; la fila de "
     "abajo cuenta cuántos pares lleva decididos.", None),
]


# ---------------------------------------------------------------- pure helpers
def format_authors(authors, max_n: int = MAX_AUTHORS) -> str:
    """First max_n authors joined by '; ', followed by 'et al.' when there are more."""
    if authors is None or isinstance(authors, float) or len(authors) == 0:
        return ""
    names = [fix_mojibake(str(a)) for a in authors]
    head = "; ".join(names[:max_n])
    return head + " et al." if len(names) > max_n else head


def clip(text, n: int = ABSTRACT_CHARS) -> str:
    """First n characters of text (mojibake repaired), with an ellipsis when cut."""
    if text is None or (isinstance(text, float) and text != text):
        return ""
    s = " ".join(fix_mojibake(str(text)).split())
    return s if len(s) <= n else s[:n].rstrip() + "…"


def doi_cluster_id(doi) -> str | None:
    """Layer-1 cluster id that a DOI-keyed record gets (deterministic._cluster_id_for)."""
    if not doi or (isinstance(doi, float) and doi != doi):
        return None
    return hashlib.sha1(f"doi::{doi}".encode("utf-8")).hexdigest()[:16]


def sample_layer1_pairs(full: pd.DataFrame, n: int, rng: np.random.Generator) -> pd.DataFrame:
    """n pairs from distinct multi-record Layer-1 clusters: the primary record
    (is_primary) against one merged record drawn at random. Returns positional
    row indices into full: a_pos (primary), b_pos (merged), plus cluster_id and
    match_basis ('doi' or 'title_year', recovered from the cluster id)."""
    full = full.reset_index(drop=True)
    size = full.groupby("cluster_id").size()
    multi = sorted(size[size > 1].index)
    if n > len(multi):
        raise ValueError(f"asked for {n} Layer-1 pairs but only {len(multi)} multi-record clusters")
    chosen = rng.choice(len(multi), size=n, replace=False)
    rows = []
    for i in sorted(chosen):
        cid = multi[i]
        members = full.index[full.cluster_id == cid]
        prim = [p for p in members if full.at[p, "is_primary"]]
        if len(prim) != 1:
            raise ValueError(f"cluster {cid} has {len(prim)} primary records")
        others = sorted((p for p in members if p != prim[0]),
                        key=lambda p: f"{full.at[p, 'source']}:{full.at[p, 'source_id']}")
        b = others[int(rng.integers(len(others)))]
        basis = "doi" if doi_cluster_id(full.at[b, "doi"]) == cid else "title_year"
        rows.append({"cluster_id": cid, "a_pos": int(prim[0]), "b_pos": int(b), "match_basis": basis})
    return pd.DataFrame(rows)


def sample_judged_pairs(judged: pd.DataFrame, quotas: dict, rng: np.random.Generator) -> pd.DataFrame:
    """Up to quotas[v] pairs per judge_verdict v, without replacement (all of them
    when fewer are available). Keeps judged's columns and order of first appearance."""
    parts = []
    for verdict, k in quotas.items():
        pool = judged.index[judged.judge_verdict == verdict]
        take = pool if len(pool) <= k else pool[np.sort(rng.choice(len(pool), size=k, replace=False))]
        parts.append(judged.loc[take])
    return pd.concat(parts)


def assemble(l1: pd.DataFrame, l3: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Stack the two strata, shuffle them together and randomise the A/B side.

    l1 / l3: one row per pair with columns a_* and b_* for every field in FIELDS
    plus a_ref, b_ref and the key columns (stratum, layer, system_verdict, ...).
    Returns the pairs in workbook order with n = 1..N and a 'swapped' flag."""
    pairs = pd.concat([l1, l3], ignore_index=True)
    pairs = pairs.iloc[rng.permutation(len(pairs))].reset_index(drop=True)
    swap = rng.random(len(pairs)) < 0.5
    for f in FIELDS + ["ref"]:
        a, b = pairs[f"a_{f}"].copy(), pairs[f"b_{f}"].copy()
        pairs.loc[swap, f"a_{f}"] = b[swap]
        pairs.loc[swap, f"b_{f}"] = a[swap]
    pairs["swapped"] = swap
    pairs.insert(0, "n", range(1, len(pairs) + 1))
    return pairs


def record_fields(rec: dict, prefix: str) -> dict:
    """Display values of one record under keys prefix_<field>."""
    year = rec.get("year")
    return {
        f"{prefix}_title": clip(rec.get("title"), 10_000),
        f"{prefix}_authors": format_authors(rec.get("authors")),
        f"{prefix}_year": None if year is None or pd.isna(year) else int(year),
        f"{prefix}_venue": clip(rec.get("venue"), 10_000),
        f"{prefix}_doi": rec.get("doi") or "",
        f"{prefix}_source": SOURCE_LABELS.get(rec.get("source"), rec.get("source") or ""),
        f"{prefix}_abstract": clip(rec.get("abstract")),
        f"{prefix}_ref": f"{rec.get('source')}:{rec.get('source_id')}",
    }


DISPLAYED = ["title", "authors", "year", "venue", "doi", "abstract", "source"]
RAW_KEY = ["source", "source_id", "title"]


def _same(x, y) -> bool:
    """Equality of two field values treating None/NaN/'' as equal and arrays by content."""
    def norm(v):
        if v is None or (isinstance(v, float) and v != v) or (isinstance(v, str) and v == ""):
            return None
        return list(v) if isinstance(v, (list, tuple, np.ndarray)) else v
    return norm(x) == norm(y)


class RawRecords:
    """Pre-merge field values of every identified record, looked up by
    (source, source_id, title). biofloc.parquet carries no cluster_id, so when
    a key is repeated (SciELO exports the same identifier twice) the candidate
    whose displayed fields equal the record's is used; otherwise the first.
    Counts records not found (shown with their current values: fallbacks) and
    repeated keys resolved by taking the first candidate (ambiguous)."""

    def __init__(self, raw: pd.DataFrame):
        self.groups = {k: g.to_dict("records") for k, g in raw.groupby(RAW_KEY, sort=False, dropna=False)}
        self.fallbacks: list[str] = []
        self.ambiguous: list[str] = []
        self.changed: list[str] = []

    def displayed(self, rec: dict) -> dict:
        """rec with its DISPLAYED fields replaced by the pre-merge values."""
        ref = f"{rec.get('source')}:{rec.get('source_id')}"
        cands = self.groups.get(tuple(rec.get(k) for k in RAW_KEY))
        if not cands:
            self.fallbacks.append(ref)
            return rec
        exact = [c for c in cands if all(_same(c.get(f), rec.get(f)) for f in DISPLAYED)]
        pick = (exact or cands)[0]
        if not exact:
            self.changed.append(ref)
            if len({repr([c.get(f) for f in DISPLAYED]) for c in cands}) > 1:
                self.ambiguous.append(ref)
        return {**rec, **{f: pick.get(f) for f in DISPLAYED}}


# ---------------------------------------------------------------- data
def load_pairs(seed: int = SEED, raw: RawRecords | None = None) -> pd.DataFrame:
    """Sampled pairs in workbook order. Displayed fields come from the pre-merge
    records of biofloc.parquet (each record as exported by its database)."""
    raw = raw or RawRecords(pd.read_parquet(DATA / f"{CORPUS}.parquet"))
    rng = np.random.default_rng(seed)
    full = pd.read_parquet(DATA / f"{CORPUS}_dedup_full.parquet").reset_index(drop=True)
    l1_idx = sample_layer1_pairs(full, N_LAYER1, rng)
    recs = full.to_dict("records")
    l1 = pd.DataFrame([{**record_fields(raw.displayed(recs[r.a_pos]), "a"),
                        **record_fields(raw.displayed(recs[r.b_pos]), "b"),
                        "stratum": "L1_merge", "layer": "L1", "system_verdict": "DUPLICATE",
                        "match_basis": r.match_basis, "cosine": np.nan, "judge_confidence": np.nan,
                        "a_cluster": r.cluster_id, "b_cluster": r.cluster_id}
                       for r in l1_idx.itertuples()])

    judged = pd.DataFrame(json.loads(
        (DATA / f"{CORPUS}_dedup_primary_near_dup_pairs_judged.json").read_text()))
    picked = sample_judged_pairs(judged, QUOTAS, rng)
    prim = (pd.read_parquet(DATA / f"{CORPUS}_dedup_primary.parquet")
              .drop_duplicates("cluster_id").set_index("cluster_id"))
    rows = []
    for p in picked.itertuples():
        a = {**prim.loc[p.a_cluster].to_dict(), "cluster_id": p.a_cluster}
        b = {**prim.loc[p.b_cluster].to_dict(), "cluster_id": p.b_cluster}
        rows.append({**record_fields(raw.displayed(a), "a"), **record_fields(raw.displayed(b), "b"),
                     "stratum": f"L3_{p.judge_verdict}", "layer": "L2/L3",
                     "system_verdict": p.judge_verdict, "match_basis": "embedding",
                     "cosine": float(p.score), "judge_confidence": int(p.judge_confidence),
                     "a_cluster": p.a_cluster, "b_cluster": p.b_cluster})
    return assemble(l1, pd.DataFrame(rows), rng)


KEY_COLUMNS = ["n", "stratum", "layer", "system_verdict", "match_basis", "cosine", "judge_confidence",
               "a_ref", "b_ref", "a_cluster", "b_cluster", "swapped", "a_title", "b_title"]


# ---------------------------------------------------------------- workbook
def write_workbook(pairs: pd.DataFrame, out: Path) -> None:
    wb = Workbook()
    last = len(pairs) + 1
    ins = add_instructions_sheet(wb, INSTRUCTIONS,
                                 counter_formula(f"Pares!B2:B{last}", len(pairs), "Pares decididos"))
    ws = wb.create_sheet("Pares")
    head = ["N°", "Decisión"]
    cols = []
    for f in FIELDS:
        for side in ("A", "B"):
            head.append(f"{FIELD_LABELS[f]} {side}")
            cols.append(f"{side.lower()}_{f}")
    head.append("Comentario")
    ws.append(head)
    for row in pairs.itertuples(index=False):
        r = row._asdict()
        ws.append([r["n"], None] + [r[c] for c in cols] + [None])

    fill_dec = PatternFill("solid", fgColor="FFF2CC")
    fill_side = {"A": PatternFill("solid", fgColor="DDEBF7"), "B": PatternFill("solid", fgColor="E2EFDA")}
    ws.column_dimensions["A"].width = 6
    ws.column_dimensions["B"].width = 20
    for j, c in enumerate(cols, start=3):
        letter = ws.cell(row=1, column=j).column_letter
        ws.column_dimensions[letter].width = FIELD_WIDTHS[c[2:]]
        ws.cell(row=1, column=j).fill = fill_side[c[0].upper()]
    comment_col = ws.cell(row=1, column=len(head)).column_letter
    ws.column_dimensions[comment_col].width = 30
    for cell in ws[1]:
        cell.font = Font(bold=True)
        cell.alignment = Alignment(wrap_text=True, vertical="top")
    for row in ws.iter_rows(min_row=2):
        for cell in row:
            cell.alignment = Alignment(wrap_text=True, vertical="top")
        row[1].fill = fill_dec
        row[-1].fill = fill_dec
    ws.freeze_panes = "C2"
    ws.print_title_rows = "1:1"
    fit_to_width(ws)
    ws.auto_filter.ref = f"A1:{comment_col}{last}"
    add_list_validation(ws, DECISIONS, f"B2:B{last}",
                        "Elija Mismo trabajo, Trabajos distintos o No se puede determinar.")

    wb.active = wb.index(ins)
    out.parent.mkdir(parents=True, exist_ok=True)
    wb.save(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True, help="workbook to send to the reviewer")
    ap.add_argument("--key", type=Path, default=DATA / "dedup_audit_key.csv")
    args = ap.parse_args()
    raw = RawRecords(pd.read_parquet(DATA / f"{CORPUS}.parquet"))
    pairs = load_pairs(raw=raw)
    write_workbook(pairs, args.out)
    print(f"[raw] pre-merge lookup: fallbacks={len(raw.fallbacks)} {raw.fallbacks}  "
          f"ambiguous={len(raw.ambiguous)} {raw.ambiguous}  "
          f"records whose shown fields differ from the merged values={len(raw.changed)} {raw.changed}")
    pairs[KEY_COLUMNS].to_csv(args.key, index=False)
    print(f"[workbook] {args.out}  pairs={len(pairs)}  key -> {args.key}")
    print(pairs.stratum.value_counts().sort_index().to_string())


if __name__ == "__main__":
    main()
