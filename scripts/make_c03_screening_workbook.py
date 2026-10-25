"""Workbook for the full manual title/abstract screening of cluster C03.

Reviewer 2 (comment 8) asked for a full manual screening of cluster C03
(applied biofloc aquaculture), in which every record was advanced by the
pipeline (INCLUDE or HUMAN_REVIEW). This script writes the 141 records of
the cluster (eligible_clusters_assignment.parquet, cluster == 3) to a blind
workbook with the same columns, decision scale and criteria sheet as the
second-reviewer workbook (scripts/make_second_reviewer_workbook.py):

- rows are shuffled with a fixed seed and numbered; the number -> record_id
  key, with the pipeline decision in the specificity and the sensitivity
  operating modes, is written to a separate file the reviewer must not see;
- no pipeline output is shown.

Run:  .venv/bin/python scripts/make_c03_screening_workbook.py --out <file.xlsx>
Out:  <file.xlsx> (to send) and data/processed/c03_screening_key.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
from openpyxl import Workbook

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_second_reviewer_workbook import (DATA, add_criteria_sheet,  # noqa: E402
                                           add_instructions_sheet, corpus_text, counter_formula,
                                           write_records_sheet)
from operating_modes_recall import sensitivity_decision  # noqa: E402

SEED = 20261026
CLUSTER = 3

INSTRUCTIONS = [
    ("Cribado manual completo del clúster C03 (biofloc aplicado)", "title"),
    ("", None),
    ("Objetivo", "head"),
    ("Cribar manualmente, por título y resumen, todos los registros (141) del clúster temático C03, que "
     "agrupa los trabajos de biofloc aplicado a la acuicultura. Para cada registro, decida si debe pasar "
     "a lectura de texto completo en una revisión sobre el modelado matemático de flóculos biológicos "
     "(biofloc, lodos activados, perifiton, agregados bacterianos) en sistemas acuáticos. Muchos "
     "trabajos de este clúster son experimentales; la pregunta clave es si contienen un modelo "
     "matemático formal (criterio I2) o solo estadística descriptiva o comparación de tratamientos "
     "(criterio E2).", None),
    ("", None),
    ("Condiciones (importante)", "head"),
    ("1. Trabajo ciego: no consulte las decisiones del algoritmo.", None),
    ("2. Decida solo con lo que ve en la hoja «Registros»: título, resumen, palabras clave, revista, "
     "año y DOI. No busque ni lea el texto completo.", None),
    ("3. Aplique los criterios de la hoja «Criterios», que son los mismos que usó el algoritmo y que se "
     "usaron en la validación ciega del cribado.", None),
    ("4. El orden de los registros es aleatorio y no significa nada.", None),
    ("5. Cerca de la mitad de los registros (sobre todo de SciELO) no tienen resumen en las bases de "
     "datos. Decida con el título, las palabras clave y la revista; si no alcanza, elija Dudoso.", None),
    ("", None),
    ("Cómo decidir (columna «Decisión», obligatoria)", "head"),
    ("Incluir: cumple I1, I2 e I3, y ninguno de los criterios de exclusión E1–E4.", None),
    ("Excluir: cumple al menos un criterio de exclusión, o claramente no cumple I1, I2 o I3.", None),
    ("Dudoso: la información no alcanza para decidir (resumen muy breve, tipo de modelo poco claro, "
     "contexto mixto). Un registro dudoso pasaría a texto completo.", None),
    ("I4 (tipo de documento): si no se puede verificar con la información disponible, no excluya "
     "por I4.", None),
    ("", None),
    ("Columnas opcionales", "head"),
    ("«Motivo principal de exclusión»: si eligió Excluir, indique el criterio que más pesó.", None),
    ("«Comentario»: cualquier observación útil (por ejemplo, por qué fue dudoso).", None),
    ("", None),
    ("Al terminar", "head"),
    ("Guarde el archivo con el mismo nombre y devuélvalo. Puede trabajar en varias sesiones; la fila "
     "de abajo cuenta cuántos registros lleva decididos.", None),
]


def shuffle_records(ids: pd.DataFrame, seed: int = SEED) -> pd.DataFrame:
    """Sort by record_id (so the order does not depend on the input order),
    shuffle with seed and number the rows n = 1..N."""
    out = ids.sort_values("record_id").reset_index(drop=True)
    out = out.sample(frac=1, random_state=seed).reset_index(drop=True)
    out.insert(0, "n", range(1, len(out) + 1))
    return out


def load_cluster(cluster: int = CLUSTER) -> pd.DataFrame:
    """Records of the cluster with their text and both operating-mode decisions."""
    asg = pd.read_parquet(DATA / "eligible_clusters_assignment.parquet", columns=["record_id", "cluster"])
    ids = asg.loc[asg.cluster == cluster, ["record_id"]].drop_duplicates()
    dec = pd.read_parquet(DATA / "biofloc_final_decisions.parquet",
                          columns=["record_id", "final_decision"]
                          + [f"E{i}_answer" for i in range(1, 6)])
    e2 = pd.read_parquet(DATA / "e2_v2_answers.parquet", columns=["record_id", "E2_v2"])
    d = (ids.merge(dec.drop_duplicates("record_id"), on="record_id", how="left")
            .merge(e2.drop_duplicates("record_id"), on="record_id", how="left"))
    d["sensitivity_decision"] = sensitivity_decision(d).to_numpy()
    d = d.rename(columns={"final_decision": "pipeline_decision"})
    return corpus_text(d[["record_id", "pipeline_decision", "sensitivity_decision"]])


def build(out: Path) -> pd.DataFrame:
    sample = shuffle_records(load_cluster())
    wb = Workbook()
    ins = add_instructions_sheet(wb, INSTRUCTIONS,
                                 counter_formula(f"Registros!H2:H{len(sample) + 1}", len(sample)))
    add_criteria_sheet(wb)
    write_records_sheet(wb, sample)
    wb.active = wb.index(ins)
    out.parent.mkdir(parents=True, exist_ok=True)
    wb.save(out)
    return sample


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True, help="workbook to send to the reviewer")
    ap.add_argument("--key", type=Path, default=DATA / "c03_screening_key.csv")
    args = ap.parse_args()
    sample = build(args.out)
    sample[["n", "record_id", "pipeline_decision", "sensitivity_decision", "title"]].to_csv(
        args.key, index=False)
    print(f"[workbook] {args.out}  records={len(sample)}  key -> {args.key}")
    print(pd.crosstab(sample.pipeline_decision, sample.sensitivity_decision).to_string())


if __name__ == "__main__":
    main()
