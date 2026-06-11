"""Extract modeling-method keyword presence from downloaded fulltext PDFs.

For each PDF in data/fulltext/, extracts text (first 25 pages to focus
on body and avoid reference bias) and computes binary presence vectors
for 9 modeling-method families plus 4 application-context families.
Outputs a parquet for downstream aggregation.
"""
from __future__ import annotations

import re
import warnings
from pathlib import Path

import pandas as pd
from pypdf import PdfReader

warnings.filterwarnings("ignore")


METHOD_PATTERNS = {
    "ODE": re.compile(
        r"\b(ordinary\s+differential\s+equation|"
        r"\bODE[s]?\b|"
        r"kinetic\s+model|monod\s+kinetics?|"
        r"michaelis[-\s]menten|haldane\s+kinetics?)\b",
        re.IGNORECASE,
    ),
    "PDE": re.compile(
        r"\b(partial\s+differential\s+equation|"
        r"\bPDE[s]?\b|"
        r"advection[-\s]diffusion|"
        r"transport\s+equation|"
        r"navier[-\s]stokes)\b",
        re.IGNORECASE,
    ),
    "ABM": re.compile(
        r"\b(agent[-\s]based\s+model|individual[-\s]based\s+model|"
        r"\bABM\b|\bIBM\b)\b",
        re.IGNORECASE,
    ),
    "PBM": re.compile(
        r"\b(population\s+balance|"
        r"breakage\s+(kernel|rate)|"
        r"aggregation\s+kernel|"
        r"coagulation\s+kernel|"
        r"floc(s|culation)?\s+size\s+distribution)\b",
        re.IGNORECASE,
    ),
    "CFD": re.compile(
        r"\b(computational\s+fluid\s+dynamics|"
        r"\bCFD\b|"
        r"fluent|openfoam|ansys|"
        r"finite\s+volume\s+method)\b",
        re.IGNORECASE,
    ),
    "ML_ANN": re.compile(
        r"\b(machine\s+learning|"
        r"neural\s+network|artificial\s+neural|"
        r"\bANN[s]?\b|deep\s+learning|"
        r"random\s+forest|support\s+vector\s+machine|"
        r"\bSVM\b|gradient\s+boosting|XGBoost)\b",
        re.IGNORECASE,
    ),
    "PINN": re.compile(
        r"\b(physics[-\s]informed\s+neural|\bPINN[s]?\b)\b",
        re.IGNORECASE,
    ),
    "Stochastic": re.compile(
        r"\b(stochastic\s+(model|differential|process)|"
        r"langevin|monte\s+carlo|"
        r"\bSDE[s]?\b|fokker[-\s]planck)\b",
        re.IGNORECASE,
    ),
    "ASM": re.compile(
        r"\b(activated\s+sludge\s+model|"
        r"\bASM[123]\b|\bASM\b)\b",
        re.IGNORECASE,
    ),
}

APPLICATION_PATTERNS = {
    "aquaculture": re.compile(
        r"\b(aquaculture|shrimp|tilapia|vannamei|"
        r"litopenaeus|biofloc\s+technolog)\b",
        re.IGNORECASE,
    ),
    "wastewater": re.compile(
        r"\b(wastewater\s+treatment|sewage|"
        r"municipal\s+effluent|WWTP\b)\b",
        re.IGNORECASE,
    ),
    "biofilm": re.compile(
        r"\b(biofilm|periphyton|moving[-\s]bed\s+biofilm|MBBR\b)\b",
        re.IGNORECASE,
    ),
    "granular_sludge": re.compile(
        r"\b(granular\s+sludge|aerobic\s+granul|anammox\s+granul)\b",
        re.IGNORECASE,
    ),
}


def extract_text(pdf_path: Path, max_pages: int = 25) -> tuple[str, int]:
    """Returns (text, num_pages_read). Errors return ('', 0)."""
    try:
        r = PdfReader(str(pdf_path))
        n_total = len(r.pages)
        n_read = min(max_pages, n_total)
        parts = []
        for i in range(n_read):
            try:
                t = r.pages[i].extract_text()
                if t:
                    parts.append(t)
            except Exception:
                continue
        return ("\n".join(parts), n_total)
    except Exception:
        return ("", 0)


def main() -> None:
    in_dir = Path("data/fulltext")
    out_path = Path("data/processed/fulltext_methods.parquet")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    pdfs = sorted(in_dir.glob("*.pdf"))
    print(f"Procesando {len(pdfs)} PDFs...")

    rows = []
    errors = 0
    for i, p in enumerate(pdfs, 1):
        rid = p.stem
        text, n_pages = extract_text(p)
        if not text:
            errors += 1
            row = {"record_id": rid, "pdf_pages": n_pages, "text_chars": 0,
                   "extract_ok": False}
            for cat in list(METHOD_PATTERNS) + list(APPLICATION_PATTERNS):
                row[f"has_{cat}"] = False
            rows.append(row)
            continue

        row = {
            "record_id": rid,
            "pdf_pages": n_pages,
            "text_chars": len(text),
            "extract_ok": True,
        }
        for cat, pat in METHOD_PATTERNS.items():
            row[f"has_{cat}"] = bool(pat.search(text))
        for cat, pat in APPLICATION_PATTERNS.items():
            row[f"has_{cat}"] = bool(pat.search(text))
        rows.append(row)

        if i % 30 == 0:
            print(f"  {i}/{len(pdfs)} done")

    df = pd.DataFrame(rows)
    df.to_parquet(out_path, index=False)
    print(f"\n  saved → {out_path}")
    print(f"  PDFs procesados: {len(df)}  errors: {errors}")
    print()
    print("=== Distribución de métodos (presence ratio sobre PDFs OK) ===")
    ok = df[df.extract_ok]
    print(f"PDFs con texto extraído: {len(ok)}/{len(df)}")
    for cat in list(METHOD_PATTERNS):
        n = ok[f"has_{cat}"].sum()
        print(f"  {cat:12} : {n:4} ({n/len(ok)*100:.1f}%)")
    print()
    print("=== Distribución de contextos de aplicación ===")
    for cat in list(APPLICATION_PATTERNS):
        n = ok[f"has_{cat}"].sum()
        print(f"  {cat:18} : {n:4} ({n/len(ok)*100:.1f}%)")


if __name__ == "__main__":
    main()
