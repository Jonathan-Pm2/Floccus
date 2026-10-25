# Data availability

This archive publishes everything needed to understand, rerun and check the
Floccus pipeline except the bibliographic content itself, which belongs to the
databases and publishers it came from.

## Published

| component | location |
|---|---|
| pipeline source code | `src/biofloc_review/` |
| analysis, workbook and export scripts | `scripts/` |
| unit tests | `tests/` |
| canonical record schema (`PaperRecord`) | `src/biofloc_review/schema.py` |
| screening criteria, prompts and decision rule | `src/biofloc_review/screening/` (`criteria.py`, `prompts.py`, `decision.py`) |
| per-database search strings, fields, restrictions and dates | `queries/` |
| configuration: dependencies and lockfile, environment template | `pyproject.toml`, `uv.lock`, `env.example` |
| derived decision tables and PRISMA counts | `release_data/` (described in `release_data/README.md`) |
| citation metadata | `CITATION.cff`, `.zenodo.json` |

The derived decision tables hold identifiers (pipeline `record_id`, source
database, source identifier, DOI, publication year, declared document type and
language) and decisions: per-criterion verdicts of both screening layers, the
final decision in both operating modes, cluster assignments, the labels of the
two human reviewers and of the two baselines on the validation sample, the
deduplication audit labels and the C03 manual screening labels. Reviewers are
identified only as `reviewer_1` and `reviewer_2`.

## Not published, and why

| not published | why |
|---|---|
| raw exports from Scopus, Web of Science, IEEE Xplore and SciELO, and the ProQuest EAASD export of the supplementary search | the licence terms of these databases do not allow redistribution; SciELO is open access but its export is withheld for consistency. The searches can be rerun from `queries/` |
| titles, abstracts, keywords, author lists and full text | copyrighted by the publishers. They are the content of every intermediate file, so none of `data/processed/` is published |
| intermediate Parquet/JSON artefacts (`data/processed/`) | they carry the fields above alongside the decisions; the decisions are published separately in `release_data/` |
| LLM evidence quotes and reasoning | they quote abstracts verbatim |
| embedding vectors and Qdrant snapshots | dense representations of the abstracts, from which content could be partially reconstructed |
| downloaded full-text PDFs | publisher copyright |
| returned human-labelling workbooks | they display titles and abstracts and contain the reviewers' free-text comments; their decisions are published in `release_data/` |
| credentials (`.env`) | private: Ollama server URL and token, contact e-mail for the OpenAlex and Europe PMC APIs. `env.example` lists the variables |
| manuscript sources | the article is distributed by the journal, not by this archive |

`data/` is excluded by `.gitignore`, and the Zenodo archive is built only from
files tracked by git (`scripts/pack_zenodo.sh`), so none of the above can
enter it.

## Reproducing the results

1. Rerun the searches in `queries/` and place the exports in the parent
   directory of the repository.
2. Install and run the pipeline as described in `README.md`.
3. Compare the resulting decisions and counts with `release_data/`. LLM
   decisions are not bit-for-bit deterministic: repeated runs at temperature 0
   changed 3–5% of decisions on a calibration sample, and the database
   contents change over time.

The analyses that need only decisions and labels (recall and WSS, agreement,
baselines comparison, audit precision) can be checked directly against the
tables in `release_data/`.
