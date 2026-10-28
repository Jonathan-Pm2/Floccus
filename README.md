# Floccus

Floccus is a reproducible, PRISMA 2020-compliant pipeline for LLM-assisted
literature screening, built for a systematic map of the mathematical modelling
of biological flocs (biofloc, activated sludge, periphyton) in aquaculture and
water treatment. It ingests exports from four bibliographic databases,
deduplicates them in three layers, screens titles and abstracts against eleven
binary criteria with a locally hosted LLM, and maps the eligible corpus into
thematic clusters. Every decision keeps an audit trail.

It is the code archive of the article

> **From Corpus to Canon: An LLM-Assisted Screening Algorithm and a
> Multi-Scale Unification of Biological-Floc Modelling Equations.**
> Under review at *Algorithms* (MDPI).

Version: **0.3.0** (round-2 revision; the manuscript as first submitted cites
v0.2.0, see [Changelog](#changelog)).

## Pipeline

```
Scopus · Web of Science · IEEE Xplore · SciELO        (22,056 records)
  → unified record schema (PaperRecord) + OpenAlex enrichment
  → deduplication  L1 deterministic (DOI / normalised title + year)
                   L2 dense embeddings (qwen3-embedding:8b, Qdrant, cosine ≥ 0.93)
                   L3 LLM judge (qwen3:14b) + union-find merge   (17,063 records)
  → screening      Layer 1: 11 binary criteria, asymmetric AND/OR rule (LangGraph)
                   Layer 2: blind second pass on HUMAN_REVIEW with enriched metadata
                   (OpenAlex topics, Europe PMC MeSH)
  → 1,385 INCLUDE + 3,498 HUMAN_REVIEW = 4,883 eligible
  → thematic map   UMAP + HDBSCAN, 25 clusters
```

The counts above are those of the archived run (round 1). The round-2 corpus
(v0.3.0, see [Round 2](#round-2-corrections-v030)) has 16,965 unique records,
1,395 INCLUDE + 3,488 HUMAN_REVIEW = 4,883 eligible.

Stack: Python 3.12, uv, Pydantic, LangGraph, Qdrant (Docker), Ollama.

## Installation

Requirements: [uv](https://docs.astral.sh/uv/), Docker (for Qdrant) and access
to an Ollama server with `qwen3-embedding:8b` and `qwen3:14b`.

```bash
uv sync
docker run -d --name qdrant-biofloc -p 6333:6333 -p 6334:6334 \
  -v "$(pwd)/qdrant_storage:/qdrant/storage" qdrant/qdrant:latest
cp env.example .env      # set OLLAMA_URL, OLLAMA_TOKEN, OPENALEX_EMAIL
uv run python -m pytest -q
```

## Running the pipeline

The database exports are read from the parent directory (`--data-dir ..`);
they are not part of this archive (see [`docs/DATA.md`](docs/DATA.md)). Run
the searches in [`queries/`](queries/) to obtain them. Intermediate artefacts
are written to `data/processed/`.

```bash
# 1. ingest + OpenAlex enrichment
uv run biofloc-review --data-dir .. --corpus biofloc
uv run biofloc-review --data-dir .. --corpus biofloc --enrich --resume --rps 0.5

# 2. deduplication: L1 deterministic, L2 embeddings, L3 LLM judge, merge
uv run python -m biofloc_review.run_dedup data/processed/biofloc.parquet
uv run python -m biofloc_review.run_embed data/processed/biofloc_dedup_primary.parquet \
  --corpus biofloc --recreate
uv run python -m biofloc_review.run_judge \
  data/processed/biofloc_dedup_primary_near_dup_pairs.json \
  data/processed/biofloc_dedup_primary.parquet --workers 2
uv run python -m biofloc_review.run_merge data/processed/biofloc_dedup_full.parquet \
  data/processed/biofloc_dedup_primary_near_dup_pairs_judged.json
# SciELO abstracts/DOIs from the article pages (→ _v2), per-paper IEEE Xplore ids (→ _v3)
uv run python -m biofloc_review.run_scielo_scrape data/processed/biofloc_final_primary.parquet --delay 2.0
uv run python -m biofloc_review.run_ieee_fix

# 3. screening: Layer 1, enrichment, Layer 2, consolidation + PRISMA counts
uv run python -m biofloc_review.run_screening \
  data/processed/biofloc_final_primary_v3.parquet --workers 2 --checkpoint-every 25
uv run python -m biofloc_review.run_enrich data/processed/biofloc_screened.parquet \
  --full-parquet data/processed/biofloc_final_primary_v3.parquet --workers 1 --rps 0.5
uv run python -m biofloc_review.run_screening_layer2 \
  data/processed/biofloc_screened.parquet data/processed/biofloc_enriched.parquet \
  --full-parquet data/processed/biofloc_final_primary_v3.parquet --workers 2
uv run python -m biofloc_review.run_consolidate \
  data/processed/biofloc_screened.parquet data/processed/biofloc_layer2_screened.parquet

# 4. blind validation sample (stratified by source × decision)
uv run python -m biofloc_review.run_validation_sample \
  data/processed/biofloc_final_decisions.parquet

# 5. thematic map of the eligible corpus
uv run python -m biofloc_review.run_thematic_clustering \
  data/processed/biofloc_final_decisions.parquet \
  data/processed/biofloc_final_primary_v3.parquet \
  --collection papers_biofloc --min-cluster-size 40
```

Each step is resumable from its output. Commands that call the LLM or the
embedding model need the variables of `.env` in the environment.

## Screening operating modes

The screener runs at two operating points. They share the eleven criteria
(I1–I4, E1–E4, D1–D3) and the asymmetric decision rule, and differ only in how
the model is asked:

| mode | prompt | criterion E2 ("no formal modelling") | effect |
|---|---|---|---|
| `specificity` (default, published) | answer first, then reasoning | asked as written (negative) | fewer records to human review |
| `sensitivity` | reasoning first, answer last | asked positively and inverted | higher recall, more human review |

```bash
uv run python -m biofloc_review.run_screening <input.parquet> --mode sensitivity
# or set FLOCCUS_SCREENING_MODE=sensitivity
```

`scripts/pilot_prompt_v2.py` reproduces the comparison on the validation
sample and `scripts/rescreen_e2_sensitivity_mode.py` the corpus counts
(4,883 vs 7,477 records to eligibility).

## Analyses reported in the article

Each script reads `data/processed/`, writes one JSON or table there, and
documents its inputs in its header. Scripts marked LLM need the Ollama server.

| analysis | script(s) |
|---|---|
| recall, specificity and WSS, stratum-reweighted | `compute_recall_wss.py` |
| operating modes on the validation sample | `operating_modes_recall.py`, `pilot_prompt_v2.py` (LLM), `rescreen_e2_sensitivity_mode.py` (LLM) |
| second reviewer: inter-human and reviewer–pipeline agreement | `make_second_reviewer_workbook.py`, `second_reviewer_agreement.py` |
| baselines: keyword rule and single-prompt LLM | `baseline_keyword.py`, `baseline_single_prompt.py` (LLM), `baselines_compare.py` |
| deduplication statistics | `dedup_audit.py` |
| manual deduplication audit (110 pairs) | `make_dedup_audit_workbook.py`, `dedup_audit_agreement.py` |
| supplementary search in ProQuest EAASD | `supplementary_search_eaasd.py` (LLM) |
| full manual screening of cluster C03 (141 records) | `make_c03_screening_workbook.py`, `c03_screening_agreement.py` |
| C03–C15 bridge score with permutation null model | `bridge_score.py` |
| C03–C15 citation links from Crossref reference lists | `bridge_citations_crossref.py` |
| cluster labels, OpenAlex-topic cross-check, reproducibility | `cluster_labels.py`, `cluster_topic_crosscheck.py`, `cluster_reproducibility.py` |
| assignment of the 201 IEEE Xplore records to clusters | `assign_ieee_clusters.py` (Ollama embeddings) |
| declared and inferred language coverage | `infer_language.py`, `scielo_language_audit.py` |
| database overlap per publisher | `source_coverage_audit.py` |
| modelling methods in full text and abstracts | `extract_methods.py`, `search_advanced_methods.py` |
| export of the published decision tables | `export_release_data.py` |

All scripts live in `scripts/` and run as `.venv/bin/python scripts/<name>.py`.

## Round 2 corrections (v0.3.0)

The first review round surfaced two defects of the archived run. Both are
corrected on top of the archived decision tables; nothing of round 1 is
overwritten, and every script below has a unit test in `tests/`.

1. **IEEE Xplore identifier collision.** The export's `Document Identifier`
   is a content-type label (11 distinct values for 545 records), and it was
   used as the IEEE `source_id`. Layer 1 keys on DOI and title, so it is
   unaffected; in Layers 2–3 the IEEE points overwrote each other in Qdrant
   and never entered the near-duplicate search. `ieee_integrity_check.py`
   re-embeds every IEEE primary under its corrected id and repeats Layers 2–3
   counterfactually: 145 candidate pairs, 124 `DUPLICATE` verdicts by the
   Layer-3 judge, of which 98 are verified by an identical DOI or normalised
   title. Only those 98 are applied.
2. **Retired language criterion E5.** Layer 1 was run while a deterministic
   language criterion E5 was still active; E5 was removed from the protocol
   afterwards without recomputing decisions. Because it matched language names
   by substring (`git show 54995c4^:src/biofloc_review/screening/prefilter.py`),
   Chinese, French and Japanese records passed it and records whose language
   field read `undefined` were flagged. E5 fired on 161 records and was
   the sole exclusion criterion for 31. `rescreen_e5.py` puts those 31 through
   the published pipeline without E5 (Layer 1 rule, enrichment, Layer 2,
   cluster assignment): 11 INCLUDE, 15 HUMAN_REVIEW, 5 EXCLUDE. The 23
   records excluded by E5 + E2 only are re-asked the corrected E2 of the
   sensitivity mode.

`build_round2_corpus.py` applies both corrections and recomputes every
reported count OLD → NEW with the existing analysis scripts (imported). Each
merged group keeps one record, chosen by the retention rule
*validation-sample member > clustered record > record with DOI > IEEE record
> smallest `record_id`*, so no validation-sample record leaves the corpus.
Headline changes: unique records 17,063 → 16,965; eligible 4,883 (unchanged:
+26 −26); INCLUDE 1,385 → 1,395; sensitivity-mode workload 7,477 → 7,466.

| step | script | output (`data/processed/`) |
|---|---|---|
| IEEE integrity check and counterfactual Layers 2–3 (Qdrant; embeddings cached, LLM judge) | `ieee_integrity_check.py` | `ieee_integrity.json`, `ieee_counterfactual_judged.json` |
| E5 re-screen (LLM; resumable, `--workers`, `--e2-workers`, `--rps`, `--skip-llm`, `--skip-clusters`) | `rescreen_e5.py` | `e5_rescreen_results.json`, `e5_rescreen_decisions.parquet` |
| round-2 corpus and all recomputed counts (`--reviewer2-workbook` required; `--quick` = 200 bootstrap draws, `--skip-bridge` skips the Qdrant block) | `build_round2_corpus.py` | `round2_final_decisions.parquet`, `round2_counts.json` |
| WSS = (N − n_adv)/N − (1 − R̂): re-weighted vs exact-workload WSS per reference (after `second_reviewer_agreement.py`) | `wss_reconciliation.py` | `wss_reconciliation.json` |
| cluster stability: 12 UMAP seeds, `min_cluster_size` 24–56, `n_neighbors` 15/50; ARI/AMI vs the published partition, per-cluster best Jaccard | `cluster_stability.py` | `cluster_stability.json` |
| PRISMA flow, cluster table, Sankey and UMAP figure with round-2 counts (`--out-dir`, `--check-round1`) | `render_round2_figures.py` | TikZ/table sources, PDF/PNG in `--out-dir` |
| published tables of the round-2 corpus | `export_release_data.py --round2` | `release_data/` |

```bash
.venv/bin/python scripts/ieee_integrity_check.py
set -a; . ../.env; set +a
.venv/bin/python scripts/rescreen_e5.py --workers 2
.venv/bin/python scripts/build_round2_corpus.py --reviewer2-workbook <returned workbook.xlsx>
.venv/bin/python scripts/wss_reconciliation.py
.venv/bin/python scripts/cluster_stability.py
.venv/bin/python scripts/render_round2_figures.py --out-dir <dir> --check-round1 ''
.venv/bin/python scripts/export_release_data.py --round2 \
  --reviewer2-workbook <...> --dedup-workbook <...> --c03-workbook <...>
```

`ieee_integrity_check.py` and `build_round2_corpus.py` need the Qdrant
container; `ieee_integrity_check.py` (first run) and `rescreen_e5.py` need the
Ollama server. `wss_reconciliation.py` and `cluster_stability.py` read only
archived artefacts.

## Published data and queries

- [`release_data/`](release_data/) — derived decision tables (identifiers and
  decisions only): screening decisions for the 17,063 round-1 records in both
  operating modes with their round-2 status (`round2_change`, `merged_into`),
  cluster assignment of the 4,883 eligible records of the round-2 corpus, the
  465-record validation sample with both reviewers and the baselines, the
  deduplication audit, the C03 manual screening and the PRISMA counts.
- [`queries/`](queries/) — the search strings of each database, with fields,
  restrictions and dates.
- [`docs/DATA.md`](docs/DATA.md) — what is and is not published, and why.

## Repository layout

```
src/biofloc_review/   pipeline package
  sources/            per-database loaders (Scopus, WoS, IEEE Xplore, SciELO, ProQuest), OpenAlex client
  dedup/              Layer 1 deterministic, Layer 2 embeddings, Layer 3 LLM judge, union-find merge
  screening/          criteria, prompts, classifier, decision rule, LangGraph graph, Layer 2, enricher
  run_*.py            command-line entry points of each step
scripts/              analyses, workbooks and release export
tests/                unit tests (no data files or LLM needed)
queries/              search strings
release_data/         published decision tables
```

## Changelog

- **v0.3.0 — round 2.** IEEE Xplore identifier-collision integrity check and
  98 verified duplicate merges; E5 re-screen of the 31 records excluded by
  the retired language criterion alone; round-2 corpus (16,965 unique, 4,883
  eligible, 1,395 INCLUDE) with every reported count recomputed; WSS
  reconciliation; cluster-stability analysis; round-2 figures;
  `release_data/` re-exported with `--round2` (`round2_change`,
  `merged_into`).
- **v0.2.0 — round 1.** Version cited by the submitted manuscript: operating
  modes (specificity / sensitivity), second-reviewer and baseline
  comparisons, deduplication audit, C03 manual screening, bridge score,
  supplementary ProQuest EAASD search, `release_data/` and `queries/`.
- **v0.1.0 — initial.** Pipeline (ingestion, three-layer deduplication,
  two-layer screening, thematic clustering) and first Zenodo deposit.

## Citation

Please cite the article and this software. Software citation metadata is in
[`CITATION.cff`](CITATION.cff); the Zenodo concept DOI, which always resolves
to the latest version, is
[10.5281/zenodo.21926426](https://doi.org/10.5281/zenodo.21926426).

## Licence

Apache License 2.0 — see [`LICENSE`](LICENSE).
