# Derived decision tables

These tables are the derived data that accompany the manuscript. They contain
**identifiers and decisions only**: no titles, abstracts, keywords, author
names, full text, embeddings, or LLM evidence and reasoning (see
[`docs/DATA.md`](../docs/DATA.md) for why). All files are UTF-8; CSV files use
`,` as separator and leave missing values empty.

They are regenerated from the pipeline artefacts (`data/processed/`, not
published) and the three returned human-labelling workbooks by:

```bash
.venv/bin/python scripts/export_release_data.py \
    --reviewer2-workbook <returned second-reviewer workbook.xlsx> \
    --dedup-workbook <returned deduplication audit workbook.xlsx> \
    --c03-workbook <returned C03 screening workbook.xlsx>
```

The script checks the counts reported in the manuscript and exits with status
1 on any mismatch.

**Round 2 (`--round2`).** After the first review round two corrections were
applied (`scripts/build_round2_corpus.py`): the 98 IEEE Xplore records verified
as duplicates of Web of Science / Scopus records (identical title or DOI) were
merged, keeping one record per pair (validation-sample member > clustered
record > record with DOI > IEEE record), and the 31 records excluded solely by
the retired language criterion E5 were re-screened without it (11 INCLUDE, 15
HUMAN_REVIEW, 5 EXCLUDE; 23 further records excluded by E5 + E2 were re-asked
the corrected E2 for the sensitivity mode). `screening_decisions.csv` keeps all
17,063 round-1 records: `round2_change` is `unchanged`, `ieee_merged_away` (98
records, `merged_into` = the retained record, excluded from every count) or
`e5_rescreened` (54 records). The round-2 corpus has 16,965 records, 4,883
eligible (1,395 INCLUDE); `eligible_clusters.csv` and `prisma_counts.json` are
computed on it.

| file | rows | content |
|---|---:|---|
| `screening_decisions.csv` | 17,063 | every deduplicated record: criterion verdicts and decisions in both operating modes; round-2 status |
| `eligible_clusters.csv` | 4,883 | eligible records (INCLUDE or HUMAN_REVIEW) of the round-2 corpus and their thematic cluster |
| `validation_sample.csv` | 465 | blind validation sample: design, pipeline decisions, two human reviewers, two baselines |
| `dedup_audit_pairs.csv` | 110 | manual audit of deduplication pairs |
| `c03_manual_screening.csv` | 141 | full manual screening of cluster C03 |
| `prisma_counts.json` | — | PRISMA 2020 flow counts |

## Identifier columns (shared)

| column | meaning |
|---|---|
| `record_id` | stable record identifier used throughout the pipeline: UUID5 (URL namespace) of `"<source>:<source_id>"` |
| `source` | database the record was kept from after deduplication: `scopus`, `wos`, `ieee`, `scielo` |
| `source_id` | identifier in that database: Scopus EID, Web of Science accession number, SciELO PID, and for IEEE Xplore the `pdf:` link (the export's `Document Identifier` is not per-paper) or `titlehash:` fallback. Withheld (empty) for one IEEE record whose export field held free text |
| `doi` | DOI as exported (lower-cased), empty when absent |
| `year` | publication year |

## `screening_decisions.csv`

| column | meaning |
|---|---|
| `dedup_cluster_id` | deduplication cluster key (truncated SHA-1); records merged as duplicates share it |
| `document_type`, `language` | as declared by the source database (empty when not exported) |
| `l1_I1` … `l1_D3` | Layer-1 verdict per criterion (`YES` / `NO` / `UNCLEAR`); criteria are defined in `src/biofloc_review/screening/criteria.py` |
| `l1_decision` | Layer-1 decision under the asymmetric rule |
| `l2_I1` … `l2_D3`, `l2_decision` | Layer-2 (second, enriched-metadata pass) verdicts and decision; only for the 4,245 Layer-1 HUMAN_REVIEW records (4,225 of the archived run plus the 20 E5 re-screened records routed to Layer 2), empty otherwise |
| `final_decision` | consolidated decision, specificity (published) mode: `INCLUDE` / `HUMAN_REVIEW` / `EXCLUDE` |
| `decision_layer` | layer that produced `final_decision` (`L1` or `L2`) |
| `decision_reason` | rule that fired: `all_inclusion_met`, `excluded_by:<criteria>`, `inclusion_not_met:<criteria>`, `routed_by:<criteria>`, `exclusion_unclear:<criteria>` |
| `e2_sensitivity` | E2 answer from the sensitivity-mode prompt (reasoning first, positive phrasing, mapped back to E2 polarity); asked only for the 9,096 records whose decision depends on E2 (9,047 of the archived run plus 49 E5 re-screened records) |
| `sensitivity_decision` | decision in the sensitivity operating mode |
| `eligible`, `eligible_sensitivity` | record advances to eligibility (INCLUDE or HUMAN_REVIEW) in each mode; always `False` for merged-away records |
| `round2_change` | `unchanged`, `ieee_merged_away` (verified IEEE duplicate of another record, not counted) or `e5_rescreened` (re-screened without the retired language criterion E5) |
| `merged_into` | for `ieee_merged_away` records, the `record_id` of the retained twin |

The `E5` columns hold a deterministic language check (`determinista sobre
rec.language`) that was part of the archived run; they are exported as stored
so that every decision can be traced to the verdicts that produced it.

## `eligible_clusters.csv`

| column | meaning |
|---|---|
| `final_decision` | `INCLUDE` or `HUMAN_REVIEW` |
| `cluster` | UMAP + HDBSCAN cluster (0–24); `-1` = no cluster (noise) |
| `cluster_label` | `C00` … `C24` as in the manuscript, `noise` for `-1` |
| `assigned_later` | `True` for the records whose embeddings were computed after the clustering and assigned to the existing partition: 199 IEEE Xplore records (`scripts/assign_ieee_clusters.py`; 201 before the round-2 merges) and the 26 records made eligible by the E5 re-screen (`scripts/rescreen_e5.py`) |
| `assignment_method` | `umap_hdbscan_fit` (4,658 records: the 4,682 of the fitted partition minus 24 merged away) or `umap_transform_hdbscan_approximate_predict` (225) |
| `membership_strength` | HDBSCAN membership strength of the later assignments, empty otherwise |

## `validation_sample.csv`

| column | meaning |
|---|---|
| `review_id` | blind identifier shown to the reviewers |
| `stratum` | sampling cell `<source>\|<final_decision>` |
| `stratum_population` | records of the corpus in that cell |
| `sampling_weight` | `stratum_population / sampled records in the cell`, used for corpus-level estimates |
| `pipeline_decision`, `pipeline_decision_sensitivity` | pipeline decision in each operating mode |
| `reviewer_1` | first reviewer, binary: `advance` / `exclude` |
| `reviewer_2` | second reviewer, three levels: `include` / `uncertain` / `exclude` |
| `reviewer_2_reason_code` | main exclusion reason given by reviewer 2: `E1`–`E4` (exclusion criterion), `not_I1`–`not_I4` (inclusion criterion not met), `other` |
| `baseline_keyword`, `baseline_keyword_strict` | keyword-rule baselines (`scripts/baseline_keyword.py`): `advance` / `exclude` |
| `baseline_single_prompt` | single-prompt LLM baseline (`scripts/baseline_single_prompt.py`): `INCLUDE` / `EXCLUDE` / `UNCLEAR` |

## `dedup_audit_pairs.csv`

| column | meaning |
|---|---|
| `pair_id` | item number in the audit workbook |
| `stratum` | `L1_merge` (deterministic merge), `L3_DUPLICATE`, `L3_NOT_DUPLICATE`, `L3_UNCLEAR` (LLM-judge verdict) |
| `layer` | `L1` or `L2/L3` |
| `match_basis` | how the pair was formed (`doi`, `title_year`, `embedding`) |
| `cosine` | embedding cosine similarity (Layer-2 pairs) |
| `judge_verdict`, `judge_confidence` | Layer-3 LLM judge verdict and confidence (0–10) |
| `a_ref`, `b_ref` | the two records as `<source>:<source_id>` |
| `a_dedup_cluster_id`, `b_dedup_cluster_id` | their Layer-1 cluster keys |
| `audit_label` | human audit: `same_work` / `different_works` / `undetermined` |

## `c03_manual_screening.csv`

| column | meaning |
|---|---|
| `item` | item number in the screening workbook |
| `pipeline_decision`, `pipeline_decision_sensitivity` | pipeline decision in each mode |
| `label` | human decision: `include` / `uncertain` / `exclude` |
| `reason_code` | main exclusion reason (same codes as `reviewer_2_reason_code`) |

## `prisma_counts.json`

Identification by source, deduplication, Layer-1 and Layer-2 screening counts
and the final counts in both operating modes. Identification and
deduplication counts are copied from the pipeline's PRISMA output; the rest
are recomputed from `screening_decisions.csv`.
