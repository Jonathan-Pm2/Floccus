# Floccus user guide

Floccus applies your systematic review's inclusion and exclusion criteria to
thousands of records with a language model, one criterion at a time, and keeps
every decision traceable for human review.

## In 5 steps

1. **Install** (once).
2. **Configure** the AI model.
3. **Create a project** and write your protocol.
4. **Upload the exports** and **run** the pipeline.
5. **Review** the results and the uncertain records.

## 1. Installation

- Recommended: install [Gentle Shell](https://github.com/Gentleman-Programming/gentle-shell),
  the AI-agent workspace Floccus is developed with.
- Install [uv](https://docs.astral.sh/uv/) and clone the repository:

```bash
git clone https://github.com/Jonathan-Pm2/Floccus.git
cd Floccus
uv sync --all-packages
uv run floccus ui          # opens http://localhost:8501
```

## 2. AI model

Copy `env.example` to `.env` and choose one option:

| Option | Cost | What to set |
|---|---|---|
| **Ollama** (local) | free | `OLLAMA_URL`, `OLLAMA_LLM_MODEL` (e.g. `qwen3:14b`) |
| **Anthropic** (API) | paid | `LLM_PROVIDER=anthropic`, `ANTHROPIC_API_KEY`, `ANTHROPIC_MODEL` |

The **AI model** tab shows the active provider and model; it never shows
addresses or keys. The published example used Ollama with `qwen3:14b` at
temperature 0.

## 3. Project and protocol

In the sidebar, **New project**: blank or from the biofloc example. In the
**Protocol** tab you define:

- The review **topic** and the **prompt language** (Spanish or English).
- **Criteria**, each a Yes/No question:

| Type | Rule |
|---|---|
| `inclusion` | all must hold |
| `exclusion` | any one excludes the record |
| `derivation` | sends the record to human review |

- **Automatic rules** (`check`, no AI): `document_type` and
  `abstract_too_short` (`params: {min_words: 100}`).

**Validate** shows the exact error; **Save** never writes an invalid protocol.

## 4. Exports and runs

In **Run**:

1. Upload each database's exports: Scopus, IEEE and SciELO as CSV; Web of
   Science and ProQuest as Excel.
2. Run the steps in order: **Import → Deduplicate → (optional) AI
   deduplication → Screening layer 1 → Screening layer 2 → (optional) Thematic
   clusters**. Each step runs in the background and shows progress.
   - **AI deduplication** finds near duplicates (translated titles, preprint
     and published version): it compares title + abstract embeddings
     (similarity ≥ 0.93) and an AI judge confirms each pair.
   - **Thematic clusters** groups the eligible records by topic (UMAP +
     HDBSCAN) and draws a thematic map in Results.
   - Both use Ollama embeddings stored in a local Qdrant inside the project
     (`data/qdrant/`): no Docker needed.
3. Try a few records first ("Try N records first").
4. With a paid API, enter your model's prices and check the **estimated cost**
   before a full run.

If **AI errors** appear, those records stay uncertain: check the model
settings. An interrupted run resumes where it stopped.

## 5. Results and human review

- **Results**: metrics, record flow (Sankey), charts per database, decision
  and criterion, and the decisions table as CSV.
- **Human review**: go through the uncertain records or a random validation
  sample. With the sample, Floccus reports agreement and Cohen's kappa between
  the pipeline and your judgement.

## Limits of this version

- AI deduplication and clusters need Ollama (embeddings), even when screening
  uses Anthropic.
- With fewer than 400 eligible records, clusters use smaller minimum sizes than
  the published study (recorded in `clusters.json`).
- Layer 2 queries OpenAlex and Europe PMC: it needs internet access and DOIs.

## Feedback

Use the buttons in the **Start** tab (report a problem, suggest an
improvement). Do not include keys or unpublished data. See `CONTRIBUTING.md`.

## How to cite

DOI [10.5281/zenodo.21926426](https://doi.org/10.5281/zenodo.21926426)
(always the latest version). The published article used v0.3.0:
[10.5281/zenodo.23719341](https://doi.org/10.5281/zenodo.23719341).
