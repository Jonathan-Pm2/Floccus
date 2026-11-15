# Floccus app

Local tool to run the Floccus LLM-assisted screening method on **any**
systematic-review topic. The published biofloc pipeline (`../src/biofloc_review`)
is left unchanged; this package reuses its generic parts and ships the biofloc
review as an example project.

## Before you start

1. Recommended: install [Gentle Shell](https://github.com/Gentleman-Programming/gentle-shell),
   the AI-agent workspace Floccus is developed with.
2. Install [uv](https://docs.astral.sh/uv/) and clone this repository.
3. Set the AI model in `.env` (see `../env.example`): local Ollama or paid Anthropic.

```bash
uv sync --all-packages
uv run floccus new my-review                      # blank protocol to edit
uv run floccus new biofloc-demo --example biofloc # the published review as example
uv run floccus list
uv run floccus ui                                 # open the interface at http://localhost:8501
```

User guide: [Español](docs/GUIA.md) · [English](docs/GUIDE.md).

`floccus ui` starts a local Streamlit app (Spanish/English selector): create
projects, edit the protocol, upload exports, run the pipeline with live
progress and an estimated API cost, see results and charts, and review
uncertain records with agreement (Cohen's kappa). Options:
`--port N`, `--headless` (do not open a browser), `--theme auto|dark|light`
(`auto`, the default, follows the system and lets users switch light/dark in
the ⋮ menu → Settings → Theme).

The same steps run from the terminal (one at a time, per-project lock):

```bash
uv run floccus run my-review import
uv run floccus run my-review dedup
uv run floccus run my-review screen-l1 --limit 20   # try a sample first
uv run floccus run my-review screen-l2
```

The analysis scripts of the published review need the optional extra:
`uv sync --all-packages --extra analysis` (a plain `uv sync` removes it).

Projects live in `./floccus-projects` (override with `--home` or `FLOCCUS_HOME`).
Each project holds `protocol.yaml` (topic, language, criteria), `data/raw/`,
`data/processed/` and `runs/`.

## Feedback

Use the *Report a problem* / *Suggest an improvement* buttons in the app's
Start tab, or `Issues → New issue` on GitHub. See `../CONTRIBUTING.md`.

Licence Apache-2.0 · cite DOI [10.5281/zenodo.21926426](https://doi.org/10.5281/zenodo.21926426).
