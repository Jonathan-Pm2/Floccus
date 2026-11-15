# Floccus app

Local tool to run the Floccus LLM-assisted screening method on **any**
systematic-review topic. The published biofloc pipeline (`../src/biofloc_review`)
is left unchanged; this package reuses its generic parts and ships the biofloc
review as an example project.

```bash
uv sync --all-packages
uv run floccus new my-review                      # blank protocol to edit
uv run floccus new biofloc-demo --example biofloc # the published review as example
uv run floccus list
uv run floccus ui                                 # open the interface at http://localhost:8501
```

`floccus ui` starts a local Streamlit app (Spanish/English selector) to create
projects, edit the protocol criteria and see the AI-model settings. Options:
`--port N`, `--headless` (do not open a browser). Run, results and human-review
pages are still under construction.

Do not run a plain `uv sync` in this repository: it removes analysis packages
installed outside the lock file. Use `uv sync --inexact --all-packages`.

Projects live in `./floccus-projects` (override with `--home` or `FLOCCUS_HOME`).
Each project holds `protocol.yaml` (topic, language, criteria), `data/raw/`,
`data/processed/` and `runs/`.
