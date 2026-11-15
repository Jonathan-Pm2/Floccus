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
```

Projects live in `./floccus-projects` (override with `--home` or `FLOCCUS_HOME`).
Each project holds `protocol.yaml` (topic, language, criteria), `data/raw/`,
`data/processed/` and `runs/`.
