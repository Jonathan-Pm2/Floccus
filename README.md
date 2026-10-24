# Floccus

**Floccus: cribado PRISMA con agentes LLM para revisiones sistemáticas en
modelado de flóculos biológicos.**

Revisión sistemática asistida por agentes de lenguaje sobre modelado de la
dinámica de flóculos biológicos (biofloc, lodos activados, periphyton) en
sistemas acuícolas y de tratamiento de aguas.

El manuscrito metodológico vive en
[`docs/revision_sistematica_biofloc_es.tex`](docs/revision_sistematica_biofloc_es.tex)
y se actualiza incrementalmente a medida que avanzan las fases del pipeline.
Publicación objetivo: *Ecological Modelling* (Elsevier).

## Arquitectura

```
4 fuentes  →  ingesta unificada  →  enriquecimiento OpenAlex  →
  Capa 1 (determinista)  →  Capa 2 (embeddings + Qdrant)  →
  Capa 3 (juez LLM)  →  corpus canónico  →
  Layer 1 (cribado: 11 criterios PRISMA binarios via LangGraph)  →
  Layer 2 enricher (OpenAlex + Europe PMC sobre advancing)  →
  Layer 2 LLM (segundo pase ciego con metadata enriquecida)  →
  consolidación + conteos PRISMA  →
  validación humana κ Cohen  →
  Layer 3 (elegibilidad por texto completo via EZproxy USAL)  →
  análisis post-cribado (clustering temático + métodos + intersección)
```

- **Stack**: Python 3.12 + uv, Pydantic, Qdrant (Docker), LangGraph 1.x, Ollama (servidor remoto con GPU).
- **Modelos**: `qwen3-embedding:8b` (4096-dim, multilingüe) para embeddings y `qwen3:14b` para los juicios LLM (Capa 3 dedup + Layer 1 cribado).
- **Fuentes**: Scopus, WoS, IEEE Xplore, SciELO como núcleo + OpenAlex y Europe PMC como enriquecimiento.

## Estado del pipeline

| Fase | Estado | Resultado |
|------|--------|-----------|
| Ingesta unificada | ✅ | 22 056 (biofloc) + 1 401 (biplot) |
| Enriquecimiento OpenAlex inicial | ✅ | corpus completo enriquecido |
| Capa 1 dedup (DOI / título normalizado + año) | ✅ | 17 326 únicos (biofloc) + 1 235 (biplot) |
| Capa 2 dedup (embeddings + Qdrant HNSW, τ=0,93) | ✅ | 567 pares candidatos (biofloc) + 25 (biplot) |
| Capa 3 dedup (juez `qwen3:14b`) | ✅ | 258 DUP / 242 NOT_DUP / 67 UNCLEAR (biofloc) |
| Merge final con Union-Find | ✅ | **17 075** (biofloc) + **1 222** (biplot) únicos |
| IEEE source_id fix (PDF Link > DOI > titlehash) | ✅ | 530/530 record_ids únicos (antes solo 11 buckets) |
| **Layer 1 cribado** (11 criterios PRISMA binarios via LangGraph) | ✅ | 17 063 records: 1 205 INCLUDE / 4 225 HR / 11 633 EXCLUDE |
| Layer 2 enricher (OpenAlex + Europe PMC sobre los 5 430 advancing) | ✅ | OpenAlex OA 11,7% + EuropePMC 4,8% (unión 13,8%) |
| Layer 2 LLM screening con metadata enriquecida | ✅ | 4 225 reprocesados: +180 INCLUDE / +547 EXCLUDE / 3 498 HR |
| **Consolidación final + conteos PRISMA** | ✅ | **1 385 INCLUDE / 3 498 HR / 12 180 EXCLUDE** → 4 883 a elegibilidad |
| Validación humana κ Cohen (CSV blind estratificado, n=465) | ✅ | κ=0.107 (Mapeo A) / 0.209 (Mapeo B), slight; precisión EXCLUDE 91% / INCLUDE 31% / HR 16% |
| Layer 3 Tier 1 fulltext fetch (OA libre sin cookies) | ✅ | **297 PDFs descargados** (604 MB), 14,5% del corpus elegible |
| Layer 3 Tier 2/3 EZproxy USAL | ⏸ bloqueado | requiere re-login CAS explícito o Playwright wrapper |
| Análisis post-cribado: caracterización del corpus elegible | ✅ | temporal 1969-2026, mediana 2015, Elsevier 38,6%, IWA 7,7% |
| Análisis post-cribado: clustering temático UMAP+HDBSCAN | ✅ | **25 clusters** sobre 4 682 embeddings, biofloc-aquaculture (C03) 3,0% |
| Análisis post-cribado: identificación de métodos en fulltext | ✅ | ASM 36,5%, ODE 23%, ML/ANN 21%, PINN 0,3% sobre 296 PDFs OK |
| Análisis post-cribado: tendencias temporales de métodos | ✅ | ML supera ODE en 2015-19; PINN emerge 2020+ |
| Análisis post-cribado: co-ocurrencia + papers híbridos | ✅ | 73 papers híbridos (24,7%): ASM+ML=20, PDE+ML=1 |
| Análisis post-cribado: búsqueda avanzada en corpus completo | ✅ | **2 PINN + 7 NeuralODE + 0 hyperbolic/Riemannian** en 4 884 abstracts |
| Análisis post-cribado: sub-clustering del cluster núcleo C03 | ✅ | 4 sub-clusters: 84% SciELO iberoamericano vs 85% Scopus moderno |
| Intersección biofloc ↔ biplot via RAG semántico | ✅ | bridge metodológico (no temático); **0 papers Poincaré en ambos corpus** |

## Análisis post-cribado y hallazgos clave

Una vez completado el cribado consolidado se ejecuta una capa de **análisis exploratorio del corpus elegible** que el manuscrito reporta en §3. Los hallazgos centrales:

### Validación con revisor humano
κ Cohen sobre n=465 records estratificados: κ=0,107 (Mapeo A PRISMA-permisivo) / κ=0,209 (Mapeo B estricto). El LLM **no reemplaza al revisor humano** para inclusiones (κ humano-humano típico = 0,5-0,8), pero **filtra exclusiones con alta especificidad (91 % precision)** y deriva correctamente los borderline a HUMAN_REVIEW (16 % precision en HR confirma su necesidad metodológica).

### Estructura temática del corpus elegible
UMAP+HDBSCAN sobre 4 682 embeddings densos detecta **25 clusters cohesivos** (73,5 % del corpus). Los más alineados con la pregunta de investigación:
- **C03 Biofloc en acuicultura** (n=141, mediana 2019, **55 % SciELO iberoamericano**)
- **C15 Estructura del flóculo / reología** (n=68, formal mecanístico)

Sub-clustering interno de C03 separa **84 % SciELO iberoamericano clásico** (n=74) de **85 % Scopus cuantitativo moderno** (n=53). El paper *Sánchez & Bargary 2025* (J Water Process Eng) — del grupo del autor — fue detectado correctamente en el sub-cluster cuantitativo.

### Métodos de modelado en el fulltext
Análisis sobre 296 PDFs descargados y validado contra 4 884 abstracts del corpus completo:

| Método | % PDFs | n abstracts corpus |
|---|---:|---:|
| ASM (Activated Sludge Model, IWA) | 36,5 % | dominante |
| ODE / Monod / Michaelis-Menten | 23,0 % | declinando desde 2015 |
| ML / ANN | 21,3 % | **supera ODE desde 2015-19** |
| Stochastic / Langevin / MC | 8,8 % | 19 |
| CFD | 7,4 % | — |
| **PINN** | 0,3 % | **2** (ambos Water Research 2024-2025, ambos sobre flóculos) |
| NeuralODE | — | 7 (frente 2024-2026) |
| **Hyperbolic / Riemannian / Poincaré** | — | **0 en ambos corpus** |

### Sweet spot doctoral identificado
La triple intersección **Poincaré biplot × modelado de flóculos × física informada** es un **nicho metodológico vacío** en la literatura indexada que el manuscrito hace navegable:

- 0 papers con Poincaré/hyperbolic en el corpus biplot completo (1 222 records)
- 0 papers con Poincaré/hyperbolic en el corpus biofloc completo (4 884 abstracts)
- 0 papers PINN en cluster C03 biofloc acuícola
- 2 papers PINN + flóculos existen en Water Research pero fuera de C03

### Manuscrito
Los hallazgos están integrados en `docs/revision_sistematica_biofloc_es.tex` (50 páginas, sin warnings al compilar). Las nueve subsecciones de §3 cubren validación κ → caracterización corpus → clustering temático → métodos → tendencias temporales → co-ocurrencia → precedentes avanzados → C03 deep-dive + sub-clustering → intersección biplot + gap Poincaré.

## Migrar a otra PC

Si necesitás levantar el proyecto en otra máquina (laboratorio, servidor,
PC de la doctora Galindo, etc.) la cadena es:

```bash
# En la PC actual:
bash scripts/pack.sh            # genera Articulos_Data_biofloc_mate.zip en el directorio padre

# Mover el zip por scp / Drive / pendrive a la otra PC, descomprimir y:
cd Articulos_Data_biofloc_mate/biofloc-review
bash scripts/setup.sh           # verifica deps, levanta Qdrant, valida Ollama
```

El detalle paso a paso (prerrequisitos por SO, troubleshooting, smoke tests)
está en [SETUP.md](SETUP.md). La plantilla de variables de entorno está en
[env.example](env.example) — copiala a `.env` y rellená los tokens.

## Quick start

### Requisitos previos

- `uv` (gestor de dependencias)
- Docker (para Qdrant)
- Acceso a un servidor Ollama (local o remoto). Si es remoto, configurar `.env` en la raíz del workspace con:
  ```
  OLLAMA_URL=https://<tunel>.example.com
  OLLAMA_TOKEN=<bearer-token>
  OPENALEX_EMAIL=<tu-correo>
  OPENALEX_RPS=0.5
  ```

### Instalación

```bash
uv sync
docker run -d --name qdrant-biofloc \
  -p 6333:6333 -p 6334:6334 \
  -v "$(pwd)/qdrant_storage:/qdrant/storage" \
  qdrant/qdrant:latest
```

### Pipeline en orden

```bash
# 1. Ingesta (lee CSV/XLS desde el workspace padre)
uv run biofloc-review --data-dir .. --corpus both

# 2. Enriquecimiento (reanudable, rate-limited a 0.5 rps)
uv run biofloc-review --data-dir .. --corpus biofloc --enrich --resume --rps 0.5

# 3. Capa 1: deduplicación determinista
uv run python -m biofloc_review.run_dedup data/processed/biofloc.parquet
uv run python -m biofloc_review.run_dedup data/processed/biplot.parquet

# 4. Capa 2: embeddings + búsqueda de near-duplicates en Qdrant
uv run python -m biofloc_review.run_embed \
  data/processed/biofloc_dedup_primary.parquet --corpus biofloc --recreate
uv run python -m biofloc_review.run_embed \
  data/processed/biplot_dedup_primary.parquet --corpus biplot --recreate

# 5. Capa 3: juez LLM sobre los pares candidatos
uv run python -m biofloc_review.run_judge \
  data/processed/biofloc_dedup_primary_near_dup_pairs.json \
  data/processed/biofloc_dedup_primary.parquet --workers 2
uv run python -m biofloc_review.run_judge \
  data/processed/biplot_dedup_primary_near_dup_pairs.json \
  data/processed/biplot_dedup_primary.parquet --workers 2

# 6. Merge final: aplicar veredictos DUPLICATE al corpus
uv run python -m biofloc_review.run_merge \
  data/processed/biofloc_dedup_full.parquet \
  data/processed/biofloc_dedup_primary_near_dup_pairs_judged.json
uv run python -m biofloc_review.run_merge \
  data/processed/biplot_dedup_full.parquet \
  data/processed/biplot_dedup_primary_near_dup_pairs_judged.json

# 7. Layer 1: cribado por criterios PRISMA binarios (11 criterios via LangGraph)
uv run python -m biofloc_review.run_screening \
  data/processed/biofloc_final_primary_v3.parquet --workers 2 --checkpoint-every 25

# 8. Layer 2: enriquecimiento extendido (OpenAlex + Europe PMC) sobre los advancing
uv run python -m biofloc_review.run_enrich \
  data/processed/biofloc_screened.parquet \
  --full-parquet data/processed/biofloc_final_primary_v3.parquet \
  --workers 1 --rps 0.5

# 9. Layer 2 LLM: segundo pase ciego sobre HUMAN_REVIEW con input enriquecido
uv run python -m biofloc_review.run_screening_layer2 \
  data/processed/biofloc_screened.parquet \
  data/processed/biofloc_enriched.parquet \
  --full-parquet data/processed/biofloc_final_primary_v3.parquet \
  --workers 2 --checkpoint-every 25

# 10. Consolidación + conteos PRISMA (Layer 1 INCLUDE + Layer 2 final)
uv run python -m biofloc_review.run_consolidate \
  data/processed/biofloc_screened.parquet \
  data/processed/biofloc_layer2_screened.parquet

# 11. Validación humana: muestra estratificada blind para κ Cohen
uv run python -m biofloc_review.run_validation_sample \
  data/processed/biofloc_final_decisions.parquet
# → llenar human_decision con "cumple"/"no cumple" en validation_sample_blind.csv
uv run python -m biofloc_review.run_kappa_analysis \
  data/processed/validation_sample_blind.csv \
  data/processed/validation_sample_truth.parquet

# 12. Layer 3: fetch fulltext PDFs (Tier 1 OA libre sin cookies)
uv run python -m biofloc_review.run_fulltext_fetch \
  data/processed/biofloc_enriched.parquet \
  data/processed/biofloc_final_decisions.parquet \
  --workers 2 --delay 0.8 --skip-session-check --resume

# 12b. Layer 3 Tier 2/3: agregar cookies USAL para descargar paywall via EZproxy
uv run python -m biofloc_review.run_fulltext_fetch \
  data/processed/biofloc_enriched.parquet \
  data/processed/biofloc_final_decisions.parquet \
  --cookies /path/to/cookies.txt --workers 2 --delay 1.5 --resume

# 13. Análisis post-cribado: clustering temático UMAP+HDBSCAN
uv run python -m biofloc_review.run_thematic_clustering \
  data/processed/biofloc_final_decisions.parquet \
  data/processed/biofloc_final_primary_v3.parquet \
  --collection papers_biofloc --min-cluster-size 40

# 14. Análisis post-cribado: identificación de métodos en los PDFs descargados
uv run python scripts/extract_methods.py
# → data/processed/fulltext_methods.parquet (regex sobre primeras 25 págs de cada PDF)
```

## Estructura del proyecto

```
biofloc-review/
├── pyproject.toml          # deps + entry points
├── uv.lock                 # lockfile reproducible
├── .env                    # credenciales Ollama / OpenAlex (no versionado)
├── docs/
│   ├── revision_sistematica_biofloc_es.tex   # manuscrito metodológico
│   └── revision_sistematica_biofloc_es.pdf   # compilado
├── data/processed/         # parquets intermedios y finales (gitignored)
├── data/fulltext/          # PDFs descargados Layer 3 (gitignored)
├── qdrant_storage/         # vector DB persistente (gitignored)
├── scripts/                # análisis post-cribado one-shot
│   └── extract_methods.py  # regex de métodos sobre PDFs descargados
└── src/biofloc_review/
    ├── schema.py           # PaperRecord (Pydantic)
    ├── normalize.py        # normalize_doi, normalize_title, ...
    ├── ingest.py           # CLI + orquestación
    ├── run_dedup.py             # CLI Capa 1 dedup
    ├── run_embed.py             # CLI Capa 2 dedup
    ├── run_judge.py             # CLI Capa 3 dedup
    ├── run_merge.py             # CLI fusión final dedup
    ├── run_screening.py         # CLI Layer 1: cribado PRISMA
    ├── run_enrich.py            # CLI Layer 2: enriquecimiento extendido
    ├── run_screening_layer2.py  # CLI Layer 2 LLM: segundo pase ciego enriquecido
    ├── run_consolidate.py       # CLI consolidación L1+L2 + conteos PRISMA
    ├── run_validation_sample.py # CLI muestreo estratificado blind para κ
    ├── run_kappa_analysis.py    # CLI κ Cohen humano vs LLM (cumple / no cumple)
    ├── run_fulltext_fetch.py    # CLI Layer 3: descarga PDFs via EZproxy
    ├── run_thematic_clustering.py # CLI UMAP+HDBSCAN sobre corpus elegible
    ├── run_validate_pipeline.py # CLI validador integridad por fuente
    ├── run_ieee_fix.py          # CLI fix IEEE source_id (one-shot, idempotente)
    ├── sources/                 # loaders por fuente
    │   ├── scopus.py
    │   ├── wos.py
    │   ├── ieee.py              # PDF Link > DOI > titlehash para source_id
    │   ├── scielo.py
    │   ├── openalex.py          # cliente + enriquecimiento + rate limiter
    │   └── ezproxy.py           # USAL URL rewriter + cookie loader Netscape
    ├── dedup/
    │   ├── deterministic.py     # Capa 1: clave canónica + clustering
    │   ├── ollama_embed.py      # cliente Ollama embed con auth Bearer
    │   ├── embeddings.py        # Capa 2: indexing + búsqueda Qdrant
    │   ├── llm_judge.py         # Capa 3: prompt + parseo de veredicto
    │   └── merge_confirmed.py   # union-find sobre veredictos DUPLICATE
    └── screening/
        ├── criteria.py          # los 11 criterios PRISMA (I1-I4, E1-E4, D1-D3)
        ├── prefilter.py         # I4/D1 deterministas (sin LLM)
        ├── prompts.py           # template por criterio (YES/NO/UNCLEAR + EVIDENCE)
        ├── classifier.py        # ask_criterion(rec, criterion) → CriterionAnswer
        ├── decision.py          # regla AND/OR + 2 variantes para sensitivity
        ├── graph.py             # StateGraph Layer 1: prefilter → LLM → supervisor
        ├── layer2.py            # Layer 2 LLM: pase ciego con metadata enriquecida
        ├── enricher.py          # Layer 2: OpenAlex + Europe PMC + arxiv (opt-in)
        └── fulltext_fetcher.py  # Layer 3: OA → PMC → DOI EZproxy multi-tier
```

## Decisiones y aprendizajes operativos

Notas que justifican configuraciones no obvias del pipeline.

- **WoS requiere export "Full Record"**. Un export con la opción por defecto omite *abstracts* y *keywords*; se detectó tras una primera corrida con 100% de los registros WoS sin *abstract*.
- **OpenAlex tiene un límite efectivo más bajo que el nominal 10 req/s**. La operación sostenida con 3 trabajadores concurrentes genera bloqueos HTTP 429 que duran hasta 24 h. La configuración actual usa un limitador explícito a 0,5 rps y *checkpoint* cada 50 registros.
- **Qwen3 en modo *chat* requiere `think: false`**. Con el modo de razonamiento habilitado (por defecto), la respuesta sale en el campo `message.thinking` y el `message.content` queda vacío.
- **El umbral coseno $\tau{=}0{,}93$ tiene ~44% de falsos positivos**. La Capa 2 sola no es suficiente para auto-merge; la Capa 3 es necesaria. Esto se verificó en ambos corpus.
- **Títulos genéricos cortos** (p. ej. "Activated sludge") generan colisiones de embedding aunque sean trabajos distintos. La Capa 3 los descarta correctamente.
- **Los rerankers nativos de Ollama** (BGE, Qwen3-Reranker) están empaquetados solo como encoder, sin la cabeza de clasificación funcional. Esto motivó usar `qwen3:14b` como pseudo-reranker vía *prompt scoring*, con la ventaja adicional de generar *audit trail* textual aprovechable para defender la metodología en el manuscrito.
- **El cribado v1 (3 clasificadores asimétricos por vista) producía 100% HUMAN_REVIEW** en la calibración n=100: el clasificador de contexto nunca votaba EXCLUDE porque la cadena de búsqueda Scopus ya filtraba por wastewater/aquaculture. El diseño se reformuló a **criterios PRISMA binarios** (4 inclusión + 4 exclusión + 3 derivación con regla AND/OR) antes del pre-registro en OSF. La nueva regla aplicada al corpus completo produce 7,1% INCLUDE / 24,8% HUMAN_REVIEW / 68,2% EXCLUDE en Layer 1.
- **E5 (idioma) removido del protocolo**. La versión inicial incluía un criterio de exclusión determinista por idioma (no inglés/español/portugués); se removió porque (i) el corpus prácticamente no incluía rows con `language` poblado por las fuentes, y (ii) Qwen3-14B resuelve chino/coreano/alemán/francés/árabe nativamente, así que una barrera lingüística arbitraria solo descartaría literatura relevante. El criterio activo es **I1-I4, E1-E4, D1-D3 (11 total)**.
- **Bug IEEE `Document Identifier`**. El export de IEEE Xplore rellena ese campo con un código institucional, no per-paper: 530 papers únicos comparten solo 11 valores distintos. El `source_id` interno los colapsaba en 11 buckets y heredaban veredictos cruzados. Fix: priorizar `PDF Link` > `DOI` > hash del título normalizado. Tras el fix: 530/530 record_ids únicos.
- **El criterio E3 ("contexto fuera del agua") tenía 5/100 falsos positivos sistemáticos**: el modelo confundía menciones colaterales de GHG/landfill/industrial con el contexto principal. El *prompt* se reformuló para explicitar que el criterio decisivo es la localización del sistema biológico estudiado, no las palabras del resumen. El fix resolvió 4/5 sin regresiones.
- **Variabilidad inter-run del 3-5% con `temperature=0`**: dos corridas idénticas sobre n=100 cambiaron la decisión final en 5/100 records. Es propiedad inherente del cómputo concurrente en GPU; no es eliminable con ajustes de prompt. Reportado en limitaciones.
- **Europe PMC duplica el acceso a fulltext** sobre OpenAlex puro: en la muestra n=5, Europe PMC marcó 3/5 (60%) con `has_fulltext=True` contra 0/5 con `pdf_url` directo de OpenAlex. La integración de PMC en Layer 2 es low-hanging fruit antes de la fase de elegibilidad.
- **arxiv API rate-limitea agresivo** (3+ seg entre calls) y nuestro corpus tiene <5% overlap con preprints arxiv. Marcado opt-in (`--with-arxiv`); útil solo para el artículo paralelo Poincaré-biplot.
- **Validación humana con etiquetado binario `cumple` / `no cumple`** en lugar de 3-clases (`INCLUDE` / `EXCLUDE` / `HUMAN_REVIEW`). El revisor humano decide si el paper avanza o no a la siguiente fase PRISMA; para el κ Cohen el analizador colapsa el 3-clases del LLM al binario humano (`INCLUDE ∪ HUMAN_REVIEW → cumple`, `EXCLUDE → no cumple`). Mapeo justificado porque `HUMAN_REVIEW` semánticamente preserva el record para la siguiente fase, igual que `INCLUDE`.
- **Layer 3 vía cookies del navegador, no automatización de login**. La autenticación al EZproxy de la USAL se hace una vez por el usuario en el navegador (Shibboleth/OpenAthens); las cookies se exportan en formato Netscape y se cargan en `MozillaCookieJar`. Esto evita problemas de TOS de las plataformas comerciales y mantiene la autenticación bajo control humano.

## Reproducibilidad

- Dependencias bloqueadas en `uv.lock`.
- Qdrant en Docker con almacenamiento en disco (`qdrant_storage/`).
- Parquets intermedios en `data/processed/` preservan el estado de cada fase, permitiendo reanudar desde cualquier punto.
- Cada `cluster_id` es un SHA-1 truncado determinista sobre la clave canónica, estable entre corridas.
- Las decisiones de fusión llevan un campo `merged_from` con los `source_id` de cada fuente absorbida, habilitando auditoría completa.

## Screening operating modes

The title-and-abstract screener can run at two operating points. They differ
only in how two things are asked of the language model; the eleven criteria and
the asymmetric decision rule are the same.

| mode | prompt | E2 ("purely experimental, no modelling") | trade-off |
|---|---|---|---|
| `specificity` (default) | answer first, then reasoning — the published configuration | asked as written (negative) | fewer records reach human review |
| `sensitivity` | reasoning first, answer last | asked positively ("does it include formal modelling?") and inverted | higher recall, more human review |

```bash
python -m biofloc_review.run_screening data/processed/biofloc_final_primary_v3.parquet --mode sensitivity
# or: FLOCCUS_SCREENING_MODE=sensitivity
```

Why two modes. In blind validation, most records the human reviewer accepted
but the pipeline excluded were rejected by E2 with a reasoning that argued the
opposite of the verdict: with the answer requested before the reasoning, and
the criterion phrased as a negation, the model committed to the wrong polarity.
Asking for the reasoning first and phrasing E2 positively removes that error,
which raises recall and lowers specificity. Measured operating points are
reported in the manuscript; `scripts/pilot_prompt_v2.py` reproduces the
validation-sample comparison and `scripts/rescreen_e2_sensitivity_mode.py` the
corpus counts.
