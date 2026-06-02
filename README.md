# biofloc-review

Revisión sistemática asistida por agentes de lenguaje sobre modelado de la
dinámica de flóculos biológicos (biofloc, lodos activados, periphyton) en
sistemas acuícolas y de tratamiento de aguas.

El manuscrito metodológico vive en
[`docs/revision_sistematica_biofloc_es.tex`](docs/revision_sistematica_biofloc_es.tex)
y se actualiza incrementalmente a medida que avanzan las fases del pipeline.

## Arquitectura

```
6 fuentes  →  ingesta unificada  →  enriquecimiento OpenAlex  →
  Capa 1 (determinista)  →  Capa 2 (embeddings + Qdrant)  →
  Capa 3 (juez LLM)  →  corpus canónico  →
  Layer 1 (cribado: criterios PRISMA binarios via LangGraph)  →
  Layer 2 (enriquecimiento extendido: OpenAlex + Europe PMC)  →
  Layer 2 LLM screening (con metadata enriquecida)  →
  [pendiente] elegibilidad por texto completo  →
  estudios incluidos + intersección biofloc ↔ biplot
```

- **Stack**: Python 3.12 + uv, Pydantic, Qdrant (Docker), LangGraph 1.x, Ollama (servidor remoto con GPU).
- **Modelos**: `qwen3-embedding:8b` (4096-dim, multilingüe) para embeddings y `qwen3:14b` para los juicios LLM (Capa 3 dedup + Layer 1 cribado).
- **Fuentes**: Scopus, WoS, IEEE Xplore, SciELO como núcleo + OpenAlex y Europe PMC como enriquecimiento.

## Estado del pipeline

| Fase | Estado | Resultado actual |
|------|--------|------------------|
| Ingesta unificada | ✅ implementada | 22.056 (biofloc) + 1.401 (biplot) |
| Enriquecimiento OpenAlex inicial | ✅ implementada, ⏳ relanzar tras espera del 429 | 131 records ya enriquecidos, ~4.991 pendientes |
| Capa 1 dedup (DOI / título normalizado) | ✅ implementada | 17.326 (biofloc) + 1.235 (biplot) |
| Capa 2 dedup (embeddings + Qdrant HNSW) | ✅ implementada | 567 pares candidatos (biofloc) + 25 (biplot) |
| Capa 3 dedup (juez `qwen3:14b`) | ✅ implementada | 258 DUP / 242 NOT_DUP / 67 UNCLEAR (biofloc) |
| Merge final con Union-Find | ✅ implementada | **17.075** (biofloc) + **1.222** (biplot) únicos |
| **Layer 1 cribado** (criterios PRISMA binarios) | 🔄 corriendo sobre 17.075 records (PID 36060) | calibración piloto n=100: 56% EXCLUDE, 36% HUMAN_REVIEW, 8% INCLUDE |
| **Layer 2 enricher** (OpenAlex + Europe PMC) | ✅ implementada, ⏳ ejecutar post-Layer-1 | smoke test: 1.9s/record, PMC fulltext en 60% |
| Layer 2 LLM screening con metadata enriquecida | ⏳ por diseñar | depende del subset post-Layer-1 |
| Elegibilidad por texto completo | ⏳ pendiente | PDFs vía OpenAlex/Europe PMC OA + EZproxy USAL |
| Intersección biofloc ↔ biplot | ⏳ pendiente | RAG semántico con el `.tex` como *query* |

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

# 7. Layer 1: cribado por criterios PRISMA binarios (12 criterios via LangGraph)
uv run python -m biofloc_review.run_screening \
  data/processed/biofloc_final_primary.parquet --workers 2 --checkpoint-every 25

# 8. Layer 2: enriquecimiento extendido (OpenAlex + Europe PMC) sobre los que avanzan
uv run python -m biofloc_review.run_enrich \
  data/processed/biofloc_screened.parquet --workers 1 --rps 0.5
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
├── qdrant_storage/         # vector DB persistente (gitignored)
└── src/biofloc_review/
    ├── schema.py           # PaperRecord (Pydantic)
    ├── normalize.py        # normalize_doi, normalize_title, ...
    ├── ingest.py           # CLI + orquestación
    ├── run_dedup.py        # CLI Capa 1 dedup
    ├── run_embed.py        # CLI Capa 2 dedup
    ├── run_judge.py        # CLI Capa 3 dedup
    ├── run_merge.py        # CLI fusión final dedup
    ├── run_screening.py    # CLI Layer 1: cribado PRISMA
    ├── run_enrich.py       # CLI Layer 2: enriquecimiento extendido
    ├── sources/            # loaders por fuente
    │   ├── scopus.py
    │   ├── wos.py
    │   ├── ieee.py
    │   ├── scielo.py
    │   └── openalex.py     # cliente + enriquecimiento + rate limiter
    ├── dedup/
    │   ├── deterministic.py    # Capa 1: clave canónica + clustering
    │   ├── ollama_embed.py     # cliente Ollama embed con auth Bearer
    │   ├── embeddings.py       # Capa 2: indexing + búsqueda Qdrant
    │   ├── llm_judge.py        # Capa 3: prompt + parseo de veredicto
    │   └── merge_confirmed.py  # union-find sobre veredictos DUPLICATE
    └── screening/
        ├── criteria.py         # los 12 criterios PRISMA (I1-I4, E1-E5, D1-D3)
        ├── prefilter.py        # E5/I4/D1 deterministas (sin LLM)
        ├── prompts.py          # template por criterio (YES/NO/UNCLEAR + EVIDENCE)
        ├── classifier.py       # ask_criterion(rec, criterion) → CriterionAnswer
        ├── decision.py         # regla AND/OR + 2 variantes para sensitivity
        ├── graph.py            # StateGraph: prefilter → llm_evaluator → supervisor
        └── enricher.py         # Layer 2: OpenAlex + Europe PMC + arxiv (opt-in)
```

## Decisiones y aprendizajes operativos

Notas que justifican configuraciones no obvias del pipeline.

- **WoS requiere export "Full Record"**. Un export con la opción por defecto omite *abstracts* y *keywords*; se detectó tras una primera corrida con 100% de los registros WoS sin *abstract*.
- **OpenAlex tiene un límite efectivo más bajo que el nominal 10 req/s**. La operación sostenida con 3 trabajadores concurrentes genera bloqueos HTTP 429 que duran hasta 24 h. La configuración actual usa un limitador explícito a 0,5 rps y *checkpoint* cada 50 registros.
- **Qwen3 en modo *chat* requiere `think: false`**. Con el modo de razonamiento habilitado (por defecto), la respuesta sale en el campo `message.thinking` y el `message.content` queda vacío.
- **El umbral coseno $\tau{=}0{,}93$ tiene ~44% de falsos positivos**. La Capa 2 sola no es suficiente para auto-merge; la Capa 3 es necesaria. Esto se verificó en ambos corpus.
- **Títulos genéricos cortos** (p. ej. "Activated sludge") generan colisiones de embedding aunque sean trabajos distintos. La Capa 3 los descarta correctamente.
- **Los rerankers nativos de Ollama** (BGE, Qwen3-Reranker) están empaquetados solo como encoder, sin la cabeza de clasificación funcional. Esto motivó usar `qwen3:14b` como pseudo-reranker vía *prompt scoring*, con la ventaja adicional de generar *audit trail* textual aprovechable para defender la metodología en el manuscrito.
- **El cribado v1 (3 clasificadores asimétricos por vista) producía 100% HUMAN_REVIEW** en la calibración n=100: el clasificador de contexto nunca votaba EXCLUDE porque la cadena de búsqueda Scopus ya filtraba por wastewater/aquaculture. El diseño se reformuló a **criterios PRISMA binarios** (4 inclusión + 5 exclusión + 3 derivación con regla AND/OR) antes del pre-registro en OSF. La nueva regla cambia el balance a 56% EXCLUDE / 36% HUMAN_REVIEW / 8% INCLUDE.
- **El criterio E3 ("contexto fuera del agua") tenía 5/100 falsos positivos sistemáticos**: el modelo confundía menciones colaterales de GHG/landfill/industrial con el contexto principal. El *prompt* se reformuló para explicitar que el criterio decisivo es la localización del sistema biológico estudiado, no las palabras del resumen. El fix resolvió 4/5 sin regresiones.
- **Variabilidad inter-run del 3-5% con `temperature=0`**: dos corridas idénticas sobre n=100 cambiaron la decisión final en 5/100 records. Es propiedad inherente del cómputo concurrente en GPU; no es eliminable con ajustes de prompt. Reportado en limitaciones.
- **Europe PMC duplica el acceso a fulltext** sobre OpenAlex puro: en la muestra n=5, Europe PMC marcó 3/5 (60%) con `has_fulltext=True` contra 0/5 con `pdf_url` directo de OpenAlex. La integración de PMC en Layer 2 es low-hanging fruit antes de la fase de elegibilidad.
- **arxiv API rate-limitea agresivo** (3+ seg entre calls) y nuestro corpus tiene <5% overlap con preprints arxiv. Marcado opt-in (`--with-arxiv`); útil solo para el artículo paralelo Poincaré-biplot.

## Reproducibilidad

- Dependencias bloqueadas en `uv.lock`.
- Qdrant en Docker con almacenamiento en disco (`qdrant_storage/`).
- Parquets intermedios en `data/processed/` preservan el estado de cada fase, permitiendo reanudar desde cualquier punto.
- Cada `cluster_id` es un SHA-1 truncado determinista sobre la clave canónica, estable entre corridas.
- Las decisiones de fusión llevan un campo `merged_from` con los `source_id` de cada fuente absorbida, habilitando auditoría completa.
