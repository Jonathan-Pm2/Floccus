# Migración a otra PC — Setup paso a paso

Esta guía describe cómo levantar el proyecto en una máquina nueva a partir
del `.zip` empaquetado con `scripts/pack.sh`. Asume Linux (Ubuntu/Debian/Mint);
para macOS o Windows los comandos de instalación de los prerrequisitos
cambian pero el resto es idéntico.

## 0. Prerrequisitos en la PC destino

Necesitás instaladas estas tres cosas. Verificá con los comandos al lado:

| Herramienta | Verificar | Si no está |
|-------------|-----------|------------|
| `docker` | `docker --version` | `curl -fsSL https://get.docker.com \| sh` y `sudo usermod -aG docker $USER` (relogin) |
| `uv` (Python) | `uv --version` | `curl -LsSf https://astral.sh/uv/install.sh \| sh` |
| `pdflatex` (opcional, solo para compilar el `.tex`) | `pdflatex --version` | `sudo apt install texlive-latex-recommended texlive-fonts-recommended texlive-latex-extra` |

Acceso al servidor Ollama remoto: la PC destino debe poder alcanzar la URL
del túnel Cloudflare (probada con `curl $OLLAMA_URL/api/tags` debería
devolver JSON).

## 1. Descomprimir y entrar

```bash
unzip Articulos_Data_biofloc_mate.zip
cd Articulos_Data_biofloc_mate/biofloc-review
```

La estructura quedará:

```
Articulos_Data_biofloc_mate/
├── biofloc-review/         # proyecto Python
│   ├── data/processed/     # parquets (estado de cada fase)
│   ├── qdrant_storage/     # vectores indexados (~17k de 4096 dim)
│   ├── docs/               # manuscrito .tex y figuras
│   └── ...
├── biplot/                 # subcorpus biplot
├── IEEE.csv, WOS.xls, ...  # corpus fuente
└── .env                    # credenciales (no versionado)
```

## 2. Restauración automática

Una sola línea:

```bash
bash scripts/setup.sh
```

El script ejecuta, en orden:

1. **Verifica prerrequisitos** (docker, uv, .env en el workspace).
2. **Reconstruye el entorno Python** con `uv sync` (lee `uv.lock`,
   garantiza versiones exactas de las dependencias).
3. **Levanta el contenedor Qdrant** con el volumen persistente
   `qdrant_storage/` montado (los vectores ya calculados se restauran tal
   cual).
4. **Verifica la conectividad a Ollama remoto** con un *ping* al `.env`.
5. **Reporta el estado** del corpus en `data/processed/` (cuántos records
   por fase, listo para reanudar desde donde quedó).

## 3. Verificar que todo funciona

```bash
# Smoke test embedding (debe imprimir "Dim: 4096")
uv run python -c "
from biofloc_review.dedup.ollama_embed import ollama_url, embed_batch
print('URL:', ollama_url())
v = embed_batch(['biofloc shrimp aquaculture'])
print('Dim:', len(v[0]))
"

# Smoke test Qdrant (debe listar las dos colecciones)
curl -s http://localhost:6333/collections | python3 -m json.tool

# Smoke test datos
uv run python -c "
import pandas as pd
df = pd.read_parquet('data/processed/biofloc.parquet')
print('biofloc records:', len(df))
print('with abstract :', df['abstract'].notna().sum())
"
```

## 4. Continuar el trabajo

Todos los CLIs están registrados como entry points o scripts. Los comandos
de uso frecuente:

```bash
# Reanudar enriquecimiento desde el checkpoint
uv run biofloc-review --data-dir .. --corpus biofloc --enrich --resume --rps 1.0

# Re-correr deduplicación Capa 1 si los parquets cambiaron
uv run python -m biofloc_review.run_dedup data/processed/biofloc.parquet

# Re-indexar embeddings (incremental, sin --recreate)
uv run python -m biofloc_review.run_embed data/processed/biofloc_dedup_primary.parquet --corpus biofloc

# Capa 3 juez LLM
uv run python -m biofloc_review.run_judge \
  data/processed/biofloc_dedup_primary_near_dup_pairs.json \
  data/processed/biofloc_dedup_primary.parquet --workers 2

# Aplicar veredictos al corpus final
uv run python -m biofloc_review.run_merge \
  data/processed/biofloc_dedup_full.parquet \
  data/processed/biofloc_dedup_primary_near_dup_pairs_judged.json

# Layer 1: cribado por 11 criterios PRISMA binarios (~42h para 17k records con 2 workers).
# Resumable con --resume. Para corrida desatachada que sobrevive suspensión:
#   setsid bash -c '... --workers 2 --checkpoint-every 25' < /dev/null & disown
uv run python -m biofloc_review.run_screening \
  data/processed/biofloc_final_primary_v3.parquet --workers 2 --checkpoint-every 25

# Layer 2 enricher: OpenAlex + Europe PMC sobre INCLUDE + HUMAN_REVIEW de L1.
# Solo metadata, no descarga PDFs. arxiv opcional via --with-arxiv.
uv run python -m biofloc_review.run_enrich \
  data/processed/biofloc_screened.parquet \
  --full-parquet data/processed/biofloc_final_primary_v3.parquet \
  --workers 1 --rps 0.5

# Layer 2 LLM: segundo pase ciego sobre HUMAN_REVIEW con metadata enriquecida (~26 min)
uv run python -m biofloc_review.run_screening_layer2 \
  data/processed/biofloc_screened.parquet \
  data/processed/biofloc_enriched.parquet \
  --full-parquet data/processed/biofloc_final_primary_v3.parquet \
  --workers 2 --checkpoint-every 25 --resume

# Consolidación final + conteos PRISMA para el flowchart del manuscrito
uv run python -m biofloc_review.run_consolidate \
  data/processed/biofloc_screened.parquet \
  data/processed/biofloc_layer2_screened.parquet

# Validación humana: generar CSV estratificado blind para κ Cohen
uv run python -m biofloc_review.run_validation_sample \
  data/processed/biofloc_final_decisions.parquet
# (el revisor completa human_decision con "cumple" / "no cumple")
uv run python -m biofloc_review.run_kappa_analysis \
  data/processed/validation_sample_blind.csv \
  data/processed/validation_sample_truth.parquet

# Layer 3 Tier 1: fetch fulltext OA libre (sin cookies)
# Recupera los ~14% de PDFs disponibles por OpenAlex OA + Europe PMC en formato OA.
uv run python -m biofloc_review.run_fulltext_fetch \
  data/processed/biofloc_enriched.parquet \
  data/processed/biofloc_final_decisions.parquet \
  --workers 2 --delay 0.8 --skip-session-check --resume

# Layer 3 Tier 2/3: fetch via EZproxy USAL con cookies del navegador
# El usuario debe autenticarse manualmente y exportar cookies.txt (Netscape).
# Si la sesión CAS está activa (cookie CASTGC en .identidad.usal.es) cubre el ~82%
# adicional con DOI; si solo hay cookies del proxy (.idm.oclc.org) el flow cae
# en login CAS y los DOIs vuelven HTML, no PDF. Reset session via re-login fresh.
uv run python -m biofloc_review.run_fulltext_fetch \
  data/processed/biofloc_enriched.parquet \
  data/processed/biofloc_final_decisions.parquet \
  --cookies /path/to/cookies.txt --workers 2 --delay 1.5 --resume

# Análisis post-cribado: clustering temático UMAP+HDBSCAN sobre el corpus elegible
# Requiere umap-learn + hdbscan + scikit-learn instalados en el venv.
# Salida: 5 archivos en data/processed/ (vectors, umap2d, clusters, assignment, summary)
uv run python -m biofloc_review.run_thematic_clustering \
  data/processed/biofloc_final_decisions.parquet \
  data/processed/biofloc_final_primary_v3.parquet \
  --collection papers_biofloc --min-cluster-size 40

# Análisis post-cribado: identificación de métodos en los PDFs descargados
# Patrones regex sobre primeras 25 págs (ASM, ODE, PDE, ABM, PBM, CFD, ML/ANN,
# PINN, Stochastic, + 4 contextos de aplicación).
uv run python scripts/extract_methods.py
# → data/processed/fulltext_methods.parquet

# Generar las figuras del manuscrito
uv run python -m biofloc_review.run_plots

# Compilar el manuscrito (dos pasadas para referencias cruzadas)
cd docs && pdflatex revision_sistematica_biofloc_es.tex && pdflatex revision_sistematica_biofloc_es.tex
```

## Troubleshooting

**`Permission denied` con Docker.**
Tu usuario no está en el grupo `docker`. Solución:
```bash
sudo usermod -aG docker $USER
# Cerrar sesión y volver a entrar, o:
newgrp docker
```

**Qdrant no levanta.**
Verificá que el puerto 6333 está libre (`sudo lsof -i :6333`) y que el
volumen `qdrant_storage/` existe y tiene permisos de lectura/escritura.

**Ollama responde 401.**
El `OLLAMA_TOKEN` en `.env` está desactualizado o ausente. Pedile al
admin del servidor un nuevo token y reemplazalo en `.env`.

**OpenAlex devuelve 429.**
Bajá el rate limit: `--rps 0.5`. Si persiste varias horas, esperá 24h
antes de reintentar (el bloqueo por IP puede durar ese tiempo).

**Falta una fuente.**
Los CSVs del workspace son los archivos crudos. Si falta uno, exportá
de nuevo desde la base correspondiente y colocalo en el directorio
padre de `biofloc-review/`.

**Layer 1 crasheó / quiero reanudar.**
El parquet `data/processed/biofloc_screened.parquet` se actualiza cada 25
records (checkpoint). Para reanudar desde el último checkpoint:
```bash
setsid bash -c 'cd $PWD && PATH="$HOME/.local/bin:$PATH" exec uv run python \
  -m biofloc_review.run_screening data/processed/biofloc_final_primary.parquet \
  --workers 2 --checkpoint-every 25 --resume \
  > screening_full.log 2>&1' < /dev/null & disown
```

**arxiv timeouts en Layer 2.**
arxiv rate-limitea fuerte (3+ seg entre calls). Layer 2 no usa arxiv por
default. Si querés activarlo: `--with-arxiv` (no recomendado para corpus
wastewater por baja cobertura; útil para el corpus biplot).
