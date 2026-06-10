# Disponibilidad de datos — Floccus

Este repositorio publica el **código del pipeline**, la **metodología**
(criterios PRISMA + reglas de agregación), el **manuscrito** y las
**figuras generadas**. No publica los archivos crudos descargados de las
bases bibliográficas comerciales por las razones detalladas abajo.

## Qué se publica

| Componente | Ubicación | Estado |
|---|---|---|
| Código fuente del pipeline | `src/biofloc_review/` | ✅ |
| Scripts de configuración | `scripts/` | ✅ |
| Manuscrito completo (LaTeX) | `docs/revision_sistematica_biofloc_es.tex` | ✅ |
| Figuras generadas | `docs/figures/*.{pdf,png,tex}` | ✅ |
| Conteos PRISMA agregados | `data/processed/prisma_counts.json` (al hacer release) | ✅ |
| Criterios PRISMA + prompts LLM | `src/biofloc_review/screening/{criteria,prompts}.py` | ✅ |
| Schema del registro unificado | `src/biofloc_review/schema.py` (`PaperRecord`) | ✅ |
| Dependencias bloqueadas | `pyproject.toml` + `uv.lock` | ✅ |

## Qué NO se publica y por qué

### Exports crudos de las bases bibliográficas

Los archivos `IEEE.csv`, `WOS.xls`, `Scielo.csv` y `scopus_export_*.csv`
del directorio padre del workspace **no se redistribuyen** porque los
términos de servicio de Scopus, Web of Science e IEEE Xplore **prohíben
la redistribución de sus exports**, incluso para uso académico
reproducible. SciELO permite redistribución (es Open Access) pero por
consistencia operativa también se omite.

**Reproducción**: las cadenas de búsqueda literales para cada base están
documentadas en el manuscrito, sección §2.2 *Fuentes y estrategia de
búsqueda*. Cualquier investigador con credenciales institucionales puede
re-ejecutar las búsquedas en el período declarado y obtener un conjunto
operativamente equivalente.

### Abstracts y texto completo

Los archivos Parquet intermedios (`data/processed/*.parquet`) contienen
títulos, abstracts y palabras clave de los artículos del corpus. Estos
campos son **contenido sujeto a copyright de los editores** (Elsevier,
Springer, Wiley, ACS, etc.) y no son redistribuibles. El repositorio
ignora `data/` por defecto en `.gitignore`.

### Vectores de embedding

Los snapshots de Qdrant en `data/qdrant_snapshots/` contienen
representaciones densas de los abstracts (4096 dimensiones) generadas por
`qwen3-embedding:8b`. Aunque son representaciones derivadas, podrían en
teoría usarse para reconstrucción parcial del contenido y, por
precaución, también se excluyen.

### Credenciales

El archivo `.env` con tokens de Ollama y email para OpenAlex/Europe PMC
es estrictamente local. La plantilla `env.example` muestra qué variables
configurar.

## Datos derivados publicables (al hacer release del paper)

Al momento de la publicación formal, se generará y publicará en este
repositorio (etiquetados como release `v1.0`) los siguientes datos
agregados que **no constituyen redistribución del contenido**:

1. **Decisiones por criterio sanitizadas** (`data/published/screening_results.parquet`):
   - `record_id` (UUID5 derivado, no reversible al contenido original)
   - `cluster_id` (SHA-1 truncado del título normalizado)
   - `doi`, `source`, `year` (metadatos bibliográficos no copyrightables)
   - Respuesta por criterio (`I1_answer`, `E1_answer`, etc.) — solo YES/NO/UNCLEAR
     sobre los 11 criterios activos (I1-I4, E1-E4, D1-D3)
   - Decisión final consolidada (INCLUDE / HUMAN_REVIEW / EXCLUDE)
   - Sin abstracts, sin evidence quotes textuales

2. **Conteos PRISMA agregados** (`data/published/prisma_counts.json`):
   - Conteos por fase del flujo PRISMA 2020
   - Distribución de decisiones por regla (primaria, agresiva, permisiva)
   - Métricas de coverage de full-text (sin URLs específicas)

3. **Validación humana κ Cohen sanitizada** (`data/published/kappa_validation.json`):
   - Estructura del muestreo estratificado (n por celda source × decisión)
   - κ overall + κ por fuente + κ por criterio, con IC 95% por bootstrap
   - Interpretación Landis & Koch (1977) — *slight, fair, moderate, substantial, almost perfect*
   - Mapeo LLM 3-clases → humano binario (`cumple` / `no cumple`) documentado
   - Sin `record_id` ni abstracts: solo agregados por estrato

4. **Snapshot del manuscrito al momento de la publicación**:
   - `docs/published/revision_sistematica_biofloc_es_v1.0.pdf`

## Reproducción completa del pipeline

Cualquier investigador puede reproducir el pipeline siguiendo este flujo:

1. Clonar este repositorio.
2. Seguir `SETUP.md` para instalar dependencias (uv, Docker, acceso a Ollama).
3. Exportar las bases bibliográficas usando las cadenas documentadas en §2.2
   del manuscrito y colocar los archivos en el directorio padre del
   workspace.
4. Ejecutar el pipeline en el orden documentado en `README.md`:
   ingest → enrich → dedup (Capas 1+2+3) → screening Layer 1 → enricher
   Layer 2 → screening Layer 2 LLM → consolidación → (opcional)
   validación κ con revisor humano → (opcional) Layer 3 fulltext via
   EZproxy.
5. Comparar los conteos PRISMA producidos con los publicados en este
   repositorio (deberían coincidir dentro del margen de varianza
   estocástica del 3–5% inherente al cómputo en GPU bajo concurrencia,
   reportado en §3 del manuscrito).

## Solicitudes de datos

Para uso colaborativo académico que requiera acceso a los datos crudos
(p. ej. validación de la regla asimétrica sobre el mismo corpus exacto),
contactar al autor de correspondencia: **Purificación Galindo
(REDACTED_EMAIL)**. Las solicitudes se evalúan caso a caso
respetando los términos de servicio de cada base.
