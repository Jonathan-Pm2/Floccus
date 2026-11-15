# Guía de uso de Floccus

Floccus aplica los criterios de inclusión y exclusión de tu revisión
sistemática a miles de registros con un modelo de lenguaje, criterio por
criterio, y deja cada decisión trazable para la revisión humana.

## En 5 pasos

1. **Instalá** (una vez).
2. **Configurá** el modelo de IA.
3. **Creá un proyecto** y escribí tu protocolo.
4. **Subí los exports** y **corré** el pipeline.
5. **Revisá** los resultados y los registros dudosos.

## 1. Instalación

- Recomendado: instalá [Gentle Shell](https://github.com/Gentleman-Programming/gentle-shell),
  el entorno de trabajo con agentes de IA con el que se desarrolla Floccus.
- Instalá [uv](https://docs.astral.sh/uv/) y cloná el repositorio:

```bash
git clone https://github.com/Jonathan-Pm2/Floccus.git
cd Floccus
uv sync --all-packages
uv run floccus ui          # abre http://localhost:8501
```

## 2. Modelo de IA

Copiá `env.example` como `.env` y elegí una opción:

| Opción | Costo | Qué configurar |
|---|---|---|
| **Ollama** (local) | gratis | `OLLAMA_URL`, `OLLAMA_LLM_MODEL` (p. ej. `qwen3:14b`) |
| **Anthropic** (API) | pago | `LLM_PROVIDER=anthropic`, `ANTHROPIC_API_KEY`, `ANTHROPIC_MODEL` |
| OpenAlex (capa 2) | gratis | `OPENALEX_EMAIL` (tu correo, recomendado) |

La pestaña **Modelo de IA** muestra qué proveedor y modelo están activos; nunca
muestra direcciones ni claves. El ejemplo publicado usó Ollama con `qwen3:14b`
a temperatura 0.

## 3. Proyecto y protocolo

En la barra lateral, **Nuevo proyecto**: en blanco o a partir del ejemplo
biofloc. En la pestaña **Protocolo** definís:

- **Tema** de la revisión e **idioma de los prompts** (español o inglés).
- **Criterios**, cada uno una pregunta Sí/No:

| Tipo | Regla |
|---|---|
| `inclusion` | todos tienen que cumplirse |
| `exclusion` | si uno se cumple, el registro se excluye |
| `derivation` | si se cumple, el registro va a revisión humana |

- **Reglas automáticas** (`check`, sin IA): `document_type` (tipo de documento)
  y `abstract_too_short` (`params: {min_words: 100}`).

**Validar** muestra el error exacto; **Guardar** nunca escribe un protocolo inválido.

## 4. Exports y ejecución

En **Ejecutar**:

1. Subí los exports de cada base: Scopus, IEEE y SciELO en CSV; Web of Science
   y ProQuest en Excel.
2. Corré los pasos en orden: **Importar → Deduplicar → (opcional) Deduplicar
   con IA → Cribado capa 1 → Cribado capa 2 → (opcional) Clusters temáticos**.
   Cada paso corre en segundo plano y muestra su progreso.
   - **Deduplicar con IA** encuentra casi duplicados (títulos traducidos,
     preprint y versión publicada): compara embeddings de título y abstract
     (similitud ≥ 0,93) y un juez de IA confirma cada par.
   - **Clusters temáticos** agrupa los registros elegibles por tema (UMAP +
     HDBSCAN) y dibuja un mapa temático en Resultados.
   - Ambos usan embeddings de Ollama guardados en un Qdrant local dentro del
     proyecto (`data/qdrant/`): no hace falta Docker.
3. Probá primero con pocos registros (campo "Probar primero con N registros").
4. Si usás una API paga, cargá los precios de tu modelo y mirá el **costo
   estimado** antes de correr todo.

Si aparecen **errores de IA**, esos registros quedan como dudosos: revisá la
configuración del modelo. Un cribado interrumpido se retoma donde quedó.

## 5. Resultados y revisión humana

- **Resultados**: métricas, flujo de registros (Sankey), gráficos por base,
  decisión y criterio, y la tabla de decisiones descargable en CSV.
- **Revisión humana**: revisá uno por uno los dudosos o una muestra de
  validación al azar. Con la muestra, Floccus calcula el acuerdo y el kappa
  de Cohen entre el pipeline y tu criterio.

## Límites de esta versión

- La deduplicación con IA y los clusters necesitan Ollama (embeddings), aun
  si el cribado usa Anthropic.
- Con menos de 400 registros elegibles, los clusters usan tamaños mínimos más
  chicos que el estudio publicado (quedan registrados en `clusters.json`).
- La capa 2 consulta OpenAlex y Europe PMC: necesita internet y DOI.

## Comentarios

Usá los botones de la pestaña **Inicio** (reportar un error, sugerir una
mejora). No incluyas claves ni datos no publicados. Ver `CONTRIBUTING.md`.

## Cómo citar

DOI [10.5281/zenodo.21926426](https://doi.org/10.5281/zenodo.21926426)
(siempre la última versión). El artículo publicado usó la v0.3.0:
[10.5281/zenodo.23719341](https://doi.org/10.5281/zenodo.23719341).
