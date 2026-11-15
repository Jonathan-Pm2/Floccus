"""User-interface strings in Spanish (default) and English."""

from __future__ import annotations

DEFAULT_LANG = "es"

STRINGS: dict[str, dict[str, str]] = {
    "app_title": {"es": "Floccus · Revisiones sistemáticas", "en": "Floccus · Systematic reviews"},
    "language": {"es": "Idioma", "en": "Language"},
    "project": {"es": "Proyecto", "en": "Project"},
    "no_projects": {"es": "Todavía no hay proyectos. Creá uno abajo.", "en": "No projects yet. Create one below."},
    "new_project": {"es": "Nuevo proyecto", "en": "New project"},
    "project_name": {"es": "Nombre (letras, números, - o _)", "en": "Name (letters, digits, - or _)"},
    "start_from": {"es": "Empezar desde", "en": "Start from"},
    "blank": {"es": "Protocolo en blanco", "en": "Blank protocol"},
    "biofloc_example": {"es": "Ejemplo: revisión biofloc publicada", "en": "Example: published biofloc review"},
    "create": {"es": "Crear", "en": "Create"},
    "created": {"es": "Proyecto creado", "en": "Project created"},
    "tab_project": {"es": "Proyecto", "en": "Project"},
    "tab_protocol": {"es": "Protocolo", "en": "Protocol"},
    "tab_run": {"es": "Ejecutar", "en": "Run"},
    "tab_results": {"es": "Resultados", "en": "Results"},
    "tab_review": {"es": "Revisión humana", "en": "Human review"},
    "tab_llm": {"es": "Modelo de IA", "en": "AI model"},
    "protocol_name": {"es": "Nombre del protocolo", "en": "Protocol name"},
    "topic": {"es": "Tema de la revisión", "en": "Review topic"},
    "prompt_language": {"es": "Idioma de los prompts", "en": "Prompt language"},
    "criteria": {"es": "Criterios", "en": "Criteria"},
    "inclusion": {"es": "Inclusión", "en": "Inclusion"},
    "exclusion": {"es": "Exclusión", "en": "Exclusion"},
    "derivation": {"es": "Derivación a revisión humana", "en": "Routed to human review"},
    "folder": {"es": "Carpeta", "en": "Folder"},
    "criteria_help": {
        "es": "Inclusión: todos deben cumplirse. Exclusión: si uno se cumple, se excluye. "
              "Derivación: si se cumple, va a revisión humana. 'check' es una regla automática sin IA "
              "(document_type o abstract_too_short) y 'params' sus parámetros en YAML, p. ej. {min_words: 100}.",
        "en": "Inclusion: all must hold. Exclusion: any one excludes. Derivation: sends the record to "
              "human review. 'check' is an automatic rule without AI (document_type or abstract_too_short) "
              "and 'params' its parameters in YAML, e.g. {min_words: 100}.",
    },
    "validate": {"es": "Validar", "en": "Validate"},
    "save": {"es": "Guardar", "en": "Save"},
    "valid": {"es": "El protocolo es válido.", "en": "The protocol is valid."},
    "saved": {"es": "Protocolo guardado.", "en": "Protocol saved."},
    "invalid": {"es": "Protocolo inválido", "en": "Invalid protocol"},
    "under_construction": {"es": "En construcción", "en": "Under construction"},
    "run_soon": {
        "es": "Acá vas a lanzar el pipeline (carga, deduplicación, cribado) y ver el avance en vivo.",
        "en": "Here you will launch the pipeline (import, dedup, screening) and watch live progress.",
    },
    "results_soon": {
        "es": "Acá vas a ver los gráficos: fuentes, deduplicación, PRISMA, Sankey y clusters.",
        "en": "Here you will see the plots: sources, dedup, PRISMA, Sankey and clusters.",
    },
    "review_soon": {
        "es": "Acá vas a revisar a mano los registros dudosos y calcular el acuerdo (kappa).",
        "en": "Here you will review uncertain records by hand and compute agreement (kappa).",
    },
    "provider": {"es": "Proveedor", "en": "Provider"},
    "model": {"es": "Modelo", "en": "Model"},
    "url": {"es": "URL del servidor", "en": "Server URL"},
    "api_key": {"es": "API key", "en": "API key"},
    "llm_help": {
        "es": "Se configura en el archivo .env (ver env.example). Ollama es local y gratis; "
              "LLM_PROVIDER=anthropic usa la API paga de Anthropic. La key nunca se muestra.",
        "en": "Set these in the .env file (see env.example). Ollama is local and free; "
              "LLM_PROVIDER=anthropic uses the paid Anthropic API. The key is never shown.",
    },
    "step": {"es": "Paso", "en": "Step"},
    "what": {"es": "Qué hace", "en": "What it does"},
    "input": {"es": "Entrada", "en": "Input"},
    "output": {"es": "Salida", "en": "Output"},
    "step_import": {"es": "1. Importación", "en": "1. Import"},
    "step_import_what": {"es": "Lee los exports de las bases (entrada = número de bases, salida = registros identificados).",
                         "en": "Reads the database exports (input = number of databases, output = identified records)."},
    "step_dedup": {"es": "2. Deduplicación", "en": "2. Deduplication"},
    "step_dedup_what": {"es": "DOI/título exactos, similitud de embeddings y un juez LLM para los pares dudosos.",
                        "en": "Exact DOI/title, embedding similarity and an LLM judge for doubtful pairs."},
    "step_l1": {"es": "3. Cribado capa 1", "en": "3. Screening layer 1"},
    "step_l1_what": {"es": "Cada criterio del protocolo es una pregunta Sí/No al LLM sobre título y abstract (salida = INCLUDE + revisión humana).",
                     "en": "Each protocol criterion is a Yes/No question to the LLM over title and abstract (output = INCLUDE + human review)."},
    "step_l2": {"es": "4. Cribado capa 2", "en": "4. Screening layer 2"},
    "step_l2_what": {"es": "Los dudosos se re-evalúan con metadatos enriquecidos de OpenAlex y Europe PMC (salida = elegibles).",
                     "en": "Uncertain records are re-assessed with enriched OpenAlex and Europe PMC metadata (output = eligible)."},
    "run_done_example": {"es": "Este proyecto de ejemplo ya fue ejecutado completo: son los resultados publicados de la revisión biofloc. Usalo para ver cómo queda un proyecto terminado.",
                         "en": "This example project has already been run end to end: these are the published results of the biofloc review. Use it to see what a finished project looks like."},
    "run_model_used": {"es": "Modelo usado", "en": "Model used"},
    "run_new_soon": {"es": "Lanzar corridas nuevas desde acá está en construcción.", "en": "Launching new runs from here is under construction."},
    "m_identified": {"es": "Identificados", "en": "Identified"},
    "m_unique": {"es": "Únicos", "en": "Unique"},
    "m_include": {"es": "Incluidos", "en": "Included"},
    "m_review": {"es": "Revisión humana", "en": "Human review"},
    "m_exclude": {"es": "Excluidos", "en": "Excluded"},
    "m_eligible": {"es": "Elegibles", "en": "Eligible"},
    "n_unique": {"es": "Registros únicos", "en": "Unique records"},
    "n_duplicates": {"es": "Duplicados eliminados", "en": "Duplicates removed"},
    "n_l1_include": {"es": "Capa 1: incluidos", "en": "Layer 1: included"},
    "n_l1_review": {"es": "Capa 1: dudosos → capa 2", "en": "Layer 1: uncertain → layer 2"},
    "n_l1_exclude": {"es": "Capa 1: excluidos", "en": "Layer 1: excluded"},
    "n_include": {"es": "Incluidos", "en": "Included"},
    "n_review": {"es": "Revisión humana", "en": "Human review"},
    "n_exclude": {"es": "Excluidos", "en": "Excluded"},
    "fig_flow": {"es": "Flujo de registros por el pipeline", "en": "Record flow through the pipeline"},
    "fig_sources": {"es": "Registros identificados por base", "en": "Records identified per database"},
    "fig_final": {"es": "Decisión final", "en": "Final decision"},
    "fig_criteria": {"es": "Registros que cumplen cada criterio (capa 1, respuesta Sí)", "en": "Records meeting each criterion (layer 1, Yes answers)"},
    "fig_clusters": {"es": "Registros elegibles por cluster temático", "en": "Eligible records per thematic cluster"},
    "decisions_table": {"es": "Decisiones por registro", "en": "Decisions per record"},
    "filter_decision": {"es": "Filtrar por decisión final", "en": "Filter by final decision"},
    "download_csv": {"es": "Descargar CSV", "en": "Download CSV"},
    "no_results": {"es": "Este proyecto todavía no tiene resultados.", "en": "This project has no results yet."},
    "validation_intro": {"es": "Muestra ciega de validación: decisiones del pipeline frente a dos revisores humanos independientes.",
                         "en": "Blind validation sample: pipeline decisions against two independent human reviewers."},
    "reviewer_1": {"es": "Pipeline (filas) vs revisor 1 (columnas)", "en": "Pipeline (rows) vs reviewer 1 (columns)"},
    "reviewer_2": {"es": "Pipeline (filas) vs revisor 2 (columnas)", "en": "Pipeline (rows) vs reviewer 2 (columns)"},
    "records": {"es": "registros", "en": "records"},
    "model_used": {
        "es": "**Usado en el proyecto de ejemplo (revisión biofloc publicada): Ollama**, IA local y gratuita.\n\n"
              "- Generación (cribado y juez de duplicados): `qwen3:14b`, temperatura 0 (determinista), hasta 400 tokens por respuesta.\n"
              "- Embeddings (deduplicación y clusters): `qwen3-embedding:8b`.",
        "en": "**Used in the example project (published biofloc review): Ollama**, local and free AI.\n\n"
              "- Generation (screening and duplicate judge): `qwen3:14b`, temperature 0 (deterministic), up to 400 tokens per answer.\n"
              "- Embeddings (deduplication and clusters): `qwen3-embedding:8b`.",
    },
    "model_alternatives": {
        "es": "**Para tus proyectos podés elegir:**\n\n"
              "- **Ollama** con cualquier modelo que tengas instalado: local y gratis.\n"
              "- **Anthropic (Claude)**: API paga, útil si no tenés IA local. El costo crece con registros × criterios; estimalo antes de correr el corpus completo.\n"
              "- **Otro proveedor**: se puede agregar en el mismo punto del código que Anthropic.\n\n"
              "Los embeddings siempre usan Ollama. La configuración va en el archivo `.env` (ver `env.example`); esta pantalla nunca muestra direcciones ni claves.",
        "en": "**For your projects you can choose:**\n\n"
              "- **Ollama** with any model you have installed: local and free.\n"
              "- **Anthropic (Claude)**: paid API, useful without local AI. Cost grows with records × criteria; estimate it before a full run.\n"
              "- **Another provider**: can be added at the same point in the code as Anthropic.\n\n"
              "Embeddings always use Ollama. Settings go in the `.env` file (see `env.example`); this screen never shows addresses or keys.",
    },
    "model_current": {"es": "Configuración actual de esta instalación", "en": "Current configuration of this installation"},

}


def t(key: str, lang: str = DEFAULT_LANG) -> str:
    texts = STRINGS[key]
    return texts.get(lang) or texts[DEFAULT_LANG]
