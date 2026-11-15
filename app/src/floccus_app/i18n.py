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
    "tab_project": {"es": "Proyecto", "en": "Project"},
    "tab_protocol": {"es": "Protocolo", "en": "Protocol"},
    "tab_run": {"es": "Ejecutar", "en": "Run"},
    "tab_results": {"es": "Resultados", "en": "Results"},
    "tab_review": {"es": "Revisión humana", "en": "Human review"},
    "tab_llm": {"es": "Modelo de IA", "en": "AI model"},
    "protocol_name": {"es": "Nombre del protocolo", "en": "Protocol name"},
    "topic": {"es": "Tema de la revisión", "en": "Review topic"},
    "prompt_language": {"es": "Idioma de los prompts", "en": "Prompt language"},
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
    "step_l1_what": {"es": "Cada criterio del protocolo es una pregunta Sí/No al LLM sobre título y abstract (salida = incluidos + revisión humana).",
                     "en": "Each protocol criterion is a Yes/No question to the LLM over title and abstract (output = included + human review)."},
    "step_l2": {"es": "4. Cribado capa 2", "en": "4. Screening layer 2"},
    "step_l2_what": {"es": "Los dudosos se re-evalúan con metadatos enriquecidos de OpenAlex y Europe PMC (salida = elegibles).",
                     "en": "Uncertain records are re-assessed with enriched OpenAlex and Europe PMC metadata (output = eligible)."},
    "run_done_example": {"es": "Este proyecto de ejemplo ya fue ejecutado completo: son los resultados publicados de la revisión biofloc. Usalo para ver cómo queda un proyecto terminado.",
                         "en": "This example project has already been run end to end: these are the published results of the biofloc review. Use it to see what a finished project looks like."},
    "run_model_used": {"es": "Modelo usado", "en": "Model used"},
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
    "fig_umap": {"es": "Mapa temático de los registros elegibles (UMAP)", "en": "Thematic map of eligible records (UMAP)"},
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

    "tab_start": {"es": "Inicio", "en": "Start"},
    "tagline": {"es": "Cribado asistido por IA para revisiones sistemáticas · libre y abierto",
                "en": "AI-assisted screening for systematic reviews · free and open"},
    "start_what": {
        "es": "Floccus aplica los criterios de inclusión y exclusión de tu revisión a miles de registros bibliográficos con un modelo de lenguaje, criterio por criterio, y deja cada decisión trazable para la revisión humana. El método fue validado en una revisión publicada sobre modelado de biofloc, incluida como proyecto de ejemplo.",
        "en": "Floccus applies your review's inclusion and exclusion criteria to thousands of bibliographic records with a language model, one criterion at a time, and keeps every decision traceable for human review. The method was validated in a published review on biofloc modelling, included as the example project.",
    },
    "start_steps_title": {"es": "Antes de empezar", "en": "Before you start"},
    "start_steps": {
        "es": "1. **Recomendado:** instalá [Gentle Shell]({gentle}), el entorno de trabajo con agentes de IA con el que se desarrolla Floccus.\n"
              "2. Instalá Floccus: `git clone {repo}` y, dentro de la carpeta, `uv sync --all-packages`.\n"
              "3. Configurá el modelo de IA en el archivo `.env` (ver `env.example`): Ollama local y gratis, o Anthropic pago.\n"
              "4. Creá un proyecto desde el **ejemplo biofloc** (barra lateral → Nuevo proyecto → Ejemplo) para ver cómo queda una revisión terminada.\n"
              "5. Creá tu proyecto en la barra lateral y escribí tu protocolo en la pestaña **Protocolo**.",
        "en": "1. **Recommended:** install [Gentle Shell]({gentle}), the AI-agent workspace Floccus is developed with.\n"
              "2. Install Floccus: `git clone {repo}` and, inside the folder, `uv sync --all-packages`.\n"
              "3. Set the AI model in the `.env` file (see `env.example`): local free Ollama, or paid Anthropic.\n"
              "4. Create a project from the **biofloc example** (sidebar → New project → Example) to see what a finished review looks like.\n"
              "5. Create your project in the sidebar and write your protocol in the **Protocol** tab.",
    },
    "cite_title": {"es": "Cómo citar", "en": "How to cite"},
    "cite_text": {"es": "Si usás Floccus en un trabajo, citá el software: DOI [{doi}]({doi_url}).",
                  "en": "If you use Floccus in your work, please cite the software: DOI [{doi}]({doi_url})."},
    "feedback_title": {"es": "Tu opinión mejora Floccus", "en": "Your feedback improves Floccus"},
    "feedback_text": {
        "es": "Los botones abren un formulario de GitHub ya completado con la versión y la pantalla. La app no envía nada: vos revisás el texto y lo publicás. No incluyas claves ni datos no publicados.",
        "en": "The buttons open a GitHub form prefilled with the version and screen. The app sends nothing: you review the text and submit it. Do not include keys or unpublished data.",
    },
    "feedback_bug": {"es": "🐞 Reportar un error", "en": "🐞 Report a problem"},
    "feedback_idea": {"es": "💡 Sugerir una mejora", "en": "💡 Suggest an improvement"},
    "theme_hint": {"es": "Modo claro u oscuro: menú ⋮ (arriba a la derecha) → Settings → Theme.",
                   "en": "Light or dark mode: ⋮ menu (top right) → Settings → Theme."},
    "run_exports": {"es": "1. Exports de las bases de datos", "en": "1. Database exports"},
    "run_exports_help": {"es": "Exportá los resultados de tu búsqueda desde cada base (Scopus y IEEE: CSV; Web of Science y ProQuest: Excel; SciELO: CSV) y subilos acá.",
                         "en": "Export your search results from each database (Scopus and IEEE: CSV; Web of Science and ProQuest: Excel; SciELO: CSV) and upload them here."},
    "source": {"es": "Base de datos", "en": "Database"},
    "files": {"es": "Archivos", "en": "Files"},
    "save_files": {"es": "Guardar archivos", "en": "Save files"},
    "saved_files": {"es": "Archivos guardados", "en": "Files saved"},
    "no_exports": {"es": "Todavía no subiste exports.", "en": "No exports uploaded yet."},
    "run_steps": {"es": "2. Pasos del pipeline", "en": "2. Pipeline steps"},
    "btn_import": {"es": "Importar", "en": "Import"},
    "btn_dedup": {"es": "Deduplicar", "en": "Deduplicate"},
    "btn_dedup_ai": {"es": "Deduplicar con IA", "en": "AI deduplication"},
    "btn_clusters": {"es": "Clusters temáticos", "en": "Thematic clusters"},
    "steps_help": {"es": "Orden: Importar → Deduplicar → (opcional) Deduplicar con IA → Cribado capa 1 → Cribado capa 2 → (opcional) Clusters temáticos. "
                         "La deduplicación con IA y los clusters usan embeddings de Ollama guardados en un Qdrant local dentro del proyecto (sin Docker).",
                   "en": "Order: Import → Deduplicate → (optional) AI deduplication → Screening layer 1 → Screening layer 2 → (optional) Thematic clusters. "
                         "AI deduplication and clusters use Ollama embeddings stored in a local Qdrant inside the project (no Docker)."},
    "btn_l1": {"es": "Cribado capa 1", "en": "Screening layer 1"},
    "btn_l2": {"es": "Cribado capa 2", "en": "Screening layer 2"},
    "limit": {"es": "Probar primero con N registros (0 = todos)", "en": "Try N records first (0 = all)"},
    "running": {"es": "Hay una corrida en curso", "en": "A run is in progress"},
    "launched": {"es": "Paso lanzado en segundo plano", "en": "Step launched in the background"},
    "status": {"es": "Estado", "en": "Status"},
    "errors": {"es": "Errores de IA", "en": "AI errors"},
    "errors_warn": {"es": "Hubo llamadas a la IA que fallaron: esos registros quedan como dudosos. Revisá la configuración del modelo.",
                    "en": "Some AI calls failed: those records stay uncertain. Check the AI model settings."},
    "log": {"es": "Registro de la corrida", "en": "Run log"},
    "cost_title": {"es": "3. Costo estimado del cribado", "en": "3. Estimated screening cost"},
    "cost_local": {"es": "Con Ollama local el cribado no tiene costo de API.", "en": "With local Ollama, screening has no API cost."},
    "price_in": {"es": "Precio de entrada (USD por millón de tokens)", "en": "Input price (USD per million tokens)"},
    "price_out": {"es": "Precio de salida (USD por millón de tokens)", "en": "Output price (USD per million tokens)"},
    "price_help": {"es": "Consultá los precios vigentes de tu modelo en la página del proveedor.",
                   "en": "Check your model's current prices on the provider's page."},
    "cost_result": {"es": "≈ {calls} llamadas · {tin} tokens de entrada · {tout} de salida · **≈ USD {cost}**",
                    "en": "≈ {calls} calls · {tin} input tokens · {tout} output · **≈ USD {cost}**"},
    "cost_need_dedup": {"es": "Deduplicá primero para estimar con el número real de registros.",
                        "en": "Deduplicate first to estimate with the real number of records."},
    "progress_title": {"es": "4. Progreso", "en": "4. Progress"},
    "own_results_note": {"es": "Resultados de la última corrida de este proyecto.", "en": "Results of this project's latest run."},
    "rv_kind": {"es": "Qué revisar", "en": "What to review"},
    "rv_uncertain": {"es": "Registros dudosos (revisión humana)", "en": "Uncertain records (human review)"},
    "rv_validation": {"es": "Muestra de validación al azar", "en": "Random validation sample"},
    "rv_size": {"es": "Tamaño de la muestra", "en": "Sample size"},
    "rv_progress": {"es": "{done} de {total} revisados", "en": "{done} of {total} reviewed"},
    "rv_record": {"es": "Registro", "en": "Record"},
    "rv_pipeline": {"es": "Decisión del pipeline", "en": "Pipeline decision"},
    "rv_note": {"es": "Nota (opcional)", "en": "Note (optional)"},
    "rv_include": {"es": "✅ Incluir", "en": "✅ Include"},
    "rv_exclude": {"es": "❌ Excluir", "en": "❌ Exclude"},
    "rv_prev": {"es": "← Anterior", "en": "← Previous"},
    "rv_next": {"es": "Siguiente →", "en": "Next →"},
    "rv_empty": {"es": "No hay registros para revisar.", "en": "No records to review."},
    "rv_no_results": {"es": "Primero corré el cribado en la pestaña Ejecutar.", "en": "Run the screening first in the Run tab."},
    "rv_agreement": {"es": "Acuerdo pipeline–revisor (muestra de validación)", "en": "Pipeline–reviewer agreement (validation sample)"},
    "rv_agreement_help": {"es": "El pipeline cuenta como 'incluir' si el registro avanza (incluido o dudoso) y 'excluir' si lo excluye.",
                          "en": "The pipeline counts as 'include' when the record advances (included or uncertain) and 'exclude' when excluded."},
    "rv_kappa": {"es": "Kappa de Cohen", "en": "Cohen's kappa"},
    "rv_percent": {"es": "Acuerdo (%)", "en": "Agreement (%)"},
    "rv_reviewed": {"es": "Revisados", "en": "Reviewed"},
    "rv_saved_as": {"es": "Tu decisión guardada", "en": "Your saved decision"},
    "step_import": {"es": "Importar", "en": "Import"},
    "step_dedup": {"es": "Deduplicar", "en": "Deduplicate"},
    "step_dedup-ai": {"es": "Deduplicar con IA", "en": "AI deduplication"},
    "step_screen-l1": {"es": "Cribado capa 1", "en": "Screening layer 1"},
    "step_screen-l2": {"es": "Cribado capa 2", "en": "Screening layer 2"},
    "step_clusters": {"es": "Clusters temáticos", "en": "Thematic clusters"},
    "step_consolidate": {"es": "Consolidar resultados", "en": "Consolidate results"},
    "status_starting": {"es": "iniciando", "en": "starting"},
    "status_running": {"es": "en curso", "en": "running"},
    "status_done": {"es": "terminado", "en": "finished"},
    "status_failed": {"es": "falló", "en": "failed"},
    "dec_INCLUDE": {"es": "Incluido", "en": "Included"},
    "dec_EXCLUDE": {"es": "Excluido", "en": "Excluded"},
    "dec_HUMAN_REVIEW": {"es": "Revisión humana", "en": "Human review"},
    "temperature": {"es": "temperatura", "en": "temperature"},
    "no_abstract": {"es": "(sin abstract)", "en": "(no abstract)"},
    "footer": {"es": "Floccus {version} · Licencia {license} · DOI [{doi}]({doi_url}) · [Código fuente]({repo})",
               "en": "Floccus {version} · {license} licence · DOI [{doi}]({doi_url}) · [Source code]({repo})"},

}


def _label(prefix: str, code: str, lang: str) -> str:
    key = f"{prefix}{code}"
    return t(key, lang) if key in STRINGS else code


def step_label(step: str, lang: str = DEFAULT_LANG) -> str:
    return _label("step_", step, lang)


def status_label(status: str, lang: str = DEFAULT_LANG) -> str:
    return _label("status_", status, lang)


def decision_label(decision: str, lang: str = DEFAULT_LANG) -> str:
    return _label("dec_", decision, lang)


def t(key: str, lang: str = DEFAULT_LANG) -> str:
    texts = STRINGS[key]
    return texts.get(lang) or texts[DEFAULT_LANG]
