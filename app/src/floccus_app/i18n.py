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
}


def t(key: str, lang: str = DEFAULT_LANG) -> str:
    texts = STRINGS[key]
    return texts.get(lang) or texts[DEFAULT_LANG]
