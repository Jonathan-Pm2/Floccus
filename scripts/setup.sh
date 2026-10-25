#!/usr/bin/env bash
# setup.sh — Restaura el proyecto en una máquina nueva tras descomprimir el zip.
#
# Verifica prerrequisitos, reconstruye el entorno Python, levanta Qdrant y
# valida la conectividad a Ollama.
#
# Uso: bash scripts/setup.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."        # biofloc-review/
PROJECT_DIR="$(pwd)"
WORKSPACE_DIR="$(cd .. && pwd)"

red()    { printf "\033[31m%s\033[0m\n" "$*"; }
green()  { printf "\033[32m%s\033[0m\n" "$*"; }
yellow() { printf "\033[33m%s\033[0m\n" "$*"; }
bold()   { printf "\033[1m%s\033[0m\n" "$*"; }

bold "== Setup biofloc-review =="
echo

# 1. Prerrequisitos
bold "1/5  Verificando prerrequisitos"

if ! command -v docker >/dev/null 2>&1; then
    red "  [X] docker no encontrado."
    echo "       Instalá con: curl -fsSL https://get.docker.com | sh"
    echo "       Luego: sudo usermod -aG docker \$USER  (y relogin)"
    exit 1
fi
green "  [OK] docker $(docker --version | awk '{print $3}' | tr -d ',')"

if ! command -v uv >/dev/null 2>&1; then
    red "  [X] uv no encontrado."
    echo "       Instalá con: curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi
green "  [OK] uv $(uv --version | awk '{print $2}')"

# 2. .env
bold "2/5  Verificando configuración"

ENV_LOCAL="${PROJECT_DIR}/.env"
ENV_WORKSPACE="${WORKSPACE_DIR}/.env"
ENV_FOUND=""
if [[ -f "$ENV_LOCAL" ]]; then
    ENV_FOUND="$ENV_LOCAL"
elif [[ -f "$ENV_WORKSPACE" ]]; then
    ENV_FOUND="$ENV_WORKSPACE"
fi

if [[ -z "$ENV_FOUND" ]]; then
    red "  [X] No se encontró .env ni en $PROJECT_DIR ni en $WORKSPACE_DIR"
    echo "       Creá uno con al menos:"
    echo "         OLLAMA_URL=https://<tu-tunel>"
    echo "         OLLAMA_TOKEN=<bearer-token>"
    echo "         OPENALEX_EMAIL=<tu-correo>"
    echo "         OPENALEX_RPS=1.0"
    exit 1
fi
green "  [OK] .env detectado en $ENV_FOUND"

# 3. uv sync
bold "3/5  Reconstruyendo entorno Python (uv sync)"
uv sync 2>&1 | tail -3
green "  [OK] entorno listo"

# 4. Qdrant
bold "4/5  Levantando Qdrant"

if docker ps -a --format '{{.Names}}' | grep -q '^qdrant-biofloc$'; then
    if docker ps --format '{{.Names}}' | grep -q '^qdrant-biofloc$'; then
        green "  [OK] qdrant-biofloc ya está corriendo"
    else
        echo "  Re-iniciando contenedor existente..."
        docker start qdrant-biofloc >/dev/null
        green "  [OK] qdrant-biofloc reiniciado"
    fi
else
    mkdir -p "${PROJECT_DIR}/qdrant_storage"
    echo "  Creando contenedor con almacenamiento persistente..."
    docker run -d \
        --name qdrant-biofloc \
        -p 6333:6333 -p 6334:6334 \
        -v "${PROJECT_DIR}/qdrant_storage:/qdrant/storage" \
        qdrant/qdrant:latest >/dev/null
    green "  [OK] qdrant-biofloc creado"
fi

# Esperar a que esté listo
for i in 1 2 3 4 5; do
    if curl -fs http://localhost:6333/healthz >/dev/null 2>&1; then
        green "  [OK] Qdrant responde en localhost:6333"
        break
    fi
    if [[ $i -eq 5 ]]; then
        red "  [X] Qdrant no respondió tras 5 intentos"
        echo "       Revisá: docker logs qdrant-biofloc"
        exit 1
    fi
    sleep 1
done

# Restaurar colecciones desde snapshots si existen
SNAP_DIR="${PROJECT_DIR}/data/qdrant_snapshots"
if [[ -d "$SNAP_DIR" ]] && [[ -n "$(ls -A "$SNAP_DIR" 2>/dev/null)" ]]; then
    echo "  Restaurando colecciones desde snapshots..."
    for snap in "$SNAP_DIR"/*.snapshot; do
        [[ -f "$snap" ]] || continue
        collection=$(basename "$snap" .snapshot)
        # Skip si la colección ya existe (no sobreescribir)
        if curl -fs "http://localhost:6333/collections/$collection" >/dev/null 2>&1; then
            yellow "    [-] $collection ya existe, no se restaura. Para forzar:"
            yellow "        curl -X DELETE http://localhost:6333/collections/$collection"
            continue
        fi
        echo "    [+] Restaurando $collection ($(du -h "$snap" | cut -f1))..."
        # Upload del snapshot al endpoint de recovery
        resp=$(curl -s -X POST \
            -F "snapshot=@${snap}" \
            "http://localhost:6333/collections/$collection/snapshots/upload?priority=snapshot")
        if echo "$resp" | grep -q '"status":"ok"'; then
            green "    [OK] $collection restaurada"
        else
            red "    [X] Falló restauración de $collection:"
            echo "       $resp" | head -c 500
        fi
    done
else
    yellow "  [!] No hay snapshots en $SNAP_DIR. Si esperabas restaurar"
    yellow "      colecciones pre-indexadas, verificá el .zip de origen."
fi

# 5. Smoke tests
bold "5/5  Smoke tests"

# Ollama
echo "  Probando Ollama (servidor remoto)..."
if uv run python -c "
from biofloc_review.dedup.ollama_embed import ollama_url, embed_batch
print('  URL:', ollama_url())
v = embed_batch(['test'])
print('  embedding dim:', len(v[0]))
" 2>&1; then
    green "  [OK] Ollama respondió correctamente"
else
    yellow "  [!] Ollama no respondió. Revisá OLLAMA_URL y OLLAMA_TOKEN en .env."
fi

# Estado del corpus
echo
echo "  Estado del corpus en data/processed/:"
uv run python -c "
import os, json
from pathlib import Path
import pandas as pd
base = Path('data/processed')
if not base.exists():
    print('    (sin datos procesados todavía)')
else:
    for name in ['biofloc.parquet','biplot.parquet','biofloc_final_primary.parquet','biplot_final_primary.parquet']:
        p = base / name
        if p.exists():
            df = pd.read_parquet(p)
            with_abs = df['abstract'].notna().sum() if 'abstract' in df.columns else None
            print(f'    {name:<40} {len(df):>6} records' + (f' ({with_abs} con abstract)' if with_abs is not None else ''))
" 2>&1

echo
bold "== Setup completo =="
echo "Próximos pasos sugeridos en README.md."
