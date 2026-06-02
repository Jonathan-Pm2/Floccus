#!/usr/bin/env bash
# pack.sh — Empaqueta el workspace para mover a otra PC.
#
# Genera Articulos_Data_biofloc_mate.zip en el directorio padre del workspace,
# excluyendo lo que se reconstruye con `uv sync` y archivos de basura.
#
# Uso: bash scripts/pack.sh

set -euo pipefail

# Asegurar ejecución desde la raíz del proyecto biofloc-review/
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."        # estamos ahora en biofloc-review/
WORKSPACE_DIR="$(cd .. && pwd)"
WORKSPACE_NAME="$(basename "$WORKSPACE_DIR")"
PARENT_DIR="$(cd "$WORKSPACE_DIR/.." && pwd)"
ZIP_PATH="$PARENT_DIR/${WORKSPACE_NAME}.zip"

echo "==> Empaquetando $WORKSPACE_NAME → $ZIP_PATH"

cd "$PARENT_DIR"

# Antes de zippear, generamos snapshots de las colecciones de Qdrant.
# Los snapshots se descargan via API y quedan como archivos del usuario, lo
# que evita el problema de portabilidad de qdrant_storage/ (escrito como root
# por el container Docker).
if docker ps --format '{{.Names}}' | grep -q '^qdrant-biofloc$'; then
    SNAP_DIR="${WORKSPACE_DIR}/biofloc-review/data/qdrant_snapshots"
    mkdir -p "$SNAP_DIR"
    echo "==> Generando snapshots de Qdrant..."
    for c in papers_biofloc papers_biplot; do
        if curl -fs "http://localhost:6333/collections/$c" >/dev/null 2>&1; then
            echo "    Colección: $c"
            curl -s -X POST "http://localhost:6333/collections/$c/snapshots" >/dev/null
            snap_name=$(curl -s "http://localhost:6333/collections/$c/snapshots" \
                | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['result'][-1]['name'])")
            curl -s -o "$SNAP_DIR/${c}.snapshot" "http://localhost:6333/collections/$c/snapshots/$snap_name"
            echo "      -> $SNAP_DIR/${c}.snapshot"
        fi
    done
else
    echo "==> Aviso: Qdrant no está corriendo. Si querés incluir vectores"
    echo "    indexados, levantá el contenedor antes y reejecutá pack.sh."
fi

# Lista de exclusiones — patrones glob relativos al directorio del workspace.
EXCLUDES=(
    "${WORKSPACE_NAME}/biofloc-review/.venv/*"
    "${WORKSPACE_NAME}/biofloc-review/.git/*"
    "${WORKSPACE_NAME}/biofloc-review/.git"
    "${WORKSPACE_NAME}/biofloc-review/**/__pycache__/*"
    "${WORKSPACE_NAME}/biofloc-review/**/*.pyc"
    "${WORKSPACE_NAME}/biofloc-review/*.log"
    "${WORKSPACE_NAME}/biofloc-review/docs/*.aux"
    "${WORKSPACE_NAME}/biofloc-review/docs/*.log"
    "${WORKSPACE_NAME}/biofloc-review/docs/*.out"
    "${WORKSPACE_NAME}/biofloc-review/docs/*.synctex.gz"
    "${WORKSPACE_NAME}/.atl/*"
    # qdrant_storage está bajo UID del container Docker (root): no es portable
    # entre máquinas. En su lugar empaquetamos los snapshots descargados via API,
    # que el setup.sh de la PC destino restaura.
    "${WORKSPACE_NAME}/biofloc-review/qdrant_storage/*"
    "${WORKSPACE_NAME}/biofloc-review/qdrant_storage"
)

EXCLUDE_ARGS=()
for pattern in "${EXCLUDES[@]}"; do
    EXCLUDE_ARGS+=("-x" "$pattern")
done

if [[ -f "$ZIP_PATH" ]]; then
    echo "==> Removiendo zip anterior"
    rm -f "$ZIP_PATH"
fi

echo "==> Comprimiendo (puede tardar 2-5 min)..."
zip -r -q "$ZIP_PATH" "$WORKSPACE_NAME" "${EXCLUDE_ARGS[@]}"

SIZE_MB=$(du -m "$ZIP_PATH" | cut -f1)
echo ""
echo "==> Listo. Archivo generado:"
echo "    $ZIP_PATH"
echo "    ${SIZE_MB} MB"
echo ""
echo "Para restaurarlo en la otra PC:"
echo "    unzip $WORKSPACE_NAME.zip"
echo "    cd $WORKSPACE_NAME/biofloc-review"
echo "    bash scripts/setup.sh"
