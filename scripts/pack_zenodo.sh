#!/usr/bin/env bash
# pack_zenodo.sh — Build the Zenodo deposit archive.
#
# NOT the same as pack.sh: that one packs the whole workspace (data, notebooks,
# manuscripts) to move it between machines. This one builds the *public*
# archive that gets a permanent DOI, so it must contain code and metadata only.
#
# The file list comes from git itself:
#   git ls-files --cached --others --exclude-standard
# which yields tracked files plus untracked ones that are NOT gitignored. The
# .gitignore is therefore the single source of truth for what stays private —
# manuscripts, cover letters, data/, PDFs and credentials never enter the zip.
#
# The archive is written OUTSIDE the repository so it can never be committed.
#
# Usage: bash scripts/pack_zenodo.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."                       # repository root
WORKSPACE_DIR="$(cd .. && pwd)"

VERSION="$(sed -n 's/^version: *"\{0,1\}\([^"]*\)"\{0,1\}/\1/p' CITATION.cff | head -1)"
[ -n "$VERSION" ] || { echo "could not read version from CITATION.cff" >&2; exit 1; }

ZIP_PATH="${WORKSPACE_DIR}/floccus-v${VERSION}.zip"
rm -f "$ZIP_PATH"

echo "==> Zenodo deposit for Floccus v${VERSION}"

# Refuse to ship anything that must stay private, even if .gitignore regresses.
FORBIDDEN='(^|/)(Anonimus|Array|Authors|cover_letter|highlights|revision_sistematica|supplementary_material)|\.(pdf|csv|xls|parquet)$|cookies|\.env$'
# release_data/ holds the derived decision tables (identifiers and decisions
# only, see docs/DATA.md): its CSV files are the one sanctioned exception.
ALLOWED='^release_data/[a-z0-9_]+\.csv$'
if git ls-files --cached --others --exclude-standard | grep -Ev "$ALLOWED" | grep -Ei "$FORBIDDEN"; then
    echo "ABORT: the files listed above must not be published." >&2
    exit 1
fi

git ls-files --cached --others --exclude-standard | sort -u | zip -q "$ZIP_PATH" -@

echo "==> $(unzip -Z1 "$ZIP_PATH" | wc -l) files, $(du -h "$ZIP_PATH" | cut -f1)"
echo "==> $ZIP_PATH"
