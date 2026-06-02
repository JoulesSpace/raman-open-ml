#!/usr/bin/env bash
# Lint: every source directory must contain a `.folderinfo` file - a one-line
# plain-text description of what lives there. Mirrors the convention from the
# acoustic-drone-detection repo. Downloaded data, caches and VCS dirs are exempt.
set -euo pipefail

cd "$(dirname "$0")/.."

missing=0
while IFS= read -r d; do
  case "$d" in
    ./.git/*|./.git) continue ;;
    ./data/bacteria_id*|./data/polystyrene*) continue ;;
    *__pycache__*|*.egg-info*|*/.venv*) continue ;;
    *.pytest_cache*|*.ruff_cache*|*/.idea*) continue ;;
  esac
  if [ ! -f "$d/.folderinfo" ]; then
    echo "MISSING: $d/.folderinfo"
    missing=1
  fi
done < <(find . -type d | sort)

if [ "$missing" -ne 0 ]; then
  echo "folderinfo lint FAILED - add a one-line .folderinfo to each dir listed above."
  exit 1
fi
echo "folderinfo lint OK"
