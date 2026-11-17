#!/bin/bash
# Pousse une commande de diagnostic en session de fond Kaggle (T4).   bash kaggle_cmd_push.sh <nom> a|b "<commande avec {ck} et {D}>" [CKPT_PAS]
NOM=${1:?nom du kernel}; SRC=${2:?a ou b}; CMD=${3:?commande}; PAS=${4:-0}; DIR=$(mktemp -d); HERE="$(cd "$(dirname "$0")" && pwd)"
{ printf 'import os; os.environ.update(CKPT_PAS=%s, CMD=%s)\n' "'$PAS'" "$(python3 -c 'import sys; print(repr(sys.argv[1]))' "$CMD")"; cat "$HERE/kaggle_cmd.py"; } > "$DIR/cmd.py"
cat > "$DIR/kernel-metadata.json" <<J
{"id": "francoispatry/$NOM", "title": "$NOM", "code_file": "cmd.py", "language": "python", "kernel_type": "script", "is_private": true,
 "enable_gpu": true, "enable_internet": true, "dataset_sources": [], "kernel_sources": ["francoispatry/bebe3c-$SRC"], "competition_sources": []}
J
head -1 "$DIR/cmd.py"; "$HERE/arc3/.venv/bin/kaggle" kernels push -p "$DIR" --accelerator NvidiaTeslaT4
