#!/bin/bash
# Pousse le test Push-T d'un bébé gelé en session de fond Kaggle (T4).   bash kaggle_pusht_push.sh a|b [CKPT_PAS] ["EXTRA"] [nom]
#   a|b : le kernel d'entraînement dont on prend la sortie ; CKPT_PAS = pas exact voulu (0 = le plus avancé)
SRC=${1:?a ou b}; PAS=${2:-0}; EXTRA=${3:-}; NOM=${4:-bebe3c-pusht}; DIR=$(mktemp -d); HERE="$(cd "$(dirname "$0")" && pwd)"
{ echo "import os; os.environ.update(CKPT_PAS='$PAS', EXTRA='$EXTRA')"; cat "$HERE/kaggle_pusht.py"; } > "$DIR/pusht.py"
cat > "$DIR/kernel-metadata.json" <<J
{"id": "francoispatry/$NOM", "title": "$NOM", "code_file": "pusht.py", "language": "python", "kernel_type": "script", "is_private": true,
 "enable_gpu": true, "enable_internet": true, "dataset_sources": [], "kernel_sources": ["francoispatry/bebe3c-$SRC"], "competition_sources": []}
J
"$HERE/arc3/.venv/bin/kaggle" kernels push -p "$DIR" --accelerator NvidiaTeslaT4
