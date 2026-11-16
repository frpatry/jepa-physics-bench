#!/bin/bash
# Pousse une session de fond Kaggle (T4) du bébé v3c.   bash kaggle_bebe3c_push.sh a|b [autre] [HEURES] [TOTAL] ["EXTRA"]
#   a|b : le kernel à lancer ; « autre » = 1 si l'autre kernel a déjà une sortie à reprendre (sinon seul le Dataset de départ est attaché)
#   relais : a (dataset) -> b (dataset + sortie de a) -> a (dataset + sortie de b) -> …
K=${1:?a ou b}; AUTRE=${2:-0}; H=${3:-11}; TOTAL=${4:-110000}; EXTRA=${5:-}
O=$([ "$K" = a ] && echo b || echo a); DIR=$(mktemp -d); HERE="$(cd "$(dirname "$0")" && pwd)"
{ echo "import os; os.environ.update(HEURES='$H', TOTAL='$TOTAL', EXTRA='$EXTRA')"; cat "$HERE/kaggle_bebe3c.py"; } > "$DIR/bebe3c.py"
SRC=$([ "$AUTRE" = 1 ] && echo "\"francoispatry/bebe3c-$O\"" || echo "")
cat > "$DIR/kernel-metadata.json" <<J
{"id": "francoispatry/bebe3c-$K", "title": "bebe3c-$K", "code_file": "bebe3c.py", "language": "python", "kernel_type": "script", "is_private": true,
 "enable_gpu": true, "enable_internet": true, "dataset_sources": ["francoispatry/bebe3c-ckpt"], "kernel_sources": [$SRC], "competition_sources": []}
J
"$HERE/arc3/.venv/bin/kaggle" kernels push -p "$DIR" --accelerator NvidiaTeslaT4
