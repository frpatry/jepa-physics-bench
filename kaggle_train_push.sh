#!/bin/bash
# Pousse une session de fond Kaggle (T4) d'entraînement.   bash kaggle_train_push.sh <nom> <recette.sh> a|b [autre] [HEURES] [TOTAL] ["EXTRA"]
#   kernels <nom>-a / <nom>-b qui se relaient ; « autre » = 1 si l'autre kernel a une sortie à reprendre (0 = départ de zéro)
NOM=${1:?nom}; REC=${2:?recette}; K=${3:?a ou b}; AUTRE=${4:-0}; H=${5:-11}; TOTAL=${6:-110000}; EXTRA=${7:-}
O=$([ "$K" = a ] && echo b || echo a); DIR=$(mktemp -d); HERE="$(cd "$(dirname "$0")" && pwd)"
{ echo "import os; os.environ.update(NOM='$NOM', RECETTE='$REC', HEURES='$H', TOTAL='$TOTAL', EXTRA='$EXTRA')"; cat "$HERE/kaggle_train.py"; } > "$DIR/train.py"
SRC=$([ "$AUTRE" = 1 ] && echo "\"francoispatry/$NOM-$O\"" || echo "")
cat > "$DIR/kernel-metadata.json" <<J
{"id": "francoispatry/$NOM-$K", "title": "$NOM-$K", "code_file": "train.py", "language": "python", "kernel_type": "script", "is_private": true,
 "enable_gpu": true, "enable_internet": true, "dataset_sources": [], "kernel_sources": [$SRC], "competition_sources": []}
J
"$HERE/arc3/.venv/bin/kaggle" kernels push -p "$DIR" --accelerator NvidiaTeslaT4
