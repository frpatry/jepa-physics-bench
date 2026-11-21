#!/bin/bash
# Pousse une session de fond Kaggle pour l'étude histo_jepa.   bash kaggle_histo_push.sh a|b <étape> [autre] [HEURES] [gpu] ["EXTRA"] [BRANCHE] [jeu_lames]
#   kernels histo-a / histo-b qui se relaient ; « autre » = 1 si l'autre kernel a une sortie à reprendre (0 = départ de zéro)
#   gpu = 1 (deux T4, compté sur le quota) ou 0 (CPU seulement : essai de tuyauterie, ne coûte aucun quota GPU)
#   jeu_lames = jeu de données Kaggle « utilisateur/nom » contenant le fichier image → lame (facultatif)
#   Le CLI Kaggle est cherché dans <dépôt>/arc3/.venv/bin/kaggle ; autre chemin : KAGGLE=/chemin/kaggle bash kaggle_histo_push.sh …
# Exemples :  essai sans GPU   bash kaggle_histo_push.sh a tout 0 1 0 "--mini 2000 --out /kaggle/working/histo_mini"
#             chronométrage    bash kaggle_histo_push.sh a chrono 0 1 1
#             vrai run         bash kaggle_histo_push.sh a tout 0 11 1     puis   bash kaggle_histo_push.sh b tout 1 11 1   (et ainsi de suite)
# ATTENTION : le code lancé est celui de la BRANCHE sur GitHub (il faut l'avoir poussée). Ne pas lancer avec gpu=1 pendant qu'une session du bébé tourne sans accord.
K=${1:?a ou b}; ETAPE=${2:?étape}; AUTRE=${3:-0}; H=${4:-11}; GPU=${5:-1}; EXTRA=${6:-}; BR=${7:-master}; LAMES=${8:-}
O=$([ "$K" = a ] && echo b || echo a); DIR=$(mktemp -d); HERE="$(cd "$(dirname "$0")" && pwd)"; RACINE="$(cd "$HERE/../.." && pwd)"
{ echo "import os; os.environ.update(ETAPE='$ETAPE', HEURES='$H', BRANCHE='$BR', EXTRA='$EXTRA')"; cat "$HERE/kaggle_histo.py"; } > "$DIR/histo.py"
SRC=$([ "$AUTRE" = 1 ] && echo "\"francoispatry/histo-$O\"" || echo ""); DS=$([ -n "$LAMES" ] && echo "\"$LAMES\"" || echo "")
cat > "$DIR/kernel-metadata.json" <<J
{"id": "francoispatry/histo-$K", "title": "histo-$K", "code_file": "histo.py", "language": "python", "kernel_type": "script", "is_private": true,
 "enable_gpu": $([ "$GPU" = 1 ] && echo true || echo false), "enable_internet": true, "dataset_sources": [$DS], "kernel_sources": [$SRC],
 "competition_sources": ["histopathologic-cancer-detection"]}
J
head -1 "$DIR/histo.py"; cat "$DIR/kernel-metadata.json"
if [ "$GPU" = 1 ]; then "${KAGGLE:-$RACINE/arc3/.venv/bin/kaggle}" kernels push -p "$DIR" --accelerator NvidiaTeslaT4
else "${KAGGLE:-$RACINE/arc3/.venv/bin/kaggle}" kernels push -p "$DIR"; fi
