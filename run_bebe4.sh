#!/bin/bash
# BÉBÉ v4 — décision user du 17 nov. 2026 (« garbage in, garbage out ») : repartir de ZÉRO avec un JEU DE POUSSÉE.
# Constat (bébé v3c gelé à 96k, parties Push-T jouées par l'utilisateur) : l'œil situe bien l'objet (0.8 px) et la main, l'imagination suit la main, mais le T poussé
# n'avance presque pas dans son imagination. Mesuré dans son jeu libre : 11 % seulement du mouvement des objets venait de LUI (64 % du parent, 25 % d'objets lancés),
# contacts de 3 images, main rapide — l'inverse de Push-T (poussées lentes, longues, toutes causées par la main).
# Recette = bébé v3c (œil calme : sans --motion_w / --roll_w / --solo_share) avec :
#   --push_play 1 --blind 0 --no_pause 1 --p_obj 0.5 --parent 0.03 : presque rien ne bouge sans lui ; traits lents et longs vers les objets, à vide et au contact
#   --oi_sens_from 60000 --oi_sens_ramp 10000 : UNE seule façon d'interroger l'imagination (ses vrais sens au 1er pas, puis ses sens imaginés) dès qu'elle démarre —
#     avant, l'entraînement ne donnait pas les sens que l'examen et le jeu donnaient (amplitude mesurée 53 % au lieu de 17 %)
# REPRISE : relancer la même commande.   bash run_bebe4.sh [ckpt] [total] [options en plus…]
CKPT=${1:-/content/drive/MyDrive/jepa_runs/bebe4.pt}; TOTAL=${2:-110000}; shift 2 2>/dev/null
cd "$(dirname "$0")"
python -u av_phase0.py --ckpt "$CKPT" --total "$TOTAL" --budgets 6000,8000,8000,12000 --world_end 60000 \
  --touch_masks 1 --contact_w 4 --p_obj 0.5 --mob_size 0.16 --calm 2 --outside 1 --parent 0.03 --inv_clip 1 \
  --ctx_w 0.5 --ctx_on enc --causal 1 --fut_share 0.7 --blind 0 --no_pause 1 --push_play 1 --curious 3 --echo 0.5 --damp 0.5 \
  --eyes_share 0.15 --replay 0.3 --motion_w 0 --shape_phys 1 --roll_w 0 --roll_k 3 --roll_obj 4 \
  --hand_hidden 0 --solo_share 0 --comove_samecol 0.3 \
  --oi_w 1.0 --oi_from 60000 --oi_warm 6000 --oi_grow 14000 --oi_kmax 12 --oi_sens_from 60000 --oi_sens_ramp 10000 --oi_nce 0.5 "$@"
