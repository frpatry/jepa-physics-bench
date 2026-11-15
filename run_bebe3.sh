#!/bin/bash
# BÉBÉ v3 — décision user du 15 nov. 2026 (après la sonde de mouvement : l'œil de l'ancien bébé à 60k suffit pour une imagination par objet à 53 %, celui du bébé v2 plafonne
# à 36 % parce que ses latents tremblent plus au repos). On repart de ZÉRO avec la recette v2 corrigée :
#   - objets PETITS (taille 1, celle du T de Push-T) : plus de --obj_scale 1.8 / --scale1_share
#   - main JAMAIS effacée de l'image (--hand_hidden 0) ; un sens peut manquer (yeux fermés, un seul sens), jamais d'incohérence
#   - objets lancés seuls REMIS (réglage du monde, comme l'ancien bébé) ; rappels d'anciennes étapes 30 % (--replay 0.3, comme l'ancien bébé)
#   - sens imaginés par l'imagination par objet : seulement quand le critère des objets est atteint (--oi_sens_from à donner à la reprise), pas à un pas fixe
# REPRISE : relancer la même commande.   bash run_bebe3.sh [ckpt] [total] [options en plus…]
#   étapes : 0a bras –6k | 0b mains –14k | 0c mobile –22k | 0d contact –34k | 0e vue nette, jeu libre –60k (monde figé ensuite) | imaginer les objets : 60k → critère
CKPT=${1:-/content/drive/MyDrive/jepa_runs/bebe3.pt}; TOTAL=${2:-110000}; shift 2 2>/dev/null
cd "$(dirname "$0")"
python -u av_phase0.py --ckpt "$CKPT" --total "$TOTAL" --budgets 6000,8000,8000,12000 --world_end 60000 \
  --touch_masks 1 --contact_w 4 --p_obj 0.8 --mob_size 0.16 --calm 2 --outside 1 --parent 0.4 --inv_clip 1 \
  --ctx_w 0.5 --ctx_on enc --causal 1 --fut_share 0.7 --blind 1 --curious 3 --echo 0.5 --damp 0.5 \
  --eyes_share 0.15 --replay 0.3 --motion_w 1.0 --shape_phys 1 --roll_w 1.0 --roll_k 3 --roll_obj 4 \
  --hand_hidden 0 --solo_share 0.1 --comove_samecol 0.3 \
  --oi_w 1.0 --oi_from 60000 --oi_warm 6000 --oi_grow 14000 --oi_kmax 12 --oi_sens_from 999999 --oi_sens_ramp 10000 --oi_nce 0.5 "$@"
