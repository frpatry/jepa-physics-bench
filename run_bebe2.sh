#!/bin/bash
# BÉBÉ v2 — entraînement complet à zéro, graduel (conception_bebe_v2.md). REPRISE : relancer la même commande (l'état est sur le Drive, tous les 2500 pas).
#   bash run_bebe2.sh [ckpt] [total] [options en plus…]
#   étapes : ventre 0–4k | 0a bras –10k | 0b mains –18k | 0c mobile –26k | 0d contact –38k | 0e vue nette, jeu libre –60k (monde figé ensuite)
#            5 imaginer les objets : 60k–80k (échauffement 6k, puis horizon 1 → 12) | 6 imaginer avec tous les sens : 80k → fin
CKPT=${1:-/content/drive/MyDrive/jepa_runs/bebe2.pt}; TOTAL=${2:-110000}; shift 2 2>/dev/null
cd "$(dirname "$0")"
python -u av_phase0.py --ckpt "$CKPT" --total "$TOTAL" --womb 4000 --budgets 10000,8000,8000,12000 --world_end 60000 \
  --touch_masks 1 --contact_w 4 --p_obj 0.8 --mob_size 0.16 --calm 2 --outside 1 --parent 0.4 --inv_clip 1 \
  --ctx_w 0.5 --ctx_on enc --causal 1 --fut_share 0.7 --blind 1 --curious 3 --echo 0.5 --damp 0.5 \
  --eyes_share 0.15 --replay 0.1 --motion_w 1.0 --shape_phys 1 --roll_w 1.0 --roll_k 3 --roll_obj 4 \
  --obj_scale 1.8 --scale1_share 0.5 --hand_hidden 0 --solo_share 0.1 --p_auto 0 --comove_samecol 0.3 \
  --oi_w 1.0 --oi_from 60000 --oi_warm 6000 --oi_grow 14000 --oi_kmax 12 --oi_sens_from 80000 --oi_sens_ramp 10000 --oi_nce 0.5 "$@"
